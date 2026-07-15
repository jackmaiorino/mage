"""Canonical JSON and atomic artifact helpers for kernel RL training."""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path
from typing import Any, Callable

from .artifact_io import (
    FORBIDDEN_TRAINING_JSON_KEYS,
    canonical_json_bytes,
    read_authoritative_json,
    read_json_file,
    sha256_bytes,
    sha256_file,
    validate_training_json_privacy,
)
from .path_safety import is_verified_output_lock_entry, mkdir_no_follow, scandir_no_follow

FaultInjector = Callable[[str, Path | None], None]
_FAULT_INJECTOR: FaultInjector | None = None

def set_fault_injector(injector: FaultInjector | None) -> FaultInjector | None:
    global _FAULT_INJECTOR
    previous = _FAULT_INJECTOR
    _FAULT_INJECTOR = injector
    return previous


def inject_fault(boundary: str, path: str | Path | None = None) -> None:
    if _FAULT_INJECTOR is not None:
        _FAULT_INJECTOR(boundary, None if path is None else Path(path))


def fsync_dir(path: str | Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_replace(src: Path, dst: Path, *, attempts: int = 6) -> None:
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) in (5, 32) and i + 1 < attempts:
                time.sleep(0.05 * (2**i))
                continue
            if exc.errno in (errno.EACCES, errno.EBUSY) and i + 1 < attempts:
                time.sleep(0.05 * (2**i))
                continue
            raise


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")


def write_bytes_atomic(path: str | Path, data: bytes) -> None:
    path = Path(path)
    mkdir_no_follow(path.parent, parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with tmp.open("xb") as fh:
            fh.write(data)
            inject_fault("json_save", tmp)
            fh.flush()
            inject_fault("json_flush", tmp)
            os.fsync(fh.fileno())
            inject_fault("json_fsync", tmp)
        inject_fault("json_replace_before", path)
        atomic_replace(tmp, path)
        fsync_dir(path.parent)
        inject_fault("json_replace_after", path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> str:
    data = canonical_json_bytes(value)
    write_bytes_atomic(path, data)
    parsed = read_json_file(path)
    if parsed != value:
        raise ValueError(f"roundtrip JSON mismatch for {path}")
    return sha256_bytes(data)


def require_new_or_empty_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise FileExistsError(f"{path} exists and is not a directory")
        if any(not is_verified_output_lock_entry(path, entry) for entry in scandir_no_follow(path)):
            raise FileExistsError("fresh training output directory must be new or empty")
    mkdir_no_follow(path, parents=True, exist_ok=True)
    return path


def generation_paths(out_dir: str | Path, update: int) -> dict[str, Path]:
    out = Path(out_dir)
    name = f"update-{update:08d}"
    return {
        "update": out / "updates" / f"{name}.json",
        "checkpoint": out / "checkpoints" / f"{name}.pt",
        "sidecar": out / "checkpoints" / f"{name}.json",
    }


def latest_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "latest.json"


def rebuild_derived_caches(out_dir: str | Path, records: list[dict[str, Any]], latest: dict[str, Any]) -> None:
    out = Path(out_dir)
    validate_training_json_privacy(latest)
    for record in records:
        validate_training_json_privacy(record)
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    summary = {
        "schema": "kernel_rl_train_summary/v2",
        "run_digest": latest["run_digest"],
        "head_update": latest["update"],
        "head": latest["head"],
        "generations": 0,
        "completed_training_updates": latest["update"],
        "episodes": 0,
        "learner_wins": 0,
        "learner_losses": 0,
        "draws": 0,
        "learner_decisions": 0,
        "optimizer_steps": 0,
    }
    for record in records:
        update_rows.append(record)
        summary["generations"] += 1
        if record.get("optimizer_step") is True:
            summary["optimizer_steps"] += 1
        summary["learner_decisions"] += int(record.get("learner_decision_count", 0))
        for row in record.get("episode_summaries", []):
            episode_rows.append(row)
            summary["episodes"] += 1
            if row["learner_return"] == 1:
                summary["learner_wins"] += 1
            elif row["learner_return"] == -1:
                summary["learner_losses"] += 1
            elif row["learner_return"] == 0:
                summary["draws"] += 1
            else:
                raise ValueError("invalid learner_return in update record")
    episodes_text = b"".join(canonical_json_bytes(row) for row in episode_rows)
    updates_text = b"".join(canonical_json_bytes(row) for row in update_rows)
    validate_training_json_privacy(summary)
    write_bytes_atomic(out / "episodes.jsonl", episodes_text)
    write_bytes_atomic(out / "updates.jsonl", updates_text)
    write_json_atomic(out / "summary.json", summary)
