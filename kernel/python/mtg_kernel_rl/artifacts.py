"""Canonical JSON and atomic artifact helpers for kernel RL training."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from .client import ProtocolError, _reject_duplicate_keys, _reject_constant, _parse_float

FaultInjector = Callable[[str, Path | None], None]
_FAULT_INJECTOR: FaultInjector | None = None

FORBIDDEN_TRAINING_JSON_KEYS = {
    "absolute_path",
    "arena_id",
    "card_name",
    "created_at",
    "display_text",
    "host",
    "hostname",
    "legal_actions",
    "observation",
    "own_hand",
    "path",
    "stable",
    "stable_id",
    "timestamp",
    "ts",
    "updated_at",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_json_tree(value: Any, context: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in JSON artifact at {context}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _validate_json_tree(item, f"{context}[{i}]")
        return
    if isinstance(value, dict):
        for key in value:
            if type(key) is not str:
                raise TypeError(f"JSON object key at {context} must be str")
        for key in sorted(value):
            _validate_json_tree(value[key], f"{context}.{key}")
        return
    raise TypeError(f"unsupported JSON artifact type at {context}: {type(value).__name__}")


def set_fault_injector(injector: FaultInjector | None) -> FaultInjector | None:
    global _FAULT_INJECTOR
    previous = _FAULT_INJECTOR
    _FAULT_INJECTOR = injector
    return previous


def inject_fault(boundary: str, path: str | Path | None = None) -> None:
    if _FAULT_INJECTOR is not None:
        _FAULT_INJECTOR(boundary, None if path is None else Path(path))


def validate_training_json_privacy(value: Any, context: str = "$") -> None:
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is str:
            normalized = value.replace("\\", "/")
            if (
                len(value) >= 3
                and value[1] == ":"
                and value[2] in ("\\", "/")
                and value[0].isalpha()
            ) or normalized.startswith(("/home/", "/Users/", "/mnt/", "/scratch/", "/tmp/")):
                raise ValueError(f"forbidden absolute path string in training artifact at {context}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            validate_training_json_privacy(item, f"{context}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_TRAINING_JSON_KEYS:
                raise ValueError(f"forbidden training artifact field {key!r} at {context}")
            validate_training_json_privacy(item, f"{context}.{key}")
        return


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    _validate_json_tree(value)
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def read_json_file(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except ProtocolError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact {path} is not an object")
    _validate_json_tree(value)
    return value


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with tmp.open("xb") as fh:
            fh.write(data)
            inject_fault("json_save", tmp)
            fh.flush()
            inject_fault("json_flush", tmp)
            os.fsync(fh.fileno())
            inject_fault("json_fsync", tmp)
        atomic_replace(tmp, path)
        fsync_dir(path.parent)
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
        if not path.is_dir():
            raise FileExistsError(f"{path} exists and is not a directory")
        if any(path.iterdir()):
            raise FileExistsError("fresh training output directory must be new or empty")
    path.mkdir(parents=True, exist_ok=True)
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
        "schema": "kernel_rl_train_summary/v1",
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
