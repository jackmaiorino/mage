"""Canonical JSON and atomic artifact helpers for kernel RL training."""

from __future__ import annotations

import errno
import os
import stat
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DerivedCachePreflight:
    out_dir: Path
    episodes: "_AppendTargetIdentity"
    updates: "_AppendTargetIdentity"
    summary: "_AppendTargetIdentity"


@dataclass(frozen=True)
class _AppendTargetIdentity:
    path: Path
    mode: int
    size: int
    mtime_ns: int
    dev: int
    ino: int
    nlink: int

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


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse_or_link_stat(st: os.stat_result) -> bool:
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _require_real_dir(path: str | Path) -> Path:
    path = _absolute(path)
    st = path.lstat()
    if _is_reparse_or_link_stat(st) or not stat.S_ISDIR(st.st_mode):
        raise ValueError(f"derived cache parent must be a real directory: {path}")
    return path


def _identity_from_lstat(path: str | Path) -> _AppendTargetIdentity:
    path = _absolute(path)
    st = path.lstat()
    if _is_reparse_or_link_stat(st) or not stat.S_ISREG(st.st_mode):
        raise ValueError(f"derived cache path must be a regular non-link file: {path}")
    nlink = int(getattr(st, "st_nlink", 1))
    if nlink != 1:
        raise ValueError(f"derived cache path must not be hardlinked: {path}")
    return _AppendTargetIdentity(
        path=path,
        mode=int(st.st_mode),
        size=int(st.st_size),
        mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
        dev=int(getattr(st, "st_dev", 0)),
        ino=int(getattr(st, "st_ino", 0)),
        nlink=nlink,
    )


def _identity_from_stat(path: Path, st: os.stat_result) -> _AppendTargetIdentity:
    if _is_reparse_or_link_stat(st) or not stat.S_ISREG(st.st_mode):
        raise ValueError(f"opened derived cache path must be a regular non-link file: {path}")
    nlink = int(getattr(st, "st_nlink", 1))
    if nlink != 1:
        raise ValueError(f"opened derived cache path must not be hardlinked: {path}")
    return _AppendTargetIdentity(
        path=path,
        mode=int(st.st_mode),
        size=int(st.st_size),
        mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
        dev=int(getattr(st, "st_dev", 0)),
        ino=int(getattr(st, "st_ino", 0)),
        nlink=nlink,
    )


def _same_identity(left: _AppendTargetIdentity, right: _AppendTargetIdentity) -> bool:
    return (
        left.mode == right.mode
        and left.size == right.size
        and left.mtime_ns == right.mtime_ns
        and left.dev == right.dev
        and left.ino == right.ino
        and left.nlink == right.nlink
    )


def _revalidate_append_identity(identity: _AppendTargetIdentity) -> None:
    current = _identity_from_lstat(identity.path)
    if not _same_identity(identity, current):
        raise ValueError(f"derived cache path changed before append: {identity.path}")


def preflight_derived_cache_append(out_dir: str | Path) -> DerivedCachePreflight:
    out = _require_real_dir(out_dir)
    _require_real_dir(out.parent)
    return DerivedCachePreflight(
        out_dir=out,
        episodes=_identity_from_lstat(out / "episodes.jsonl"),
        updates=_identity_from_lstat(out / "updates.jsonl"),
        summary=_identity_from_lstat(out / "summary.json"),
    )


def _open_append_no_follow(identity: _AppendTargetIdentity, *, boundary_prefix: str) -> int:
    _revalidate_append_identity(identity)
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    inject_fault(f"{boundary_prefix}_open_before", identity.path)
    fd = os.open(str(identity.path), flags)
    try:
        os.set_inheritable(fd, False)
        opened = _identity_from_stat(identity.path, os.fstat(fd))
        if not _same_identity(identity, opened):
            raise ValueError(f"derived cache descriptor identity mismatch: {identity.path}")
        post_open = _identity_from_lstat(identity.path)
        if not _same_identity(identity, post_open):
            raise ValueError(f"derived cache path identity changed during open: {identity.path}")
        inject_fault(f"{boundary_prefix}_open_after", identity.path)
        return fd
    except BaseException:
        try:
            os.close(fd)
        except BaseException:
            pass
        raise


def _write_all(fd: int, data: bytes, path: Path) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError(f"short write made no progress: {path}")
        offset += written


def _append_jsonl_no_read(identity: _AppendTargetIdentity, data: bytes, *, boundary_prefix: str) -> None:
    fd = _open_append_no_follow(identity, boundary_prefix=boundary_prefix)
    try:
        inject_fault(f"{boundary_prefix}_write_before", identity.path)
        _write_all(fd, data, identity.path)
        inject_fault(f"{boundary_prefix}_write_after", identity.path)
        inject_fault(f"{boundary_prefix}_fsync_before", identity.path)
        os.fsync(fd)
        inject_fault(f"{boundary_prefix}_fsync_after", identity.path)
    finally:
        os.close(fd)
    fsync_dir(identity.path.parent)


def _seat_counter_sum(payload: dict[str, Any], counter: str) -> int:
    outcomes = payload["outcomes_by_learner_seat"]
    return int(outcomes["p0"][counter]) + int(outcomes["p1"][counter])


def _incremental_summary(*, latest: dict[str, Any], checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    completed_update = int(checkpoint_payload["completed_update"])
    if completed_update != latest["update"]:
        raise ValueError("incremental summary update mismatch")
    learner_wins = _seat_counter_sum(checkpoint_payload, "win")
    learner_losses = _seat_counter_sum(checkpoint_payload, "loss")
    draws = _seat_counter_sum(checkpoint_payload, "draw")
    episodes = int(checkpoint_payload["next_episode"])
    if episodes != learner_wins + learner_losses + draws:
        raise ValueError("checkpoint episode aggregate mismatch")
    learner_decisions = int(checkpoint_payload["learner_decisions_by_seat"]["p0"]) + int(checkpoint_payload["learner_decisions_by_seat"]["p1"])
    summary = {
        "schema": "kernel_rl_train_summary/v2",
        "run_digest": latest["run_digest"],
        "head_update": latest["update"],
        "head": latest["head"],
        "generations": completed_update + 1,
        "completed_training_updates": completed_update,
        "episodes": episodes,
        "learner_wins": learner_wins,
        "learner_losses": learner_losses,
        "draws": draws,
        "learner_decisions": learner_decisions,
        "optimizer_steps": int(checkpoint_payload["optimizer_step_count"]),
    }
    validate_training_json_privacy(summary)
    return summary


def append_current_derived_caches(
    out_dir: str | Path,
    *,
    current_record: dict[str, Any],
    latest: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    preflight: DerivedCachePreflight | None = None,
) -> None:
    out = _absolute(out_dir)
    if preflight is None:
        preflight = preflight_derived_cache_append(out)
    if preflight.out_dir != out:
        raise ValueError("derived cache preflight root mismatch")
    validate_training_json_privacy(current_record)
    validate_training_json_privacy(latest)
    if current_record["update"] != latest["update"]:
        raise ValueError("derived cache update mismatch")
    if current_record["run_digest"] != latest["run_digest"]:
        raise ValueError("derived cache run digest mismatch")
    if checkpoint_payload["next_episode"] != current_record["episode_end_exclusive"]:
        raise ValueError("derived cache episode range mismatch")
    episode_bytes = b""
    for row in current_record["episode_summaries"]:
        validate_training_json_privacy(row)
        episode_bytes += canonical_json_bytes(row)
    update_bytes = canonical_json_bytes(current_record)
    summary = _incremental_summary(latest=latest, checkpoint_payload=checkpoint_payload)
    summary_bytes = canonical_json_bytes(summary)

    _revalidate_append_identity(preflight.episodes)
    _revalidate_append_identity(preflight.updates)
    _revalidate_append_identity(preflight.summary)
    _append_jsonl_no_read(preflight.episodes, episode_bytes, boundary_prefix="derived_episodes_append")
    _append_jsonl_no_read(preflight.updates, update_bytes, boundary_prefix="derived_updates_append")
    _revalidate_append_identity(preflight.summary)
    inject_fault("derived_summary_publish_before", preflight.summary.path)
    write_bytes_atomic(preflight.summary.path, summary_bytes)
    inject_fault("derived_summary_publish_after", preflight.summary.path)


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
