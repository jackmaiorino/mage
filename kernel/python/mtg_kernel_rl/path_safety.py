"""Lexical no-follow helpers for trainer artifact trees."""

from __future__ import annotations

import os
import re
import shutil
import stat
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
OUTPUT_LOCK_FILE_NAME = ".mtg-kernel-train.lock"
_QUARANTINE_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _absolute_lexical(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def canonical_identity(path: str | Path) -> str:
    text = os.path.abspath(os.fspath(path))
    text = os.path.normpath(text)
    if os.name == "nt":
        text = os.path.normcase(text).replace("/", "\\")
    return text


@dataclass(frozen=True)
class PhysicalOutputIdentity:
    """Stable lock identity for one output spelling after physical resolution."""

    identity: str
    lock_root: Path
    display_path: str


@dataclass(frozen=True)
class PathIdentity:
    path: Path
    mode: int
    size: int
    mtime_ns: int
    dev: int
    ino: int
    nlink: int


def _normalized_suffix(parts: tuple[str, ...]) -> str:
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("output path has an unsafe unresolved suffix")
    if os.name == "nt":
        return "\\".join(unicodedata.normalize("NFC", part).casefold() for part in parts)
    return "/".join(unicodedata.normalize("NFC", part) for part in parts)


def _stat_identity(st: os.stat_result) -> str:
    dev = int(getattr(st, "st_dev", 0))
    ino = int(getattr(st, "st_ino", 0))
    if dev == 0 and ino == 0:
        raise ValueError("platform did not expose a stable file identity")
    return f"dev={dev:x}:ino={ino:x}"


def _resolved_existing_path(path: Path) -> str:
    try:
        resolved = os.path.realpath(os.fspath(path))
    except OSError as exc:
        raise ValueError(f"could not resolve physical output path: {path}") from exc
    resolved = os.path.normpath(resolved)
    if os.name == "nt":
        resolved = os.path.normcase(resolved).replace("/", "\\")
    return resolved


def prepare_physical_output_root(path: str | Path) -> PhysicalOutputIdentity:
    """Create and validate the real output root used as the lock namespace.

    Ancestor aliases may converge on the same physical root for cooperating local
    processes. The final output-root component itself must be a real directory,
    not a link or reparse point. This helper intentionally does not defend
    against a hostile concurrent namespace swap outside the cooperating-process
    local-filesystem threat model; artifact reads still use strict no-follow
    validation after the lock is held.
    """

    path_abs = _absolute_lexical(path)
    try:
        st_final = path_abs.lstat()
    except FileNotFoundError:
        os.makedirs(path_abs, exist_ok=True)
        st_final = path_abs.lstat()
    if _is_reparse_or_symlink_stat(st_final):
        raise ValueError(f"output root must not be a direct link/reparse point: {path_abs}")
    if not stat.S_ISDIR(st_final.st_mode):
        raise ValueError(f"output root must be a real directory when it exists: {path_abs}")
    real_root = Path(os.path.realpath(os.fspath(path_abs)))
    real_lstat = real_root.lstat()
    if _is_reparse_or_symlink_stat(real_lstat) or not stat.S_ISDIR(real_lstat.st_mode):
        raise ValueError(f"resolved output root must be a real directory: {real_root}")
    final_stat = real_root.stat()
    final_id = _stat_identity(final_stat)
    final_real = _resolved_existing_path(real_root)
    identity = f"output-root:v3:object={final_id}:real_path={final_real}"
    return PhysicalOutputIdentity(identity=identity, lock_root=real_root, display_path=final_real)


def physical_output_identity(path: str | Path) -> PhysicalOutputIdentity:
    return prepare_physical_output_root(path)


def same_lexical_path(left: str | Path, right: str | Path) -> bool:
    return canonical_identity(left) == canonical_identity(right)


def _is_reparse_or_symlink_stat(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & REPARSE_ATTRIBUTE)


def _reject_relative_parts(parts: tuple[str, ...]) -> None:
    for part in parts:
        if part in ("", ".", ".."):
            raise ValueError(f"invalid artifact relative path component: {part!r}")


def relative_to_root(root: str | Path, path: str | Path) -> Path:
    root_id = canonical_identity(root)
    path_id = canonical_identity(path)
    if os.name == "nt":
        prefix = root_id.rstrip("\\") + "\\"
        if path_id != root_id and not path_id.startswith(prefix):
            raise ValueError(f"path escapes artifact root: {path}")
    else:
        prefix = root_id.rstrip("/") + "/"
        if path_id != root_id and not path_id.startswith(prefix):
            raise ValueError(f"path escapes artifact root: {path}")
    rel_text = os.path.relpath(path_id, root_id)
    rel = Path(rel_text)
    if rel_text == ".":
        return Path()
    _reject_relative_parts(rel.parts)
    return rel


def ensure_no_follow_path(root: str | Path, path: str | Path, *, expected: str = "any", reject_hardlinks: bool = False) -> Path:
    root_abs = _absolute_lexical(root)
    path_abs = _absolute_lexical(path)
    rel = relative_to_root(root_abs, path_abs)
    current = root_abs
    try:
        root_stat = current.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"artifact root does not exist: {root_abs}") from exc
    if _is_reparse_or_symlink_stat(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"artifact root must be a real directory: {root_abs}")
    for part in rel.parts:
        current = current / part
        try:
            st = current.lstat()
        except FileNotFoundError:
            break
        if _is_reparse_or_symlink_stat(st):
            raise ValueError(f"artifact path component must not be a link/reparse point: {current}")
        is_final = current == path_abs
        if is_final:
            if expected == "file" and not stat.S_ISREG(st.st_mode):
                raise ValueError(f"artifact path must be a regular file: {current}")
            if expected == "dir" and not stat.S_ISDIR(st.st_mode):
                raise ValueError(f"artifact path must be a directory: {current}")
            if expected == "any" and not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
                raise ValueError(f"artifact path has unsupported file type: {current}")
            if reject_hardlinks and stat.S_ISREG(st.st_mode) and getattr(st, "st_nlink", 1) > 1:
                raise ValueError(f"artifact file must not be hardlinked: {current}")
        elif not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"artifact path parent must be a directory: {current}")
    return path_abs


def ensure_real_dir(path: str | Path) -> Path:
    path_abs = _absolute_lexical(path)
    st = path_abs.lstat()
    if _is_reparse_or_symlink_stat(st) or not stat.S_ISDIR(st.st_mode):
        raise ValueError(f"expected real directory: {path_abs}")
    return path_abs


def ensure_real_file(root: str | Path, path: str | Path, *, reject_hardlinks: bool = True) -> Path:
    return ensure_no_follow_path(root, path, expected="file", reject_hardlinks=reject_hardlinks)


def ensure_real_child_dir(root: str | Path, path: str | Path) -> Path:
    return ensure_no_follow_path(root, path, expected="dir")


def mkdir_no_follow(path: str | Path, *, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> Path:
    path_abs = _absolute_lexical(path)
    if path_abs == Path(path_abs.anchor):
        st = path_abs.lstat()
        if _is_reparse_or_symlink_stat(st) or not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"artifact directory anchor is not real: {path_abs}")
        return path_abs
    if not parents:
        try:
            path_abs.mkdir(mode=mode)
        except FileExistsError:
            if not exist_ok:
                raise
        st = path_abs.lstat()
        if _is_reparse_or_symlink_stat(st) or not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"created artifact directory is not real: {path_abs}")
        return path_abs
    anchor = Path(path_abs.anchor)
    if not anchor.exists():
        raise FileNotFoundError(f"artifact path anchor does not exist: {anchor}")
    anchor_st = anchor.lstat()
    if _is_reparse_or_symlink_stat(anchor_st) or not stat.S_ISDIR(anchor_st.st_mode):
        raise ValueError(f"artifact path anchor is not a real directory: {anchor}")
    current = anchor
    parts = path_abs.parts[1:] if path_abs.anchor else path_abs.parts
    for index, part in enumerate(parts):
        if part in ("", ".", ".."):
            raise ValueError(f"invalid artifact directory component: {part!r}")
        current = current / part
        is_final = index == len(parts) - 1
        try:
            st = current.lstat()
        except FileNotFoundError:
            if not parents and not is_final:
                raise
            os.mkdir(current, mode if is_final else 0o777)
            st = current.lstat()
        else:
            if is_final and not exist_ok:
                raise FileExistsError(f"artifact directory already exists: {current}")
        if _is_reparse_or_symlink_stat(st) or not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"artifact directory component is not real: {current}")
    st = path_abs.lstat()
    if _is_reparse_or_symlink_stat(st) or not stat.S_ISDIR(st.st_mode):
        raise ValueError(f"created artifact directory is not real: {path_abs}")
    return path_abs


def lock_file_path(lock_root: str | Path) -> Path:
    root = ensure_real_dir(lock_root)
    return root / OUTPUT_LOCK_FILE_NAME


def _capture_lstat_identity(path: str | Path) -> PathIdentity:
    path_abs = _absolute_lexical(path)
    st = path_abs.lstat()
    if _is_reparse_or_symlink_stat(st):
        raise ValueError(f"artifact path must not be a link/reparse point: {path_abs}")
    if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
        raise ValueError(f"artifact path has unsupported file type: {path_abs}")
    return PathIdentity(
        path=path_abs,
        mode=int(st.st_mode),
        size=int(st.st_size),
        mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
        dev=int(getattr(st, "st_dev", 0)),
        ino=int(getattr(st, "st_ino", 0)),
        nlink=int(getattr(st, "st_nlink", 1)),
    )


def revalidate_path_identity(identity: PathIdentity) -> None:
    current = _capture_lstat_identity(identity.path)
    if (
        current.mode != identity.mode
        or current.size != identity.size
        or current.mtime_ns != identity.mtime_ns
        or current.dev != identity.dev
        or current.ino != identity.ino
        or current.nlink != identity.nlink
    ):
        raise ValueError(f"artifact source changed before mutation: {identity.path}")


def capture_path_identity(path: str | Path) -> PathIdentity:
    return _capture_lstat_identity(path)


def validate_output_lock_file(path: str | Path) -> Path:
    path_abs = _absolute_lexical(path)
    if path_abs.name != OUTPUT_LOCK_FILE_NAME:
        raise ValueError(f"unexpected output lock file name: {path_abs.name}")
    st = path_abs.lstat()
    if _is_reparse_or_symlink_stat(st) or not stat.S_ISREG(st.st_mode):
        raise ValueError(f"output lock path is not a regular non-link file: {path_abs}")
    if int(getattr(st, "st_nlink", 1)) != 1:
        raise ValueError(f"output lock file must not be hardlinked: {path_abs}")
    if int(st.st_size) != 1:
        raise ValueError(f"output lock file must be exactly one byte: {path_abs}")
    return path_abs


def filesystem_file_identity(path: str | Path) -> tuple[Any, ...]:
    path_abs = _absolute_lexical(path)
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class FILE_ID_128(ctypes.Structure):
                _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

            class FILE_ID_INFO(ctypes.Structure):
                _fields_ = [("VolumeSerialNumber", ctypes.c_ulonglong), ("FileId", FILE_ID_128)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            CreateFileW = kernel32.CreateFileW
            CreateFileW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            CreateFileW.restype = wintypes.HANDLE
            GetFileInformationByHandleEx = kernel32.GetFileInformationByHandleEx
            GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD]
            GetFileInformationByHandleEx.restype = wintypes.BOOL
            CloseHandle = kernel32.CloseHandle
            CloseHandle.argtypes = [wintypes.HANDLE]
            CloseHandle.restype = wintypes.BOOL
            handle = CreateFileW(
                str(path_abs),
                0,
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,
                0x02000000,
                None,
            )
            if handle == wintypes.HANDLE(-1).value:
                raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path_abs}")
            try:
                info = FILE_ID_INFO()
                if not GetFileInformationByHandleEx(handle, 18, ctypes.byref(info), ctypes.sizeof(info)):
                    raise OSError(ctypes.get_last_error(), f"GetFileInformationByHandleEx failed for {path_abs}")
                return ("windows-file-id", int(info.VolumeSerialNumber), bytes(info.FileId.Identifier).hex())
            finally:
                CloseHandle(handle)
        except Exception:
            pass
    st = path_abs.stat()
    return ("stat", int(getattr(st, "st_dev", 0)), int(getattr(st, "st_ino", 0)))


def is_verified_output_lock_entry(root: str | Path, entry: os.DirEntry[str] | Path) -> bool:
    name = entry.name if isinstance(entry, os.DirEntry) else Path(entry).name
    if name != OUTPUT_LOCK_FILE_NAME:
        return False
    path = Path(entry.path) if isinstance(entry, os.DirEntry) else Path(entry)
    ensure_no_follow_path(root, path, expected="file", reject_hardlinks=True)
    validate_output_lock_file(path)
    return True


def scandir_no_follow(path: str | Path) -> list[os.DirEntry[str]]:
    with os.scandir(path) as it:
        entries = list(it)
    for entry in entries:
        st = entry.stat(follow_symlinks=False)
        if _is_reparse_or_symlink_stat(st):
            raise ValueError(f"artifact tree contains link/reparse entry: {entry.name}")
    return entries


def _validate_quarantine_reason(reason: str) -> str:
    if type(reason) is not str or _QUARANTINE_REASON_RE.fullmatch(reason) is None:
        raise ValueError("quarantine reason must be one safe ASCII component")
    return reason


def _snapshot_tree_no_follow(path: Path) -> tuple[PathIdentity, ...]:
    root_identity = _capture_lstat_identity(path)
    snapshots = [root_identity]
    if stat.S_ISREG(root_identity.mode):
        return tuple(snapshots)
    entries = scandir_no_follow(path)
    for entry in sorted(entries, key=lambda item: item.name):
        child = Path(entry.path)
        child_identity = _capture_lstat_identity(child)
        if stat.S_ISDIR(child_identity.mode):
            snapshots.extend(_snapshot_tree_no_follow(child))
        elif stat.S_ISREG(child_identity.mode):
            snapshots.append(child_identity)
        else:
            raise ValueError(f"quarantine source contains unsupported entry: {child}")
    return tuple(snapshots)


def _revalidate_tree_snapshot(snapshot: tuple[PathIdentity, ...]) -> None:
    for identity in snapshot:
        revalidate_path_identity(identity)


def atomic_quarantine(root: str | Path, path: str | Path, reason: str) -> Path:
    reason = _validate_quarantine_reason(reason)
    root_abs = ensure_real_dir(root)
    path_abs = ensure_no_follow_path(root_abs, path, expected="any", reject_hardlinks=False)
    rel = relative_to_root(root_abs, path_abs)
    source_snapshot = _snapshot_tree_no_follow(path_abs)
    quarantine_root = root_abs / ".quarantine"
    mkdir_no_follow(quarantine_root, mode=0o700, parents=False, exist_ok=True)
    ensure_no_follow_path(root_abs, quarantine_root, expected="dir")
    quarantine_batch = quarantine_root / f"{reason}-{time.monotonic_ns()}"
    quarantine = quarantine_batch / rel
    relative_to_root(quarantine_root, quarantine)
    mkdir_no_follow(quarantine.parent, parents=True, exist_ok=True)
    ensure_no_follow_path(quarantine_root, quarantine.parent, expected="dir")
    _revalidate_tree_snapshot(source_snapshot)
    os.replace(path_abs, quarantine)
    return quarantine


def remove_tree_no_follow(path: str | Path) -> None:
    path_abs = _absolute_lexical(path)
    try:
        st = path_abs.lstat()
    except FileNotFoundError:
        return
    if _is_reparse_or_symlink_stat(st):
        raise ValueError(f"refusing to remove link/reparse tree root: {path_abs}")
    if not stat.S_ISDIR(st.st_mode):
        path_abs.unlink()
        return
    with os.scandir(path_abs) as it:
        entries = list(it)
    for entry in entries:
        child = path_abs / entry.name
        child_st = entry.stat(follow_symlinks=False)
        if _is_reparse_or_symlink_stat(child_st):
            raise ValueError(f"refusing to remove link/reparse entry: {child}")
        if stat.S_ISDIR(child_st.st_mode):
            remove_tree_no_follow(child)
        elif stat.S_ISREG(child_st.st_mode):
            os.unlink(child)
        else:
            raise ValueError(f"refusing to remove unsupported artifact entry: {child}")
    os.rmdir(path_abs)


def copytree_no_symlinks(src: str | Path, dst: str | Path) -> None:
    def ignore_links(dir_name: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            path = Path(dir_name) / name
            try:
                st = path.lstat()
            except FileNotFoundError:
                continue
            if _is_reparse_or_symlink_stat(st):
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, symlinks=False, ignore=ignore_links)
