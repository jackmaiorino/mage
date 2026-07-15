"""Lexical no-follow helpers for trainer artifact trees."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path


REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


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
    lock_parent: Path
    display_path: str


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


def _find_existing_physical_anchor(path_abs: Path) -> tuple[Path, tuple[str, ...]]:
    missing: list[str] = []
    current = path_abs
    while True:
        try:
            st = current.lstat()
        except FileNotFoundError:
            missing.append(current.name)
            parent = current.parent
            if parent == current:
                raise ValueError(f"could not find existing physical ancestor for {path_abs}")
            current = parent
            continue
        if _is_reparse_or_symlink_stat(st):
            resolved = Path(os.path.realpath(os.fspath(current)))
            resolved_st = resolved.lstat()
            if _is_reparse_or_symlink_stat(resolved_st) or not stat.S_ISDIR(resolved_st.st_mode):
                raise ValueError(f"output path unresolved ancestor could not be resolved physically: {current}")
            return resolved, tuple(reversed(missing))
        if not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"nearest existing output ancestor must be a real directory: {current}")
        return current, tuple(reversed(missing))


def physical_output_identity(path: str | Path) -> PhysicalOutputIdentity:
    """Return a stable lock identity that follows physical aliases for locking only.

    Lexical artifact containment and resume validation remain separate and strict.
    """

    path_abs = _absolute_lexical(path)
    try:
        st_final = path_abs.lstat()
    except FileNotFoundError:
        anchor, suffix = _find_existing_physical_anchor(path_abs)
        anchor_stat = anchor.stat()
        anchor_id = _stat_identity(anchor_stat)
        anchor_real = _resolved_existing_path(anchor)
        suffix_text = _normalized_suffix(suffix)
        identity = f"missing-output:v2:anchor={anchor_id}:anchor_path={anchor_real}:suffix={suffix_text}"
        return PhysicalOutputIdentity(identity=identity, lock_parent=anchor, display_path=f"{anchor_real}{os.sep}{suffix_text}")
    if _is_reparse_or_symlink_stat(st_final):
        raise ValueError(f"output root must not be a direct link/reparse point: {path_abs}")
    if not stat.S_ISDIR(st_final.st_mode):
        raise ValueError(f"output root must be a real directory when it exists: {path_abs}")
    final_stat = path_abs.stat()
    final_id = _stat_identity(final_stat)
    final_real = _resolved_existing_path(path_abs)
    parent = Path(os.path.realpath(os.fspath(path_abs.parent)))
    parent_stat = parent.lstat()
    if _is_reparse_or_symlink_stat(parent_stat) or not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError(f"resolved output lock parent is not a real directory: {parent}")
    identity = f"existing-output:v2:object={final_id}:real_path={final_real}"
    return PhysicalOutputIdentity(identity=identity, lock_parent=parent, display_path=final_real)


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


def lock_file_path(lock_parent: str | Path, identity: str) -> Path:
    parent = mkdir_no_follow(Path(lock_parent) / ".mtg-kernel-train-locks", mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return parent / f"{digest}.lock"


def scandir_no_follow(path: str | Path) -> list[os.DirEntry[str]]:
    with os.scandir(path) as it:
        entries = list(it)
    for entry in entries:
        st = entry.stat(follow_symlinks=False)
        if _is_reparse_or_symlink_stat(st):
            raise ValueError(f"artifact tree contains link/reparse entry: {entry.name}")
    return entries


def atomic_quarantine(root: str | Path, path: str | Path, reason: str) -> Path:
    root_abs = ensure_real_dir(root)
    path_abs = ensure_no_follow_path(root_abs, path, expected="any", reject_hardlinks=False)
    rel = relative_to_root(root_abs, path_abs)
    quarantine_root = root_abs / ".quarantine"
    mkdir_no_follow(quarantine_root, mode=0o700, parents=False, exist_ok=True)
    ensure_no_follow_path(root_abs, quarantine_root, expected="dir")
    quarantine = quarantine_root / f"{reason}-{time.monotonic_ns()}" / rel
    mkdir_no_follow(quarantine.parent, parents=True, exist_ok=True)
    ensure_no_follow_path(root_abs, quarantine.parent, expected="dir")
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
