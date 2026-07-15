"""Lexical no-follow helpers for trainer artifact trees."""

from __future__ import annotations

import os
import shutil
import stat
import time
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
    if parents:
        path_abs.mkdir(mode=mode, parents=True, exist_ok=exist_ok)
    else:
        path_abs.mkdir(mode=mode, exist_ok=exist_ok)
    st = path_abs.lstat()
    if _is_reparse_or_symlink_stat(st) or not stat.S_ISDIR(st.st_mode):
        raise ValueError(f"created artifact directory is not real: {path_abs}")
    return path_abs


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
    quarantine = root_abs / ".quarantine" / f"{reason}-{time.monotonic_ns()}" / rel
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
