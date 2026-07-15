"""Cross-platform fail-fast output locks for trainer artifact roots."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .path_safety import lock_file_path, prepare_physical_output_root


class OutputLockError(RuntimeError):
    pass


class OutputLock:
    def __init__(self, out_dir: str | Path):
        physical = prepare_physical_output_root(out_dir)
        self.identity = physical.identity
        self.display_path = physical.display_path
        self.path = lock_file_path(physical.lock_root)
        self._fd: int | None = None

    def __enter__(self) -> "OutputLock":
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            st_pre: os.stat_result | None = None
            try:
                st_pre = self.path.lstat()
                if _is_link_or_reparse(st_pre) or not stat.S_ISREG(st_pre.st_mode):
                    raise OutputLockError(f"output lock path is not a regular non-link file: {self.path}")
                if getattr(st_pre, "st_nlink", 1) != 1:
                    raise OutputLockError(f"output lock path must not be hardlinked: {self.path}")
            except FileNotFoundError:
                pass
            fd = os.open(str(self.path), flags, 0o600)
            os.set_inheritable(fd, False)
            st = os.fstat(fd)
            if _is_link_or_reparse(st) or not stat.S_ISREG(st.st_mode):
                os.close(fd)
                raise OutputLockError(f"output lock path is not a regular non-link file: {self.path}")
            if st_pre is not None and hasattr(os.path, "samestat") and not os.path.samestat(st_pre, st):
                os.close(fd)
                raise OutputLockError(f"output lock path changed during open: {self.path}")
            if getattr(st, "st_nlink", 1) != 1:
                os.close(fd)
                raise OutputLockError(f"output lock path must not be hardlinked: {self.path}")
            if st.st_size < 1:
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, b"\0")
                os.fsync(fd)
                st = os.fstat(fd)
            if st.st_size != 1:
                os.close(fd)
                raise OutputLockError(f"output lock path must be exactly one byte: {self.path}")
            self._fd = fd
            self._lock_fd(fd)
            return self
        except Exception:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raise

    def __exit__(self, _exc_type, _exc, _tb) -> None:  # type: ignore[no-untyped-def]
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            self._unlock_fd(fd)
        finally:
            os.close(fd)

    def _lock_fd(self, fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise OutputLockError(f"output root is already locked: {self.identity}") from exc
            return
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OutputLockError(f"output root is already locked: {self.identity}") from exc

    def _unlock_fd(self, fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _is_link_or_reparse(st: os.stat_result) -> bool:
    attrs = getattr(st, "st_file_attributes", 0)
    return stat.S_ISLNK(st.st_mode) or bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
