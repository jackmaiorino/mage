"""Cross-platform fail-fast output locks for trainer artifact roots."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from .path_safety import canonical_identity


class OutputLockError(RuntimeError):
    pass


class OutputLock:
    def __init__(self, out_dir: str | Path):
        self.identity = canonical_identity(out_dir)
        self.digest = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()
        parent = Path(os.path.abspath(os.fspath(out_dir))).parent
        parent.mkdir(parents=True, exist_ok=True)
        self.path = parent / f".mtg-kernel-train-{self.digest}.lock"
        self._fd: int | None = None

    def __enter__(self) -> "OutputLock":
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            try:
                st_pre = self.path.lstat()
                if _is_link_or_reparse(st_pre) or not stat.S_ISREG(st_pre.st_mode):
                    raise OutputLockError(f"output lock path is not a regular non-link file: {self.path}")
            except FileNotFoundError:
                pass
            fd = os.open(str(self.path), flags, 0o600)
            os.set_inheritable(fd, False)
            st = os.fstat(fd)
            if _is_link_or_reparse(st) or not stat.S_ISREG(st.st_mode):
                os.close(fd)
                raise OutputLockError(f"output lock path is not a regular non-link file: {self.path}")
            if st.st_size < 1:
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, b"\0")
                os.fsync(fd)
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
