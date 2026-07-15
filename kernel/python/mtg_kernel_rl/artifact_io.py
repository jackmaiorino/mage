"""Bounded artifact file, JSON, and privacy validation primitives."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable

from .client import ProtocolError, _parse_float, _reject_constant, _reject_duplicate_keys


@dataclasses.dataclass(frozen=True)
class CapturedFile:
    data: bytes
    size: int
    sha256: str


MAX_RUN_JSON_BYTES = 256 * 1024
MAX_SMALL_JSON_BYTES = 16 * 1024
MAX_UPDATE_JSON_BYTES = 16 * 1024 * 1024
MAX_DEFAULT_JSON_BYTES = MAX_UPDATE_JSON_BYTES
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 250_000
MAX_JSON_ITEMS = 250_000
MAX_JSON_STRING_BYTES = 16 * 1024 * 1024
MAX_JSON_ONE_STRING_BYTES = 2 * 1024 * 1024
MAX_JSON_NUMERIC_DIGITS = 128
MAX_JSON_INTEGER_BITS = 64

AUTHORITATIVE_JSON_LIMITS = {
    "run": MAX_RUN_JSON_BYTES,
    "latest": MAX_SMALL_JSON_BYTES,
    "sidecar": MAX_SMALL_JSON_BYTES,
    "update": MAX_UPDATE_JSON_BYTES,
    "summary": MAX_UPDATE_JSON_BYTES,
}

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

_WINDOWS_DRIVE_RE = re.compile(r"(^|[^A-Za-z0-9_])([A-Za-z]:[\\/](?:[^\\/\s\"'<>|]+[\\/]*)*)")
_WINDOWS_EXTENDED_RE = re.compile(r"(^|[\s\"'(<])(?:\\\\[.?]\\|\\\\\?\\|\\\?\\|\\Device\\|//[.?]/|//\?/)", re.IGNORECASE)
_WINDOWS_UNC_RE = re.compile(r"(^|[\s\"'(<])(?:\\\\|//)[^\\/\s:\"'<>|]+[\\/][^\\/\s:\"'<>|]+")
_WINDOWS_ROOT_REL_RE = re.compile(r"(^|[\s\"'(<])\\[^\\/\s:\"'<>|]+[\\/][^\\/\s:\"'<>|]+")
_POSIX_ABSOLUTE_RE = re.compile(
    r"(^|[\s\"'(<:=,])/(?:home|Users|mnt|scratch|tmp|var|etc|opt|usr|root|private|Volumes|workspace)(?:/|$)"
)
_FILE_URI_RE = re.compile(r"file://", re.IGNORECASE)
_NON_FILE_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_reparse_or_link_stat(st: os.stat_result) -> bool:
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _open_regular_no_follow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    pre = path.lstat()
    if _is_reparse_or_link_stat(pre) or not stat.S_ISREG(pre.st_mode):
        raise ValueError(f"artifact file must be a regular non-link file: {path}")
    fd = os.open(str(path), flags)
    os.set_inheritable(fd, False)
    try:
        post = os.fstat(fd)
        if _is_reparse_or_link_stat(post) or not stat.S_ISREG(post.st_mode):
            raise ValueError(f"artifact file must be a regular non-link file: {path}")
        if hasattr(os.path, "samestat") and not os.path.samestat(pre, post):
            raise ValueError(f"artifact file changed during open: {path}")
    except Exception:
        os.close(fd)
        raise
    return fd


def read_regular_file_bytes(path: str | Path, *, max_bytes: int, allow_empty: bool = False) -> CapturedFile:
    path = Path(path)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    fd = _open_regular_no_follow(path)
    try:
        st = os.fstat(fd)
        expected_size = int(st.st_size)
        if expected_size < 0 or expected_size > max_bytes:
            raise ValueError(f"artifact file size out of bounds: {path}")
        if expected_size == 0 and not allow_empty:
            raise ValueError(f"artifact file is empty: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"artifact file exceeded bounded read: {path}")
        if total != expected_size:
            raise ValueError(f"artifact file size changed during read: {path}")
        data = b"".join(chunks)
        return CapturedFile(data=data, size=len(data), sha256=sha256_bytes(data))
    finally:
        os.close(fd)


def sha256_file(path: str | Path, *, max_bytes: int | None = None, allow_empty: bool = True) -> str:
    if max_bytes is None:
        h = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    return read_regular_file_bytes(path, max_bytes=max_bytes, allow_empty=allow_empty).sha256


def _validate_json_tree(value: Any, context: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        if type(value) is int and abs(value).bit_length() > MAX_JSON_INTEGER_BITS:
            raise ValueError(f"JSON integer out of bounds at {context}")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in JSON artifact at {context}")
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _validate_json_tree(item, f"{context}[{i}]")
        return
    if type(value) is dict:
        for key in value:
            if type(key) is not str:
                raise TypeError(f"JSON object key at {context} must be str")
        for key in sorted(value):
            _validate_json_tree(value[key], f"{context}.{key}")
        return
    raise TypeError(f"unsupported JSON artifact type at {context}: {type(value).__name__}")


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    _validate_json_tree(value)
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _preflight_json_bytes(data: bytes) -> None:
    if not data:
        raise ValueError("JSON artifact is empty")
    depth = 0
    max_depth = 0
    in_string = False
    escape = False
    string_bytes = 0
    i = 0
    while i < len(data):
        ch = data[i]
        if in_string:
            if escape:
                escape = False
            elif ch == 0x5C:
                escape = True
            elif ch == 0x22:
                in_string = False
                if string_bytes > MAX_JSON_ONE_STRING_BYTES:
                    raise ValueError("JSON string exceeds per-string limit")
                string_bytes = 0
            else:
                string_bytes += 1
                if string_bytes > MAX_JSON_ONE_STRING_BYTES:
                    raise ValueError("JSON string exceeds per-string limit")
            i += 1
            continue
        if ch == 0x22:
            in_string = True
            string_bytes = 0
        elif ch in (0x7B, 0x5B):
            depth += 1
            max_depth = max(max_depth, depth)
            if max_depth > MAX_JSON_DEPTH:
                raise ValueError("JSON nesting exceeds depth limit")
        elif ch in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise ValueError("JSON nesting underflow")
        elif ch == 0x2D or 0x30 <= ch <= 0x39:
            start = i
            digits = 0
            while i < len(data) and data[i] in b"0123456789+-.eE":
                if 0x30 <= data[i] <= 0x39:
                    digits += 1
                    if digits > MAX_JSON_NUMERIC_DIGITS:
                        raise ValueError("JSON numeric literal has too many digits")
                i += 1
            if i == start:
                i += 1
            continue
        i += 1
    if in_string or depth != 0:
        raise ValueError("JSON artifact has unterminated string or container")


def _parse_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > MAX_JSON_NUMERIC_DIGITS:
        raise ValueError("JSON integer literal has too many digits")
    parsed = int(value)
    if abs(parsed).bit_length() > MAX_JSON_INTEGER_BITS:
        raise ValueError("JSON integer exceeds bit-size limit")
    return parsed


def _parse_json_bytes(data: bytes, path: str | Path) -> dict[str, Any]:
    _preflight_json_bytes(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 JSON artifact {path}: {exc}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            parse_int=_parse_int,
        )
    except ProtocolError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if type(value) is not dict:
        raise ValueError(f"JSON artifact {path} is not an object")
    _validate_json_tree(value)
    _validate_json_aggregate(value)
    return value


def _validate_json_aggregate(value: Any) -> None:
    counters = {"nodes": 0, "items": 0, "string_bytes": 0}

    def walk(item: Any, depth: int, context: str) -> None:
        counters["nodes"] += 1
        if counters["nodes"] > MAX_JSON_NODES:
            raise ValueError("JSON artifact has too many nodes")
        if depth > MAX_JSON_DEPTH:
            raise ValueError("JSON artifact exceeds depth limit")
        if type(item) is str:
            counters["string_bytes"] += len(item.encode("utf-8"))
            if counters["string_bytes"] > MAX_JSON_STRING_BYTES:
                raise ValueError("JSON artifact has too many string bytes")
            return
        if item is None or type(item) in (bool, int, float):
            return
        if type(item) is list:
            counters["items"] += len(item)
            if counters["items"] > MAX_JSON_ITEMS:
                raise ValueError("JSON artifact has too many collection items")
            for index, child in enumerate(item):
                walk(child, depth + 1, f"{context}[{index}]")
            return
        if type(item) is dict:
            counters["items"] += len(item)
            if counters["items"] > MAX_JSON_ITEMS:
                raise ValueError("JSON artifact has too many collection items")
            for key, child in item.items():
                counters["string_bytes"] += len(key.encode("utf-8"))
                if counters["string_bytes"] > MAX_JSON_STRING_BYTES:
                    raise ValueError("JSON artifact has too many string bytes")
                walk(child, depth + 1, f"{context}.{key}")
            return
        raise TypeError(f"unsupported JSON object at {context}: {type(item).__name__}")

    walk(value, 0, "$")


def read_json_file(
    path: str | Path,
    *,
    max_bytes: int = MAX_DEFAULT_JSON_BYTES,
    require_canonical: bool = True,
) -> dict[str, Any]:
    captured = read_regular_file_bytes(path, max_bytes=max_bytes, allow_empty=False)
    value = _parse_json_bytes(captured.data, path)
    if require_canonical and captured.data != canonical_json_bytes(value):
        raise ValueError(f"JSON artifact {path} is not canonical sorted ASCII JSON")
    return value


def read_authoritative_json(path: str | Path, kind: str) -> dict[str, Any]:
    if kind not in AUTHORITATIVE_JSON_LIMITS:
        raise ValueError(f"unknown authoritative JSON kind: {kind}")
    return read_json_file(path, max_bytes=AUTHORITATIVE_JSON_LIMITS[kind], require_canonical=True)


def _looks_like_absolute_path(value: str) -> bool:
    if not value:
        return False
    if _FILE_URI_RE.search(value):
        return True
    if _NON_FILE_URI_RE.match(value):
        return False
    if value in ("/", "\\"):
        return True
    if _WINDOWS_EXTENDED_RE.search(value):
        return True
    if _WINDOWS_DRIVE_RE.search(value):
        return True
    if _WINDOWS_UNC_RE.search(value):
        return True
    if _WINDOWS_ROOT_REL_RE.search(value):
        return True
    if _POSIX_ABSOLUTE_RE.search(value):
        return True
    return False


def validate_training_json_privacy(value: Any, context: str = "$") -> None:
    if value is None or type(value) in (bool, int, float):
        return
    if type(value) is str:
        if _looks_like_absolute_path(value):
            raise ValueError(f"forbidden absolute path string in training artifact at {context}")
        return
    if type(value) is list or type(value) is tuple:
        for i, item in enumerate(value):
            validate_training_json_privacy(item, f"{context}[{i}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"privacy scan key must be str at {context}")
            if key in FORBIDDEN_TRAINING_JSON_KEYS or _looks_like_absolute_path(key):
                raise ValueError(f"forbidden training artifact field {key!r} at {context}")
            validate_training_json_privacy(item, f"{context}.{key}")
        return
    raise TypeError(f"unsupported privacy scan type at {context}: {type(value).__name__}")


def walk_checkpoint_scalar_metadata(value: Any, visitor: Callable[[str, str], None], context: str = "$") -> None:
    if value is None or type(value) in (bool, int, float):
        return
    if type(value) is str:
        visitor(value, context)
        return
    if type(value) is list or type(value) is tuple:
        for i, item in enumerate(value):
            walk_checkpoint_scalar_metadata(item, visitor, f"{context}[{i}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"checkpoint metadata key must be str at {context}")
            visitor(key, f"{context}.<key>")
            walk_checkpoint_scalar_metadata(item, visitor, f"{context}.{key}")
        return
    try:
        import torch

        if type(value) is torch.Tensor:
            return
    except Exception:
        pass
    raise TypeError(f"unsupported checkpoint metadata type at {context}: {type(value).__name__}")
