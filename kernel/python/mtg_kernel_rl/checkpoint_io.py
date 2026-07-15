"""Bounded Torch ZIP checkpoint admission."""

from __future__ import annotations

import dataclasses
import inspect
import io
import pickletools
import re
import zipfile
from pathlib import Path
from typing import Any

import torch

from .artifact_io import CapturedFile, read_regular_file_bytes


MAX_CHECKPOINT_FILE_BYTES = 64 * 1024 * 1024
MAX_TORCH_ZIP_ENTRIES = 512
MAX_TORCH_ZIP_UNCOMPRESSED_BYTES = MAX_CHECKPOINT_FILE_BYTES
MAX_TORCH_ZIP_STORAGE_BYTES = MAX_CHECKPOINT_FILE_BYTES
MAX_TORCH_DATA_PKL_BYTES = 2 * 1024 * 1024
MAX_PICKLE_OPCODES = 100_000
MAX_PICKLE_MEMO_WRITES = 100_000
TORCH_ZIP_ROOT = "archive"

ALLOWED_PICKLE_GLOBALS = {
    "collections OrderedDict",
    "torch BoolStorage",
    "torch BFloat16Storage",
    "torch ByteStorage",
    "torch CharStorage",
    "torch ComplexDoubleStorage",
    "torch ComplexFloatStorage",
    "torch DoubleStorage",
    "torch FloatStorage",
    "torch HalfStorage",
    "torch IntStorage",
    "torch LongStorage",
    "torch ShortStorage",
    "torch._utils _rebuild_tensor_v2",
}

ALLOWED_METADATA_MEMBERS = {
    "data.pkl",
    "byteorder",
    "version",
    ".data/serialization_id",
    ".format_version",
    ".storage_alignment",
}

_STORAGE_MEMBER_RE = re.compile(r"^data/(\d+)$")


@dataclasses.dataclass(frozen=True)
class TorchZipPreflight:
    entries: int
    storage_entries: int
    total_uncompressed_bytes: int
    total_storage_bytes: int
    data_pkl_bytes: int
    pickle_opcodes: int


@dataclasses.dataclass(frozen=True)
class LoadedCheckpoint:
    payload: Any
    sha256: str
    size: int
    preflight: TorchZipPreflight


def load_torch_zip_checkpoint(path: str | Path) -> LoadedCheckpoint:
    captured = read_regular_file_bytes(path, max_bytes=MAX_CHECKPOINT_FILE_BYTES, allow_empty=False)
    return load_torch_zip_checkpoint_bytes(captured)


def load_torch_zip_checkpoint_bytes(captured: CapturedFile) -> LoadedCheckpoint:
    preflight = preflight_torch_zip(captured.data)
    _require_weights_only_torch_load()
    try:
        payload = torch.load(io.BytesIO(captured.data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("Torch safe checkpoint loading failed") from exc
    return LoadedCheckpoint(payload=payload, sha256=captured.sha256, size=captured.size, preflight=preflight)


def preflight_torch_zip(data: bytes) -> TorchZipPreflight:
    if len(data) <= 0 or len(data) > MAX_CHECKPOINT_FILE_BYTES:
        raise ValueError("checkpoint byte size out of bounds")
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise RuntimeError("checkpoint must use modern Torch ZIP serialization")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            infos = zf.infolist()
            if not infos or len(infos) > MAX_TORCH_ZIP_ENTRIES:
                raise ValueError("checkpoint ZIP entry count out of bounds")
            names = [info.filename for info in infos]
            if len(set(names)) != len(names):
                raise ValueError("checkpoint ZIP contains duplicate member names")
            total_uncompressed = 0
            total_storage = 0
            storage_names: set[str] = set()
            data_pkl: bytes | None = None
            for info in infos:
                parts = info.filename.split("/")
                if len(parts) < 2 or parts[0] != TORCH_ZIP_ROOT:
                    raise ValueError("checkpoint ZIP root mismatch")
                member = "/".join(parts[1:])
                if not member or member.startswith("/") or "\\" in member:
                    raise ValueError("checkpoint ZIP member path is not relative")
                if any(part in ("", ".", "..") for part in member.split("/")):
                    raise ValueError("checkpoint ZIP member path traverses")
                if info.flag_bits & 0x1:
                    raise ValueError("checkpoint ZIP encryption is not allowed")
                if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
                    raise ValueError("checkpoint ZIP compression is not allowed")
                total_uncompressed += int(info.file_size)
                if total_uncompressed > MAX_TORCH_ZIP_UNCOMPRESSED_BYTES:
                    raise ValueError("checkpoint ZIP aggregate bytes exceed limit")
                storage_match = _STORAGE_MEMBER_RE.fullmatch(member)
                if storage_match is not None:
                    storage_name = storage_match.group(1)
                    if storage_name in storage_names:
                        raise ValueError("checkpoint ZIP duplicate storage member")
                    storage_names.add(storage_name)
                    total_storage += int(info.file_size)
                    if total_storage > MAX_TORCH_ZIP_STORAGE_BYTES:
                        raise ValueError("checkpoint ZIP storage bytes exceed limit")
                    continue
                if member not in ALLOWED_METADATA_MEMBERS:
                    raise ValueError(f"checkpoint ZIP member not in storage contract: {member}")
                if member == "data.pkl":
                    if info.file_size > MAX_TORCH_DATA_PKL_BYTES:
                        raise ValueError("checkpoint data.pkl exceeds limit")
                    data_pkl = zf.read(info)
            if data_pkl is None:
                raise ValueError("checkpoint ZIP missing data.pkl")
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"checkpoint ZIP CRC failed for {bad}")
            pickle_counts = _preflight_pickle(data_pkl)
            return TorchZipPreflight(
                entries=len(infos),
                storage_entries=len(storage_names),
                total_uncompressed_bytes=total_uncompressed,
                total_storage_bytes=total_storage,
                data_pkl_bytes=len(data_pkl),
                pickle_opcodes=pickle_counts[0],
            )
    except zipfile.BadZipFile as exc:
        raise RuntimeError("checkpoint must use modern Torch ZIP serialization") from exc


def _preflight_pickle(data: bytes) -> tuple[int, int]:
    opcodes = 0
    memo_writes = 0
    for opcode, arg, _pos in pickletools.genops(data):
        opcodes += 1
        if opcodes > MAX_PICKLE_OPCODES:
            raise ValueError("checkpoint pickle has too many opcodes")
        if opcode.name in {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"}:
            memo_writes += 1
            if memo_writes > MAX_PICKLE_MEMO_WRITES:
                raise ValueError("checkpoint pickle memo exceeds limit")
        if opcode.name == "GLOBAL":
            if arg not in ALLOWED_PICKLE_GLOBALS:
                raise ValueError(f"checkpoint pickle global is not allowed: {arg}")
        elif opcode.name == "STACK_GLOBAL":
            raise ValueError("checkpoint pickle STACK_GLOBAL is not allowed")
        elif opcode.name in {"INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4"}:
            raise ValueError(f"checkpoint pickle opcode is not allowed: {opcode.name}")
    return opcodes, memo_writes


def _require_weights_only_torch_load() -> None:
    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Torch safe checkpoint loading is unavailable") from exc
    if "weights_only" not in signature.parameters:
        raise RuntimeError("Torch safe checkpoint loading is unavailable")
