"""Bounded Torch ZIP checkpoint admission."""

from __future__ import annotations

import dataclasses
import inspect
import io
import math
import pickletools
import re
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any

import torch

from .artifact_io import CapturedFile, read_regular_file_bytes


MAX_CHECKPOINT_FILE_BYTES = 64 * 1024 * 1024
MAX_TORCH_ZIP_ENTRIES = 512
MAX_TORCH_ZIP_UNCOMPRESSED_BYTES = MAX_CHECKPOINT_FILE_BYTES
MAX_TORCH_ZIP_STORAGE_BYTES = MAX_CHECKPOINT_FILE_BYTES
MAX_TORCH_DATA_PKL_BYTES = 2 * 1024 * 1024
MAX_TORCH_CENTRAL_DIRECTORY_BYTES = 4 * 1024 * 1024
MAX_PICKLE_OPCODES = 100_000
MAX_PICKLE_MEMO_WRITES = 100_000
MAX_PICKLE_STACK_DEPTH = 20_000
MAX_PICKLE_MARKS = 4096
MAX_PICKLE_CONTAINER_ITEMS = 4096
MAX_PICKLE_TOTAL_ITEMS = 100_000
MAX_PICKLE_GRAPH_NODES = 100_000
MAX_PICKLE_GRAPH_DEPTH = 16
MAX_PICKLE_STRING_BYTES = 4 * 1024 * 1024
MAX_PICKLE_TENSORS = 512
MAX_PICKLE_TENSOR_RANK = 16
MAX_PICKLE_TENSOR_ELEMENTS = 20_000_000
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

_STORAGE_DTYPE_BYTES = {
    "torch BoolStorage": ("torch.bool", 1),
    "torch ByteStorage": ("torch.uint8", 1),
    "torch CharStorage": ("torch.int8", 1),
    "torch ShortStorage": ("torch.int16", 2),
    "torch IntStorage": ("torch.int32", 4),
    "torch LongStorage": ("torch.int64", 8),
    "torch HalfStorage": ("torch.float16", 2),
    "torch BFloat16Storage": ("torch.bfloat16", 2),
    "torch FloatStorage": ("torch.float32", 4),
    "torch DoubleStorage": ("torch.float64", 8),
    "torch ComplexFloatStorage": ("torch.complex64", 8),
    "torch ComplexDoubleStorage": ("torch.complex128", 16),
}


@dataclasses.dataclass(frozen=True)
class TorchZipPreflight:
    entries: int
    storage_entries: int
    tensor_declarations: int
    central_directory_bytes: int
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


@dataclasses.dataclass(frozen=True)
class _RawZipEntry:
    filename: str
    filename_bytes: bytes
    member: str
    version_needed: int
    flag_bits: int
    compress_type: int
    mod_time: int
    mod_date: int
    crc: int
    compress_size: int
    file_size: int
    header_offset: int


@dataclasses.dataclass(frozen=True)
class _LocalZipRecord:
    start: int
    header_end: int
    data_start: int
    data_end: int
    descriptor_start: int
    descriptor_end: int
    end: int


@dataclasses.dataclass(frozen=True)
class _RawZipLayout:
    entries: list[_RawZipEntry]
    central_directory_offset: int
    central_directory_bytes: int


@dataclasses.dataclass(frozen=True)
class _GlobalRef:
    name: str


@dataclasses.dataclass(frozen=True)
class _StorageRef:
    key: str
    storage_type: str
    dtype: str
    location: str
    elements: int
    byte_count: int


@dataclasses.dataclass(frozen=True)
class _TensorDecl:
    storage_key: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_count: int


_MARK = object()
_ORDERED_DICT = object()


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
    layout = _raw_zip_preflight(data)
    entries = layout.entries
    if not entries or len(entries) > MAX_TORCH_ZIP_ENTRIES:
        raise ValueError("checkpoint ZIP entry count out of bounds")
    names = [entry.filename for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("checkpoint ZIP contains duplicate member names")
    total_uncompressed = 0
    total_storage = 0
    storage_names: set[str] = set()
    storage_member_sizes: dict[str, int] = {}
    data_pkl: bytes | None = None
    for entry in entries:
        parts = entry.filename.split("/")
        if len(parts) < 2 or parts[0] != TORCH_ZIP_ROOT:
            raise ValueError("checkpoint ZIP root mismatch")
        member = "/".join(parts[1:])
        if not member or member.startswith("/") or "\\" in member:
            raise ValueError("checkpoint ZIP member path is not relative")
        if any(part in ("", ".", "..") for part in member.split("/")):
            raise ValueError("checkpoint ZIP member path traverses")
        if entry.flag_bits & 0x1:
            raise ValueError("checkpoint ZIP encryption is not allowed")
        if entry.compress_type != zipfile.ZIP_STORED or entry.compress_size != entry.file_size:
            raise ValueError("checkpoint ZIP compression is not allowed")
        payload_start, payload_end = _local_member_data_bounds(data, entry, central_directory_offset=layout.central_directory_offset)
        payload = data[payload_start:payload_end]
        if zlib.crc32(payload) & 0xFFFF_FFFF != entry.crc:
            raise ValueError(f"checkpoint ZIP CRC failed for {entry.filename}")
        total_uncompressed += int(entry.file_size)
        if total_uncompressed > MAX_TORCH_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError("checkpoint ZIP aggregate bytes exceed limit")
        storage_match = _STORAGE_MEMBER_RE.fullmatch(member)
        if storage_match is not None:
            storage_name = storage_match.group(1)
            if storage_name in storage_names:
                raise ValueError("checkpoint ZIP duplicate storage member")
            storage_names.add(storage_name)
            storage_member_sizes[storage_name] = int(entry.file_size)
            total_storage += int(entry.file_size)
            if total_storage > MAX_TORCH_ZIP_STORAGE_BYTES:
                raise ValueError("checkpoint ZIP storage bytes exceed limit")
            continue
        if member not in ALLOWED_METADATA_MEMBERS:
            raise ValueError(f"checkpoint ZIP member not in storage contract: {member}")
        if member == "data.pkl":
            if entry.file_size > MAX_TORCH_DATA_PKL_BYTES:
                raise ValueError("checkpoint data.pkl exceeds limit")
            data_pkl = payload
    if data_pkl is None:
        raise ValueError("checkpoint ZIP missing data.pkl")
    pickle_counts = _preflight_pickle(data_pkl, storage_member_sizes=storage_member_sizes)
    return TorchZipPreflight(
        entries=len(entries),
        storage_entries=len(storage_names),
        tensor_declarations=pickle_counts[2],
        central_directory_bytes=layout.central_directory_bytes,
        total_uncompressed_bytes=total_uncompressed,
        total_storage_bytes=total_storage,
        data_pkl_bytes=len(data_pkl),
        pickle_opcodes=pickle_counts[0],
    )


def _raw_zip_preflight(data: bytes) -> _RawZipLayout:
    eocd_pos = _find_single_eocd(data)
    if eocd_pos + 22 > len(data):
        raise ValueError("checkpoint ZIP EOCD is truncated")
    (
        _sig,
        disk_number,
        cd_start_disk,
        entries_this_disk,
        entries_total,
        cd_size_32,
        cd_offset_32,
        _comment_len,
    ) = struct.unpack_from("<IHHHHIIH", data, eocd_pos)
    if disk_number != 0 or cd_start_disk != 0 or entries_this_disk != entries_total:
        raise ValueError("checkpoint ZIP multi-disk archives are not allowed")
    zip64_needed = entries_total == 0xFFFF or cd_size_32 == 0xFFFF_FFFF or cd_offset_32 == 0xFFFF_FFFF
    zip64_present = eocd_pos >= 20 and data[eocd_pos - 20 : eocd_pos - 16] == b"PK\x06\x07"
    if zip64_needed:
        entries_total, cd_size, cd_offset, cd_end_boundary = _read_zip64_eocd(
            data,
            eocd_pos,
            classic_entries_total=entries_total,
            classic_cd_size=cd_size_32,
            classic_cd_offset=cd_offset_32,
        )
    elif zip64_present:
        entries_total, cd_size, cd_offset, cd_end_boundary = _read_zip64_eocd(
            data,
            eocd_pos,
            classic_entries_total=entries_total,
            classic_cd_size=cd_size_32,
            classic_cd_offset=cd_offset_32,
        )
    else:
        cd_size = int(cd_size_32)
        cd_offset = int(cd_offset_32)
        cd_end_boundary = eocd_pos
    if entries_total <= 0 or entries_total > MAX_TORCH_ZIP_ENTRIES:
        raise ValueError("checkpoint ZIP entry count out of bounds")
    if cd_size <= 0 or cd_size > MAX_TORCH_CENTRAL_DIRECTORY_BYTES:
        raise ValueError("checkpoint ZIP central directory byte work exceeds limit")
    if cd_offset < 0 or cd_offset + cd_size != cd_end_boundary or cd_end_boundary > len(data):
        raise ValueError("checkpoint ZIP central directory layout is malformed")
    entries = _parse_central_directory(data, cd_offset=cd_offset, cd_size=cd_size, expected_count=entries_total)
    _validate_local_member_layout(data, entries, central_directory_offset=cd_offset)
    return _RawZipLayout(entries=entries, central_directory_offset=cd_offset, central_directory_bytes=cd_size)


def _find_single_eocd(data: bytes) -> int:
    sig = b"PK\x05\x06"
    start = max(0, len(data) - (0xFFFF + 22))
    candidates: list[int] = []
    pos = data.find(sig, start)
    while pos != -1:
        if pos + 22 <= len(data):
            comment_len = struct.unpack_from("<H", data, pos + 20)[0]
            if comment_len == 0 and pos + 22 == len(data):
                candidates.append(pos)
        pos = data.find(sig, pos + 1)
    if not candidates:
        raise RuntimeError("checkpoint must use modern Torch ZIP serialization")
    if len(candidates) != 1:
        raise ValueError("checkpoint ZIP EOCD is ambiguous")
    return candidates[0]


def _read_zip64_eocd(
    data: bytes,
    eocd_pos: int,
    *,
    classic_entries_total: int,
    classic_cd_size: int,
    classic_cd_offset: int,
) -> tuple[int, int, int, int]:
    locator_pos = eocd_pos - 20
    if locator_pos < 0 or data[locator_pos : locator_pos + 4] != b"PK\x06\x07":
        raise ValueError("checkpoint ZIP64 EOCD locator is missing")
    _sig, disk_with_eocd, zip64_offset, total_disks = struct.unpack_from("<IIQI", data, locator_pos)
    if disk_with_eocd != 0 or total_disks != 1:
        raise ValueError("checkpoint ZIP64 multi-disk archives are not allowed")
    if zip64_offset < 0 or zip64_offset + 56 > locator_pos:
        raise ValueError("checkpoint ZIP64 EOCD offset is malformed")
    if data[zip64_offset : zip64_offset + 4] != b"PK\x06\x06":
        raise ValueError("checkpoint ZIP64 EOCD signature is missing")
    size = struct.unpack_from("<Q", data, zip64_offset + 4)[0]
    if size != 44 or zip64_offset + 12 + size != locator_pos:
        raise ValueError("checkpoint ZIP64 EOCD size is malformed")
    (
        _version_made,
        _version_needed,
        disk_number,
        cd_start_disk,
        entries_this_disk,
        entries_total,
        cd_size,
        cd_offset,
    ) = struct.unpack_from("<HHIIQQQQ", data, zip64_offset + 12)
    if disk_number != 0 or cd_start_disk != 0 or entries_this_disk != entries_total:
        raise ValueError("checkpoint ZIP64 multi-disk archives are not allowed")
    if classic_entries_total != 0xFFFF and int(entries_total) != int(classic_entries_total):
        raise ValueError("checkpoint ZIP64 entry count disagrees with ordinary EOCD")
    if classic_cd_size != 0xFFFF_FFFF and int(cd_size) != int(classic_cd_size):
        raise ValueError("checkpoint ZIP64 central directory size disagrees with ordinary EOCD")
    if classic_cd_offset != 0xFFFF_FFFF and int(cd_offset) != int(classic_cd_offset):
        raise ValueError("checkpoint ZIP64 central directory offset disagrees with ordinary EOCD")
    if int(cd_offset) < 0 or int(cd_offset) + int(cd_size) != int(zip64_offset):
        raise ValueError("checkpoint ZIP64 EOCD placement is malformed")
    return int(entries_total), int(cd_size), int(cd_offset), int(zip64_offset)


def _parse_central_directory(data: bytes, *, cd_offset: int, cd_size: int, expected_count: int) -> list[_RawZipEntry]:
    entries: list[_RawZipEntry] = []
    pos = cd_offset
    end = cd_offset + cd_size
    while pos < end:
        if pos + 46 > end or data[pos : pos + 4] != b"PK\x01\x02":
            raise ValueError("checkpoint ZIP central directory entry is malformed")
        fields = struct.unpack_from("<HHHHHHIIIHHHHHII", data, pos + 4)
        (
            _version_made,
            version_needed,
            flag_bits,
            compress_type,
            mod_time,
            mod_date,
            crc,
            compress_size_32,
            file_size_32,
            name_len,
            extra_len,
            comment_len,
            disk_start,
            _internal_attr,
            _external_attr,
            header_offset_32,
        ) = fields
        name_start = pos + 46
        extra_start = name_start + name_len
        comment_start = extra_start + extra_len
        next_pos = comment_start + comment_len
        if next_pos > end:
            raise ValueError("checkpoint ZIP central directory entry lengths are malformed")
        name_bytes = data[name_start:extra_start]
        extra = data[extra_start:comment_start]
        if comment_len != 0:
            raise ValueError("checkpoint ZIP member comments are not allowed")
        try:
            filename = name_bytes.decode("utf-8" if flag_bits & 0x800 else "cp437")
        except UnicodeDecodeError as exc:
            raise ValueError("checkpoint ZIP member name is not decodable") from exc
        if flag_bits & ~(0x8 | 0x800):
            raise ValueError("checkpoint ZIP member flags are not allowed")
        if _has_zip64_extra(extra):
            raise ValueError("checkpoint ZIP per-entry ZIP64 extras are not supported under trainer limits")
        if compress_size_32 == 0xFFFF_FFFF or file_size_32 == 0xFFFF_FFFF or header_offset_32 == 0xFFFF_FFFF or disk_start == 0xFFFF:
            raise ValueError("checkpoint ZIP per-entry ZIP64 values exceed trainer limits")
        compress_size = int(compress_size_32)
        file_size = int(file_size_32)
        header_offset = int(header_offset_32)
        if disk_start != 0:
            raise ValueError("checkpoint ZIP member starts on a nonzero disk")
        entries.append(
            _RawZipEntry(
                filename=filename,
                filename_bytes=name_bytes,
                member="/".join(filename.split("/")[1:]) if "/" in filename else "",
                version_needed=int(version_needed),
                flag_bits=flag_bits,
                compress_type=compress_type,
                mod_time=int(mod_time),
                mod_date=int(mod_date),
                crc=int(crc),
                compress_size=compress_size,
                file_size=file_size,
                header_offset=header_offset,
            )
        )
        pos = next_pos
    if pos != end or len(entries) != expected_count:
        raise ValueError("checkpoint ZIP central directory count mismatch")
    return entries


def _has_zip64_extra(extra: bytes) -> bool:
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, pos)
        payload_start = pos + 4
        payload_end = payload_start + size
        if payload_end > len(extra):
            raise ValueError("checkpoint ZIP extra field is malformed")
        if header_id == 0x0001:
            return True
        pos = payload_end
    if pos != len(extra):
        raise ValueError("checkpoint ZIP extra field has malformed residual bytes")
    return False


def _parse_zip_extra_fields(extra: bytes, *, context: str) -> list[tuple[int, bytes]]:
    fields: list[tuple[int, bytes]] = []
    seen: set[int] = set()
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, pos)
        if header_id in seen:
            raise ValueError(f"checkpoint ZIP {context} extra field contains duplicate IDs")
        seen.add(header_id)
        payload_start = pos + 4
        payload_end = payload_start + size
        if payload_end > len(extra):
            raise ValueError(f"checkpoint ZIP {context} extra field is truncated")
        fields.append((header_id, extra[payload_start:payload_end]))
        pos = payload_end
    if pos != len(extra):
        raise ValueError(f"checkpoint ZIP {context} extra field has malformed residual bytes")
    return fields


def _validate_local_extra(extra: bytes, entry: _RawZipEntry, *, data_start: int) -> None:
    fields = _parse_zip_extra_fields(extra, context="local")
    if entry.flag_bits & 0x8:
        if len(fields) != 1:
            raise ValueError("checkpoint ZIP Torch local header must contain exactly one alignment extra")
        header_id, payload = fields[0]
        if header_id == 0x0001:
            raise ValueError("checkpoint ZIP local ZIP64 extra is not allowed")
        if header_id != 0x4246:
            raise ValueError("checkpoint ZIP local extra field is not recognized")
        if any(ch != ord("Z") for ch in payload):
            raise ValueError("checkpoint ZIP Torch local alignment padding is malformed")
        if data_start % 64 != 0:
            raise ValueError("checkpoint ZIP Torch local payload is not 64-byte aligned")
        return
    if fields:
        header_id, _payload = fields[0]
        if header_id == 0x0001:
            raise ValueError("checkpoint ZIP local ZIP64 extra is not allowed")
        raise ValueError("checkpoint ZIP seekable local extra field is not allowed")


def _validate_local_member_layout(data: bytes, entries: list[_RawZipEntry], *, central_directory_offset: int) -> None:
    records: list[tuple[int, int, str]] = []
    seen_names: set[str] = set()
    for entry in entries:
        if entry.filename in seen_names:
            raise ValueError("checkpoint ZIP contains duplicate member names")
        seen_names.add(entry.filename)
        if entry.compress_type != zipfile.ZIP_STORED or entry.compress_size != entry.file_size:
            raise ValueError("checkpoint ZIP compression is not allowed")
        if entry.compress_size < 0 or entry.file_size < 0:
            raise ValueError("checkpoint ZIP member size is negative")
        record = _parse_local_member_record(data, entry, central_directory_offset=central_directory_offset)
        records.append((record.start, record.end, entry.filename))
    records.sort()
    previous_end = 0
    for start, end, name in records:
        if start != previous_end:
            raise ValueError(f"checkpoint ZIP local records are not contiguous before central directory: {name}")
        previous_end = end
    if previous_end != central_directory_offset:
        raise ValueError("checkpoint ZIP local records do not end at central directory")


def _local_member_data_bounds(data: bytes, entry: _RawZipEntry, *, central_directory_offset: int) -> tuple[int, int]:
    record = _parse_local_member_record(data, entry, central_directory_offset=central_directory_offset)
    return record.data_start, record.data_end


def _parse_local_member_record(data: bytes, entry: _RawZipEntry, *, central_directory_offset: int) -> _LocalZipRecord:
    off = entry.header_offset
    if off < 0 or off + 30 > central_directory_offset or data[off : off + 4] != b"PK\x03\x04":
        raise ValueError("checkpoint ZIP local header offset is malformed")
    (
        version_needed,
        local_flags,
        local_compress,
        mod_time,
        mod_date,
        local_crc,
        local_comp_size,
        local_file_size,
        name_len,
        extra_len,
    ) = struct.unpack_from("<HHHHHIIIHH", data, off + 4)
    name_start = off + 30
    extra_start = name_start + name_len
    data_start = extra_start + extra_len
    data_end = data_start + entry.compress_size
    if data_end > central_directory_offset:
        raise ValueError("checkpoint ZIP member overlaps central directory")
    local_name = data[name_start:extra_start]
    if (
        local_name != entry.filename_bytes
        or int(version_needed) != entry.version_needed
        or int(local_flags) != entry.flag_bits
        or int(local_compress) != entry.compress_type
        or int(mod_time) != entry.mod_time
        or int(mod_date) != entry.mod_date
    ):
        raise ValueError("checkpoint ZIP local header disagrees with central directory")
    _validate_local_extra(data[extra_start:data_start], entry, data_start=data_start)
    if entry.flag_bits & 0x8:
        if local_crc != 0 or local_comp_size != 0 or local_file_size != 0:
            raise ValueError("checkpoint ZIP data-descriptor local header must have zero CRC and sizes")
        descriptor_start = data_end
        descriptor_end = descriptor_start + 16
        if descriptor_end > central_directory_offset:
            raise ValueError("checkpoint ZIP data descriptor is truncated")
        if data[descriptor_start : descriptor_start + 4] != b"PK\x07\x08":
            raise ValueError("checkpoint ZIP data descriptor signature is missing")
        desc_crc, desc_comp_size, desc_file_size = struct.unpack_from("<III", data, descriptor_start + 4)
        if int(desc_crc) != entry.crc or int(desc_comp_size) != entry.compress_size or int(desc_file_size) != entry.file_size:
            raise ValueError("checkpoint ZIP data descriptor disagrees with central directory")
        return _LocalZipRecord(
            start=off,
            header_end=data_start,
            data_start=data_start,
            data_end=data_end,
            descriptor_start=descriptor_start,
            descriptor_end=descriptor_end,
            end=descriptor_end,
        )
    if local_crc != entry.crc or local_comp_size != entry.compress_size or local_file_size != entry.file_size:
        raise ValueError("checkpoint ZIP local header CRC or sizes disagree with central directory")
    return _LocalZipRecord(
        start=off,
        header_end=data_start,
        data_start=data_start,
        data_end=data_end,
        descriptor_start=data_end,
        descriptor_end=data_end,
        end=data_end,
    )


def _preflight_pickle(data: bytes, *, storage_member_sizes: dict[str, int]) -> tuple[int, int, int]:
    opcodes = 0
    memo_writes = 0
    tensor_decls = 0
    next_memo_index = 0
    stop_seen = False
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    storage_decls: dict[str, _StorageRef] = {}
    storage_used: set[str] = set()
    tensor_decl_ids: set[int] = set()

    def push(value: Any) -> None:
        stack.append(value)
        if len(stack) > MAX_PICKLE_STACK_DEPTH:
            raise ValueError("checkpoint pickle stack exceeds limit")

    def pop() -> Any:
        if not stack:
            raise ValueError("checkpoint pickle stack underflow")
        return stack.pop()

    def mark_index() -> int:
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is _MARK:
                return index
        raise ValueError("checkpoint pickle mark underflow")

    def memo_put(index: int) -> None:
        nonlocal memo_writes, next_memo_index
        if not stack:
            raise ValueError("checkpoint pickle memo write without stack value")
        if index != next_memo_index:
            raise ValueError("checkpoint pickle memo writes must be sequential from zero")
        if index in memo:
            raise ValueError("checkpoint pickle memo overwrite is not allowed")
        memo[index] = stack[-1]
        next_memo_index += 1
        memo_writes += 1
        if memo_writes > MAX_PICKLE_MEMO_WRITES:
            raise ValueError("checkpoint pickle memo exceeds limit")

    def memo_get(index: int) -> Any:
        if index not in memo:
            raise ValueError("checkpoint pickle memo read before write")
        value = memo[index]
        if not (value is None or type(value) in (str, bool, int, float) or isinstance(value, _GlobalRef)):
            raise ValueError("checkpoint pickle memo read of mutable or special object is not allowed")
        return value

    def dict_insert(target: dict[Any, Any], key: Any, value: Any) -> None:
        if type(key) is not str:
            raise ValueError("checkpoint pickle dict keys must be exact strings")
        if key in target:
            raise ValueError("checkpoint pickle duplicate dict key")
        if len(target) + 1 > MAX_PICKLE_CONTAINER_ITEMS:
            raise ValueError("checkpoint pickle dict exceeds limit")
        target[key] = value

    for opcode, arg, pos in pickletools.genops(data):
        opcodes += 1
        if opcodes > MAX_PICKLE_OPCODES:
            raise ValueError("checkpoint pickle has too many opcodes")
        name = opcode.name
        if opcodes == 1 and (name != "PROTO" or pos != 0 or arg != 2):
            raise ValueError("checkpoint pickle must start with exactly one PROTO 2")
        if name == "PROTO":
            if opcodes != 1 or pos != 0 or arg != 2:
                raise ValueError("checkpoint pickle must start with exactly one PROTO 2")
        elif name == "GLOBAL":
            if arg not in ALLOWED_PICKLE_GLOBALS:
                raise ValueError(f"checkpoint pickle global is not allowed: {arg}")
            push(_GlobalRef(str(arg)))
        elif name == "STACK_GLOBAL":
            raise ValueError("checkpoint pickle STACK_GLOBAL is not allowed")
        elif name in {"INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "BUILD"}:
            raise ValueError(f"checkpoint pickle opcode is not allowed: {name}")
        elif name == "BINUNICODE":
            push(str(arg))
        elif name in {"BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}:
            if type(arg) is not int:
                raise ValueError("checkpoint pickle integer argument is malformed")
            push(int(arg))
        elif name == "BINFLOAT":
            push(float(arg))
        elif name == "NONE":
            push(None)
        elif name == "NEWTRUE":
            push(True)
        elif name == "NEWFALSE":
            push(False)
        elif name == "EMPTY_DICT":
            push({})
        elif name == "EMPTY_LIST":
            push([])
        elif name == "EMPTY_TUPLE":
            push(())
        elif name == "MARK":
            if sum(1 for item in stack if item is _MARK) >= MAX_PICKLE_MARKS:
                raise ValueError("checkpoint pickle mark depth exceeds limit")
            push(_MARK)
        elif name == "TUPLE":
            index = mark_index()
            items = tuple(stack[index + 1 :])
            if len(items) > MAX_PICKLE_CONTAINER_ITEMS:
                raise ValueError("checkpoint pickle tuple exceeds limit")
            del stack[index:]
            push(items)
        elif name == "TUPLE1":
            item = pop()
            push((item,))
        elif name == "TUPLE2":
            b = pop()
            a = pop()
            push((a, b))
        elif name == "TUPLE3":
            c = pop()
            b = pop()
            a = pop()
            push((a, b, c))
        elif name == "SETITEM":
            value = pop()
            key = pop()
            target = stack[-1] if stack else None
            if type(target) is not dict:
                raise ValueError("checkpoint pickle SETITEM target is not a dict")
            dict_insert(target, key, value)
        elif name == "SETITEMS":
            index = mark_index()
            target = stack[index - 1] if index > 0 else None
            if type(target) is not dict:
                raise ValueError("checkpoint pickle SETITEMS target is not a dict")
            items = stack[index + 1 :]
            if len(items) % 2 != 0:
                raise ValueError("checkpoint pickle SETITEMS has odd item count")
            for i in range(0, len(items), 2):
                dict_insert(target, items[i], items[i + 1])
            del stack[index:]
        elif name == "APPEND":
            value = pop()
            target = stack[-1] if stack else None
            if type(target) is not list:
                raise ValueError("checkpoint pickle APPEND target is not a list")
            if len(target) + 1 > MAX_PICKLE_CONTAINER_ITEMS:
                raise ValueError("checkpoint pickle list exceeds limit")
            target.append(value)
        elif name == "APPENDS":
            index = mark_index()
            target = stack[index - 1] if index > 0 else None
            if type(target) is not list:
                raise ValueError("checkpoint pickle APPENDS target is not a list")
            items = stack[index + 1 :]
            if len(items) + len(target) > MAX_PICKLE_CONTAINER_ITEMS:
                raise ValueError("checkpoint pickle list exceeds limit")
            target.extend(items)
            del stack[index:]
        elif name in {"BINPUT", "LONG_BINPUT"}:
            memo_put(int(arg))
        elif name == "MEMOIZE":
            raise ValueError("checkpoint pickle MEMOIZE is not allowed")
        elif name in {"BINGET", "LONG_BINGET"}:
            push(memo_get(int(arg)))
        elif name == "BINPERSID":
            persistent_id = pop()
            push(_validate_storage_persistent_id(persistent_id, storage_member_sizes, storage_decls))
        elif name == "REDUCE":
            args = pop()
            func = pop()
            if isinstance(func, _GlobalRef) and func.name == "collections OrderedDict":
                if args != ():
                    raise ValueError("checkpoint pickle OrderedDict reduce args mismatch")
                push(_ORDERED_DICT)
            elif isinstance(func, _GlobalRef) and func.name == "torch._utils _rebuild_tensor_v2":
                tensor = _validate_tensor_reduce(args, storage_decls, storage_used)
                tensor_decls += 1
                if tensor_decls > MAX_PICKLE_TENSORS:
                    raise ValueError("checkpoint pickle has too many tensor declarations")
                tensor_decl_ids.add(id(tensor))
                push(tensor)
            else:
                raise ValueError("checkpoint pickle REDUCE target is not allowed")
        elif name == "STOP":
            if pos != len(data) - 1:
                raise ValueError("checkpoint pickle STOP must be final byte")
            if len(stack) != 1:
                raise ValueError("checkpoint pickle STOP with malformed stack")
            _validate_pickle_root_graph(
                stack[0],
                tensor_decl_ids=tensor_decl_ids,
                storage_used=storage_used,
                storage_decls=storage_decls,
                storage_member_sizes=storage_member_sizes,
            )
            stop_seen = True
            break
        else:
            raise ValueError(f"checkpoint pickle opcode is not allowed: {name}")
    else:
        raise ValueError("checkpoint pickle missing STOP")
    if not stop_seen:
        raise ValueError("checkpoint pickle missing STOP")
    if set(storage_decls) != set(storage_member_sizes):
        raise ValueError("checkpoint pickle storage declarations do not match ZIP storage members")
    if set(storage_used) != set(storage_decls):
        raise ValueError("checkpoint pickle storage declarations were not all consumed by tensors")
    return opcodes, memo_writes, tensor_decls


def _validate_pickle_root_graph(
    root: Any,
    *,
    tensor_decl_ids: set[int],
    storage_used: set[str],
    storage_decls: dict[str, _StorageRef],
    storage_member_sizes: dict[str, int],
) -> None:
    if type(root) is not dict:
        raise ValueError("checkpoint pickle root must be an exact plain dict")
    counters = {
        "nodes": 0,
        "items": 0,
        "strings": 0,
        "tensors": 0,
        "tensor_elements": 0,
        "tensor_bytes": 0,
        "storage_bytes": 0,
    }
    seen_containers: set[int] = set()
    active_containers: set[int] = set()
    reachable_tensor_ids: set[int] = set()
    reachable_storage_keys: set[str] = set()
    stack: list[tuple[Any, int, bool]] = [(root, 0, False)]
    while stack:
        item, depth, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(item))
            continue
        counters["nodes"] += 1
        if counters["nodes"] > MAX_PICKLE_GRAPH_NODES:
            raise ValueError("checkpoint pickle object graph has too many nodes")
        if depth > MAX_PICKLE_GRAPH_DEPTH:
            raise ValueError("checkpoint pickle object graph is too deep")
        if item is None or type(item) in (bool, int):
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("checkpoint pickle has a non-finite float")
            continue
        if type(item) is str:
            counters["strings"] += len(item.encode("utf-8"))
            if counters["strings"] > MAX_PICKLE_STRING_BYTES:
                raise ValueError("checkpoint pickle has too many string bytes")
            continue
        if type(item) is _TensorDecl:
            tensor_id = id(item)
            if tensor_id not in tensor_decl_ids:
                raise ValueError("checkpoint pickle has an unknown tensor declaration")
            if tensor_id in reachable_tensor_ids:
                raise ValueError("checkpoint pickle repeats a tensor declaration")
            reachable_tensor_ids.add(tensor_id)
            reachable_storage_keys.add(item.storage_key)
            counters["tensors"] += 1
            if counters["tensors"] > MAX_PICKLE_TENSORS:
                raise ValueError("checkpoint pickle has too many tensors")
            counters["tensor_elements"] += _shape_product(item.shape)
            counters["tensor_bytes"] += item.byte_count
            storage = storage_decls.get(item.storage_key)
            if storage is None:
                raise ValueError("checkpoint pickle tensor references missing storage")
            counters["storage_bytes"] += storage.byte_count
            if counters["tensor_elements"] > MAX_PICKLE_TENSOR_ELEMENTS:
                raise ValueError("checkpoint pickle tensor element aggregate exceeds limit")
            if counters["tensor_bytes"] > MAX_TORCH_ZIP_STORAGE_BYTES:
                raise ValueError("checkpoint pickle tensor byte aggregate exceeds limit")
            if counters["storage_bytes"] > MAX_TORCH_ZIP_STORAGE_BYTES:
                raise ValueError("checkpoint pickle tensor storage aggregate exceeds limit")
            continue
        if type(item) is dict:
            container_id = id(item)
            if container_id in active_containers:
                raise ValueError("checkpoint pickle object graph contains a cycle")
            if container_id in seen_containers:
                raise ValueError("checkpoint pickle repeats a container object identity")
            seen_containers.add(container_id)
            active_containers.add(container_id)
            if len(item) > MAX_PICKLE_CONTAINER_ITEMS:
                raise ValueError("checkpoint pickle dict exceeds limit")
            counters["items"] += len(item)
            if counters["items"] > MAX_PICKLE_TOTAL_ITEMS:
                raise ValueError("checkpoint pickle collection aggregate exceeds limit")
            stack.append((item, depth, True))
            for key, child in reversed(list(item.items())):
                if type(key) is not str:
                    raise ValueError("checkpoint pickle dict key must be an exact string")
                counters["strings"] += len(key.encode("utf-8"))
                if counters["strings"] > MAX_PICKLE_STRING_BYTES:
                    raise ValueError("checkpoint pickle has too many string bytes")
                stack.append((child, depth + 1, False))
            continue
        if type(item) is list or type(item) is tuple:
            container_id = id(item)
            if container_id in active_containers:
                raise ValueError("checkpoint pickle object graph contains a cycle")
            if container_id in seen_containers:
                raise ValueError("checkpoint pickle repeats a container object identity")
            seen_containers.add(container_id)
            active_containers.add(container_id)
            if len(item) > MAX_PICKLE_CONTAINER_ITEMS:
                raise ValueError("checkpoint pickle container exceeds limit")
            counters["items"] += len(item)
            if counters["items"] > MAX_PICKLE_TOTAL_ITEMS:
                raise ValueError("checkpoint pickle collection aggregate exceeds limit")
            stack.append((item, depth, True))
            for child in reversed(item):
                stack.append((child, depth + 1, False))
            continue
        raise ValueError(f"checkpoint pickle object type is not allowed: {type(item).__name__}")
    if tensor_decl_ids != reachable_tensor_ids:
        raise ValueError("checkpoint pickle tensor declarations are not exactly reachable from root")
    if reachable_storage_keys != set(storage_used):
        raise ValueError("checkpoint pickle reachable tensor storage keys do not match constructed tensors")
    if reachable_storage_keys != set(storage_decls):
        raise ValueError("checkpoint pickle reachable tensor storage keys do not match storage declarations")
    if reachable_storage_keys != set(storage_member_sizes):
        raise ValueError("checkpoint pickle reachable tensor storage keys do not match ZIP storage members")


def _validate_storage_persistent_id(
    persistent_id: Any,
    storage_member_sizes: dict[str, int],
    storage_decls: dict[str, _StorageRef],
) -> _StorageRef:
    if not isinstance(persistent_id, tuple) or len(persistent_id) != 5:
        raise ValueError("checkpoint pickle storage persistent ID shape mismatch")
    tag, storage_type, key, location, element_count = persistent_id
    if tag != "storage" or not isinstance(storage_type, _GlobalRef):
        raise ValueError("checkpoint pickle storage persistent ID tag/type mismatch")
    if storage_type.name not in _STORAGE_DTYPE_BYTES:
        raise ValueError("checkpoint pickle storage type is not allowed")
    if type(key) is not str or not re.fullmatch(r"(0|[1-9][0-9]*)", key):
        raise ValueError("checkpoint pickle storage key is not canonical")
    if key in storage_decls:
        raise ValueError("checkpoint pickle storage key is reused")
    if location != "cpu":
        raise ValueError("checkpoint pickle storage location must be cpu")
    if type(element_count) is not int or element_count < 0:
        raise ValueError("checkpoint pickle storage element count is invalid")
    dtype, element_size = _STORAGE_DTYPE_BYTES[storage_type.name]
    byte_count = element_count * element_size
    if byte_count > MAX_TORCH_ZIP_STORAGE_BYTES:
        raise ValueError("checkpoint pickle declared storage bytes exceed limit")
    if key not in storage_member_sizes:
        raise ValueError("checkpoint pickle storage member is missing")
    if storage_member_sizes[key] != byte_count:
        raise ValueError("checkpoint pickle storage byte count disagrees with ZIP member")
    ref = _StorageRef(
        key=key,
        storage_type=storage_type.name,
        dtype=dtype,
        location=location,
        elements=element_count,
        byte_count=byte_count,
    )
    storage_decls[key] = ref
    return ref


def _validate_tensor_reduce(
    args: Any,
    storage_decls: dict[str, _StorageRef],
    storage_used: set[str],
) -> _TensorDecl:
    if not isinstance(args, tuple) or len(args) != 6:
        raise ValueError("checkpoint pickle tensor rebuild args mismatch")
    storage, offset, shape, strides, requires_grad, metadata = args
    if not isinstance(storage, _StorageRef):
        raise ValueError("checkpoint pickle tensor storage reference mismatch")
    if storage.key not in storage_decls:
        raise ValueError("checkpoint pickle tensor references unknown storage")
    if storage.key in storage_used:
        raise ValueError("checkpoint pickle reuses a storage alias")
    if type(offset) is not int or offset != 0:
        raise ValueError("checkpoint pickle tensor storage offset must be zero")
    if type(shape) is not tuple or type(strides) is not tuple or len(shape) != len(strides):
        raise ValueError("checkpoint pickle tensor shape/stride mismatch")
    if len(shape) > MAX_PICKLE_TENSOR_RANK:
        raise ValueError("checkpoint pickle tensor rank exceeds limit")
    if requires_grad is not False:
        raise ValueError("checkpoint pickle tensor requires_grad must be false")
    if metadata is not _ORDERED_DICT:
        raise ValueError("checkpoint pickle tensor metadata must be empty OrderedDict")
    shape_ints: list[int] = []
    for dim in shape:
        if type(dim) is not int or dim < 0:
            raise ValueError("checkpoint pickle tensor shape contains invalid dimension")
        shape_ints.append(dim)
    stride_ints: list[int] = []
    for stride in strides:
        if type(stride) is not int or stride <= 0:
            raise ValueError("checkpoint pickle tensor strides must be positive")
        stride_ints.append(stride)
    expected_strides = _contiguous_strides(tuple(shape_ints))
    if tuple(stride_ints) != expected_strides:
        raise ValueError("checkpoint pickle tensor strides are not exact contiguous strides")
    elements = _shape_product(tuple(shape_ints))
    if elements > 20_000_000:
        raise ValueError("checkpoint pickle tensor declares too many elements")
    _dtype_name, element_size = _STORAGE_DTYPE_BYTES[storage.storage_type]
    byte_count = elements * element_size
    if byte_count > MAX_TORCH_ZIP_STORAGE_BYTES:
        raise ValueError("checkpoint pickle tensor declares too many bytes")
    if byte_count != storage.byte_count:
        raise ValueError("checkpoint pickle tensor bytes do not exactly cover storage")
    storage_used.add(storage.key)
    return _TensorDecl(storage_key=storage.key, shape=tuple(shape_ints), strides=tuple(stride_ints), byte_count=byte_count)


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= dim
        if product > 20_000_000:
            raise ValueError("checkpoint pickle tensor shape product exceeds limit")
    return product


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    out: list[int] = []
    running = 1
    for dim in reversed(shape):
        out.append(running)
        running *= max(dim, 1)
    return tuple(reversed(out))


def _require_weights_only_torch_load() -> None:
    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Torch safe checkpoint loading is unavailable") from exc
    if "weights_only" not in signature.parameters:
        raise RuntimeError("Torch safe checkpoint loading is unavailable")
