"""Public side-effect-free TrainingStore reader for kernel RL artifacts."""

from __future__ import annotations

import copy
import dataclasses
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .artifact_io import (
    FORBIDDEN_TRAINING_JSON_KEYS,
    read_authoritative_json_capture,
    sha256_bytes,
    validate_training_json_privacy,
)
from .artifacts import generation_paths, latest_path
from .checkpoint import (
    CHECKPOINT_SCHEMA,
    LATEST_SCHEMA,
    SIDECAR_SCHEMA,
    UPDATE_RECORD_SCHEMA,
    adam_config,
    build_sidecar,
    compute_head,
    create_adam,
    logical_state_hash,
    load_adam_state,
    load_checkpoint_file_with_digest,
    update_record_hash,
    validate_adam_state_for_model,
    validate_checkpoint_payload,
    validate_model_state,
    validate_torch_rng_state,
)
from .determinism import (
    TrainerSeedDerivation,
    configure_torch_determinism,
    derive_model_init_seed,
    derive_train_env_seed,
    validate_positive_int,
    validate_uint63,
)
from .model import INITIALIZER_TRAINER_SEEDED_V1, KernelPolicyValueNet, ModelConfig
from .path_safety import ensure_real_child_dir, ensure_real_file


RUN_SCHEMA = "kernel_rl_train_run/v11"
ALGORITHM_NAME = "terminal_reinforce_value/v1"
MAX_UPDATES = 1_000_000
MAX_BATCH_EPISODES = 10_000
MAX_DECISIONS = 10_000_000
EPISODE_SUMMARY_SCHEMA = "kernel_rl_train_episode_summary/v2"
SUMMARY_SCHEMA = "kernel_rl_train_summary/v2"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclasses.dataclass(frozen=True, slots=True)
class StoreReadCounts:
    run: int
    latest: int
    updates: int
    sidecars: int
    checkpoints: int

    @property
    def total(self) -> int:
        return self.run + self.latest + self.updates + self.sidecars + self.checkpoints


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotRef:
    root: Path
    update: int
    run_digest: str
    head: str
    parent_head: str | None
    update_path: Path
    sidecar_path: Path
    checkpoint_path: Path
    update_record_sha256: str
    checkpoint_sha256: str
    logical_state_sha256: str
    model_config: ModelConfig


@dataclasses.dataclass(frozen=True, slots=True)
class PolicySnapshot:
    ref: SnapshotRef
    model: KernelPolicyValueNet


@dataclasses.dataclass(slots=True)
class ResumeSnapshot:
    ref: SnapshotRef
    model: KernelPolicyValueNet
    optimizer: torch.optim.Adam
    completed_update: int
    optimizer_step_count: int
    next_episode: int
    outcomes_by_learner_seat: dict[str, dict[str, int]]
    learner_decisions_by_seat: dict[str, int]
    python_rng_state: dict[str, Any]
    torch_cpu_rng_state: torch.Tensor
    checkpoint_payload: dict[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class ValidatedChain:
    snapshots: tuple[SnapshotRef, ...]
    head: SnapshotRef
    read_counts: StoreReadCounts
    _root: Path = dataclasses.field(repr=False, compare=False)
    _run: dict[str, Any] = dataclasses.field(repr=False, compare=False)
    _latest: dict[str, Any] = dataclasses.field(repr=False, compare=False)
    _records: tuple[dict[str, Any], ...] = dataclasses.field(repr=False, compare=False)
    _member_ref_ids: frozenset[int] = dataclasses.field(repr=False, compare=False)
    _snapshots_by_update: dict[int, SnapshotRef] = dataclasses.field(repr=False, compare=False)

    @property
    def run_record(self) -> dict[str, Any]:
        return _clone_value(self._run)

    @property
    def latest_record(self) -> dict[str, Any]:
        return _clone_value(self._latest)

    @property
    def update_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(_clone_value(record) for record in self._records)

    def update_record(self, ref: SnapshotRef | None = None) -> dict[str, Any]:
        selected = self._validate_member_ref(self.head if ref is None else ref)
        return _clone_value(self._records[selected.update])

    def load_policy(self, ref: SnapshotRef | None = None) -> PolicySnapshot:
        selected = self._validate_member_ref(self.head if ref is None else ref)
        payload = self._load_snapshot_payload(selected)
        model = _model_from_payload(payload, selected.model_config, initializer_seed=self._run["initializer"]["seed"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return PolicySnapshot(ref=selected, model=model)

    def load_resume(self, ref: SnapshotRef | None = None) -> ResumeSnapshot:
        selected = self._validate_member_ref(self.head if ref is None else ref)
        if selected is not self.head:
            raise ValueError("resume loading is only supported for the pinned head")
        payload = self._load_snapshot_payload(selected)
        model = _model_from_payload(payload, selected.model_config, initializer_seed=self._run["initializer"]["seed"])
        model.train()
        optimizer = create_adam(model, self._run["optimizer"]["lr"])
        load_adam_state(
            optimizer,
            model,
            payload["optimizer_state"],
            self._run["optimizer"]["lr"],
            expected_step_count=payload["optimizer_step_count"],
        )
        return ResumeSnapshot(
            ref=selected,
            model=model,
            optimizer=optimizer,
            completed_update=payload["completed_update"],
            optimizer_step_count=payload["optimizer_step_count"],
            next_episode=payload["next_episode"],
            outcomes_by_learner_seat=_clone_value(payload["outcomes_by_learner_seat"]),
            learner_decisions_by_seat=_clone_value(payload["learner_decisions_by_seat"]),
            python_rng_state=_clone_value(payload["python_rng_state"]),
            torch_cpu_rng_state=payload["torch_cpu_rng_state"].detach().contiguous().clone(),
            checkpoint_payload=_clone_value(payload),
        )

    def _validate_member_ref(self, ref: SnapshotRef) -> SnapshotRef:
        if type(ref) is not SnapshotRef:
            raise TypeError("snapshot ref must be a SnapshotRef")
        if id(ref) not in self._member_ref_ids:
            raise ValueError("snapshot ref is not a member of this validated chain")
        expected = self._snapshots_by_update.get(ref.update)
        if expected is not ref:
            raise ValueError("snapshot ref identity mismatch")
        if ref.root != self._root:
            raise ValueError("snapshot root mismatch")
        if ref.update < 0 or ref.update >= len(self.snapshots):
            raise ValueError("snapshot update out of bounds")
        paths = generation_paths(self._root, ref.update)
        if ref.update_path != paths["update"] or ref.sidecar_path != paths["sidecar"] or ref.checkpoint_path != paths["checkpoint"]:
            raise ValueError("snapshot paths mismatch")
        _validate_hash(ref.run_digest, "snapshot run_digest")
        _validate_hash(ref.head, "snapshot head")
        if ref.parent_head is not None:
            _validate_hash(ref.parent_head, "snapshot parent_head")
        _validate_hash(ref.update_record_sha256, "snapshot update_record_sha256")
        _validate_hash(ref.checkpoint_sha256, "snapshot checkpoint_sha256")
        _validate_hash(ref.logical_state_sha256, "snapshot logical_state_sha256")
        return ref

    def _load_snapshot_payload(self, ref: SnapshotRef) -> dict[str, Any]:
        for path in (ref.update_path, ref.sidecar_path, ref.checkpoint_path):
            ensure_real_file(self._root, path)
        record_capture = read_authoritative_json_capture(ref.update_path, "update")
        sidecar_capture = read_authoritative_json_capture(ref.sidecar_path, "sidecar")
        sidecar = sidecar_capture.value
        if record_capture.file.sha256 != ref.update_record_sha256:
            raise ValueError("snapshot update record hash mismatch")
        if update_record_hash(record_capture.value) != ref.update_record_sha256:
            raise ValueError("snapshot update record canonical hash mismatch")
        _validate_sidecar(sidecar, update=ref.update, run_digest=ref.run_digest, parent_head=ref.parent_head)
        loaded = load_checkpoint_file_with_digest(ref.checkpoint_path)
        if loaded.sha256 != ref.checkpoint_sha256:
            raise ValueError("snapshot checkpoint byte hash mismatch")
        payload = loaded.payload
        _strict_validate_checkpoint_for_model(
            payload,
            run_digest=ref.run_digest,
            compatibility=self._run["compatibility"],
            learning_rate=self._run["optimizer"]["lr"],
        )
        logical_hash = logical_state_hash(payload)
        if logical_hash != ref.logical_state_sha256:
            raise ValueError("snapshot logical state hash mismatch")
        expected_head = compute_head(
            parent_head=ref.parent_head,
            checkpoint_byte_hash=ref.checkpoint_sha256,
            logical_hash=ref.logical_state_sha256,
            update_hash=ref.update_record_sha256,
        )
        if expected_head != ref.head:
            raise ValueError("snapshot head mismatch")
        expected_sidecar = build_sidecar(
            update=ref.update,
            run_digest=ref.run_digest,
            parent_head=ref.parent_head,
            checkpoint_sha256=ref.checkpoint_sha256,
            logical_hash=ref.logical_state_sha256,
            update_hash=ref.update_record_sha256,
        )
        if sidecar != expected_sidecar:
            raise ValueError("snapshot sidecar content mismatch")
        return payload


class TrainingStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def validate_latest(self) -> ValidatedChain:
        root = ensure_real_child_dir(self.root.parent, self.root)
        run_capture = read_authoritative_json_capture(root / "run.json", "run")
        run = _validate_run_manifest(run_capture.value)
        run_digest = run_capture.file.sha256
        latest_capture = read_authoritative_json_capture(latest_path(root), "latest")
        latest = _validate_latest(latest_capture.value, run_digest=run_digest)
        head_update = latest["update"]
        parent_head: str | None = None
        previous_payload: dict[str, Any] | None = None
        snapshots: list[SnapshotRef] = []
        records: list[dict[str, Any]] = []
        for update in range(head_update + 1):
            ref, record, payload = _validate_generation_capture(
                root=root,
                update=update,
                run=run,
                run_digest=run_digest,
                parent_head=parent_head,
                previous_payload=previous_payload,
            )
            snapshots.append(ref)
            records.append(_clone_value(record))
            parent_head = ref.head
            previous_payload = payload
        if parent_head != latest["head"]:
            raise ValueError("latest head mismatch")
        if not snapshots:
            raise ValueError("empty checkpoint chain")
        snapshot_tuple = tuple(snapshots)
        read_counts = StoreReadCounts(
            run=1,
            latest=1,
            updates=len(snapshot_tuple),
            sidecars=len(snapshot_tuple),
            checkpoints=len(snapshot_tuple),
        )
        return ValidatedChain(
            snapshots=snapshot_tuple,
            head=snapshot_tuple[-1],
            read_counts=read_counts,
            _root=root,
            _run=_clone_value(run),
            _latest=_clone_value(latest),
            _records=tuple(records),
            _member_ref_ids=frozenset(id(ref) for ref in snapshot_tuple),
            _snapshots_by_update={ref.update: ref for ref in snapshot_tuple},
        )


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().contiguous().clone()
    if type(value) is dict:
        return {key: _clone_value(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone_value(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone_value(item) for item in value)
    return copy.deepcopy(value)


def _model_from_payload(payload: dict[str, Any], config: ModelConfig, *, initializer_seed: int) -> KernelPolicyValueNet:
    model = KernelPolicyValueNet(
        config,
        initializer=INITIALIZER_TRAINER_SEEDED_V1,
        initializer_seed=initializer_seed,
        configure_runtime=False,
    )
    model.load_state_dict(validate_model_state(model, payload["model_state"]), strict=True)
    return model


def _validate_generation_capture(
    *,
    root: Path,
    update: int,
    run: dict[str, Any],
    run_digest: str,
    parent_head: str | None,
    previous_payload: dict[str, Any] | None,
) -> tuple[SnapshotRef, dict[str, Any], dict[str, Any]]:
    paths = generation_paths(root, update)
    return _validate_generation_at_paths(
        root=root,
        paths=paths,
        update=update,
        run=run,
        run_digest=run_digest,
        parent_head=parent_head,
        previous_payload=previous_payload,
        compatibility=run["compatibility"],
        learning_rate=run["optimizer"]["lr"],
    )


def _validate_generation_bundle(
    *,
    paths: dict[str, Path],
    update: int,
    run: dict[str, Any],
    run_digest: str,
    parent_head: str | None,
    previous_payload: dict[str, Any] | None,
    compatibility: dict[str, Any],
    learning_rate: float,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    root = Path(run["_artifact_root"])
    ref, record, payload = _validate_generation_at_paths(
        root=root,
        paths=paths,
        update=update,
        run=run,
        run_digest=run_digest,
        parent_head=parent_head,
        previous_payload=previous_payload,
        compatibility=compatibility,
        learning_rate=learning_rate,
    )
    return ref.head, record, payload


def _validate_generation_at_paths(
    *,
    root: Path,
    paths: dict[str, Path],
    update: int,
    run: dict[str, Any],
    run_digest: str,
    parent_head: str | None,
    previous_payload: dict[str, Any] | None,
    compatibility: dict[str, Any],
    learning_rate: float,
) -> tuple[SnapshotRef, dict[str, Any], dict[str, Any]]:
    for path in paths.values():
        ensure_real_file(root, path)
    record_capture = read_authoritative_json_capture(paths["update"], "update")
    sidecar_capture = read_authoritative_json_capture(paths["sidecar"], "sidecar")
    sidecar = _validate_sidecar(sidecar_capture.value, update=update, run_digest=run_digest, parent_head=parent_head)
    loaded_checkpoint = load_checkpoint_file_with_digest(paths["checkpoint"])
    if sidecar["checkpoint_sha256"] != loaded_checkpoint.sha256:
        raise ValueError("checkpoint byte hash mismatch")
    payload = loaded_checkpoint.payload
    _strict_validate_checkpoint_for_model(
        payload,
        run_digest=run_digest,
        compatibility=run["compatibility"],
        learning_rate=run["optimizer"]["lr"],
    )
    if payload["base_seed"] != run["trainer"]["base_seed"]:
        raise ValueError("checkpoint base_seed mismatch")
    if payload["seed_derivation"] != run["seed_derivation"]:
        raise ValueError("checkpoint seed derivation mismatch")
    if payload["provenance"] != run["protocol_provenance"]:
        raise ValueError("checkpoint provenance mismatch")
    if payload["model_config"] != run["model"]["config"]:
        raise ValueError("checkpoint model config mismatch")
    validate_torch_rng_state(payload["torch_cpu_rng_state"])
    logical_hash = logical_state_hash(payload)
    if sidecar["logical_state_sha256"] != logical_hash:
        raise ValueError("logical state hash mismatch")
    record = _validate_update_record(
        record_capture.value,
        update=update,
        run=run,
        run_digest=run_digest,
        parent_head=parent_head,
        previous_payload=previous_payload,
        payload=payload,
        logical_hash=logical_hash,
    )
    update_hash = record_capture.file.sha256
    if update_record_hash(record) != update_hash:
        raise ValueError("update record canonical hash mismatch")
    if sidecar["update_record_sha256"] != update_hash:
        raise ValueError("update record hash mismatch")
    expected_sidecar = build_sidecar(
        update=update,
        run_digest=run_digest,
        parent_head=parent_head,
        checkpoint_sha256=loaded_checkpoint.sha256,
        logical_hash=logical_hash,
        update_hash=update_hash,
    )
    if sidecar != expected_sidecar:
        raise ValueError("sidecar content mismatch")
    return (
        SnapshotRef(
            root=root,
            update=update,
            run_digest=run_digest,
            head=sidecar["head"],
            parent_head=parent_head,
            update_path=paths["update"],
            sidecar_path=paths["sidecar"],
            checkpoint_path=paths["checkpoint"],
            update_record_sha256=update_hash,
            checkpoint_sha256=loaded_checkpoint.sha256,
            logical_state_sha256=logical_hash,
            model_config=ModelConfig.from_dict(payload["model_config"]),
        ),
        record,
        payload,
    )


def _finite_positive_float(value: Any, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _validate_batch_episodes(value: Any) -> int:
    out = validate_positive_int(value, "batch_episodes", maximum=MAX_BATCH_EPISODES)
    if out < 2 or out % 2 != 0:
        raise ValueError("batch_episodes must be even and at least 2")
    return out


def _validate_until_update(value: Any) -> int:
    if type(value) is not int or value < 0 or value > MAX_UPDATES:
        raise ValueError(f"until_update must be in [0, {MAX_UPDATES}]")
    return value


def _validate_max_decisions(value: Any) -> int:
    return validate_positive_int(value, "max_decisions", maximum=MAX_DECISIONS)


def _learner_seat(episode: int) -> str:
    return "p0" if episode % 2 == 0 else "p1"


def _opponent_seat(learner: str) -> str:
    return "p1" if learner == "p0" else "p0"


def _env_seed_for_episode(base_seed: int, episode: int) -> int:
    return derive_train_env_seed(base_seed, episode // 2)


def _compatibility_tuple() -> dict[str, Any]:
    configure_torch_determinism()
    torch_config_digest = sha256_bytes(str(torch.__config__.show()).encode("utf-8"))
    if torch.empty(0).device.type != "cpu":
        raise RuntimeError("Torch default tensor device is not CPU")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_byteorder": sys.byteorder,
        "torch_version": str(torch.__version__),
        "torch_config_sha256": torch_config_digest,
        "os_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "cpu_only": True,
        "default_device": "cpu",
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def runtime_compatibility() -> dict[str, Any]:
    """Return the deterministic runtime contract used by TrainingStore runs."""

    return _compatibility_tuple()


def _seed_derivation_dict() -> dict[str, Any]:
    data = dataclasses.asdict(TrainerSeedDerivation())
    data["namespaces"] = list(data["namespaces"])
    return data


def _artifact_schemas() -> dict[str, str]:
    return {
        "run": RUN_SCHEMA,
        "latest": LATEST_SCHEMA,
        "update_record": UPDATE_RECORD_SCHEMA,
        "checkpoint": CHECKPOINT_SCHEMA,
        "sidecar": SIDECAR_SCHEMA,
        "episode_summary": EPISODE_SUMMARY_SCHEMA,
        "summary": SUMMARY_SCHEMA,
    }


def _artifact_boundary_contract() -> dict[str, Any]:
    from . import artifact_io as json_contract
    from . import checkpoint as checkpoint_contract
    from . import checkpoint_io as zip_contract

    return {
        "schema": "kernel_rl_artifact_boundary/v9",
        "format": {
            "checkpoint_container": "torch-zip",
            "checkpoint_zip_root": zip_contract.TORCH_ZIP_ROOT,
            "authoritative_json": "canonical-sorted-ascii-json-lf",
        },
        "safe_load": {
            "checkpoint_read": "single bounded regular non-link file read",
            "raw_zip": "EOCD/ZIP64 locator/ZIP64 EOCD, central directory, complete local records including authenticated local extra fields, and signed Torch descriptors parsed from captured bytes before ZipFile",
            "pickle_preflight": "restricted protocol-2 opcode interpreter validates exact dict root graph, storage persistent IDs, tensor reachability, and tensor rebuild metadata before torch.load",
            "torch_load": "io.BytesIO(the same captured bytes), map_location=cpu, weights_only=True, never falling back",
            "residual": "local trainer-artifact byte/object/tensor bounds, not a general arbitrary hostile-pickle sandbox",
        },
        "byte_limits": {
            "checkpoint_file": zip_contract.MAX_CHECKPOINT_FILE_BYTES,
            "zip_entries": zip_contract.MAX_TORCH_ZIP_ENTRIES,
            "zip_central_directory": zip_contract.MAX_TORCH_CENTRAL_DIRECTORY_BYTES,
            "zip_uncompressed": zip_contract.MAX_TORCH_ZIP_UNCOMPRESSED_BYTES,
            "zip_storage": zip_contract.MAX_TORCH_ZIP_STORAGE_BYTES,
            "data_pkl": zip_contract.MAX_TORCH_DATA_PKL_BYTES,
            "run_json": json_contract.MAX_RUN_JSON_BYTES,
            "latest_json": json_contract.MAX_SMALL_JSON_BYTES,
            "sidecar_json": json_contract.MAX_SMALL_JSON_BYTES,
            "update_json": json_contract.MAX_UPDATE_JSON_BYTES,
        },
        "pickle_limits": {
            "opcodes": zip_contract.MAX_PICKLE_OPCODES,
            "memo_writes": zip_contract.MAX_PICKLE_MEMO_WRITES,
            "stack_depth": zip_contract.MAX_PICKLE_STACK_DEPTH,
            "marks": zip_contract.MAX_PICKLE_MARKS,
            "container_items": zip_contract.MAX_PICKLE_CONTAINER_ITEMS,
            "total_items": zip_contract.MAX_PICKLE_TOTAL_ITEMS,
            "graph_nodes": zip_contract.MAX_PICKLE_GRAPH_NODES,
            "graph_depth": zip_contract.MAX_PICKLE_GRAPH_DEPTH,
            "tensor_rank": zip_contract.MAX_PICKLE_TENSOR_RANK,
            "allowed_globals": sorted(zip_contract.ALLOWED_PICKLE_GLOBALS),
            "storage": "BINPERSID must be ('storage', allowed CPU storage type, canonical unique decimal key, 'cpu', bounded element count) and exactly match archive/data/<key> bytes",
            "tensor_rebuild": "_rebuild_tensor_v2 only; unique storage reference, exact integer zero offset, rank <= 16, bounded nonnegative shape, exact positive contiguous strides, exact full-storage byte coverage, false requires_grad, empty OrderedDict metadata, and every constructed tensor declaration must be reachable from the exact dict root",
        },
        "json_limits": {
            "depth": json_contract.MAX_JSON_DEPTH,
            "nodes": json_contract.MAX_JSON_NODES,
            "items": json_contract.MAX_JSON_ITEMS,
            "string_bytes": json_contract.MAX_JSON_STRING_BYTES,
            "one_string_bytes": json_contract.MAX_JSON_ONE_STRING_BYTES,
            "numeric_digits": json_contract.MAX_JSON_NUMERIC_DIGITS,
            "integer_bits": json_contract.MAX_JSON_INTEGER_BITS,
        },
        "object_limits": {
            "depth": checkpoint_contract.MAX_CHECKPOINT_DEPTH,
            "nodes": checkpoint_contract.MAX_CHECKPOINT_NODES,
            "items": checkpoint_contract.MAX_CHECKPOINT_TOTAL_ITEMS,
            "string_bytes": checkpoint_contract.MAX_CHECKPOINT_STRING_BYTES,
        },
        "tensor_limits": {
            "count": checkpoint_contract.MAX_CHECKPOINT_TENSORS,
            "one_tensor_elements": checkpoint_contract.MAX_CHECKPOINT_TENSOR_ELEMENTS,
            "one_tensor_bytes": checkpoint_contract.MAX_CHECKPOINT_TENSOR_BYTES,
            "total_elements": checkpoint_contract.MAX_CHECKPOINT_TOTAL_TENSOR_ELEMENTS,
            "total_bytes": checkpoint_contract.MAX_CHECKPOINT_TOTAL_TENSOR_BYTES,
            "total_storage_bytes": checkpoint_contract.MAX_CHECKPOINT_TOTAL_STORAGE_BYTES,
            "policy": "plain CPU dense contiguous tensors only; no subclasses, views, aliases, shared storage, or nonzero offsets",
        },
        "lock": {
            "algorithm": "one constant persistent child lock file named .mtg-kernel-train.lock under the physical output root",
            "identity": "component-create and validate the real output directory first; ancestor aliases converge through the resolved physical root while direct output-root links/reparse points are rejected",
            "windows": "msvcrt.locking LK_NBLCK on persistent one-byte regular file",
            "posix": "fcntl.flock LOCK_EX|LOCK_NB on persistent regular file",
            "lifecycle": "held for complete train call including client lifetime, commit, recovery, no-op resume, and shutdown",
            "file_policy": "persistent lock file is never unlinked or truncated; descriptor is non-inheritable",
            "threat_boundary": "cooperating local processes on a local filesystem; hostile concurrent namespace swaps outside the validated output-root lock file are not claimed",
        },
        "path_policy": {
            "containment": "lexical artifact-root containment",
            "links": "reject symlink and Windows reparse components, including dangling or in-root links",
            "generation_names": "writer-created committed generation files and transaction directories use exact ASCII decimal update names; Unicode decimal digits and bounded component overflows fail closed before recovery or environment launch",
            "creation": "all output, transaction, quarantine, lock-parent, and generation-directory creation is component-wise no-follow",
            "traversal": "os.scandir/lstat no-follow traversal and contained atomic quarantine rename after destination parents validate",
            "resume": "resume path must lexically equal selected latest.json",
            "pre_manifest_recovery": "under the output lock, fresh bootstrap validates the entire root then may remove only empty real updates/checkpoints directories and canonical bounded .run.json.<pid>.<monotonic_ns>.tmp files before run.json exists; unknown entries, nonempty directories, links, reparse points, malformed temps, and lock impostors fail closed with zero cleanup",
            "resume_recovery": "resume builds an immutable debris plan, validates every latest-reachable committed generation while tolerating only planned debris, prevalidates the entire plan with zero mutation, then immediately revalidates identities per action before cleanup and derived-cache rebuild",
        },
        "privacy": {
            "scan": "authoritative JSON keys and values plus checkpoint scalar metadata",
            "rejects": "generic POSIX absolute roots including whitespace-leading first components, spaced-slash text with any additional slash/backslash separator, chained division outside the single-separator arithmetic exemption, Windows drive-root, all Windows root-relative separators including whitespace-leading first components, UNC, device/extended, file URI, controls/format-boundary embedded fragments, combining-mark-hidden path boundaries after punctuation/assignment, invalid or diagnostic URI spellings, malformed HTTP(S) authorities, and schema-fragment prefixes followed by local absolute paths",
            "allows": "numeric versions, digests, narrow arithmetic slash text only when a single slash is the sole slash/backslash in the complete string, relative namespace labels including word-like bases with combining marks, exact ordinary whole HTTP(S) URIs with case-insensitive scheme and validated authority, and validated whole schema reference tokens",
        },
    }


def _run_manifest_from_config(
    *,
    env_sha: str,
    provenance: dict[str, Any],
    model_config: dict[str, Any],
    base_seed: int,
    batch_episodes: int,
    learning_rate: float,
    value_coef: float,
    max_decisions: int,
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    cfg = ModelConfig.from_dict(model_config)
    optimizer = adam_config(learning_rate)
    return {
        "schema": RUN_SCHEMA,
        "package": {"name": "mtg-kernel-rl", "version": __version__},
        "algorithm": {
            "name": ALGORITHM_NAME,
            "loss": (
                "loss = (sum(-log_prob(selected) * (terminal_return - value.detach())) "
                "+ value_coef * sum((value - terminal_return)^2)) divided by learner_decision_count"
            ),
            "discount": None,
            "entropy_bonus": None,
            "bootstrap": None,
        },
        "environment": {"binary_sha256": env_sha},
        "protocol": {"schema_version": 2, "protocol": "kernel_rl_jsonl", "protocol_version": 2},
        "protocol_provenance": provenance,
        "feature_contract": {
            "feature_schema_version": model_config["feature_schema_version"],
            "feature_registry_version": model_config["feature_registry_version"],
            "feature_contract_digest": model_config["feature_contract_digest"],
            "feature_encoding_digest": model_config["feature_encoding_digest"],
        },
        "model": {"config": model_config, "contract_fingerprint": cfg.contract_fingerprint()},
        "initializer": {
            "name": INITIALIZER_TRAINER_SEEDED_V1,
            "seed": derive_model_init_seed(base_seed),
            "namespace": "model-init/base_seed",
        },
        "optimizer": optimizer,
        "samplers": {
            "learner": {
                "algorithm": "torch.multinomial(softmax(logits), replacement=false, generator=actor-local-cpu)",
                "seed_namespace": "train-learner-action/base_seed/episode_index/learner_decision_index",
                "global_python_rng": "unused",
                "global_torch_rng": "unused",
            },
            "opponent": {
                "algorithm": "uniform-index = train-opponent-action-seed % legal_action_count",
                "seed_namespace": "train-opponent-action/base_seed/episode_index/opponent_decision_index",
                "counter_scope": "actor-local opponent decisions within episode",
            },
        },
        "schedule": {
            "learner_seat": "p0 for even global episodes, p1 for odd global episodes",
            "paired_env_seed": "episodes 2k and 2k+1 share train-env/base_seed/pair_index",
            "batch_episodes": batch_episodes,
        },
        "trainer": {
            "base_seed": base_seed,
            "value_coef": value_coef,
            "max_decisions": max_decisions,
            "terminal_returns": {"learner_win": 1, "draw": 0, "learner_loss": -1},
        },
        "seed_derivation": _seed_derivation_dict(),
        "compatibility": compatibility,
        "artifact_schemas": _artifact_schemas(),
        "artifact_boundary": _artifact_boundary_contract(),
        "privacy_contract": {
            "forbidden_raw_fields": sorted(FORBIDDEN_TRAINING_JSON_KEYS),
        },
    }


def _run_manifest(
    *,
    env_sha: str,
    provenance: dict[str, Any],
    model: KernelPolicyValueNet,
    base_seed: int,
    batch_episodes: int,
    learning_rate: float,
    value_coef: float,
    max_decisions: int,
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    return _run_manifest_from_config(
        env_sha=env_sha,
        provenance=provenance,
        model_config=model.config.to_dict(),
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=compatibility,
    )


def _is_hex64(value: Any) -> bool:
    return type(value) is str and HEX64_RE.fullmatch(value) is not None


def _validate_hash(value: Any, name: str) -> str:
    if not _is_hex64(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _validate_provenance(value: Any) -> None:
    required = {"protocol", "protocol_version", "schema_version", "kernel_version", "surface_version", "card_db_hash"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("run provenance keys mismatch")
    if value["protocol"] != "kernel_rl_jsonl":
        raise ValueError("run provenance protocol mismatch")
    for key in ("protocol_version", "schema_version", "surface_version"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"run provenance {key} must be a nonnegative int")
    if type(value["kernel_version"]) is not str or not value["kernel_version"]:
        raise ValueError("run provenance kernel_version must be nonempty")
    if type(value["card_db_hash"]) is not int or value["card_db_hash"] < 0 or value["card_db_hash"] > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("run provenance card_db_hash out of range")


def _validate_run_manifest(
    run: Any,
    *,
    env_sha: str | None = None,
    compatibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema",
        "package",
        "algorithm",
        "environment",
        "protocol",
        "protocol_provenance",
        "feature_contract",
        "model",
        "initializer",
        "optimizer",
        "samplers",
        "schedule",
        "trainer",
        "seed_derivation",
        "compatibility",
        "artifact_schemas",
        "artifact_boundary",
        "privacy_contract",
    }
    if not isinstance(run, dict) or set(run) != required:
        raise ValueError("run.json keys mismatch")
    if run["schema"] != RUN_SCHEMA:
        raise ValueError("run.json schema mismatch")
    validate_training_json_privacy(run)
    committed_env_sha = _validate_hash(run.get("environment", {}).get("binary_sha256"), "run environment binary_sha256")
    if env_sha is not None and committed_env_sha != env_sha:
        raise ValueError("environment executable SHA-256 drift")
    if compatibility is not None and run["compatibility"] != compatibility:
        raise ValueError("runtime compatibility tuple drift")
    _validate_provenance(run["protocol_provenance"])
    model_entry = run["model"]
    if not isinstance(model_entry, dict) or set(model_entry) != {"config", "contract_fingerprint"}:
        raise ValueError("run model keys mismatch")
    model_config = ModelConfig.from_dict(model_entry["config"]).to_dict()
    base_seed = validate_uint63(run.get("trainer", {}).get("base_seed"), "base_seed")
    batch_episodes = _validate_batch_episodes(run.get("schedule", {}).get("batch_episodes"))
    learning_rate = _finite_positive_float(run.get("optimizer", {}).get("lr"), "learning_rate")
    value_coef = _finite_positive_float(run.get("trainer", {}).get("value_coef"), "value_coef")
    max_decisions = _validate_max_decisions(run.get("trainer", {}).get("max_decisions"))
    expected = _run_manifest_from_config(
        env_sha=committed_env_sha,
        provenance=run["protocol_provenance"],
        model_config=model_config,
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=run["compatibility"],
    )
    if run != expected:
        raise ValueError("run.json training contract drift")
    return run


def _validate_resume_overrides(
    *,
    base_seed: int | None,
    batch_episodes: int | None,
    learning_rate: float | None,
    value_coef: float | None,
    max_decisions: int | None,
) -> tuple[int | None, int | None, float | None, float | None, int | None]:
    return (
        None if base_seed is None else validate_uint63(base_seed, "base_seed"),
        None if batch_episodes is None else _validate_batch_episodes(batch_episodes),
        None if learning_rate is None else _finite_positive_float(learning_rate, "learning_rate"),
        None if value_coef is None else _finite_positive_float(value_coef, "value_coef"),
        None if max_decisions is None else _validate_max_decisions(max_decisions),
    )


def _assert_run_matches_options(
    run: dict[str, Any],
    *,
    env_sha: str,
    base_seed: int | None,
    batch_episodes: int | None,
    learning_rate: float | None,
    value_coef: float | None,
    max_decisions: int | None,
    compatibility: dict[str, Any],
) -> None:
    _validate_run_manifest(run, env_sha=env_sha, compatibility=compatibility)
    checks = {
        "base_seed": (base_seed, run["trainer"]["base_seed"]),
        "batch_episodes": (batch_episodes, run["schedule"]["batch_episodes"]),
        "learning_rate": (learning_rate, run["optimizer"]["lr"]),
        "value_coef": (value_coef, run["trainer"]["value_coef"]),
        "max_decisions": (max_decisions, run["trainer"]["max_decisions"]),
    }
    for name, (explicit, committed) in checks.items():
        if explicit is not None and explicit != committed:
            raise ValueError(f"resume override conflicts with committed {name}")


def _validate_latest(latest: Any, *, run_digest: str) -> dict[str, Any]:
    if not isinstance(latest, dict) or set(latest) != {"schema", "update", "run_digest", "head"}:
        raise ValueError("latest.json keys mismatch")
    validate_training_json_privacy(latest)
    if latest["schema"] != LATEST_SCHEMA or latest["run_digest"] != run_digest:
        raise ValueError("latest.json schema or run digest mismatch")
    head_update = latest["update"]
    if type(head_update) is not int or head_update < 0 or head_update > MAX_UPDATES:
        raise ValueError("latest update out of bounds")
    _validate_hash(latest["head"], "latest head")
    return latest


def _validate_sidecar(sidecar: Any, *, update: int, run_digest: str, parent_head: str | None) -> dict[str, Any]:
    if not isinstance(sidecar, dict) or set(sidecar) != {
        "schema",
        "update",
        "run_digest",
        "parent_head",
        "checkpoint_sha256",
        "logical_state_sha256",
        "update_record_sha256",
        "head",
    }:
        raise ValueError("sidecar keys mismatch")
    validate_training_json_privacy(sidecar)
    if sidecar["schema"] != SIDECAR_SCHEMA or sidecar["update"] != update or sidecar["run_digest"] != run_digest:
        raise ValueError("sidecar generation mismatch")
    if sidecar["parent_head"] != parent_head:
        raise ValueError("sidecar parent mismatch")
    _validate_hash(sidecar["checkpoint_sha256"], "sidecar checkpoint_sha256")
    _validate_hash(sidecar["logical_state_sha256"], "sidecar logical_state_sha256")
    _validate_hash(sidecar["update_record_sha256"], "sidecar update_record_sha256")
    _validate_hash(sidecar["head"], "sidecar head")
    return sidecar


def _validate_loss_fields(loss: Any, *, optimizer_step: bool) -> None:
    if not isinstance(loss, dict) or set(loss) != {"policy_sum_hex", "value_sum_hex", "loss_hex"}:
        raise ValueError("loss field keys mismatch")
    if not optimizer_step:
        if loss != {"policy_sum_hex": None, "value_sum_hex": None, "loss_hex": None}:
            raise ValueError("non-step loss fields must be null")
        return
    for key, value in loss.items():
        if type(value) is not str:
            raise ValueError(f"{key} must be a float hex string")
        parsed = float.fromhex(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{key} must be finite")


def _validate_episode_summary(summary: Any, *, expected_episode: int, base_seed: int) -> dict[str, Any]:
    required = {
        "schema",
        "episode",
        "env_seed",
        "learner_seat",
        "terminal_outcome",
        "winner",
        "learner_return",
        "decision_count",
        "learner_decision_count",
        "opponent_decision_count",
        "trajectory_digest",
    }
    if not isinstance(summary, dict) or set(summary) != required:
        raise ValueError("episode summary keys mismatch")
    validate_training_json_privacy(summary)
    if summary["schema"] != EPISODE_SUMMARY_SCHEMA:
        raise ValueError("episode summary schema mismatch")
    if summary["episode"] != expected_episode:
        raise ValueError("episode summary index mismatch")
    if summary["env_seed"] != _env_seed_for_episode(base_seed, expected_episode):
        raise ValueError("episode env seed mismatch")
    learner = _learner_seat(expected_episode)
    if summary["learner_seat"] != learner:
        raise ValueError("episode learner seat mismatch")
    outcome = summary["terminal_outcome"]
    winner = summary["winner"]
    if outcome == "draw":
        if winner is not None:
            raise ValueError("draw episode winner must be null")
        derived_return = 0
    elif outcome == "p0_win":
        if winner != "p0":
            raise ValueError("p0_win winner mismatch")
        derived_return = 1 if learner == "p0" else -1
    elif outcome == "p1_win":
        if winner != "p1":
            raise ValueError("p1_win winner mismatch")
        derived_return = 1 if learner == "p1" else -1
    else:
        raise ValueError("episode terminal outcome must be natural win/draw")
    if summary["learner_return"] != derived_return:
        raise ValueError("episode learner return mismatch")
    for key in ("decision_count", "learner_decision_count", "opponent_decision_count"):
        if type(summary[key]) is not int or summary[key] < 0:
            raise ValueError(f"episode {key} must be a nonnegative int")
    if summary["decision_count"] != summary["learner_decision_count"] + summary["opponent_decision_count"]:
        raise ValueError("episode decision count mismatch")
    _validate_hash(summary["trajectory_digest"], "episode trajectory_digest")
    return summary


def _validate_update_record(
    record: Any,
    *,
    update: int,
    run: dict[str, Any],
    run_digest: str,
    parent_head: str | None,
    previous_payload: dict[str, Any] | None,
    payload: dict[str, Any],
    logical_hash: str,
) -> dict[str, Any]:
    required = {
        "schema",
        "run_digest",
        "update",
        "parent_head",
        "episode_start",
        "episode_count",
        "episode_end_exclusive",
        "optimizer_step",
        "learner_decision_count",
        "loss",
        "episode_summaries",
        "post_update_logical_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("update record keys mismatch")
    validate_training_json_privacy(record)
    if record["schema"] != UPDATE_RECORD_SCHEMA:
        raise ValueError("update record schema mismatch")
    if record["update"] != update:
        raise ValueError("update record index mismatch")
    if record["run_digest"] != run_digest:
        raise ValueError("update record run digest mismatch")
    if record["parent_head"] != parent_head:
        raise ValueError("update record parent mismatch")
    if record["post_update_logical_sha256"] != logical_hash:
        raise ValueError("update record logical digest mismatch")
    if type(record["optimizer_step"]) is not bool:
        raise ValueError("optimizer_step must be bool")
    if type(record["learner_decision_count"]) is not int or record["learner_decision_count"] < 0:
        raise ValueError("learner_decision_count must be a nonnegative int")
    if not isinstance(record["episode_summaries"], list):
        raise ValueError("episode summaries must be a list")

    if update == 0:
        if previous_payload is not None:
            raise ValueError("update 0 must not have a previous payload")
        if record["parent_head"] is not None or record["episode_start"] != 0 or record["episode_count"] != 0:
            raise ValueError("update 0 record range mismatch")
        if record["episode_end_exclusive"] != 0 or record["optimizer_step"] or record["learner_decision_count"] != 0:
            raise ValueError("update 0 record counters mismatch")
        if record["episode_summaries"]:
            raise ValueError("update 0 must not contain episodes")
        _validate_loss_fields(record["loss"], optimizer_step=False)
        if payload["completed_update"] != 0 or payload["optimizer_step_count"] != 0 or payload["next_episode"] != 0:
            raise ValueError("update 0 checkpoint counters mismatch")
        expected_outcomes = {"p0": {"win": 0, "loss": 0, "draw": 0}, "p1": {"win": 0, "loss": 0, "draw": 0}}
        if payload["outcomes_by_learner_seat"] != expected_outcomes or payload["learner_decisions_by_seat"] != {"p0": 0, "p1": 0}:
            raise ValueError("update 0 checkpoint aggregates mismatch")
        return record

    if previous_payload is None:
        raise ValueError("trained update requires previous payload")
    if payload["completed_update"] != previous_payload["completed_update"] + 1 or payload["completed_update"] != update:
        raise ValueError("checkpoint completed update mismatch")
    episode_start = previous_payload["next_episode"]
    batch_episodes = run["schedule"]["batch_episodes"]
    if type(record["episode_start"]) is not int or record["episode_start"] != episode_start:
        raise ValueError("update episode_start mismatch")
    if record["episode_start"] % 2 != 0:
        raise ValueError("update episode_start must be pair-aligned")
    if record["episode_count"] != batch_episodes or record["episode_end_exclusive"] != episode_start + batch_episodes:
        raise ValueError("update episode range mismatch")
    if payload["next_episode"] != record["episode_end_exclusive"]:
        raise ValueError("checkpoint next_episode mismatch")
    if len(record["episode_summaries"]) != batch_episodes:
        raise ValueError("episode summary count mismatch")

    expected_outcomes = {
        "p0": dict(previous_payload["outcomes_by_learner_seat"]["p0"]),
        "p1": dict(previous_payload["outcomes_by_learner_seat"]["p1"]),
    }
    expected_decisions = dict(previous_payload["learner_decisions_by_seat"])
    learner_decision_total = 0
    for offset, summary in enumerate(record["episode_summaries"]):
        row = _validate_episode_summary(summary, expected_episode=episode_start + offset, base_seed=run["trainer"]["base_seed"])
        learner_decision_total += row["learner_decision_count"]
        seat = row["learner_seat"]
        if row["learner_return"] == 1:
            expected_outcomes[seat]["win"] += 1
        elif row["learner_return"] == -1:
            expected_outcomes[seat]["loss"] += 1
        elif row["learner_return"] == 0:
            expected_outcomes[seat]["draw"] += 1
        expected_decisions[seat] += row["learner_decision_count"]
    if learner_decision_total != record["learner_decision_count"]:
        raise ValueError("learner decision total mismatch")
    if record["optimizer_step"] != (learner_decision_total > 0):
        raise ValueError("optimizer step flag mismatch")
    _validate_loss_fields(record["loss"], optimizer_step=record["optimizer_step"])
    expected_step_count = previous_payload["optimizer_step_count"] + (1 if record["optimizer_step"] else 0)
    if payload["optimizer_step_count"] != expected_step_count:
        raise ValueError("checkpoint optimizer_step_count mismatch")
    if payload["outcomes_by_learner_seat"] != expected_outcomes:
        raise ValueError("checkpoint outcome aggregates mismatch")
    if payload["learner_decisions_by_seat"] != expected_decisions:
        raise ValueError("checkpoint learner decision aggregates mismatch")
    return record


def _strict_validate_checkpoint_for_model(
    payload: dict[str, Any],
    *,
    run_digest: str,
    compatibility: dict[str, Any],
    learning_rate: float,
) -> None:
    validate_checkpoint_payload(payload, run_digest=run_digest, compatibility=compatibility)
    cfg = ModelConfig.from_dict(payload["model_config"])
    model = KernelPolicyValueNet(cfg, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=0, configure_runtime=False)
    model.load_state_dict(validate_model_state(model, payload["model_state"]), strict=True)
    validate_adam_state_for_model(
        model,
        payload["optimizer_state"],
        learning_rate,
        expected_step_count=payload["optimizer_step_count"],
    )


def _update_record_zero(run_digest: str, logical_hash: str) -> dict[str, Any]:
    return {
        "schema": UPDATE_RECORD_SCHEMA,
        "run_digest": run_digest,
        "update": 0,
        "parent_head": None,
        "episode_start": 0,
        "episode_count": 0,
        "episode_end_exclusive": 0,
        "optimizer_step": False,
        "learner_decision_count": 0,
        "loss": {"policy_sum_hex": None, "value_sum_hex": None, "loss_hex": None},
        "episode_summaries": [],
        "post_update_logical_sha256": logical_hash,
    }


__all__ = [
    "PolicySnapshot",
    "ResumeSnapshot",
    "SnapshotRef",
    "StoreReadCounts",
    "TrainingStore",
    "ValidatedChain",
    "runtime_compatibility",
]
