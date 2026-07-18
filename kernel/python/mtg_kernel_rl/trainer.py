"""Terminal-only REINFORCE/value trainer with atomic exact resume."""

from __future__ import annotations

import dataclasses
import copy
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch

from .action_sampling import sample_fixed_categorical
from .artifacts import (
    append_current_derived_caches,
    atomic_replace,
    canonical_json_bytes,
    fsync_dir,
    generation_paths,
    inject_fault,
    latest_path,
    preflight_derived_cache_append,
    read_authoritative_json,
    read_json_file,
    rebuild_derived_caches,
    require_new_or_empty_dir,
    sha256_bytes,
    sha256_file,
    validate_training_json_privacy,
    write_bytes_atomic,
    write_json_atomic,
)
from .checkpoint import (
    SIDECAR_SCHEMA,
    UPDATE_RECORD_SCHEMA,
    build_checkpoint_payload,
    build_latest,
    build_sidecar,
    create_adam,
    export_adam_state,
    logical_state_hash,
    load_checkpoint_file,
    load_checkpoint_file_with_digest,
    restore_python_rng_state,
    restore_torch_rng_state,
    save_checkpoint_file,
    update_record_hash,
    validate_checkpoint_payload,
)
from .client import Decision, KernelRlClient, Terminal
from .determinism import (
    configure_torch_determinism,
    derive_model_init_seed,
    derive_train_learner_action_seed,
    derive_train_opponent_action_seed,
    deterministic_index_from_seed,
)
from .features import encode_decision
from .model import INITIALIZER_TRAINER_SEEDED_V1, KernelPolicyValueNet, ModelConfig, model_config_from_encoded
from .output_lock import OutputLock
from .path_safety import (
    atomic_quarantine,
    capture_path_identity,
    ensure_no_follow_path,
    ensure_real_child_dir,
    ensure_real_file,
    is_verified_output_lock_entry,
    mkdir_no_follow,
    revalidate_path_identity,
    remove_tree_no_follow,
    same_lexical_path,
    scandir_no_follow,
)
from .training_store import (
    EPISODE_SUMMARY_SCHEMA,
    MAX_UPDATES,
    RUN_SCHEMA,
    TrainingStore,
    _assert_run_matches_options,
    _compatibility_tuple,
    _env_seed_for_episode,
    _validate_generation_bundle,
    _learner_seat,
    _opponent_seat,
    _run_manifest,
    _strict_validate_checkpoint_for_model,
    _update_record_zero,
    _validate_latest,
    _validate_resume_overrides,
    _validate_run_manifest,
    _validate_until_update,
    _validate_update_record,
)

GENERATION_RE = re.compile(r"^update-([0-9]{8})\.(json|pt)$")
TRANSACTION_RE = re.compile(r"^update-([0-9]{8})-([1-9][0-9]*)\.([1-9][0-9]*)$")

_DECIMAL_COMPONENT_RE = re.compile(r"^[1-9][0-9]*$")
_PREMANIFEST_TEMP_TARGETS = frozenset({"run.json"})
_COMMITTED_ROOT_TEMP_TARGETS = frozenset({"latest.json", "episodes.jsonl", "updates.jsonl", "summary.json"})
_TRANSACTION_TEMP_TARGETS = frozenset({"update.json", "sidecar.json"})
_MAX_PID_COMPONENT = (1 << 32) - 1
_MAX_MONOTONIC_COMPONENT = (1 << 64) - 1
DEFAULT_DECK_IDS = ("Burn", "Burn")

# Legacy private trainer names remain bound for ownership audits; definitions live in training_store.
_PERSISTED_CONTRACT_OWNER_HELPERS = (_validate_latest, _validate_run_manifest, _validate_update_record)


def _validate_requested_deck_ids(value: Any) -> tuple[str, str]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("deck_ids must be an exact two-item tuple")
    if any(type(item) is not str or not item for item in value):
        raise ValueError("deck_ids entries must be nonempty strings")
    return value[0], value[1]


def _assert_response_deck_identity(run: dict[str, Any], response: Decision | Terminal) -> None:
    expected_ids = tuple(run["environment"]["deck_ids"])
    expected_hashes = tuple(run["environment"]["deck_hashes"])
    if response.deck_ids != expected_ids:
        raise ValueError("environment deck_ids drift")
    if response.deck_hashes != expected_hashes:
        raise ValueError("environment deck_hashes drift")


@dataclasses.dataclass
class TrainState:
    out_dir: Path
    run: dict[str, Any]
    run_digest: str
    model: KernelPolicyValueNet
    optimizer: torch.optim.Adam
    completed_update: int
    optimizer_step_count: int
    next_episode: int
    outcomes_by_learner_seat: dict[str, dict[str, int]]
    learner_decisions_by_seat: dict[str, int]
    parent_head: str | None
    head_payload: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _PlannedMkdir:
    path: Path
    parent_identity: Any


@dataclasses.dataclass(frozen=True)
class _RecoveryPlan:
    root_identity: Any
    mkdirs: tuple[_PlannedMkdir, ...] = ()
    quarantines: tuple[tuple[Any, str], ...] = ()
    remove_dirs: tuple[Any, ...] = ()


def _write_run_json(out_dir: Path, run: dict[str, Any]) -> str:
    validate_training_json_privacy(run)
    path = out_dir / "run.json"
    try:
        path.lstat()
        exists = True
    except FileNotFoundError:
        exists = False
    if exists:
        raise FileExistsError("run.json already exists")
    return write_json_atomic(path, run)


def _assert_first_decision_matches_run(run: dict[str, Any], decision: Decision) -> None:
    _assert_response_deck_identity(run, decision)
    if decision.provenance != run["protocol_provenance"]:
        raise ValueError("environment provenance drift on first decision")
    encoded = encode_decision(decision.observation, decision.legal_actions)
    cfg = ModelConfig.from_dict(run["model"]["config"])
    if encoded.schema.version != cfg.feature_schema_version:
        raise ValueError("feature schema drift on first decision")
    if encoded.schema.registry_version != cfg.feature_registry_version:
        raise ValueError("feature registry drift on first decision")
    if encoded.schema.contract_digest != cfg.feature_contract_digest:
        raise ValueError("feature contract drift on first decision")
    if encoded.schema.encoding_digest != cfg.feature_encoding_digest:
        raise ValueError("feature encoding drift on first decision")


def _staged_generation_paths(transaction_dir: Path) -> dict[str, Path]:
    return {
        "update": transaction_dir / "update.json",
        "checkpoint": transaction_dir / "checkpoint.pt",
        "sidecar": transaction_dir / "sidecar.json",
    }


def _new_transaction_dir(out_dir: Path, update: int) -> Path:
    root = out_dir / ".transactions"
    mkdir_no_follow(root, parents=True, exist_ok=True)
    path = root / f"update-{update:08d}-{os.getpid()}.{time.monotonic_ns()}"
    mkdir_no_follow(path, mode=0o700)
    return path


def _validate_path_contained(root: Path, path: Path) -> Path:
    return ensure_no_follow_path(root, path, expected="any")


def _quarantine_file(out_dir: Path, path: Path, reason: str) -> None:
    atomic_quarantine(out_dir, path, reason)


def _quarantine_tree(out_dir: Path, path: Path, reason: str) -> None:
    atomic_quarantine(out_dir, path, reason)


def _entries_excluding_verified_lock(root: Path) -> list[os.DirEntry[str]]:
    entries: list[os.DirEntry[str]] = []
    for entry in scandir_no_follow(root):
        if is_verified_output_lock_entry(root, entry):
            continue
        entries.append(entry)
    return entries


def _atomic_temp_target(name: str, allowed_targets: frozenset[str]) -> str | None:
    if not name.startswith(".") or not name.endswith(".tmp"):
        return None
    for target in allowed_targets:
        prefix = f".{target}."
        if not name.startswith(prefix):
            continue
        body = name[len(prefix) : -len(".tmp")]
        parts = body.split(".")
        if len(parts) != 2:
            raise ValueError(f"malformed temp file: {name}")
        pid, monotonic_ns = parts
        if not _valid_decimal_component(pid, _MAX_PID_COMPONENT) or not _valid_decimal_component(monotonic_ns, _MAX_MONOTONIC_COMPONENT):
            raise ValueError(f"malformed temp file: {name}")
        return target
    return None


def _valid_decimal_component(value: str, maximum: int) -> bool:
    if _DECIMAL_COMPONENT_RE.fullmatch(value) is None:
        return False
    return int(value) <= maximum


def _reject_malformed_temp_for_allowed_targets(name: str, allowed_targets: frozenset[str]) -> None:
    if not name.startswith(".") or not name.endswith(".tmp"):
        return
    for target in allowed_targets:
        if name.startswith(f".{target}."):
            raise ValueError(f"malformed temp file: {name}")


def _revalidate_then_unlink(identity: Any) -> None:
    revalidate_path_identity(identity)
    Path(identity.path).unlink()


def _revalidate_then_rmdir(identity: Any) -> None:
    revalidate_path_identity(identity)
    Path(identity.path).rmdir()


def _revalidate_then_quarantine(root: Path, identity: Any, reason: str) -> None:
    revalidate_path_identity(identity)
    atomic_quarantine(root, Path(identity.path), reason)


def _validate_orphan_generation_file(
    out_dir: Path,
    path: Path,
    *,
    update: int,
    kind: str,
    run_digest: str,
    compatibility: dict[str, Any],
) -> None:
    ensure_real_file(out_dir, path)
    if kind == "update":
        record = read_authoritative_json(path, "update")
        if record.get("schema") != UPDATE_RECORD_SCHEMA or record.get("update") != update or record.get("run_digest") != run_digest:
            raise ValueError(f"uncommitted update record schema mismatch: {path}")
        validate_training_json_privacy(record)
        return
    if kind == "sidecar":
        sidecar = read_authoritative_json(path, "sidecar")
        if sidecar.get("schema") != SIDECAR_SCHEMA or sidecar.get("update") != update or sidecar.get("run_digest") != run_digest:
            raise ValueError(f"uncommitted sidecar schema mismatch: {path}")
        validate_training_json_privacy(sidecar)
        return
    if kind == "checkpoint":
        payload = load_checkpoint_file(path)
        validate_checkpoint_payload(payload, run_digest=run_digest, compatibility=compatibility)
        if payload["completed_update"] != update:
            raise ValueError(f"uncommitted checkpoint update mismatch: {path}")
        return
    raise AssertionError(kind)


def _validate_transaction_tree(out_dir: Path, transaction_dir: Path, *, head_update: int) -> None:
    ensure_real_child_dir(out_dir, transaction_dir)
    match = TRANSACTION_RE.fullmatch(transaction_dir.name)
    if match is None:
        raise ValueError(f"unknown transaction staging directory name: {transaction_dir.name}")
    update = int(match.group(1))
    if not _valid_decimal_component(match.group(2), _MAX_PID_COMPONENT) or not _valid_decimal_component(match.group(3), _MAX_MONOTONIC_COMPONENT):
        raise ValueError(f"unknown transaction staging directory name: {transaction_dir.name}")
    if update < 0 or update > MAX_UPDATES:
        raise ValueError("staging directory update out of bounds")
    allowed = {"checkpoint.pt", "update.json", "sidecar.json"}
    for entry in scandir_no_follow(transaction_dir):
        child = Path(entry.path)
        _validate_path_contained(out_dir, child)
        if entry.is_dir(follow_symlinks=False):
            raise ValueError(f"unexpected nested transaction directory: {child.name}")
        if not entry.is_file(follow_symlinks=False):
            raise ValueError(f"unknown transaction staging entry: {child.name}")
        if child.parent != transaction_dir:
            raise ValueError(f"unexpected nested transaction file: {child.name}")
        if child.name in allowed:
            ensure_real_file(out_dir, child)
            continue
        if _atomic_temp_target(child.name, _TRANSACTION_TEMP_TARGETS) is not None:
            ensure_real_file(out_dir, child)
            continue
        _reject_malformed_temp_for_allowed_targets(child.name, _TRANSACTION_TEMP_TARGETS)
        if child.name not in allowed:
            raise ValueError(f"unknown transaction staging file: {child.name}")


def _plan_uncommitted_artifact_recovery(
    out_dir: Path,
    *,
    head_update: int,
    run_digest: str,
    compatibility: dict[str, Any],
) -> _RecoveryPlan:
    ensure_real_child_dir(out_dir.parent, out_dir)
    root_identity = capture_path_identity(out_dir)
    mkdirs: list[_PlannedMkdir] = []
    quarantine_actions: list[tuple[Any, str]] = []
    remove_dirs: list[Any] = []
    transactions = out_dir / ".transactions"
    if transactions.exists():
        ensure_real_child_dir(out_dir, transactions)
        transaction_entries = scandir_no_follow(transactions)
        if transaction_entries:
            for entry in transaction_entries:
                child = Path(entry.path)
                if not entry.is_dir(follow_symlinks=False):
                    raise ValueError(f"unknown transaction entry: {child.name}")
                _validate_transaction_tree(out_dir, child, head_update=head_update)
            quarantine_actions.append((capture_path_identity(transactions), "staging"))
        else:
            remove_dirs.append(capture_path_identity(transactions))

    allowed_root_files = {"run.json", "latest.json", "episodes.jsonl", "updates.jsonl", "summary.json"}
    allowed_root_dirs = {"updates", "checkpoints", ".transactions", ".quarantine"}
    for entry in _entries_excluding_verified_lock(out_dir):
        path = Path(entry.path)
        if entry.name in allowed_root_files:
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"root artifact is not a file: {entry.name}")
            ensure_real_file(out_dir, path)
            continue
        if entry.name in allowed_root_dirs:
            if not entry.is_dir(follow_symlinks=False):
                raise ValueError(f"root artifact directory is not a directory: {entry.name}")
            ensure_real_child_dir(out_dir, path)
            continue
        if _atomic_temp_target(entry.name, _COMMITTED_ROOT_TEMP_TARGETS) is not None:
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"temp artifact is not a file: {entry.name}")
            ensure_real_file(out_dir, path, reject_hardlinks=False)
            quarantine_actions.append((capture_path_identity(path), "temp"))
            continue
        _reject_malformed_temp_for_allowed_targets(entry.name, _COMMITTED_ROOT_TEMP_TARGETS)
        raise ValueError(f"unknown root artifact entry: {entry.name}")

    for directory_name, mapping in (
        ("updates", {"json": "update"}),
        ("checkpoints", {"pt": "checkpoint", "json": "sidecar"}),
    ):
        directory = out_dir / directory_name
        if not directory.exists():
            mkdirs.append(_PlannedMkdir(path=directory, parent_identity=capture_path_identity(directory.parent)))
            continue
        ensure_real_child_dir(out_dir, directory)
        for entry in scandir_no_follow(directory):
            path = Path(entry.path)
            _validate_path_contained(out_dir, path)
            if entry.is_dir(follow_symlinks=False):
                raise ValueError(f"unknown directory under {directory_name}: {path.name}")
            match = GENERATION_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unknown generation artifact name: {path.name}")
            update = int(match.group(1))
            suffix = match.group(2)
            kind = mapping.get(suffix)
            if kind is None:
                raise ValueError(f"generation artifact in wrong directory: {path.name}")
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"generation artifact is not a file: {path.name}")
            ensure_real_file(out_dir, path)
            if update <= head_update:
                continue
            _validate_orphan_generation_file(out_dir, path, update=update, kind=kind, run_digest=run_digest, compatibility=compatibility)
            quarantine_actions.append((capture_path_identity(path), "uncommitted"))
    return _RecoveryPlan(
        root_identity=root_identity,
        mkdirs=tuple(mkdirs),
        quarantines=tuple(quarantine_actions),
        remove_dirs=tuple(remove_dirs),
    )


def _apply_uncommitted_artifact_recovery(out_dir: Path, plan: _RecoveryPlan) -> None:
    _prevalidate_uncommitted_artifact_recovery(out_dir, plan)
    for planned in plan.mkdirs:
        try:
            planned.path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"planned recovery directory already exists: {planned.path}")
        mkdir_no_follow(planned.path, parents=True, exist_ok=False)
    for identity, reason in plan.quarantines:
        _revalidate_then_quarantine(out_dir, identity, reason)
    for identity in plan.remove_dirs:
        if any(scandir_no_follow(Path(identity.path))):
            raise ValueError(f"planned empty recovery directory became nonempty: {identity.path}")
        _revalidate_then_rmdir(identity)


def _prevalidate_uncommitted_artifact_recovery(out_dir: Path, plan: _RecoveryPlan) -> None:
    revalidate_path_identity(plan.root_identity)
    for planned in plan.mkdirs:
        revalidate_path_identity(planned.parent_identity)
        try:
            planned.path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"planned recovery directory already exists: {planned.path}")
    for identity, reason in plan.quarantines:
        revalidate_path_identity(identity)
        _prevalidate_quarantine_destination(out_dir, Path(identity.path), reason)
    for identity in plan.remove_dirs:
        revalidate_path_identity(identity)
        if any(scandir_no_follow(Path(identity.path))):
            raise ValueError(f"planned empty recovery directory became nonempty: {identity.path}")


def _prevalidate_quarantine_destination(out_dir: Path, path: Path, reason: str) -> None:
    if type(reason) is not str or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", reason) is None:
        raise ValueError("quarantine reason must be one safe ASCII component")
    _validate_path_contained(out_dir, path)
    quarantine_root = out_dir / ".quarantine"
    if quarantine_root.exists():
        ensure_real_child_dir(out_dir, quarantine_root)


def _commit_generation(
    *,
    out_dir: Path,
    run: dict[str, Any],
    payload: dict[str, Any],
    update_record: dict[str, Any],
    run_digest: str,
    parent_head: str | None,
    previous_payload: dict[str, Any] | None,
    compatibility: dict[str, Any],
    learning_rate: float,
) -> tuple[str, dict[str, Any]]:
    update = payload["completed_update"]
    cache_preflight = preflight_derived_cache_append(out_dir) if update > 0 else None
    mkdir_no_follow(out_dir, parents=True, exist_ok=True)
    mkdir_no_follow(out_dir / "updates", parents=True, exist_ok=True)
    mkdir_no_follow(out_dir / "checkpoints", parents=True, exist_ok=True)
    paths = generation_paths(out_dir, update)
    for path in paths.values():
        ensure_no_follow_path(out_dir, path.parent, expected="dir")
        try:
            path.lstat()
            exists = True
        except FileNotFoundError:
            exists = False
        if exists:
            raise FileExistsError(f"immutable generation already exists: {path}")

    tx_dir = _new_transaction_dir(out_dir, update)
    staged = _staged_generation_paths(tx_dir)
    try:
        save_checkpoint_file(staged["checkpoint"], payload)
        loaded_checkpoint = load_checkpoint_file_with_digest(staged["checkpoint"])
        loaded = loaded_checkpoint.payload
        _strict_validate_checkpoint_for_model(loaded, run_digest=run_digest, compatibility=compatibility, learning_rate=learning_rate)
        logical_hash = logical_state_hash(loaded)
        checkpoint_hash = loaded_checkpoint.sha256
        update_bytes = canonical_json_bytes(update_record)
        update_hash = sha256_bytes(update_bytes)
        write_bytes_atomic(staged["update"], update_bytes)
        if update_hash != update_record_hash(update_record):
            raise ValueError("update record canonical hash mismatch")
        sidecar = build_sidecar(
            update=update,
            run_digest=run_digest,
            parent_head=parent_head,
            checkpoint_sha256=checkpoint_hash,
            logical_hash=logical_hash,
            update_hash=update_hash,
        )
        write_bytes_atomic(staged["sidecar"], canonical_json_bytes(sidecar))
        run_for_validation = dict(run)
        run_for_validation["_artifact_root"] = str(out_dir)
        _validate_generation_bundle(
            paths=staged,
            update=update,
            run=run_for_validation,
            run_digest=run_digest,
            parent_head=parent_head,
            previous_payload=previous_payload,
            compatibility=compatibility,
            learning_rate=learning_rate,
        )
        inject_fault("generation_validate", tx_dir)

        for key in ("checkpoint", "update", "sidecar"):
            ensure_real_file(out_dir, staged[key])
            ensure_no_follow_path(out_dir, paths[key].parent, expected="dir")
            try:
                paths[key].lstat()
                raise FileExistsError(f"immutable generation already exists: {paths[key]}")
            except FileNotFoundError:
                pass
            inject_fault(f"final_replace_{key}_before", staged[key])
            atomic_replace(staged[key], paths[key])
            fsync_dir(paths[key].parent)
            inject_fault(f"final_replace_{key}_after", paths[key])
        final_head, final_record, final_payload = _validate_generation_bundle(
            paths=paths,
            update=update,
            run=run_for_validation,
            run_digest=run_digest,
            parent_head=parent_head,
            previous_payload=previous_payload,
            compatibility=compatibility,
            learning_rate=learning_rate,
        )
        latest = build_latest(update=update, run_digest=run_digest, head=final_head)
        validate_training_json_privacy(latest)
        inject_fault("latest_replace_before", latest_path(out_dir))
        write_bytes_atomic(latest_path(out_dir), canonical_json_bytes(latest))
        inject_fault("latest_replace_after", latest_path(out_dir))
        if update == 0:
            rebuild_derived_caches(out_dir, [final_record], latest)
        else:
            append_current_derived_caches(
                out_dir,
                current_record=final_record,
                latest=latest,
                checkpoint_payload=final_payload,
                preflight=cache_preflight,
            )
        inject_fault("post_latest_cleanup_before", tx_dir)
        return final_head, final_payload
    finally:
        if tx_dir.exists():
            remove_tree_no_follow(tx_dir)
        root = out_dir / ".transactions"
        if root.exists():
            try:
                if not any(scandir_no_follow(root)):
                    root.rmdir()
            except OSError:
                pass


def _load_chain(
    *,
    out_dir: Path,
    env_sha: str,
    until_update: int,
    deck_ids: tuple[str, str],
    base_seed: int | None,
    batch_episodes: int | None,
    learning_rate: float | None,
    value_coef: float | None,
    max_decisions: int | None,
    compatibility: dict[str, Any],
    before_mutation: Callable[[dict[str, Any], Any], None] | None = None,
) -> TrainState:
    chain = TrainingStore(out_dir).validate_latest()
    run = chain.run_record
    run_digest = chain.head.run_digest
    (
        base_seed,
        batch_episodes,
        learning_rate,
        value_coef,
        max_decisions,
    ) = _validate_resume_overrides(
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
    )
    _assert_run_matches_options(
        run,
        env_sha=env_sha,
        deck_ids=deck_ids,
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=compatibility,
    )
    if until_update < chain.head.update:
        raise ValueError("until_update must be at least the committed update")
    recovery_plan = _plan_uncommitted_artifact_recovery(
        out_dir,
        head_update=chain.head.update,
        run_digest=run_digest,
        compatibility=compatibility,
    )
    resume_snapshot = chain.load_resume(chain.head)
    records = list(chain.update_records)
    latest = chain.latest_record
    if before_mutation is not None:
        before_mutation(run, resume_snapshot)
    _apply_uncommitted_artifact_recovery(out_dir, recovery_plan)
    rebuild_derived_caches(out_dir, records, latest)
    restore_python_rng_state(resume_snapshot.python_rng_state)
    restore_torch_rng_state(resume_snapshot.torch_cpu_rng_state)
    return TrainState(
        out_dir=out_dir,
        run=run,
        run_digest=run_digest,
        model=resume_snapshot.model,
        optimizer=resume_snapshot.optimizer,
        completed_update=resume_snapshot.completed_update,
        optimizer_step_count=resume_snapshot.optimizer_step_count,
        next_episode=resume_snapshot.next_episode,
        outcomes_by_learner_seat=copy.deepcopy(resume_snapshot.outcomes_by_learner_seat),
        learner_decisions_by_seat=copy.deepcopy(resume_snapshot.learner_decisions_by_seat),
        parent_head=chain.head.head,
        head_payload=resume_snapshot.checkpoint_payload,
    )


def _bootstrap_fresh(
    *,
    env_bin: Path,
    out_dir: Path,
    deck_ids: tuple[str, str],
    base_seed: int,
    batch_episodes: int,
    learning_rate: float,
    value_coef: float,
    max_decisions: int,
    compatibility: dict[str, Any],
) -> tuple[TrainState, KernelRlClient, Decision]:
    require_new_or_empty_dir(out_dir)
    env_sha = sha256_file(env_bin)
    client = KernelRlClient(env_bin, timeout_s=10.0)
    try:
        first = client.reset(
            episode_id=0,
            env_seed=_env_seed_for_episode(base_seed, 0),
            max_decisions=max_decisions,
            deck_ids=deck_ids,
        )
        if not isinstance(first, Decision):
            raise ValueError("fresh trainer reset must return an initial decision")
        if first.deck_ids != deck_ids:
            raise ValueError("environment deck_ids differ from requested deck_ids")
        encoded = encode_decision(first.observation, first.legal_actions)
        random.seed(derive_model_init_seed(base_seed))
        torch.manual_seed(derive_model_init_seed(base_seed))
        model = KernelPolicyValueNet(
            model_config_from_encoded(encoded),
            initializer=INITIALIZER_TRAINER_SEEDED_V1,
            initializer_seed=derive_model_init_seed(base_seed),
        )
        optimizer = create_adam(model, learning_rate)
        run = _run_manifest(
            env_sha=env_sha,
            deck_ids=first.deck_ids,
            deck_hashes=first.deck_hashes,
            provenance=first.provenance,
            model=model,
            base_seed=base_seed,
            batch_episodes=batch_episodes,
            learning_rate=learning_rate,
            value_coef=value_coef,
            max_decisions=max_decisions,
            compatibility=compatibility,
        )
        run_digest = _write_run_json(out_dir, run)
        if run_digest != sha256_file(out_dir / "run.json", max_bytes=256 * 1024, allow_empty=False):
            raise ValueError("run.json digest mismatch")
        outcomes = {"p0": {"win": 0, "loss": 0, "draw": 0}, "p1": {"win": 0, "loss": 0, "draw": 0}}
        decisions = {"p0": 0, "p1": 0}
        payload = build_checkpoint_payload(
            run_digest=run_digest,
            completed_update=0,
            optimizer_step_count=0,
            next_episode=0,
            outcomes_by_learner_seat=outcomes,
            learner_decisions_by_seat=decisions,
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            base_seed=base_seed,
            seed_derivation=run["seed_derivation"],
            provenance=first.provenance,
            compatibility=compatibility,
        )
        update0 = _update_record_zero(run_digest, logical_state_hash(payload))
        _head, _payload = _commit_generation(
            out_dir=out_dir,
            run=run,
            payload=payload,
            update_record=update0,
            run_digest=run_digest,
            parent_head=None,
            previous_payload=None,
            compatibility=compatibility,
            learning_rate=learning_rate,
        )
        reloaded = _load_chain(
            out_dir=out_dir,
            env_sha=env_sha,
            until_update=0,
            deck_ids=deck_ids,
            base_seed=base_seed,
            batch_episodes=batch_episodes,
            learning_rate=learning_rate,
            value_coef=value_coef,
            max_decisions=max_decisions,
            compatibility=compatibility,
        )
        return reloaded, client, first
    except Exception:
        client.close()
        raise


def _recover_pre_manifest_bootstrap(out_dir: Path) -> None:
    ensure_real_child_dir(out_dir.parent, out_dir)
    actions: list[tuple[str, Any]] = []
    for entry in _entries_excluding_verified_lock(out_dir):
        child = Path(entry.path)
        if entry.name in {"updates", "checkpoints"}:
            if not entry.is_dir(follow_symlinks=False):
                raise ValueError(f"pre-manifest bootstrap debris is not a directory: {entry.name}")
            ensure_real_child_dir(out_dir, child)
            if any(scandir_no_follow(child)):
                raise ValueError(f"pre-manifest bootstrap directory is not empty: {entry.name}")
            actions.append(("rmdir", capture_path_identity(child)))
            continue
        if _atomic_temp_target(entry.name, _PREMANIFEST_TEMP_TARGETS) is not None:
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"pre-manifest run.json temp is malformed: {entry.name}")
            ensure_real_file(out_dir, child, reject_hardlinks=False)
            temp_value = read_json_file(child, max_bytes=256 * 1024, require_canonical=True)
            if not isinstance(temp_value, dict) or temp_value.get("schema") != RUN_SCHEMA:
                raise ValueError(f"pre-manifest run.json temp content is malformed: {entry.name}")
            validate_training_json_privacy(temp_value)
            actions.append(("unlink", capture_path_identity(child)))
            continue
        _reject_malformed_temp_for_allowed_targets(entry.name, _PREMANIFEST_TEMP_TARGETS)
        raise ValueError(f"unknown pre-manifest bootstrap debris: {entry.name}")
    for action, identity in actions:
        if action == "rmdir":
            _revalidate_then_rmdir(identity)
        elif action == "unlink":
            _revalidate_then_unlink(identity)
        else:
            raise AssertionError(action)


def _bootstrap_incomplete_fresh(
    *,
    env_bin: Path,
    out_dir: Path,
    deck_ids: tuple[str, str],
    base_seed: int,
    batch_episodes: int,
    learning_rate: float,
    value_coef: float,
    max_decisions: int,
    compatibility: dict[str, Any],
) -> tuple[TrainState, KernelRlClient, Decision]:
    env_sha = sha256_file(env_bin)
    run_path = out_dir / "run.json"
    ensure_real_file(out_dir, run_path)
    run = read_authoritative_json(run_path, "run")
    run_digest = sha256_file(run_path, max_bytes=256 * 1024, allow_empty=False)
    _assert_run_matches_options(
        run,
        env_sha=env_sha,
        deck_ids=deck_ids,
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=compatibility,
    )
    recovery_plan = _plan_uncommitted_artifact_recovery(
        out_dir,
        head_update=-1,
        run_digest=run_digest,
        compatibility=compatibility,
    )
    client = KernelRlClient(env_bin, timeout_s=10.0)
    try:
        first = client.reset(
            episode_id=0,
            env_seed=_env_seed_for_episode(base_seed, 0),
            max_decisions=max_decisions,
            deck_ids=deck_ids,
        )
        if not isinstance(first, Decision):
            raise ValueError("incomplete fresh trainer reset must return an initial decision")
        _assert_first_decision_matches_run(run, first)
        _apply_uncommitted_artifact_recovery(out_dir, recovery_plan)
        cfg = ModelConfig.from_dict(run["model"]["config"])
        random.seed(run["initializer"]["seed"])
        torch.manual_seed(run["initializer"]["seed"])
        model = KernelPolicyValueNet(
            cfg,
            initializer=INITIALIZER_TRAINER_SEEDED_V1,
            initializer_seed=run["initializer"]["seed"],
        )
        optimizer = create_adam(model, learning_rate)
        outcomes = {"p0": {"win": 0, "loss": 0, "draw": 0}, "p1": {"win": 0, "loss": 0, "draw": 0}}
        decisions = {"p0": 0, "p1": 0}
        payload = build_checkpoint_payload(
            run_digest=run_digest,
            completed_update=0,
            optimizer_step_count=0,
            next_episode=0,
            outcomes_by_learner_seat=outcomes,
            learner_decisions_by_seat=decisions,
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            base_seed=base_seed,
            seed_derivation=run["seed_derivation"],
            provenance=run["protocol_provenance"],
            compatibility=compatibility,
        )
        update0 = _update_record_zero(run_digest, logical_state_hash(payload))
        _commit_generation(
            out_dir=out_dir,
            run=run,
            payload=payload,
            update_record=update0,
            run_digest=run_digest,
            parent_head=None,
            previous_payload=None,
            compatibility=compatibility,
            learning_rate=learning_rate,
        )
        reloaded = _load_chain(
            out_dir=out_dir,
            env_sha=env_sha,
            until_update=0,
            deck_ids=deck_ids,
            base_seed=base_seed,
            batch_episodes=batch_episodes,
            learning_rate=learning_rate,
            value_coef=value_coef,
            max_decisions=max_decisions,
            compatibility=compatibility,
        )
        return reloaded, client, first
    except Exception:
        client.close()
        raise


def _terminal_return(terminal: Terminal, learner_seat: str) -> int:
    if terminal.terminal_outcome == "draw":
        derived = 0
    elif terminal.winner == learner_seat:
        derived = 1
    elif terminal.winner == _opponent_seat(learner_seat):
        derived = -1
    else:
        raise ValueError("terminal winner does not match natural outcome")
    idx = 0 if learner_seat == "p0" else 1
    if terminal.terminal_reward[idx] != derived:
        raise ValueError("terminal reward vector disagrees with derived learner return")
    return derived


def _select_learner_action(logits: torch.Tensor, action_seed: int) -> tuple[int, torch.Tensor]:
    """Select without Torch RNG while retaining differentiable Torch loss math."""

    selected = sample_fixed_categorical(logits, action_seed)
    log_prob = torch.log_softmax(logits, dim=0)[selected]
    if not torch.isfinite(log_prob):
        raise ValueError("selected learner log probability is non-finite")
    return selected, log_prob


def _encoded_policy_digest(encoded: Any, selected_index: int) -> str:
    return logical_state_hash(
        {
            "schema": dataclasses.asdict(encoded.schema),
            "state": encoded.state,
            "object_features": encoded.object_features,
            "object_card_ids": encoded.object_card_ids,
            "object_groups": encoded.object_groups,
            "object_node_ids": encoded.object_node_ids,
            "edge_features": encoded.edge_features,
            "edge_source_indices": encoded.edge_source_indices,
            "edge_target_indices": encoded.edge_target_indices,
            "action_features": encoded.action_features,
            "action_ref_features": encoded.action_ref_features,
            "action_ref_card_ids": encoded.action_ref_card_ids,
            "action_ref_action_indices": encoded.action_ref_action_indices,
            "action_ref_node_indices": encoded.action_ref_node_indices,
            "selected_index": selected_index,
            "selected_action_features": encoded.action_features[selected_index],
        }
    )


def _episode_digest(decision_digests: list[str]) -> str:
    return sha256_bytes(canonical_json_bytes({"learner_decision_digests": decision_digests}))


def _run_episode(
    *,
    client: KernelRlClient,
    state: TrainState,
    episode: int,
    max_decisions: int,
    first_decision: Decision | None,
) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor, int]]]:
    learner = _learner_seat(episode)
    env_seed = _env_seed_for_episode(state.run["trainer"]["base_seed"], episode)
    current: Decision | Terminal
    if first_decision is not None:
        current = first_decision
        if current.episode_id != episode or current.step != 0:
            raise ValueError("carried first decision does not match expected episode")
    else:
        current = client.reset(
            episode_id=episode,
            env_seed=env_seed,
            max_decisions=max_decisions,
            deck_ids=tuple(state.run["environment"]["deck_ids"]),
        )
    if not isinstance(current, Decision):
        raise ValueError("trainer reset must return an initial decision")
    _assert_first_decision_matches_run(state.run, current)
    learner_decision_index = 0
    opponent_decision_index = 0
    learner_digests: list[str] = []
    learner_terms: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    total_decisions = 0
    while isinstance(current, Decision):
        if total_decisions >= max_decisions:
            raise RuntimeError("decision cap reached before terminal")
        if current.acting_player == learner:
            encoded = encode_decision(current.observation, current.legal_actions)
            logits, value = state.model(encoded)
            if logits.device.type != "cpu" or value.device.type != "cpu":
                raise ValueError("model outputs must remain on CPU")
            if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                raise ValueError("model produced non-finite learner output")
            seed = derive_train_learner_action_seed(
                state.run["trainer"]["base_seed"],
                episode,
                learner_decision_index,
            )
            selected, log_prob = _select_learner_action(logits, seed)
            learner_terms.append((log_prob, value, 0))
            learner_digests.append(_encoded_policy_digest(encoded, selected))
            learner_decision_index += 1
        elif current.acting_player == _opponent_seat(learner):
            seed = derive_train_opponent_action_seed(
                state.run["trainer"]["base_seed"],
                episode,
                opponent_decision_index,
            )
            selected = deterministic_index_from_seed(seed, len(current.legal_actions))
            opponent_decision_index += 1
        else:
            raise ValueError("unknown acting player")
        action = current.legal_actions[selected]
        current = client.step(action["selected_index"], action["stable_id"])
        _assert_response_deck_identity(state.run, current)
        total_decisions += 1
    terminal_return = _terminal_return(current, learner)
    learner_terms = [(log_prob, value, terminal_return) for log_prob, value, _unused in learner_terms]
    summary = {
        "schema": EPISODE_SUMMARY_SCHEMA,
        "episode": episode,
        "env_seed": env_seed,
        "deck_ids": list(current.deck_ids),
        "deck_hashes": list(current.deck_hashes),
        "learner_seat": learner,
        "terminal_outcome": current.terminal_outcome,
        "winner": current.winner,
        "learner_return": terminal_return,
        "decision_count": current.decision_count,
        "learner_decision_count": learner_decision_index,
        "opponent_decision_count": opponent_decision_index,
        "trajectory_digest": _episode_digest(learner_digests),
    }
    return summary, learner_terms


def _apply_batch_loss(
    *,
    state: TrainState,
    terms: list[tuple[torch.Tensor, torch.Tensor, int]],
    value_coef: float,
) -> tuple[bool, dict[str, str | None]]:
    if not terms:
        return False, {"policy_sum_hex": None, "value_sum_hex": None, "loss_hex": None}
    policy_sum, value_sum, loss = _compute_loss_tensors(terms, value_coef)
    state.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name, param in state.model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise ValueError(f"gradient for {name} is non-finite")
    state.optimizer.step()
    for name, tensor in state.model.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"model state {name} became non-finite")
    export_adam_state(state.optimizer, state.model, state.run["optimizer"]["lr"])
    return True, {
        "policy_sum_hex": float(policy_sum.detach().item()).hex(),
        "value_sum_hex": float(value_sum.detach().item()).hex(),
        "loss_hex": float(loss.detach().item()).hex(),
    }


def _compute_loss_tensors(
    terms: list[tuple[torch.Tensor, torch.Tensor, int]],
    value_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_terms = []
    value_terms = []
    for log_prob, value, terminal_return in terms:
        if value.device.type != "cpu" or log_prob.device.type != "cpu":
            raise ValueError("loss tensors must remain on CPU")
        target = torch.tensor(float(terminal_return), dtype=value.dtype)
        advantage = target - value.detach()
        policy_terms.append(-log_prob * advantage)
        value_terms.append((value - target) ** 2)
    policy_sum = torch.stack(policy_terms).sum()
    value_sum = torch.stack(value_terms).sum()
    loss = (policy_sum + float(value_coef) * value_sum) / len(terms)
    for name, tensor in (("policy_sum", policy_sum), ("value_sum", value_sum), ("loss", loss)):
        if not torch.isfinite(tensor):
            raise ValueError(f"{name} is non-finite")
    return policy_sum, value_sum, loss


def _record_outcome(state: TrainState, episode_summary: dict[str, Any]) -> None:
    seat = episode_summary["learner_seat"]
    ret = episode_summary["learner_return"]
    if ret == 1:
        state.outcomes_by_learner_seat[seat]["win"] += 1
    elif ret == -1:
        state.outcomes_by_learner_seat[seat]["loss"] += 1
    elif ret == 0:
        state.outcomes_by_learner_seat[seat]["draw"] += 1
    else:
        raise ValueError("invalid learner return")
    state.learner_decisions_by_seat[seat] += episode_summary["learner_decision_count"]


def _train_until(
    *,
    state: TrainState,
    client: KernelRlClient,
    until_update: int,
    first_decision: Decision | None,
) -> TrainState:
    if until_update < state.completed_update:
        raise ValueError("until_update is before committed update")
    if until_update == state.completed_update:
        return state
    batch_episodes = state.run["schedule"]["batch_episodes"]
    max_decisions = state.run["trainer"]["max_decisions"]
    value_coef = state.run["trainer"]["value_coef"]
    carried = first_decision
    while state.completed_update < until_update:
        if state.next_episode % 2 != 0:
            raise ValueError("next episode is not pair-aligned")
        previous_payload = state.head_payload
        episode_start = state.next_episode
        batch_summaries: list[dict[str, Any]] = []
        terms: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for episode in range(episode_start, episode_start + batch_episodes):
            summary, episode_terms = _run_episode(
                client=client,
                state=state,
                episode=episode,
                max_decisions=max_decisions,
                first_decision=carried if episode == episode_start else None,
            )
            carried = None
            batch_summaries.append(summary)
            terms.extend(episode_terms)
        optimizer_step, loss_fields = _apply_batch_loss(state=state, terms=terms, value_coef=value_coef)
        next_update = state.completed_update + 1
        next_episode = episode_start + batch_episodes
        for summary in batch_summaries:
            _record_outcome(state, summary)
        if optimizer_step:
            state.optimizer_step_count += 1
        state.completed_update = next_update
        state.next_episode = next_episode
        payload = build_checkpoint_payload(
            run_digest=state.run_digest,
            completed_update=state.completed_update,
            optimizer_step_count=state.optimizer_step_count,
            next_episode=state.next_episode,
            outcomes_by_learner_seat=state.outcomes_by_learner_seat,
            learner_decisions_by_seat=state.learner_decisions_by_seat,
            model=state.model,
            optimizer=state.optimizer,
            learning_rate=state.run["optimizer"]["lr"],
            base_seed=state.run["trainer"]["base_seed"],
            seed_derivation=state.run["seed_derivation"],
            provenance=state.run["protocol_provenance"],
            compatibility=state.run["compatibility"],
        )
        logical = logical_state_hash(payload)
        record = {
            "schema": UPDATE_RECORD_SCHEMA,
            "run_digest": state.run_digest,
            "update": state.completed_update,
            "parent_head": state.parent_head,
            "episode_start": episode_start,
            "episode_count": batch_episodes,
            "episode_end_exclusive": next_episode,
            "optimizer_step": optimizer_step,
            "learner_decision_count": len(terms),
            "loss": loss_fields,
            "episode_summaries": batch_summaries,
            "post_update_logical_sha256": logical,
        }
        head, committed_payload = _commit_generation(
            out_dir=state.out_dir,
            run=state.run,
            payload=payload,
            update_record=record,
            run_digest=state.run_digest,
            parent_head=state.parent_head,
            previous_payload=previous_payload,
            compatibility=state.run["compatibility"],
            learning_rate=state.run["optimizer"]["lr"],
        )
        state.parent_head = head
        state.head_payload = committed_payload
    return state


def train(
    *,
    env_bin: str | Path,
    out_dir: str | Path,
    until_update: int,
    deck_ids: tuple[str, str] = DEFAULT_DECK_IDS,
    resume: str | Path | None = None,
    base_seed: int | None = None,
    batch_episodes: int | None = None,
    learning_rate: float | None = None,
    value_coef: float | None = None,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    deck_ids = _validate_requested_deck_ids(deck_ids)
    with OutputLock(out_dir):
        return _train_locked(
            env_bin=env_bin,
            out_dir=out_dir,
            until_update=until_update,
            deck_ids=deck_ids,
            resume=resume,
            base_seed=base_seed,
            batch_episodes=batch_episodes,
            learning_rate=learning_rate,
            value_coef=value_coef,
            max_decisions=max_decisions,
        )


def _train_locked(
    *,
    env_bin: str | Path,
    out_dir: str | Path,
    until_update: int,
    deck_ids: tuple[str, str] = DEFAULT_DECK_IDS,
    resume: str | Path | None = None,
    base_seed: int | None = None,
    batch_episodes: int | None = None,
    learning_rate: float | None = None,
    value_coef: float | None = None,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    configure_torch_determinism()
    deck_ids = _validate_requested_deck_ids(deck_ids)
    until_update = _validate_until_update(until_update)
    env_path = Path(env_bin)
    if not env_path.is_file():
        raise FileNotFoundError(env_path)
    out_path = Path(out_dir)
    compatibility = _compatibility_tuple()
    env_sha = sha256_file(env_path)
    if resume is None:
        if None in (base_seed, batch_episodes, learning_rate, value_coef, max_decisions):
            raise ValueError("fresh train requires base_seed, batch_episodes, learning_rate, value_coef, and max_decisions")
        base_seed, batch_episodes, learning_rate, value_coef, max_decisions = _validate_resume_overrides(
            base_seed=base_seed,
            batch_episodes=batch_episodes,
            learning_rate=learning_rate,
            value_coef=value_coef,
            max_decisions=max_decisions,
        )
        if out_path.exists() and (out_path.is_symlink() or not out_path.is_dir()):
            require_new_or_empty_dir(out_path)
            raise AssertionError("unreachable")
        if out_path.exists() and any(_entries_excluding_verified_lock(out_path)):
            if (out_path / "run.json").is_file() and not latest_path(out_path).exists():
                state, client, first_decision = _bootstrap_incomplete_fresh(
                    env_bin=env_path,
                    out_dir=out_path,
                    deck_ids=deck_ids,
                    base_seed=base_seed,
                    batch_episodes=batch_episodes,
                    learning_rate=learning_rate,
                    value_coef=value_coef,
                    max_decisions=max_decisions,
                    compatibility=compatibility,
                )
            else:
                _recover_pre_manifest_bootstrap(out_path)
                state, client, first_decision = _bootstrap_fresh(
                    env_bin=env_path,
                    out_dir=out_path,
                    deck_ids=deck_ids,
                    base_seed=base_seed,
                    batch_episodes=batch_episodes,
                    learning_rate=learning_rate,
                    value_coef=value_coef,
                    max_decisions=max_decisions,
                    compatibility=compatibility,
                )
        else:
            state, client, first_decision = _bootstrap_fresh(
                env_bin=env_path,
                out_dir=out_path,
                deck_ids=deck_ids,
                base_seed=base_seed,
                batch_episodes=batch_episodes,
                learning_rate=learning_rate,
                value_coef=value_coef,
                max_decisions=max_decisions,
                compatibility=compatibility,
            )
        try:
            state = _train_until(state=state, client=client, until_update=until_update, first_decision=first_decision)
        finally:
            client.close()
        return _result(state)

    if not same_lexical_path(Path(resume), latest_path(out_path)):
        raise ValueError("resume path must be exactly the selected out-dir latest.json")
    ensure_real_file(out_path, latest_path(out_path))
    client: KernelRlClient | None = None
    first: Decision | None = None

    def preflight_environment(run: dict[str, Any], resume_snapshot: Any) -> None:
        nonlocal client, first
        client = KernelRlClient(env_path, timeout_s=10.0)
        try:
            response = client.reset(
                episode_id=resume_snapshot.next_episode,
                env_seed=_env_seed_for_episode(run["trainer"]["base_seed"], resume_snapshot.next_episode),
                max_decisions=run["trainer"]["max_decisions"],
                deck_ids=deck_ids,
            )
            if not isinstance(response, Decision):
                raise ValueError("resume trainer reset must return an initial decision")
            _assert_first_decision_matches_run(run, response)
            first = response
        except Exception:
            client.close()
            client = None
            raise

    try:
        state = _load_chain(
            out_dir=out_path,
            env_sha=env_sha,
            until_update=until_update,
            deck_ids=deck_ids,
            base_seed=base_seed,
            batch_episodes=batch_episodes,
            learning_rate=learning_rate,
            value_coef=value_coef,
            max_decisions=max_decisions,
            compatibility=compatibility,
            before_mutation=preflight_environment,
        )
        if until_update < state.completed_update:
            raise ValueError("until_update must be at least the committed update")
        if until_update == state.completed_update:
            return _result(state)
        if client is None or first is None:
            raise AssertionError("resume environment preflight did not provide a client and first decision")
        state = _train_until(state=state, client=client, until_update=until_update, first_decision=first)
    finally:
        if client is not None:
            client.close()
    return _result(state)


def _result(state: TrainState) -> dict[str, Any]:
    return {
        "run_digest": state.run_digest,
        "completed_update": state.completed_update,
        "next_episode": state.next_episode,
        "optimizer_step_count": state.optimizer_step_count,
        "head": state.parent_head,
        "logical_state_sha256": logical_state_hash(
            build_checkpoint_payload(
                run_digest=state.run_digest,
                completed_update=state.completed_update,
                optimizer_step_count=state.optimizer_step_count,
                next_episode=state.next_episode,
                outcomes_by_learner_seat=state.outcomes_by_learner_seat,
                learner_decisions_by_seat=state.learner_decisions_by_seat,
                model=state.model,
                optimizer=state.optimizer,
                learning_rate=state.run["optimizer"]["lr"],
                base_seed=state.run["trainer"]["base_seed"],
                seed_derivation=state.run["seed_derivation"],
                provenance=state.run["protocol_provenance"],
                compatibility=state.run["compatibility"],
            )
        ),
    }
