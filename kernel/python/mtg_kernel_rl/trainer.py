"""Terminal-only REINFORCE/value trainer with atomic exact resume."""

from __future__ import annotations

import dataclasses
import copy
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .artifacts import (
    atomic_replace,
    canonical_json_bytes,
    fsync_dir,
    generation_paths,
    inject_fault,
    latest_path,
    read_authoritative_json,
    read_json_file,
    rebuild_derived_caches,
    require_new_or_empty_dir,
    sha256_bytes,
    sha256_file,
    validate_training_json_privacy,
    write_json_atomic,
)
from .artifact_io import FORBIDDEN_TRAINING_JSON_KEYS
from .checkpoint import (
    CHECKPOINT_SCHEMA,
    LATEST_SCHEMA,
    SIDECAR_SCHEMA,
    UPDATE_RECORD_SCHEMA,
    adam_config,
    build_checkpoint_payload,
    build_latest,
    build_sidecar,
    create_adam,
    export_adam_state,
    logical_state_hash,
    load_adam_state,
    load_checkpoint_file,
    load_checkpoint_file_with_digest,
    restore_python_rng_state,
    restore_torch_rng_state,
    save_checkpoint_file,
    update_record_hash,
    validate_checkpoint_payload,
    validate_model_state,
    validate_torch_rng_state,
)
from .client import Decision, KernelRlClient, Terminal
from .determinism import (
    TrainerSeedDerivation,
    configure_torch_determinism,
    derive_model_init_seed,
    derive_train_env_seed,
    derive_train_learner_action_seed,
    derive_train_opponent_action_seed,
    deterministic_index_from_seed,
    validate_positive_int,
    validate_uint63,
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

RUN_SCHEMA = "kernel_rl_train_run/v5"
ALGORITHM_NAME = "terminal_reinforce_value/v1"
MAX_UPDATES = 1_000_000
MAX_BATCH_EPISODES = 10_000
MAX_DECISIONS = 10_000_000
EPISODE_SUMMARY_SCHEMA = "kernel_rl_train_episode_summary/v2"
SUMMARY_SCHEMA = "kernel_rl_train_summary/v2"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
GENERATION_RE = re.compile(r"^update-(\d{8})\.(json|pt)$")
TEMP_RE = re.compile(r"^\..+\.\d+\.\d+\.tmp$")
TRANSACTION_RE = re.compile(r"^update-(\d{8})-[0-9a-f]+$")
TRANSACTION_TEMP_RE = re.compile(r"^\.(update\.json|sidecar\.json)\.\d+\.\d+\.tmp$")


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
    records: list[dict[str, Any]]
    parent_head: str | None
    head_payload: dict[str, Any]


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
    from . import checkpoint as checkpoint_contract
    from . import checkpoint_io as zip_contract
    from . import artifact_io as json_contract

    return {
        "schema": "kernel_rl_artifact_boundary/v3",
        "format": {
            "checkpoint_container": "torch-zip",
            "checkpoint_zip_root": zip_contract.TORCH_ZIP_ROOT,
            "authoritative_json": "canonical-sorted-ascii-json-lf",
        },
        "safe_load": {
            "checkpoint_read": "single bounded regular non-link file read",
            "raw_zip": "EOCD/ZIP64 locator/ZIP64 EOCD and bounded central directory parsed from captured bytes before ZipFile",
            "pickle_preflight": "restricted protocol/opcode interpreter validates storage persistent IDs and tensor rebuild metadata before torch.load",
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
            "allowed_globals": sorted(zip_contract.ALLOWED_PICKLE_GLOBALS),
            "storage": "BINPERSID must be ('storage', allowed CPU storage type, canonical unique decimal key, 'cpu', bounded element count) and exactly match archive/data/<key> bytes",
            "tensor_rebuild": "_rebuild_tensor_v2 only; unique storage reference, zero offset, bounded nonnegative shape, exact positive contiguous strides, exact full-storage byte coverage, false requires_grad, empty OrderedDict metadata",
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
            "algorithm": "one constant persistent child lock file at <physical-output-root>/.mtg-kernel-train.lock",
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
            "creation": "all output, transaction, quarantine, lock-parent, and generation-directory creation is component-wise no-follow",
            "traversal": "os.scandir/lstat no-follow traversal and contained atomic quarantine rename after destination parents validate",
            "resume": "resume path must lexically equal selected latest.json",
            "pre_manifest_recovery": "under the output lock, fresh bootstrap validates the entire root then may remove only empty real updates/checkpoints directories and canonical .run.json.<pid>.<n>.tmp files before run.json exists; unknown entries, nonempty directories, links, reparse points, malformed temps, and lock impostors fail closed with zero cleanup",
        },
        "privacy": {
            "scan": "authoritative JSON keys and values plus checkpoint scalar metadata",
            "rejects": "generic POSIX absolute roots including / plus Windows drive-root, Windows root-relative, UNC, device/extended, file URI, and embedded absolute path fragments",
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
                "+ value_coef * sum((value - terminal_return)^2)) / learner_decision_count"
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
    model_config = model.config.to_dict()
    return _run_manifest_from_config(
        env_sha=env_sha,
        provenance=provenance,
        model_config=model_config,
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=compatibility,
    )


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


def _validate_run_manifest(run: Any, *, env_sha: str, compatibility: dict[str, Any]) -> dict[str, Any]:
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
    _validate_hash(run.get("environment", {}).get("binary_sha256"), "run environment binary_sha256")
    if run["environment"]["binary_sha256"] != env_sha:
        raise ValueError("environment executable SHA-256 drift")
    if run["compatibility"] != compatibility:
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
        env_sha=env_sha,
        provenance=run["protocol_provenance"],
        model_config=model_config,
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=compatibility,
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


def _assert_first_decision_matches_run(run: dict[str, Any], decision: Decision) -> None:
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
    model = KernelPolicyValueNet(cfg, initializer=INITIALIZER_TRAINER_SEEDED_V1, initializer_seed=0)
    model.load_state_dict(validate_model_state(model, payload["model_state"]), strict=True)
    optimizer = create_adam(model, learning_rate)
    load_adam_state(
        optimizer,
        model,
        payload["optimizer_state"],
        learning_rate,
        expected_step_count=payload["optimizer_step_count"],
    )


def _staged_generation_paths(transaction_dir: Path) -> dict[str, Path]:
    return {
        "update": transaction_dir / "update.json",
        "checkpoint": transaction_dir / "checkpoint.pt",
        "sidecar": transaction_dir / "sidecar.json",
    }


def _new_transaction_dir(out_dir: Path, update: int) -> Path:
    root = out_dir / ".transactions"
    mkdir_no_follow(root, parents=True, exist_ok=True)
    path = root / f"update-{update:08d}-{os.getpid():x}{time.monotonic_ns():x}"
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


def _revalidate_then_unlink(identity: Any) -> None:
    revalidate_path_identity(identity)
    Path(identity.path).unlink()


def _revalidate_then_rmdir(identity: Any) -> None:
    revalidate_path_identity(identity)
    Path(identity.path).rmdir()


def _revalidate_then_quarantine(root: Path, identity: Any, reason: str) -> None:
    revalidate_path_identity(identity)
    atomic_quarantine(root, Path(identity.path), reason)


def _validate_orphan_generation_file(path: Path, *, update: int, kind: str, run_digest: str, compatibility: dict[str, Any]) -> None:
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
        if child.name not in allowed and TRANSACTION_TEMP_RE.fullmatch(child.name) is None:
            raise ValueError(f"unknown transaction staging file: {child.name}")


def _reconcile_uncommitted_artifacts(
    out_dir: Path,
    *,
    head_update: int,
    run_digest: str,
    compatibility: dict[str, Any],
) -> None:
    ensure_real_child_dir(out_dir.parent, out_dir)
    mkdirs: list[Path] = []
    quarantine_actions: list[tuple[Any, str]] = []
    remove_dirs: list[Any] = []
    transactions = out_dir / ".transactions"
    if transactions.exists():
        ensure_real_child_dir(out_dir, transactions)
        transaction_entries = scandir_no_follow(transactions)
        for entry in transaction_entries:
            child = Path(entry.path)
            if not entry.is_dir(follow_symlinks=False):
                raise ValueError(f"unknown transaction entry: {child.name}")
            _validate_transaction_tree(out_dir, child, head_update=head_update)
            quarantine_actions.append((capture_path_identity(child), "staging"))
        if not transaction_entries:
            remove_dirs.append(capture_path_identity(transactions))

    allowed_root_files = {"run.json", "latest.json", "episodes.jsonl", "updates.jsonl", "summary.json"}
    allowed_root_dirs = {"updates", "checkpoints", ".transactions", ".quarantine"}
    direct_temp_prefixes = (".latest.json", ".episodes.jsonl", ".updates.jsonl", ".summary.json")
    for entry in _entries_excluding_verified_lock(out_dir):
        path = Path(entry.path)
        if entry.name in allowed_root_files:
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"root artifact is not a file: {entry.name}")
            ensure_real_file(out_dir, path, reject_hardlinks=False)
            continue
        if entry.name in allowed_root_dirs:
            if not entry.is_dir(follow_symlinks=False):
                raise ValueError(f"root artifact directory is not a directory: {entry.name}")
            ensure_real_child_dir(out_dir, path)
            continue
        if any(entry.name.startswith(f"{prefix}.") and entry.name.endswith(".tmp") for prefix in direct_temp_prefixes):
            if TEMP_RE.fullmatch(entry.name) is None:
                raise ValueError(f"unknown temp file: {entry.name}")
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"temp artifact is not a file: {entry.name}")
            ensure_real_file(out_dir, path, reject_hardlinks=False)
            quarantine_actions.append((capture_path_identity(path), "temp"))
            continue
        raise ValueError(f"unknown root artifact entry: {entry.name}")

    for directory_name, mapping in (
        ("updates", {"json": "update"}),
        ("checkpoints", {"pt": "checkpoint", "json": "sidecar"}),
    ):
        directory = out_dir / directory_name
        if not directory.exists():
            mkdirs.append(directory)
            continue
        ensure_real_child_dir(out_dir, directory)
        for entry in scandir_no_follow(directory):
            path = Path(entry.path)
            _validate_path_contained(out_dir, path)
            if entry.is_dir(follow_symlinks=False):
                raise ValueError(f"unknown directory under {directory_name}: {path.name}")
            if TEMP_RE.fullmatch(path.name):
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"temp artifact is not a file: {path.name}")
                quarantine_actions.append((capture_path_identity(path), "temp"))
                continue
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
            if update <= head_update:
                continue
            _validate_orphan_generation_file(path, update=update, kind=kind, run_digest=run_digest, compatibility=compatibility)
            quarantine_actions.append((capture_path_identity(path), "uncommitted"))
    for directory in mkdirs:
        mkdir_no_follow(directory, parents=True, exist_ok=True)
    for identity, reason in quarantine_actions:
        _revalidate_then_quarantine(out_dir, identity, reason)
    for identity in remove_dirs:
        try:
            if not any(scandir_no_follow(Path(identity.path))):
                _revalidate_then_rmdir(identity)
        except OSError:
            pass
    if transactions.exists():
        ensure_real_child_dir(out_dir, transactions)
        try:
            if not any(scandir_no_follow(transactions)):
                identity = capture_path_identity(transactions)
                _revalidate_then_rmdir(identity)
        except OSError:
            pass


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
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    for path in paths.values():
        ensure_real_file(Path(run["_artifact_root"]), path)
    record_raw = read_authoritative_json(paths["update"], "update")
    sidecar = read_authoritative_json(paths["sidecar"], "sidecar")
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
    loaded_checkpoint = load_checkpoint_file_with_digest(paths["checkpoint"])
    checkpoint_hash = loaded_checkpoint.sha256
    if sidecar["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint byte hash mismatch")
    _validate_hash(sidecar["checkpoint_sha256"], "sidecar checkpoint_sha256")
    _validate_hash(sidecar["logical_state_sha256"], "sidecar logical_state_sha256")
    _validate_hash(sidecar["update_record_sha256"], "sidecar update_record_sha256")
    _validate_hash(sidecar["head"], "sidecar head")

    payload = loaded_checkpoint.payload
    _strict_validate_checkpoint_for_model(
        payload,
        run_digest=run_digest,
        compatibility=compatibility,
        learning_rate=learning_rate,
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
        record_raw,
        update=update,
        run=run,
        run_digest=run_digest,
        parent_head=parent_head,
        previous_payload=previous_payload,
        payload=payload,
        logical_hash=logical_hash,
    )
    update_hash = update_record_hash(record)
    if sidecar["update_record_sha256"] != update_hash:
        raise ValueError("update record hash mismatch")
    expected_sidecar = build_sidecar(
        update=update,
        run_digest=run_digest,
        parent_head=parent_head,
        checkpoint_sha256=checkpoint_hash,
        logical_hash=logical_hash,
        update_hash=update_hash,
    )
    if sidecar != expected_sidecar:
        raise ValueError("sidecar content mismatch")
    records = _records_through(Path(run["_artifact_root"]), update) if paths == generation_paths(Path(run["_artifact_root"]), update) else []
    return sidecar["head"], records, payload


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
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    update = payload["completed_update"]
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
        update_hash = write_json_atomic(staged["update"], update_record)
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
        write_json_atomic(staged["sidecar"], sidecar)
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
        final_head, _unused_records, final_payload = _validate_generation_bundle(
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
        write_json_atomic(latest_path(out_dir), latest)
        inject_fault("latest_replace_after", latest_path(out_dir))
        records = _records_through(out_dir, update)
        rebuild_derived_caches(out_dir, records, latest)
        inject_fault("post_latest_cleanup_before", tx_dir)
        return final_head, records, final_payload
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


def _records_through(out_dir: Path, update: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(update + 1):
        records.append(read_authoritative_json(generation_paths(out_dir, i)["update"], "update"))
    return records


def _load_chain(
    *,
    out_dir: Path,
    env_sha: str,
    base_seed: int | None,
    batch_episodes: int | None,
    learning_rate: float | None,
    value_coef: float | None,
    max_decisions: int | None,
    compatibility: dict[str, Any],
) -> TrainState:
    run_path = out_dir / "run.json"
    run = read_authoritative_json(run_path, "run")
    run_digest = sha256_file(run_path, max_bytes=256 * 1024, allow_empty=False)
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
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=compatibility,
    )
    latest = read_authoritative_json(latest_path(out_dir), "latest")
    latest = _validate_latest(latest, run_digest=run_digest)
    head_update = latest["update"]
    _reconcile_uncommitted_artifacts(out_dir, head_update=head_update, run_digest=run_digest, compatibility=compatibility)
    parent_head: str | None = None
    records: list[dict[str, Any]] = []
    head_payload: dict[str, Any] | None = None
    for update in range(head_update + 1):
        paths = generation_paths(out_dir, update)
        run_for_validation = dict(run)
        run_for_validation["_artifact_root"] = str(out_dir)
        expected_head, _unused_records, payload = _validate_generation_bundle(
            paths=paths,
            update=update,
            run=run_for_validation,
            run_digest=run_digest,
            parent_head=parent_head,
            previous_payload=head_payload,
            compatibility=compatibility,
            learning_rate=run["optimizer"]["lr"],
        )
        record = read_authoritative_json(paths["update"], "update")
        records.append(record)
        parent_head = expected_head
        head_payload = payload
    if parent_head != latest.get("head"):
        raise ValueError("latest head mismatch")
    if head_payload is None:
        raise ValueError("empty checkpoint chain")
    rebuild_derived_caches(out_dir, records, latest)
    cfg = ModelConfig.from_dict(head_payload["model_config"])
    model = KernelPolicyValueNet(
        cfg,
        initializer=INITIALIZER_TRAINER_SEEDED_V1,
        initializer_seed=run["initializer"]["seed"],
    )
    model.load_state_dict(validate_model_state(model, head_payload["model_state"]), strict=True)
    optimizer = create_adam(model, run["optimizer"]["lr"])
    load_adam_state(
        optimizer,
        model,
        head_payload["optimizer_state"],
        run["optimizer"]["lr"],
        expected_step_count=head_payload["optimizer_step_count"],
    )
    restore_python_rng_state(head_payload["python_rng_state"])
    restore_torch_rng_state(head_payload["torch_cpu_rng_state"])
    return TrainState(
        out_dir=out_dir,
        run=run,
        run_digest=run_digest,
        model=model,
        optimizer=optimizer,
        completed_update=head_payload["completed_update"],
        optimizer_step_count=head_payload["optimizer_step_count"],
        next_episode=head_payload["next_episode"],
        outcomes_by_learner_seat=copy.deepcopy(head_payload["outcomes_by_learner_seat"]),
        learner_decisions_by_seat=copy.deepcopy(head_payload["learner_decisions_by_seat"]),
        records=records,
        parent_head=parent_head,
        head_payload=head_payload,
    )


def _bootstrap_fresh(
    *,
    env_bin: Path,
    out_dir: Path,
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
        first = client.reset(episode_id=0, env_seed=_env_seed_for_episode(base_seed, 0), max_decisions=max_decisions)
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
        _head, _records, _payload = _commit_generation(
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
        if entry.name.startswith(".run.json.") and entry.name.endswith(".tmp"):
            if TEMP_RE.fullmatch(entry.name) is None or not entry.is_file(follow_symlinks=False):
                raise ValueError(f"pre-manifest run.json temp is malformed: {entry.name}")
            ensure_real_file(out_dir, child, reject_hardlinks=False)
            temp_value = read_json_file(child, max_bytes=256 * 1024, require_canonical=True)
            if not isinstance(temp_value, dict) or temp_value.get("schema") != RUN_SCHEMA:
                raise ValueError(f"pre-manifest run.json temp content is malformed: {entry.name}")
            validate_training_json_privacy(temp_value)
            actions.append(("unlink", capture_path_identity(child)))
            continue
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
    base_seed: int,
    batch_episodes: int,
    learning_rate: float,
    value_coef: float,
    max_decisions: int,
    compatibility: dict[str, Any],
) -> tuple[TrainState, KernelRlClient, Decision]:
    env_sha = sha256_file(env_bin)
    run_path = out_dir / "run.json"
    run = read_authoritative_json(run_path, "run")
    run_digest = sha256_file(run_path, max_bytes=256 * 1024, allow_empty=False)
    _assert_run_matches_options(
        run,
        env_sha=env_sha,
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=compatibility,
    )
    _reconcile_uncommitted_artifacts(out_dir, head_update=-1, run_digest=run_digest, compatibility=compatibility)
    client = KernelRlClient(env_bin, timeout_s=10.0)
    try:
        first = client.reset(episode_id=0, env_seed=_env_seed_for_episode(base_seed, 0), max_decisions=max_decisions)
        _assert_first_decision_matches_run(run, first)
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
        current = client.reset(episode_id=episode, env_seed=env_seed, max_decisions=max_decisions)
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
            probabilities = torch.softmax(logits, dim=0)
            if not torch.isfinite(probabilities).all() or float(probabilities.sum().item()) <= 0.0:
                raise ValueError("learner probabilities are invalid")
            seed = derive_train_learner_action_seed(
                state.run["trainer"]["base_seed"],
                episode,
                learner_decision_index,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            selected = int(torch.multinomial(probabilities.detach(), 1, generator=generator).item())
            log_prob = torch.log_softmax(logits, dim=0)[selected]
            if not torch.isfinite(log_prob):
                raise ValueError("selected learner log probability is non-finite")
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
        total_decisions += 1
    terminal_return = _terminal_return(current, learner)
    learner_terms = [(log_prob, value, terminal_return) for log_prob, value, _unused in learner_terms]
    summary = {
        "schema": EPISODE_SUMMARY_SCHEMA,
        "episode": episode,
        "env_seed": env_seed,
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
        head, records, committed_payload = _commit_generation(
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
        state.records = records
        state.head_payload = committed_payload
    return state


def train(
    *,
    env_bin: str | Path,
    out_dir: str | Path,
    until_update: int,
    resume: str | Path | None = None,
    base_seed: int | None = None,
    batch_episodes: int | None = None,
    learning_rate: float | None = None,
    value_coef: float | None = None,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    with OutputLock(out_dir):
        return _train_locked(
            env_bin=env_bin,
            out_dir=out_dir,
            until_update=until_update,
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
    resume: str | Path | None = None,
    base_seed: int | None = None,
    batch_episodes: int | None = None,
    learning_rate: float | None = None,
    value_coef: float | None = None,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    configure_torch_determinism()
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
    state = _load_chain(
        out_dir=out_path,
        env_sha=env_sha,
        base_seed=base_seed,
        batch_episodes=batch_episodes,
        learning_rate=learning_rate,
        value_coef=value_coef,
        max_decisions=max_decisions,
        compatibility=compatibility,
    )
    if until_update < state.completed_update:
        raise ValueError("until_update must be at least the committed update")
    if until_update == state.completed_update:
        return _result(state)
    client = KernelRlClient(env_path, timeout_s=10.0)
    try:
        first = client.reset(
            episode_id=state.next_episode,
            env_seed=_env_seed_for_episode(state.run["trainer"]["base_seed"], state.next_episode),
            max_decisions=state.run["trainer"]["max_decisions"],
        )
        _assert_first_decision_matches_run(state.run, first)
        state = _train_until(state=state, client=client, until_update=until_update, first_decision=first)
    finally:
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
