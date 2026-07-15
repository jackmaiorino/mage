"""Terminal-only REINFORCE/value trainer with atomic exact resume."""

from __future__ import annotations

import dataclasses
import math
import os
import platform
import random
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
    latest_path,
    read_json_file,
    rebuild_derived_caches,
    require_new_or_empty_dir,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from .checkpoint import (
    LATEST_SCHEMA,
    UPDATE_RECORD_SCHEMA,
    build_checkpoint_payload,
    build_latest,
    build_sidecar,
    create_adam,
    export_adam_state,
    logical_state_hash,
    load_adam_state,
    load_checkpoint_file,
    restore_python_rng_state,
    restore_torch_rng_state,
    save_checkpoint_file,
    update_record_hash,
    validate_checkpoint_payload,
    validate_model_state,
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

RUN_SCHEMA = "kernel_rl_train_run/v1"
ALGORITHM_NAME = "terminal_reinforce_value/v1"
MAX_UPDATES = 1_000_000
MAX_BATCH_EPISODES = 10_000
MAX_DECISIONS = 10_000_000


@dataclasses.dataclass
class TrainState:
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


def _finite_positive_float(value: Any, name: str) -> float:
    if type(value) not in (float, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


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
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_byteorder": sys.byteorder,
        "torch_version": str(torch.__version__),
        "torch_build": str(torch.__config__.show()),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "cpu_only": True,
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def _seed_derivation_dict() -> dict[str, Any]:
    data = dataclasses.asdict(TrainerSeedDerivation())
    data["namespaces"] = list(data["namespaces"])
    return data


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
    return {
        "schema": RUN_SCHEMA,
        "package": {"name": "mtg-kernel-rl", "version": __version__},
        "algorithm": {
            "name": ALGORITHM_NAME,
            "loss": "policy_sum + value_coef * value_sum over complete paired terminal batches",
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
        "model": {"config": model_config, "contract_fingerprint": model.config.contract_fingerprint()},
        "initializer": {
            "name": INITIALIZER_TRAINER_SEEDED_V1,
            "seed": derive_model_init_seed(base_seed),
            "namespace": "model-init/base_seed",
        },
        "optimizer": {
            "algorithm": "adam/torch-cpu-canonical-v1",
            "lr": learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
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
    }


def _write_run_json(out_dir: Path, run: dict[str, Any]) -> str:
    path = out_dir / "run.json"
    if path.exists():
        raise FileExistsError("run.json already exists")
    return write_json_atomic(path, run)


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
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError("run.json schema mismatch")
    if run["environment"]["binary_sha256"] != env_sha:
        raise ValueError("environment executable SHA-256 drift")
    if run["compatibility"] != compatibility:
        raise ValueError("runtime compatibility tuple drift")
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


def _validate_update_record(record: dict[str, Any], update: int, run_digest: str, parent_head: str | None) -> None:
    if record.get("schema") != UPDATE_RECORD_SCHEMA:
        raise ValueError("update record schema mismatch")
    if record.get("update") != update:
        raise ValueError("update record index mismatch")
    if record.get("run_digest") != run_digest:
        raise ValueError("update record run digest mismatch")
    if record.get("parent_head") != parent_head:
        raise ValueError("update record parent mismatch")


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
    load_adam_state(optimizer, model, payload["optimizer_state"], learning_rate)


def _commit_generation(
    *,
    out_dir: Path,
    payload: dict[str, Any],
    update_record: dict[str, Any],
    run_digest: str,
    parent_head: str | None,
    compatibility: dict[str, Any],
    learning_rate: float,
) -> tuple[str, list[dict[str, Any]]]:
    update = payload["completed_update"]
    paths = generation_paths(out_dir, update)
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"immutable generation already exists: {path}")

    checkpoint_tmp = paths["checkpoint"].with_name(f".{paths['checkpoint'].name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        save_checkpoint_file(checkpoint_tmp, payload)
        loaded = load_checkpoint_file(checkpoint_tmp)
        _strict_validate_checkpoint_for_model(
            loaded,
            run_digest=run_digest,
            compatibility=compatibility,
            learning_rate=learning_rate,
        )
        logical_hash = logical_state_hash(loaded)
        checkpoint_hash = sha256_file(checkpoint_tmp)
        atomic_replace(checkpoint_tmp, paths["checkpoint"])
        fsync_dir(paths["checkpoint"].parent)
    finally:
        if checkpoint_tmp.exists():
            try:
                checkpoint_tmp.unlink()
            except OSError:
                pass

    update_hash = write_json_atomic(paths["update"], update_record)
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
    write_json_atomic(paths["sidecar"], sidecar)
    parsed_sidecar = read_json_file(paths["sidecar"])
    if parsed_sidecar != sidecar:
        raise ValueError("sidecar roundtrip mismatch")
    latest = build_latest(update=update, run_digest=run_digest, head=sidecar["head"])
    write_json_atomic(latest_path(out_dir), latest)
    records = _records_through(out_dir, update)
    rebuild_derived_caches(out_dir, records, latest)
    return sidecar["head"], records


def _records_through(out_dir: Path, update: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(update + 1):
        records.append(read_json_file(generation_paths(out_dir, i)["update"]))
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
    run = read_json_file(run_path)
    run_digest = sha256_file(run_path)
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
    latest = read_json_file(latest_path(out_dir))
    if latest.get("schema") != LATEST_SCHEMA or latest.get("run_digest") != run_digest:
        raise ValueError("latest.json schema or run digest mismatch")
    head_update = latest.get("update")
    if type(head_update) is not int or head_update < 0 or head_update > MAX_UPDATES:
        raise ValueError("latest update out of bounds")
    parent_head: str | None = None
    records: list[dict[str, Any]] = []
    head_payload: dict[str, Any] | None = None
    for update in range(head_update + 1):
        paths = generation_paths(out_dir, update)
        record = read_json_file(paths["update"])
        _validate_update_record(record, update, run_digest, parent_head)
        sidecar = read_json_file(paths["sidecar"])
        if sidecar.get("schema") != "kernel_rl_train_checkpoint_sidecar/v1":
            raise ValueError("sidecar schema mismatch")
        if sidecar.get("update") != update or sidecar.get("run_digest") != run_digest:
            raise ValueError("sidecar generation mismatch")
        if sidecar.get("parent_head") != parent_head:
            raise ValueError("sidecar parent mismatch")
        update_hash = update_record_hash(record)
        if sidecar.get("update_record_sha256") != update_hash:
            raise ValueError("update record hash mismatch")
        if sha256_file(paths["checkpoint"]) != sidecar.get("checkpoint_sha256"):
            raise ValueError("checkpoint byte hash mismatch")
        payload = load_checkpoint_file(paths["checkpoint"])
        _strict_validate_checkpoint_for_model(
            payload,
            run_digest=run_digest,
            compatibility=compatibility,
            learning_rate=run["optimizer"]["lr"],
        )
        if payload["completed_update"] != update:
            raise ValueError("checkpoint update mismatch")
        logical_hash = logical_state_hash(payload)
        if sidecar.get("logical_state_sha256") != logical_hash:
            raise ValueError("logical state hash mismatch")
        expected_head = build_sidecar(
            update=update,
            run_digest=run_digest,
            parent_head=parent_head,
            checkpoint_sha256=sidecar["checkpoint_sha256"],
            logical_hash=logical_hash,
            update_hash=update_hash,
        )["head"]
        if sidecar.get("head") != expected_head:
            raise ValueError("sidecar head mismatch")
        records.append(record)
        parent_head = sidecar["head"]
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
    load_adam_state(optimizer, model, head_payload["optimizer_state"], run["optimizer"]["lr"])
    restore_python_rng_state(head_payload["python_rng_state"])
    restore_torch_rng_state(head_payload["torch_cpu_rng_state"])
    return TrainState(
        run=run,
        run_digest=run_digest,
        model=model,
        optimizer=optimizer,
        completed_update=head_payload["completed_update"],
        optimizer_step_count=head_payload["optimizer_step_count"],
        next_episode=head_payload["next_episode"],
        outcomes_by_learner_seat=head_payload["outcomes_by_learner_seat"],
        learner_decisions_by_seat=head_payload["learner_decisions_by_seat"],
        records=records,
        parent_head=parent_head,
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
    (out_dir / "updates").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
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
        if run_digest != sha256_file(out_dir / "run.json"):
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
        head, records = _commit_generation(
            out_dir=out_dir,
            payload=payload,
            update_record=update0,
            run_digest=run_digest,
            parent_head=None,
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
        "schema": "kernel_rl_train_episode_summary/v1",
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
        head, records = _commit_generation(
            out_dir=Path(state.run["_out_dir"]),
            payload=payload,
            update_record=record,
            run_digest=state.run_digest,
            parent_head=state.parent_head,
            compatibility=state.run["compatibility"],
            learning_rate=state.run["optimizer"]["lr"],
        )
        state.parent_head = head
        state.records = records
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
        base_seed = validate_uint63(base_seed, "base_seed")
        batch_episodes = _validate_batch_episodes(batch_episodes)
        learning_rate = _finite_positive_float(learning_rate, "learning_rate")
        value_coef = _finite_positive_float(value_coef, "value_coef")
        max_decisions = _validate_max_decisions(max_decisions)
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
        state.run["_out_dir"] = str(out_path)
        try:
            state = _train_until(state=state, client=client, until_update=until_update, first_decision=first_decision)
        finally:
            client.close()
        return _result(state)

    if Path(resume).resolve() != latest_path(out_path).resolve():
        raise ValueError("resume path must be exactly the selected out-dir latest.json")
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
    state.run["_out_dir"] = str(out_path)
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
