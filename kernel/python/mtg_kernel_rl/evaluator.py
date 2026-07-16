"""Deterministic greedy head-versus-update-zero paired evaluation."""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .artifact_io import json_values_equal_strict, read_regular_file_bytes
from .artifacts import require_new_or_empty_dir
from .client import Decision, KernelRlClient, Terminal
from .determinism import (
    EvaluatorSeedDerivation,
    derive_evaluation_bootstrap_seed,
    derive_evaluation_env_seed,
    validate_positive_int,
    validate_uint63,
)
from .evaluation_stats import PairedGamePoints, score_pair_half_points, summarize_paired_game_points
from .evaluation_store import (
    ALGORITHM_CONTRACT,
    GAME_SCHEMA,
    PAIR_SCHEMA,
    RUN_SCHEMA,
    MAX_BOOTSTRAP_DRAWS,
    MAX_BOOTSTRAP_REPLICATES,
    MAX_DECISIONS,
    MAX_PAIR_COUNT,
    MAX_TIMEOUT_MS,
    MIN_BOOTSTRAP_REPLICATES,
    _publish_evaluation,
    statistics_payload,
)
from .features import encode_decision
from .model import KernelPolicyValueNet, greedy_action
from .output_lock import OutputLock
from .training_store import SnapshotRef, TrainingStore, ValidatedChain, runtime_compatibility


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ENV_BINARY_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Compact path-free result for a committed paired evaluation."""

    run_sha256: str
    candidate_head: str
    baseline_head: str
    pair_count: int
    game_count: int
    total_half_points: int
    estimate: float


def _validate_exact_head(value: Any) -> str:
    if type(value) is not str or _HEX64_RE.fullmatch(value) is None:
        raise ValueError("expected_candidate_head must be exactly 64 lowercase hexadecimal characters")
    return value


def _validate_request(
    *,
    expected_candidate_head: Any,
    pairs: Any,
    base_seed: Any,
    bootstrap_replicates: Any,
    max_decisions: Any,
    timeout_ms: Any,
) -> tuple[str, int, int, int, int, int]:
    expected_head = _validate_exact_head(expected_candidate_head)
    base_seed = validate_uint63(base_seed, "base_seed")
    pairs = validate_positive_int(pairs, "pairs", maximum=MAX_PAIR_COUNT)
    if type(bootstrap_replicates) is not int:
        raise TypeError("bootstrap_replicates must be an integer and not bool")
    if bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES or bootstrap_replicates > MAX_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"bootstrap_replicates must be in [{MIN_BOOTSTRAP_REPLICATES}, {MAX_BOOTSTRAP_REPLICATES}]"
        )
    if pairs * bootstrap_replicates > MAX_BOOTSTRAP_DRAWS:
        raise ValueError(f"pairs * bootstrap_replicates must be at most {MAX_BOOTSTRAP_DRAWS}")
    max_decisions = validate_positive_int(max_decisions, "max_decisions", maximum=MAX_DECISIONS)
    if type(timeout_ms) is not int:
        raise TypeError("timeout_ms must be an integer and not bool")
    if timeout_ms <= 0 or timeout_ms > MAX_TIMEOUT_MS:
        raise ValueError(f"timeout_ms must be in [1, {MAX_TIMEOUT_MS}]")
    return expected_head, pairs, base_seed, bootstrap_replicates, max_decisions, timeout_ms


def _normalized_real(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.realpath(os.path.abspath(os.fspath(path)))))


def _paths_overlap(left: str | Path, right: str | Path) -> bool:
    left_real = _normalized_real(left)
    right_real = _normalized_real(right)
    try:
        common = os.path.commonpath((left_real, right_real))
    except ValueError:
        return False
    return common == left_real or common == right_real


def _validate_loaded_model(model: KernelPolicyValueNet, context: str) -> None:
    if model.training:
        raise ValueError(f"{context} model must be in eval mode")
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type != "cpu":
            raise ValueError(f"{context} model tensor {name} must be on CPU")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"{context} model tensor {name} must use float32")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(f"{context} model parameters must be frozen")


def _require_model_matches_attestation(
    used: KernelPolicyValueNet,
    attested: KernelPolicyValueNet,
    context: str,
) -> None:
    if used.config != attested.config:
        raise ValueError(f"{context} model config changed during evaluation")
    used_state = used.state_dict()
    attested_state = attested.state_dict()
    if tuple(used_state) != tuple(attested_state):
        raise ValueError(f"{context} model state keys changed during evaluation")
    for name in used_state:
        left = used_state[name]
        right = attested_state[name]
        if left.dtype != right.dtype or left.device != right.device or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"{context} model tensor {name} differs from selected checkpoint")


def _select_greedy_action(model: Any, decision: Decision) -> int:
    encoded = encode_decision(decision.observation, decision.legal_actions)
    with torch.inference_mode():
        output = model(encoded)
    if type(output) is tuple:
        if len(output) != 2:
            raise ValueError("evaluation model tuple output must contain logits and value")
        logits, value = output
    else:
        logits, value = output, None
    if not isinstance(logits, torch.Tensor):
        raise TypeError("evaluation model logits must be a tensor")
    if logits.device.type != "cpu":
        raise ValueError("evaluation model logits must be on CPU")
    if logits.dtype != torch.float32 or logits.shape != (len(decision.legal_actions),):
        raise ValueError("evaluation model logits must exactly cover all legal actions as a float32 vector")
    selected = greedy_action(logits)
    if value is not None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("evaluation model value must be a tensor when present")
        if value.device.type != "cpu" or value.dtype != torch.float32 or value.ndim != 0:
            raise ValueError("evaluation model value must be a CPU float32 tensor")
        if not torch.isfinite(value).all():
            raise ValueError("evaluation model produced non-finite value")
    return selected


def _candidate_result(terminal: Terminal, candidate_seat: str) -> tuple[str, int]:
    if terminal.terminal_classification != "natural" or terminal.terminal_code != "natural_game_over":
        raise ValueError("evaluation admitted a non-natural terminal")
    expected = {
        "p0_win": ("p0", [1, -1]),
        "p1_win": ("p1", [-1, 1]),
        "draw": (None, [0, 0]),
    }
    if terminal.terminal_outcome not in expected:
        raise ValueError("evaluation terminal outcome is not a natural win/draw")
    expected_winner, expected_reward = expected[terminal.terminal_outcome]
    if terminal.winner != expected_winner or not json_values_equal_strict(terminal.terminal_reward, expected_reward):
        raise ValueError("evaluation terminal outcome/winner/reward tuple mismatch")
    seat_index = 0 if candidate_seat == "p0" else 1
    points = terminal.terminal_reward[seat_index] + 1
    result = {0: "loss", 1: "draw", 2: "win"}[points]
    if (result == "win") != (terminal.winner == candidate_seat):
        raise ValueError("candidate terminal result disagrees with winner")
    return result, points


def _run_game(
    client: KernelRlClient,
    *,
    pair_index: int,
    game_in_pair: int,
    env_seed: int,
    max_decisions: int,
    candidate_model: KernelPolicyValueNet,
    baseline_model: KernelPolicyValueNet,
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    candidate_seat = "p0" if game_in_pair == 0 else "p1"
    baseline_seat = "p1" if candidate_seat == "p0" else "p0"
    episode_id = 2 * pair_index + game_in_pair
    current: Decision | Terminal = client.reset(
        episode_id=episode_id,
        env_seed=env_seed,
        max_decisions=max_decisions,
    )
    if not json_values_equal_strict(current.provenance, expected_provenance):
        raise ValueError("evaluation environment protocol provenance differs from source training run")
    decisions = 0
    candidate_decisions = 0
    baseline_decisions = 0
    while isinstance(current, Decision):
        if decisions >= max_decisions:
            raise RuntimeError("evaluation exceeded the local max_decisions guard before a natural terminal")
        is_candidate = current.acting_player == candidate_seat
        model = candidate_model if is_candidate else baseline_model
        selected_position = _select_greedy_action(model, current)
        if selected_position < 0 or selected_position >= len(current.legal_actions):
            raise ValueError("greedy model selected an out-of-range legal action")
        selected_action = current.legal_actions[selected_position]
        current = client.step(selected_action["selected_index"], selected_action["stable_id"])
        decisions += 1
        if is_candidate:
            candidate_decisions += 1
        else:
            baseline_decisions += 1
        if not json_values_equal_strict(current.provenance, expected_provenance):
            raise ValueError("evaluation environment protocol provenance drift")
    if not isinstance(current, Terminal):
        raise TypeError("evaluation client returned an unsupported response")
    if current.decision_count != decisions:
        raise ValueError("terminal decision_count differs from evaluator routing count")
    result, points = _candidate_result(current, candidate_seat)
    return {
        "baseline_decisions": baseline_decisions,
        "baseline_seat": baseline_seat,
        "candidate_decisions": candidate_decisions,
        "candidate_half_points": points,
        "candidate_result": result,
        "candidate_seat": candidate_seat,
        "decision_count": current.decision_count,
        "env_seed": env_seed,
        "episode_id": episode_id,
        "game_in_pair": game_in_pair,
        "pair_index": pair_index,
        "schema": GAME_SCHEMA,
        "terminal_classification": current.terminal_classification,
        "terminal_code": current.terminal_code,
        "terminal_outcome": current.terminal_outcome,
        "terminal_reward": current.terminal_reward,
        "winner": current.winner,
    }


def _pair_row(pair_index: int, env_seed: int, points: PairedGamePoints) -> dict[str, Any]:
    return {
        "candidate_as_p0_episode_id": 2 * pair_index,
        "candidate_as_p0_half_points": points.candidate_as_p0,
        "candidate_as_p1_episode_id": 2 * pair_index + 1,
        "candidate_as_p1_half_points": points.candidate_as_p1,
        "env_seed": env_seed,
        "pair_index": pair_index,
        "schema": PAIR_SCHEMA,
        "total_half_points": points.total_half_points,
    }


def _snapshot_manifest(ref: SnapshotRef, role: str) -> dict[str, Any]:
    return {
        "checkpoint_sha256": ref.checkpoint_sha256,
        "head": ref.head,
        "logical_state_sha256": ref.logical_state_sha256,
        "model_contract_fingerprint": ref.model_config.contract_fingerprint(),
        "parent_head": ref.parent_head,
        "role": role,
        "run_digest": ref.run_digest,
        "update": ref.update,
        "update_record_sha256": ref.update_record_sha256,
    }


def _manifest_without_files(
    *,
    chain: ValidatedChain,
    candidate_ref: SnapshotRef,
    baseline_ref: SnapshotRef,
    env_sha: str,
    pair_count: int,
    base_seed: int,
    bootstrap_replicates: int,
    max_decisions: int,
    timeout_ms: int,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    source = chain.run_record
    seed_derivation = dataclasses.asdict(EvaluatorSeedDerivation())
    seed_derivation["namespaces"] = list(seed_derivation["namespaces"])
    return {
        "algorithm": ALGORITHM_CONTRACT,
        "action_selection": {
            "action_rng": "unused",
            "algorithm": "argmax over finite CPU float32 logits",
            "inference": "torch.inference_mode",
            "mode": "greedy",
            "tie_break": "lowest legal-action index",
        },
        "artifact_schemas": {"game": GAME_SCHEMA, "pair": PAIR_SCHEMA, "run": RUN_SCHEMA},
        "configuration": {
            "base_seed": base_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "game_count": 2 * pair_count,
            "max_decisions": max_decisions,
            "pair_count": pair_count,
            "timeout_ms": timeout_ms,
        },
        "environment": {
            "binary_sha256": env_sha,
            "protocol": source["protocol"],
            "protocol_provenance": source["protocol_provenance"],
        },
        "evaluator_runtime_compatibility": source["compatibility"],
        "publication": {
            "authoritative_file": "run.json",
            "data_files_published_first": ["games.jsonl", "pairs.jsonl"],
            "fresh_only": True,
            "resume": False,
        },
        "package": {"name": "mtg-kernel-rl", "version": __version__},
        "schema": RUN_SCHEMA,
        "scoring": {"candidate_draw": 1, "candidate_loss": 0, "candidate_win": 2, "unit": "half_point"},
        "seat_schedule": {
            "candidate_as_p0": "episode 2k",
            "candidate_as_p1": "episode 2k+1",
            "paired_environment_seed": "both games use evaluation-env/base_seed/pair_index",
        },
        "seed_derivation": seed_derivation,
        "snapshots": [_snapshot_manifest(candidate_ref, "candidate"), _snapshot_manifest(baseline_ref, "baseline")],
        "source_training": {
            "environment": source["environment"],
            "feature_contract": source["feature_contract"],
            "model_contract": source["model"],
            "protocol": source["protocol"],
            "protocol_provenance": source["protocol_provenance"],
            "run": {"schema": source["schema"], "sha256": candidate_ref.run_digest},
            "runtime_compatibility": source["compatibility"],
            "trainer_max_decisions": source["trainer"]["max_decisions"],
        },
        "statistics": statistics,
    }


def _preflight_store(
    training_store: str | Path,
    *,
    expected_candidate_head: str,
    env_bin: str | Path,
    out_dir: str | Path,
    max_decisions: int,
) -> tuple[ValidatedChain, SnapshotRef, SnapshotRef, KernelPolicyValueNet, KernelPolicyValueNet, str]:
    chain = TrainingStore(training_store).validate_latest()
    if not chain.snapshots or chain.snapshots[0].update != 0:
        raise ValueError("validated training chain must begin at update zero")
    candidate_ref = chain.head
    baseline_ref = chain.snapshots[0]
    if candidate_ref.head != expected_candidate_head:
        raise ValueError("expected_candidate_head does not match the validated training head")
    source = chain.run_record
    if max_decisions != source["trainer"]["max_decisions"]:
        raise ValueError("max_decisions must exactly match the validated training run contract")
    current_compatibility = runtime_compatibility()
    if not json_values_equal_strict(current_compatibility, source["compatibility"]):
        raise ValueError("current runtime compatibility differs from the validated training run")
    if _paths_overlap(chain.head.root, out_dir):
        raise ValueError("evaluation output root must not overlap the training store")

    env_path = Path(env_bin)
    captured_env = read_regular_file_bytes(env_path, max_bytes=_MAX_ENV_BINARY_BYTES, allow_empty=False)
    if captured_env.sha256 != source["environment"]["binary_sha256"]:
        raise ValueError("evaluation environment binary hash differs from the validated training run")
    candidate = chain.load_policy(candidate_ref)
    baseline = chain.load_policy(baseline_ref)
    if candidate.ref.model_config != baseline.ref.model_config:
        raise ValueError("candidate and baseline model contracts differ")
    if candidate.ref.model_config.contract_fingerprint() != source["model"]["contract_fingerprint"]:
        raise ValueError("loaded model contract differs from the source training run")
    _validate_loaded_model(candidate.model, "candidate")
    _validate_loaded_model(baseline.model, "baseline")
    return chain, candidate_ref, baseline_ref, candidate.model, baseline.model, captured_env.sha256


def evaluate(
    *,
    training_store: str | Path,
    expected_candidate_head: str,
    env_bin: str | Path,
    out_dir: str | Path,
    pairs: int,
    base_seed: int,
    bootstrap_replicates: int,
    max_decisions: int,
    timeout_ms: int,
) -> EvaluationResult:
    """Evaluate the validated head against update zero with fixed greedy pairs."""

    expected_head, pairs, base_seed, bootstrap_replicates, max_decisions, timeout_ms = _validate_request(
        expected_candidate_head=expected_candidate_head,
        pairs=pairs,
        base_seed=base_seed,
        bootstrap_replicates=bootstrap_replicates,
        max_decisions=max_decisions,
        timeout_ms=timeout_ms,
    )
    chain, candidate_ref, baseline_ref, candidate_model, baseline_model, env_sha = _preflight_store(
        training_store,
        expected_candidate_head=expected_head,
        env_bin=env_bin,
        out_dir=out_dir,
        max_decisions=max_decisions,
    )
    expected_provenance = chain.run_record["protocol_provenance"]

    with OutputLock(out_dir) as output_lock:
        # OutputLock resolves cooperating ancestor aliases to one physical root
        # while rejecting a direct alias at the final output component. Keep
        # artifact validation and publication in that same physical namespace.
        root = require_new_or_empty_dir(output_lock.path.parent)
        game_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        paired_points: list[PairedGamePoints] = []
        with KernelRlClient(env_bin, timeout_s=timeout_ms / 1000.0) as client:
            for pair_index in range(pairs):
                env_seed = derive_evaluation_env_seed(base_seed, pair_index)
                candidate_as_p0 = _run_game(
                    client,
                    pair_index=pair_index,
                    game_in_pair=0,
                    env_seed=env_seed,
                    max_decisions=max_decisions,
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                    expected_provenance=expected_provenance,
                )
                candidate_as_p1 = _run_game(
                    client,
                    pair_index=pair_index,
                    game_in_pair=1,
                    env_seed=env_seed,
                    max_decisions=max_decisions,
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                    expected_provenance=expected_provenance,
                )
                points = PairedGamePoints(
                    candidate_as_p0["candidate_half_points"],
                    candidate_as_p1["candidate_half_points"],
                )
                game_rows.extend((candidate_as_p0, candidate_as_p1))
                pair_rows.append(_pair_row(pair_index, env_seed, points))
                paired_points.append(points)

        score = score_pair_half_points(
            [points.total_half_points for points in paired_points],
            derive_evaluation_bootstrap_seed(base_seed),
            bootstrap_replicates,
        )
        game_summary = summarize_paired_game_points(paired_points)
        statistics = statistics_payload(score, game_summary)

        recaptured_env = read_regular_file_bytes(Path(env_bin), max_bytes=_MAX_ENV_BINARY_BYTES, allow_empty=False)
        if recaptured_env.sha256 != env_sha:
            raise ValueError("evaluation environment binary changed during the run")

        # Re-attest only the immutable selected generations. A newer latest.json
        # is a valid concurrent append and must not invalidate this evaluation.
        attested_candidate = chain.load_policy(candidate_ref).model
        _require_model_matches_attestation(candidate_model, attested_candidate, "candidate")
        if baseline_ref is candidate_ref:
            attested_baseline = attested_candidate
        else:
            attested_baseline = chain.load_policy(baseline_ref).model
        _require_model_matches_attestation(baseline_model, attested_baseline, "baseline")
        _validate_loaded_model(candidate_model, "candidate")
        _validate_loaded_model(baseline_model, "baseline")

        manifest = _manifest_without_files(
            chain=chain,
            candidate_ref=candidate_ref,
            baseline_ref=baseline_ref,
            env_sha=env_sha,
            pair_count=pairs,
            base_seed=base_seed,
            bootstrap_replicates=bootstrap_replicates,
            max_decisions=max_decisions,
            timeout_ms=timeout_ms,
            statistics=statistics,
        )
        validated = _publish_evaluation(root, games=game_rows, pairs=pair_rows, manifest_without_files=manifest)
    return EvaluationResult(
        run_sha256=validated.run_sha256,
        candidate_head=validated.candidate_head,
        baseline_head=validated.baseline_head,
        pair_count=validated.pair_count,
        game_count=validated.game_count,
        total_half_points=validated.total_half_points,
        estimate=validated.estimate,
    )


__all__ = ["EvaluationResult", "evaluate"]
