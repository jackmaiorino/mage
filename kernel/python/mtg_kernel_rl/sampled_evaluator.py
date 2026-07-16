"""Deterministic sampled head-versus-update-zero paired evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .action_sampling import sample_fixed_categorical
from .artifact_io import json_values_equal_strict, read_regular_file_bytes
from .artifacts import require_new_or_empty_dir
from .client import Decision, KernelRlClient, Terminal
from .determinism import (
    derive_evaluation_action_seed,
    derive_evaluation_bootstrap_seed,
    derive_evaluation_env_seed,
    validate_uint63,
)
from .evaluation_stats import PairedGamePoints, score_pair_half_points, summarize_paired_game_points
from .evaluation_store import statistics_payload
from .evaluator import (
    EvaluationResult,
    _MAX_ENV_BINARY_BYTES,
    _candidate_result,
    _manifest_without_files,
    _preflight_store,
    _require_model_matches_attestation,
    _validate_loaded_model,
    _validate_request,
)
from .features import encode_decision
from .model import KernelPolicyValueNet
from .output_lock import OutputLock
from .sampled_evaluation_store import (
    ACTION_SEED_DERIVATION_CONTRACT,
    ACTION_SELECTION_CONTRACT,
    ALGORITHM_CONTRACT,
    GAME_SCHEMA,
    PAIR_SCHEMA,
    RUN_SCHEMA,
    SEAT_SCHEDULE_CONTRACT,
    _publish_sampled_evaluation,
)
from .training_store import SnapshotRef, ValidatedChain


def _select_sampled_action(model: Any, decision: Decision, action_seed: int) -> int:
    action_seed = validate_uint63(action_seed, "action_seed")
    encoded = encode_decision(decision.observation, decision.legal_actions)
    with torch.inference_mode():
        output = model(encoded)
    if type(output) is tuple:
        if len(output) != 2:
            raise ValueError("sampled evaluation model tuple output must contain logits and value")
        logits, value = output
    else:
        logits, value = output, None
    if not isinstance(logits, torch.Tensor):
        raise TypeError("sampled evaluation model logits must be a tensor")
    if logits.device.type != "cpu":
        raise ValueError("sampled evaluation model logits must be on CPU")
    if logits.dtype != torch.float32 or logits.shape != (len(decision.legal_actions),):
        raise ValueError("sampled evaluation model logits must exactly cover all legal actions as a float32 vector")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("sampled evaluation model produced non-finite logits")
    if value is not None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("sampled evaluation model value must be a tensor when present")
        if value.device.type != "cpu" or value.dtype != torch.float32 or value.ndim != 0:
            raise ValueError("sampled evaluation model value must be a CPU float32 tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("sampled evaluation model produced non-finite value")

    selected_position = sample_fixed_categorical(logits, action_seed)
    if selected_position < 0 or selected_position >= len(decision.legal_actions):
        raise ValueError("sampled model selected an out-of-range legal action")
    return selected_position


def _run_sampled_game(
    client: KernelRlClient,
    *,
    pair_index: int,
    game_in_pair: int,
    env_seed: int,
    base_seed: int,
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
        raise ValueError("sampled evaluation environment protocol provenance differs from source training run")
    decisions = 0
    candidate_decisions = 0
    baseline_decisions = 0
    physical_decision_counts = {"p0": 0, "p1": 0}
    while isinstance(current, Decision):
        if decisions >= max_decisions:
            raise RuntimeError("sampled evaluation exceeded the local max_decisions guard before a natural terminal")
        actor = current.acting_player
        if actor not in physical_decision_counts:
            raise ValueError("sampled evaluation decision actor must be p0 or p1")
        is_candidate = actor == candidate_seat
        model = candidate_model if is_candidate else baseline_model
        local_decision_index = physical_decision_counts[actor]
        action_seed = derive_evaluation_action_seed(base_seed, pair_index, actor, local_decision_index)
        selected_position = _select_sampled_action(model, current, action_seed)
        physical_decision_counts[actor] += 1
        selected_action = current.legal_actions[selected_position]
        current = client.step(selected_action["selected_index"], selected_action["stable_id"])
        decisions += 1
        if is_candidate:
            candidate_decisions += 1
        else:
            baseline_decisions += 1
        if not json_values_equal_strict(current.provenance, expected_provenance):
            raise ValueError("sampled evaluation environment protocol provenance drift")
    if not isinstance(current, Terminal):
        raise TypeError("sampled evaluation client returned an unsupported response")
    if current.decision_count != decisions:
        raise ValueError("sampled terminal decision_count differs from evaluator routing count")
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


def _sampled_manifest_without_files(
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
    manifest = _manifest_without_files(
        chain=chain,
        candidate_ref=candidate_ref,
        baseline_ref=baseline_ref,
        env_sha=env_sha,
        pair_count=pair_count,
        base_seed=base_seed,
        bootstrap_replicates=bootstrap_replicates,
        max_decisions=max_decisions,
        timeout_ms=timeout_ms,
        statistics=statistics,
    )
    manifest["schema"] = RUN_SCHEMA
    manifest["algorithm"] = ALGORITHM_CONTRACT
    manifest["artifact_schemas"] = {"game": GAME_SCHEMA, "pair": PAIR_SCHEMA, "run": RUN_SCHEMA}
    manifest["action_seed_derivation"] = ACTION_SEED_DERIVATION_CONTRACT
    manifest["action_selection"] = ACTION_SELECTION_CONTRACT
    manifest["seat_schedule"] = SEAT_SCHEDULE_CONTRACT
    return manifest


def evaluate_sampled(
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
    """Evaluate the validated head against update zero with sampled CRN pairs."""

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
        root = require_new_or_empty_dir(output_lock.path.parent)
        game_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        paired_points: list[PairedGamePoints] = []
        with KernelRlClient(env_bin, timeout_s=timeout_ms / 1000.0) as client:
            for pair_index in range(pairs):
                env_seed = derive_evaluation_env_seed(base_seed, pair_index)
                candidate_as_p0 = _run_sampled_game(
                    client,
                    pair_index=pair_index,
                    game_in_pair=0,
                    env_seed=env_seed,
                    base_seed=base_seed,
                    max_decisions=max_decisions,
                    candidate_model=candidate_model,
                    baseline_model=baseline_model,
                    expected_provenance=expected_provenance,
                )
                candidate_as_p1 = _run_sampled_game(
                    client,
                    pair_index=pair_index,
                    game_in_pair=1,
                    env_seed=env_seed,
                    base_seed=base_seed,
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
            raise ValueError("sampled evaluation environment binary changed during the run")

        attested_candidate = chain.load_policy(candidate_ref).model
        _require_model_matches_attestation(candidate_model, attested_candidate, "sampled candidate")
        if baseline_ref is candidate_ref:
            attested_baseline = attested_candidate
        else:
            attested_baseline = chain.load_policy(baseline_ref).model
        _require_model_matches_attestation(baseline_model, attested_baseline, "sampled baseline")
        _validate_loaded_model(candidate_model, "sampled candidate")
        _validate_loaded_model(baseline_model, "sampled baseline")

        manifest = _sampled_manifest_without_files(
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
        validated = _publish_sampled_evaluation(
            root,
            games=game_rows,
            pairs=pair_rows,
            manifest_without_files=manifest,
        )
    return EvaluationResult(
        run_sha256=validated.run_sha256,
        candidate_head=validated.candidate_head,
        baseline_head=validated.baseline_head,
        pair_count=validated.pair_count,
        game_count=validated.game_count,
        total_half_points=validated.total_half_points,
        estimate=validated.estimate,
    )


__all__ = ["EvaluationResult", "evaluate_sampled"]
