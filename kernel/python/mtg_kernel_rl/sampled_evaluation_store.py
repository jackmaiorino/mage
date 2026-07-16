"""Fresh-only sampled paired-evaluation artifacts and strict V3 validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_sampling import (
    fixed_categorical_sampler_contract,
)
from .artifact_io import (
    CapturedFile,
    canonical_json_bytes,
    parse_canonical_json_bytes,
    read_regular_file_bytes,
    sha256_bytes,
    validate_training_json_privacy,
)
from .artifacts import write_bytes_atomic
from .determinism import (
    EVALUATOR_ACTION_SEED_DERIVATION_VERSION,
    derive_evaluation_bootstrap_seed,
    derive_evaluation_env_seed,
)
from .evaluation_stats import PairedGamePoints, ScoreSummary, score_pair_half_points, summarize_paired_game_points
from . import evaluation_store as v1
from .path_safety import (
    OUTPUT_LOCK_FILE_NAME,
    ensure_real_dir,
    ensure_real_file,
    is_verified_output_lock_entry,
    scandir_no_follow,
)


RUN_SCHEMA = "kernel_rl_paired_evaluation/v3"
GAME_SCHEMA = "kernel_rl_paired_evaluation_game/v3"
PAIR_SCHEMA = "kernel_rl_paired_evaluation_pair/v3"
RUN_FILE_NAME = v1.RUN_FILE_NAME
GAMES_FILE_NAME = v1.GAMES_FILE_NAME
PAIRS_FILE_NAME = v1.PAIRS_FILE_NAME

MAX_PAIR_COUNT = v1.MAX_PAIR_COUNT
MIN_BOOTSTRAP_REPLICATES = v1.MIN_BOOTSTRAP_REPLICATES
MAX_BOOTSTRAP_REPLICATES = v1.MAX_BOOTSTRAP_REPLICATES
MAX_BOOTSTRAP_DRAWS = v1.MAX_BOOTSTRAP_DRAWS
MAX_DECISIONS = v1.MAX_DECISIONS
MAX_TIMEOUT_MS = v1.MAX_TIMEOUT_MS
MAX_RUN_BYTES = v1.MAX_RUN_BYTES
MAX_GAME_ROW_BYTES = v1.MAX_GAME_ROW_BYTES
MAX_PAIR_ROW_BYTES = v1.MAX_PAIR_ROW_BYTES
MAX_GAMES_BYTES = v1.MAX_GAMES_BYTES
MAX_PAIRS_BYTES = v1.MAX_PAIRS_BYTES

ALGORITHM_CONTRACT = {
    **v1.ALGORITHM_CONTRACT,
    "name": "sampled_head_vs_update_zero_paired/v2",
}
ACTION_SEED_DERIVATION_CONTRACT = {
    "algorithm": "sha256(type-tagged big-endian length-prefixed fields)[:8] & 0x7fff_ffff_ffff_ffff",
    "namespaces": ["evaluation-action/base_seed/pair_index/physical_seat/local_decision_index"],
    "physical_seat_encoding": {"p0": 0, "p1": 1},
    "version": EVALUATOR_ACTION_SEED_DERIVATION_VERSION,
}
ACTION_SELECTION_CONTRACT = {
    "categorical_sampler": fixed_categorical_sampler_contract(),
    "inference": "torch.inference_mode",
    "mode": "sampled_softmax",
    "replacement": False,
    "temperature_hex": "0x1.0000000000000p+0",
}

# V2 is intentionally frozen rather than reinterpreted as the V3 selector. These
# identities make the release boundary explicit and support corruption tests.
V2_RUN_SCHEMA = "kernel_rl_paired_evaluation/v2"
V2_GAME_SCHEMA = "kernel_rl_paired_evaluation_game/v2"
V2_PAIR_SCHEMA = "kernel_rl_paired_evaluation_pair/v2"
V2_ALGORITHM_CONTRACT = {
    **v1.ALGORITHM_CONTRACT,
    "name": "sampled_head_vs_update_zero_paired/v1",
}
V2_ACTION_SELECTION_CONTRACT = {
    "action_rng": "fresh CPU torch.Generator per decision seeded by action_seed_derivation",
    "algorithm": "torch.multinomial(torch.softmax(finite CPU float32 logits, dim=0).detach(), 1, replacement=False, generator=action_generator)",
    "inference": "torch.inference_mode",
    "mode": "sampled_softmax",
    "replacement": False,
    "temperature_hex": "0x1.0000000000000p+0",
}
SEAT_SCHEDULE_CONTRACT = {
    "candidate_as_p0": "episode 2k",
    "candidate_as_p1": "episode 2k+1",
    "paired_environment_seed": "both games use evaluation-env/base_seed/pair_index",
    "paired_physical_action_streams": "both games use evaluation-action/base_seed/pair_index/physical_seat/local_decision_index",
}

_V1_ACTION_SELECTION_CONTRACT = {
    "action_rng": "unused",
    "algorithm": "argmax over finite CPU float32 logits",
    "inference": "torch.inference_mode",
    "mode": "greedy",
    "tie_break": "lowest legal-action index",
}
_V1_SEAT_SCHEDULE_CONTRACT = {
    "candidate_as_p0": "episode 2k",
    "candidate_as_p1": "episode 2k+1",
    "paired_environment_seed": "both games use evaluation-env/base_seed/pair_index",
}

ValidatedEvaluation = v1.ValidatedEvaluation


@dataclass(frozen=True, slots=True)
class _ValidatedContent:
    candidate_head: str
    baseline_head: str
    pair_count: int
    score: ScoreSummary


def _project_manifest_to_v1(manifest: dict[str, Any]) -> dict[str, Any]:
    projected = dict(manifest)
    del projected["action_seed_derivation"]
    projected["schema"] = v1.RUN_SCHEMA
    projected["algorithm"] = v1.ALGORITHM_CONTRACT
    projected["artifact_schemas"] = {
        "game": v1.GAME_SCHEMA,
        "pair": v1.PAIR_SCHEMA,
        "run": v1.RUN_SCHEMA,
    }
    projected["action_selection"] = _V1_ACTION_SELECTION_CONTRACT
    projected["seat_schedule"] = _V1_SEAT_SCHEDULE_CONTRACT
    return projected


def _validate_manifest(manifest: dict[str, Any]) -> tuple[int, int, int, int, int, str, str]:
    expected_root = {
        "schema",
        "package",
        "algorithm",
        "artifact_schemas",
        "environment",
        "evaluator_runtime_compatibility",
        "source_training",
        "snapshots",
        "configuration",
        "seed_derivation",
        "action_seed_derivation",
        "action_selection",
        "seat_schedule",
        "scoring",
        "statistics",
        "files",
        "publication",
    }
    v1._require_keys(manifest, expected_root, RUN_FILE_NAME)
    if manifest["schema"] != RUN_SCHEMA:
        raise ValueError("sampled run.json schema mismatch")
    v1._require_strict_equal(manifest["algorithm"], ALGORITHM_CONTRACT, "sampled evaluation algorithm identity")
    v1._require_strict_equal(
        manifest["artifact_schemas"],
        {"game": GAME_SCHEMA, "pair": PAIR_SCHEMA, "run": RUN_SCHEMA},
        "sampled artifact schemas",
    )
    v1._require_strict_equal(
        manifest["action_seed_derivation"],
        ACTION_SEED_DERIVATION_CONTRACT,
        "sampled action seed derivation contract",
    )
    v1._require_strict_equal(
        manifest["action_selection"],
        ACTION_SELECTION_CONTRACT,
        "sampled action selection contract",
    )
    v1._require_strict_equal(
        manifest["seat_schedule"],
        SEAT_SCHEDULE_CONTRACT,
        "sampled seat schedule contract",
    )
    return v1._validate_manifest(_project_manifest_to_v1(manifest))


def _game_points_from_row(
    row: dict[str, Any],
    *,
    pair_index: int,
    game_in_pair: int,
    base_seed: int,
    max_decisions: int,
) -> int:
    expected_keys = {
        "schema",
        "pair_index",
        "game_in_pair",
        "episode_id",
        "env_seed",
        "candidate_seat",
        "baseline_seat",
        "terminal_outcome",
        "terminal_classification",
        "terminal_code",
        "terminal_reward",
        "winner",
        "candidate_result",
        "candidate_half_points",
        "decision_count",
        "candidate_decisions",
        "baseline_decisions",
    }
    v1._require_keys(row, expected_keys, "sampled game row")
    if row["schema"] != GAME_SCHEMA:
        raise ValueError("sampled game row schema mismatch")
    projected = {**row, "schema": v1.GAME_SCHEMA}
    return v1._game_points_from_row(
        projected,
        pair_index=pair_index,
        game_in_pair=game_in_pair,
        base_seed=base_seed,
        max_decisions=max_decisions,
    )


def _expected_pair_row(pair_index: int, env_seed: int, points: PairedGamePoints) -> dict[str, Any]:
    return {**v1._expected_pair_row(pair_index, env_seed, points), "schema": PAIR_SCHEMA}


def _validate_aa_pair(
    candidate_as_p0: dict[str, Any],
    candidate_as_p1: dict[str, Any],
    points: PairedGamePoints,
) -> None:
    for field in (
        "terminal_outcome",
        "terminal_classification",
        "terminal_code",
        "terminal_reward",
        "winner",
    ):
        v1._require_strict_equal(
            candidate_as_p0[field],
            candidate_as_p1[field],
            f"sampled A/A physical terminal {field}",
        )
    if candidate_as_p0["decision_count"] != candidate_as_p1["decision_count"]:
        raise ValueError("sampled A/A legs must have identical decision_count")
    if candidate_as_p0["candidate_decisions"] != candidate_as_p1["baseline_decisions"]:
        raise ValueError("sampled A/A p0 physical decision counts differ across legs")
    if candidate_as_p0["baseline_decisions"] != candidate_as_p1["candidate_decisions"]:
        raise ValueError("sampled A/A p1 physical decision counts differ across legs")
    if points.total_half_points != 2:
        raise ValueError("sampled A/A pair must total exactly two half-points")


def _validate_captured_payload(
    manifest: dict[str, Any],
    captured_games: CapturedFile,
    captured_pairs: CapturedFile,
) -> _ValidatedContent:
    validate_training_json_privacy(manifest)
    pair_count, base_seed, bootstrap_replicates, max_decisions, _trainer_cap, candidate_head, baseline_head = (
        _validate_manifest(manifest)
    )
    game_rows = v1._parse_jsonl(
        captured_games,
        row_limit=MAX_GAME_ROW_BYTES,
        row_count_limit=2 * MAX_PAIR_COUNT,
        label=GAMES_FILE_NAME,
    )
    pair_rows = v1._parse_jsonl(
        captured_pairs,
        row_limit=MAX_PAIR_ROW_BYTES,
        row_count_limit=MAX_PAIR_COUNT,
        label=PAIRS_FILE_NAME,
    )
    if len(game_rows) != 2 * pair_count or len(pair_rows) != pair_count:
        raise ValueError("sampled evaluation row counts do not match configuration")

    paired_points: list[PairedGamePoints] = []
    for pair_index in range(pair_count):
        candidate_as_p0 = game_rows[2 * pair_index]
        candidate_as_p1 = game_rows[2 * pair_index + 1]
        p0_points = _game_points_from_row(
            candidate_as_p0,
            pair_index=pair_index,
            game_in_pair=0,
            base_seed=base_seed,
            max_decisions=max_decisions,
        )
        p1_points = _game_points_from_row(
            candidate_as_p1,
            pair_index=pair_index,
            game_in_pair=1,
            base_seed=base_seed,
            max_decisions=max_decisions,
        )
        points = PairedGamePoints(p0_points, p1_points)
        paired_points.append(points)
        expected_pair = _expected_pair_row(pair_index, derive_evaluation_env_seed(base_seed, pair_index), points)
        v1._require_strict_equal(pair_rows[pair_index], expected_pair, "sampled pair row derived from game rows")
        if candidate_head == baseline_head:
            _validate_aa_pair(candidate_as_p0, candidate_as_p1, points)

    score = score_pair_half_points(
        [points.total_half_points for points in paired_points],
        derive_evaluation_bootstrap_seed(base_seed),
        bootstrap_replicates,
    )
    games = summarize_paired_game_points(paired_points)
    expected_statistics = v1.statistics_payload(score, games)
    v1._require_strict_equal(manifest["statistics"], expected_statistics, "sampled evaluation statistics")
    sign = manifest["statistics"]["paired"]["sign_test"]
    if sign.get("exact_fraction_encoding") != v1.EXACT_FRACTION_ENCODING:
        raise ValueError("sampled sign-test exact fraction encoding mismatch")
    numerator = v1._parse_magnitude_hex(sign.get("p_value_numerator_hex"), "sampled sign numerator")
    denominator = v1._parse_magnitude_hex(
        sign.get("p_value_denominator_hex"),
        "sampled sign denominator",
        positive=True,
    )
    if numerator != score.sign_test.p_value_numerator or denominator != score.sign_test.p_value_denominator:
        raise ValueError("sampled sign-test exact fraction mismatch")

    files = v1._require_keys(manifest["files"], {GAMES_FILE_NAME, PAIRS_FILE_NAME}, "sampled files")
    v1._validate_file_entry(files[GAMES_FILE_NAME], captured_games, rows=len(game_rows), context=GAMES_FILE_NAME)
    v1._validate_file_entry(files[PAIRS_FILE_NAME], captured_pairs, rows=len(pair_rows), context=PAIRS_FILE_NAME)
    return _ValidatedContent(
        candidate_head=candidate_head,
        baseline_head=baseline_head,
        pair_count=pair_count,
        score=score,
    )


def _publish_sampled_evaluation(
    root: str | Path,
    *,
    games: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    manifest_without_files: dict[str, Any],
) -> ValidatedEvaluation:
    """Atomically publish V2 data files and the authoritative manifest last."""

    root = ensure_real_dir(root)
    initial_entries = scandir_no_follow(root)
    if {entry.name for entry in initial_entries} != {OUTPUT_LOCK_FILE_NAME}:
        raise FileExistsError("fresh sampled evaluation root must contain only the verified persistent lock")
    is_verified_output_lock_entry(root, initial_entries[0])
    configuration = v1._require_keys(
        manifest_without_files.get("configuration"),
        {"base_seed", "bootstrap_replicates", "game_count", "max_decisions", "pair_count", "timeout_ms"},
        "sampled configuration",
    )
    pair_count = v1._int(configuration["pair_count"], "pair_count", minimum=1, maximum=MAX_PAIR_COUNT)
    game_count = v1._int(configuration["game_count"], "game_count", minimum=1, maximum=2 * MAX_PAIR_COUNT)
    if len(pairs) != pair_count or len(games) != 2 * pair_count or game_count != len(games):
        raise ValueError("sampled evaluation rows do not match configured exact pair/game counts")
    games_data = v1._jsonl_bytes(
        games,
        row_limit=MAX_GAME_ROW_BYTES,
        row_count_limit=2 * MAX_PAIR_COUNT,
        total_limit=MAX_GAMES_BYTES,
        label="games",
    )
    pairs_data = v1._jsonl_bytes(
        pairs,
        row_limit=MAX_PAIR_ROW_BYTES,
        row_count_limit=MAX_PAIR_COUNT,
        total_limit=MAX_PAIRS_BYTES,
        label="pairs",
    )
    if "files" in manifest_without_files:
        raise ValueError("sampled manifest files metadata is owned by the evaluation store")
    manifest = {
        **manifest_without_files,
        "files": {
            GAMES_FILE_NAME: {
                "row_count": len(games),
                "sha256": sha256_bytes(games_data),
                "size_bytes": len(games_data),
            },
            PAIRS_FILE_NAME: {
                "row_count": len(pairs),
                "sha256": sha256_bytes(pairs_data),
                "size_bytes": len(pairs_data),
            },
        },
    }
    validate_training_json_privacy(manifest)
    run_data = canonical_json_bytes(manifest)
    if len(run_data) > MAX_RUN_BYTES:
        raise ValueError("sampled evaluation run manifest exceeds byte limit")
    parsed_manifest = parse_canonical_json_bytes(run_data, source=RUN_FILE_NAME, max_bytes=MAX_RUN_BYTES)

    games_path = root / GAMES_FILE_NAME
    pairs_path = root / PAIRS_FILE_NAME
    run_path = root / RUN_FILE_NAME
    write_bytes_atomic(games_path, games_data)
    captured_games = read_regular_file_bytes(games_path, max_bytes=MAX_GAMES_BYTES, allow_empty=False)
    if captured_games.data != games_data:
        raise ValueError("sampled games artifact changed after atomic publication")
    write_bytes_atomic(pairs_path, pairs_data)
    captured_pairs = read_regular_file_bytes(pairs_path, max_bytes=MAX_PAIRS_BYTES, allow_empty=False)
    if captured_pairs.data != pairs_data:
        raise ValueError("sampled pairs artifact changed after atomic publication")
    content = _validate_captured_payload(parsed_manifest, captured_games, captured_pairs)
    write_bytes_atomic(run_path, run_data)
    captured_run = read_regular_file_bytes(run_path, max_bytes=MAX_RUN_BYTES, allow_empty=False)
    if captured_run.data != run_data:
        raise ValueError("sampled run manifest changed after atomic publication")
    return ValidatedEvaluation(
        run_sha256=captured_run.sha256,
        candidate_head=content.candidate_head,
        baseline_head=content.baseline_head,
        pair_count=content.pair_count,
        game_count=2 * content.pair_count,
        total_half_points=content.score.total_half_points,
        estimate=content.score.estimate,
    )


def validate_sampled_evaluation(root: str | Path) -> ValidatedEvaluation:
    """Validate one complete sampled V3 evaluation without side effects."""

    root = ensure_real_dir(root)
    entries = scandir_no_follow(root)
    expected_names = {OUTPUT_LOCK_FILE_NAME, GAMES_FILE_NAME, PAIRS_FILE_NAME, RUN_FILE_NAME}
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ValueError(
            f"sampled evaluation root entries mismatch: missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    by_name = {entry.name: entry for entry in entries}
    is_verified_output_lock_entry(root, by_name[OUTPUT_LOCK_FILE_NAME])
    for name in (GAMES_FILE_NAME, PAIRS_FILE_NAME, RUN_FILE_NAME):
        ensure_real_file(root, root / name, reject_hardlinks=True)

    captured_run = read_regular_file_bytes(root / RUN_FILE_NAME, max_bytes=MAX_RUN_BYTES, allow_empty=False)
    manifest = parse_canonical_json_bytes(captured_run.data, source=RUN_FILE_NAME, max_bytes=MAX_RUN_BYTES)
    captured_games = read_regular_file_bytes(root / GAMES_FILE_NAME, max_bytes=MAX_GAMES_BYTES, allow_empty=False)
    captured_pairs = read_regular_file_bytes(root / PAIRS_FILE_NAME, max_bytes=MAX_PAIRS_BYTES, allow_empty=False)
    content = _validate_captured_payload(manifest, captured_games, captured_pairs)
    return ValidatedEvaluation(
        run_sha256=captured_run.sha256,
        candidate_head=content.candidate_head,
        baseline_head=content.baseline_head,
        pair_count=content.pair_count,
        game_count=2 * content.pair_count,
        total_half_points=content.score.total_half_points,
        estimate=content.score.estimate,
    )


__all__ = [
    "ACTION_SEED_DERIVATION_CONTRACT",
    "ACTION_SELECTION_CONTRACT",
    "ALGORITHM_CONTRACT",
    "GAME_SCHEMA",
    "GAMES_FILE_NAME",
    "PAIR_SCHEMA",
    "PAIRS_FILE_NAME",
    "RUN_FILE_NAME",
    "RUN_SCHEMA",
    "SEAT_SCHEDULE_CONTRACT",
    "V2_ACTION_SELECTION_CONTRACT",
    "V2_ALGORITHM_CONTRACT",
    "V2_GAME_SCHEMA",
    "V2_PAIR_SCHEMA",
    "V2_RUN_SCHEMA",
    "ValidatedEvaluation",
    "validate_sampled_evaluation",
]
