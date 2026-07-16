"""Fresh-only paired-evaluation artifacts and side-effect-free validation."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .artifact_io import (
    CapturedFile,
    canonical_json_bytes,
    json_values_equal_strict,
    parse_canonical_json_bytes,
    read_regular_file_bytes,
    sha256_bytes,
    validate_training_json_privacy,
)
from .artifacts import write_bytes_atomic
from .checkpoint import compute_head
from .determinism import (
    EvaluatorSeedDerivation,
    derive_evaluation_bootstrap_seed,
    derive_evaluation_env_seed,
    validate_positive_int,
    validate_uint63,
)
from .evaluation_stats import (
    GameOutcomeSummary,
    PairedGamePoints,
    ScoreSummary,
    score_pair_half_points,
    summarize_paired_game_points,
)
from .model import ModelConfig
from .path_safety import (
    OUTPUT_LOCK_FILE_NAME,
    ensure_real_dir,
    ensure_real_file,
    is_verified_output_lock_entry,
    scandir_no_follow,
)


RUN_SCHEMA = "kernel_rl_paired_evaluation/v1"
GAME_SCHEMA = "kernel_rl_paired_evaluation_game/v1"
PAIR_SCHEMA = "kernel_rl_paired_evaluation_pair/v1"
RUN_FILE_NAME = "run.json"
GAMES_FILE_NAME = "games.jsonl"
PAIRS_FILE_NAME = "pairs.jsonl"
MAX_PAIR_COUNT = 50_000
MIN_BOOTSTRAP_REPLICATES = 1_000
MAX_BOOTSTRAP_REPLICATES = 100_000
MAX_BOOTSTRAP_DRAWS = 50_000_000
MAX_DECISIONS = 10_000_000
MAX_TIMEOUT_MS = 3_600_000
MAX_RUN_BYTES = 2 * 1024 * 1024
MAX_GAME_ROW_BYTES = 64 * 1024
MAX_PAIR_ROW_BYTES = 16 * 1024
MAX_GAMES_BYTES = 128 * 1024 * 1024
MAX_PAIRS_BYTES = 64 * 1024 * 1024
EXACT_FRACTION_ENCODING = "unsigned-lowercase-hex-magnitude/v1"
SOURCE_RUN_SCHEMA = "kernel_rl_train_run/v11"
ALGORITHM_CONTRACT = {
    "descriptive_intervals": "fixed 95% Wilson over game-level outcomes",
    "name": "greedy_head_vs_update_zero_paired/v1",
    "primary_statistic": "mean candidate half-points per pair divided by 4",
}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_MAGNITUDE_HEX_RE = re.compile(r"^0x(?:0|[1-9a-f][0-9a-f]*)$")
_MAX_SIGN_HEX_DIGITS = (MAX_PAIR_COUNT + 3) // 4 + 1


@dataclass(frozen=True, slots=True)
class ValidatedEvaluation:
    """Scalar result of validating one complete committed evaluation."""

    run_sha256: str
    candidate_head: str
    baseline_head: str
    pair_count: int
    game_count: int
    total_half_points: int
    estimate: float


@dataclass(frozen=True, slots=True)
class _ValidatedContent:
    candidate_head: str
    baseline_head: str
    pair_count: int
    score: ScoreSummary


def _require_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{context} keys mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    return value


def _require_strict_equal(actual: Any, expected: Any, context: str) -> None:
    if not json_values_equal_strict(actual, expected):
        raise ValueError(f"{context} mismatch")


def _int(value: Any, context: str, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer and not bool")
    if value < minimum or value > maximum:
        raise ValueError(f"{context} must be in [{minimum}, {maximum}]")
    return value


def _str(value: Any, context: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise ValueError(f"{context} must be {'a nonempty ' if nonempty else ''}string")
    return value


def _hash(value: Any, context: str) -> str:
    value = _str(value, context)
    if _HEX64_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _seat(value: Any, context: str) -> str:
    value = _str(value, context)
    if value not in ("p0", "p1"):
        raise ValueError(f"{context} must be p0 or p1")
    return value


def _magnitude_hex(value: int) -> str:
    if type(value) is not int or value < 0:
        raise ValueError("exact fraction magnitude must be a nonnegative integer")
    return hex(value)


def _parse_magnitude_hex(value: Any, context: str, *, positive: bool = False) -> int:
    value = _str(value, context)
    if len(value) > _MAX_SIGN_HEX_DIGITS + 2 or _MAGNITUDE_HEX_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be bounded canonical lowercase magnitude hex")
    parsed = int(value[2:], 16)
    if positive and parsed <= 0:
        raise ValueError(f"{context} must be positive")
    if hex(parsed) != value:
        raise ValueError(f"{context} is not canonical magnitude hex")
    return parsed


def _wilson_payload(interval: Any) -> dict[str, Any]:
    return {
        "confidence_level_hex": float(0.95).hex(),
        "estimate_hex": interval.estimate.hex(),
        "lower_hex": interval.lower.hex(),
        "successes": interval.successes,
        "trials": interval.trials,
        "upper_hex": interval.upper.hex(),
    }


def statistics_payload(score: ScoreSummary, games: GameOutcomeSummary) -> dict[str, Any]:
    """Serialize complete paired and game-level statistics without JSON floats."""

    bootstrap = score.bootstrap
    sign = score.sign_test
    return {
        "games": {
            "baseline_wins": games.baseline_wins,
            "candidate_as_p0_wins": games.candidate_as_p0_wins,
            "candidate_as_p1_wins": games.candidate_as_p1_wins,
            "candidate_wins": games.candidate_wins,
            "draws": games.draws,
            "game_count": games.game_count,
            "pair_count": games.pair_count,
            "wilson": {
                "baseline_win": _wilson_payload(games.baseline_win),
                "candidate_as_p0_win": _wilson_payload(games.candidate_as_p0_win),
                "candidate_as_p1_win": _wilson_payload(games.candidate_as_p1_win),
                "candidate_win": _wilson_payload(games.candidate_win),
                "draw": _wilson_payload(games.draw),
            },
        },
        "interpretation": {
            "paired": "primary analysis over complete seat-swapped pair totals",
            "wilson": "descriptive game marginals; paired games share one environment seed",
        },
        "paired": {
            "bootstrap": {
                "bootstrap_replicates": bootstrap.bootstrap_replicates,
                "bootstrap_seed": bootstrap.bootstrap_seed,
                "confidence_level_hex": bootstrap.confidence_level.hex(),
                "lower_hex": bootstrap.lower.hex(),
                "lower_percentile_index": bootstrap.lower_percentile_index,
                "lower_sum": bootstrap.lower_sum,
                "replicate_sums_sha256": bootstrap.replicate_sums_sha256,
                "upper_hex": bootstrap.upper.hex(),
                "upper_percentile_index": bootstrap.upper_percentile_index,
                "upper_sum": bootstrap.upper_sum,
            },
            "estimate_hex": score.estimate.hex(),
            "pair_count": score.pair_count,
            "score_denominator_half_points": 4 * score.pair_count,
            "sign_test": {
                "exact_fraction_encoding": EXACT_FRACTION_ENCODING,
                "losses": sign.losses,
                "non_ties": sign.non_ties,
                "p_value_denominator_hex": _magnitude_hex(sign.p_value_denominator),
                "p_value_hex": sign.p_value.hex(),
                "p_value_numerator_hex": _magnitude_hex(sign.p_value_numerator),
                "ties": sign.ties,
                "wins": sign.wins,
            },
            "total_half_points": score.total_half_points,
        },
    }


def _jsonl_bytes(
    rows: Iterable[dict[str, Any]],
    *,
    row_limit: int,
    row_count_limit: int,
    total_limit: int,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for index, row in enumerate(rows):
        if index >= row_count_limit:
            raise ValueError(f"{label} row count exceeds limit")
        validate_training_json_privacy(row, f"$.{label}[{index}]")
        data = canonical_json_bytes(row)
        if len(data) > row_limit:
            raise ValueError(f"{label} row exceeds byte limit")
        total += len(data)
        if total > total_limit:
            raise ValueError(f"{label} file exceeds byte limit")
        chunks.append(data)
    if not chunks:
        raise ValueError(f"{label} must contain at least one row")
    return b"".join(chunks)


def _publish_evaluation(
    root: str | Path,
    *,
    games: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    manifest_without_files: dict[str, Any],
) -> ValidatedEvaluation:
    """Atomically publish data files and the authoritative run manifest last."""

    root = ensure_real_dir(root)
    initial_entries = scandir_no_follow(root)
    if {entry.name for entry in initial_entries} != {OUTPUT_LOCK_FILE_NAME}:
        raise FileExistsError("fresh evaluation root must contain only the verified persistent lock")
    is_verified_output_lock_entry(root, initial_entries[0])
    configuration = _require_keys(
        manifest_without_files.get("configuration"),
        {"base_seed", "bootstrap_replicates", "game_count", "max_decisions", "pair_count", "timeout_ms"},
        "configuration",
    )
    pair_count = _int(configuration["pair_count"], "pair_count", minimum=1, maximum=MAX_PAIR_COUNT)
    game_count = _int(configuration["game_count"], "game_count", minimum=1, maximum=2 * MAX_PAIR_COUNT)
    if len(pairs) != pair_count or len(games) != 2 * pair_count or game_count != len(games):
        raise ValueError("evaluation rows do not match configured exact pair/game counts")
    games_data = _jsonl_bytes(
        games,
        row_limit=MAX_GAME_ROW_BYTES,
        row_count_limit=2 * MAX_PAIR_COUNT,
        total_limit=MAX_GAMES_BYTES,
        label="games",
    )
    pairs_data = _jsonl_bytes(
        pairs,
        row_limit=MAX_PAIR_ROW_BYTES,
        row_count_limit=MAX_PAIR_COUNT,
        total_limit=MAX_PAIRS_BYTES,
        label="pairs",
    )
    if "files" in manifest_without_files:
        raise ValueError("manifest files metadata is owned by the evaluation store")
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
        raise ValueError("evaluation run manifest exceeds byte limit")
    parsed_manifest = parse_canonical_json_bytes(run_data, source=RUN_FILE_NAME, max_bytes=MAX_RUN_BYTES)

    games_path = root / GAMES_FILE_NAME
    pairs_path = root / PAIRS_FILE_NAME
    run_path = root / RUN_FILE_NAME
    write_bytes_atomic(games_path, games_data)
    captured_games = read_regular_file_bytes(games_path, max_bytes=MAX_GAMES_BYTES, allow_empty=False)
    if captured_games.data != games_data:
        raise ValueError("games artifact changed after atomic publication")
    write_bytes_atomic(pairs_path, pairs_data)
    captured_pairs = read_regular_file_bytes(pairs_path, max_bytes=MAX_PAIRS_BYTES, allow_empty=False)
    if captured_pairs.data != pairs_data:
        raise ValueError("pairs artifact changed after atomic publication")
    content = _validate_captured_payload(parsed_manifest, captured_games, captured_pairs)
    write_bytes_atomic(run_path, run_data)
    captured_run = read_regular_file_bytes(run_path, max_bytes=MAX_RUN_BYTES, allow_empty=False)
    if captured_run.data != run_data:
        raise ValueError("run manifest changed after atomic publication")
    return ValidatedEvaluation(
        run_sha256=captured_run.sha256,
        candidate_head=content.candidate_head,
        baseline_head=content.baseline_head,
        pair_count=content.pair_count,
        game_count=2 * content.pair_count,
        total_half_points=content.score.total_half_points,
        estimate=content.score.estimate,
    )


def _parse_jsonl(
    captured: CapturedFile,
    *,
    row_limit: int,
    row_count_limit: int,
    label: str,
) -> tuple[dict[str, Any], ...]:
    data = captured.data
    if not data.endswith(b"\n"):
        raise ValueError(f"{label} must end with exactly LF-delimited canonical rows")
    if data.count(b"\n") > row_count_limit:
        raise ValueError(f"{label} row count exceeds limit")
    raw_rows = data[:-1].split(b"\n")
    if not raw_rows or any(not row for row in raw_rows):
        raise ValueError(f"{label} contains an empty row")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        if len(row) + 1 > row_limit:
            raise ValueError(f"{label} row exceeds byte limit")
        value = parse_canonical_json_bytes(row + b"\n", source=f"{label} row {index}", max_bytes=row_limit)
        validate_training_json_privacy(value, f"$.{label}[{index}]")
        rows.append(value)
    return tuple(rows)


def _validate_provenance(value: Any, context: str) -> dict[str, Any]:
    value = _require_keys(
        value,
        {"protocol", "protocol_version", "schema_version", "kernel_version", "surface_version", "card_db_hash"},
        context,
    )
    if value["protocol"] != "kernel_rl_jsonl":
        raise ValueError(f"{context}.protocol mismatch")
    _int(value["protocol_version"], f"{context}.protocol_version", maximum=(1 << 32) - 1)
    _int(value["schema_version"], f"{context}.schema_version", maximum=(1 << 32) - 1)
    _str(value["kernel_version"], f"{context}.kernel_version")
    _int(value["surface_version"], f"{context}.surface_version", maximum=(1 << 32) - 1)
    _int(value["card_db_hash"], f"{context}.card_db_hash", maximum=(1 << 64) - 1)
    return value


def _validate_snapshot(value: Any, *, role: str, source_digest: str, model_fingerprint: str) -> dict[str, Any]:
    expected = {
        "role",
        "update",
        "run_digest",
        "head",
        "parent_head",
        "update_record_sha256",
        "checkpoint_sha256",
        "logical_state_sha256",
        "model_contract_fingerprint",
    }
    value = _require_keys(value, expected, f"snapshot {role}")
    if value["role"] != role:
        raise ValueError(f"snapshot {role} role mismatch")
    _int(value["update"], f"snapshot {role}.update", maximum=1_000_000)
    if _hash(value["run_digest"], f"snapshot {role}.run_digest") != source_digest:
        raise ValueError(f"snapshot {role} source digest mismatch")
    _hash(value["head"], f"snapshot {role}.head")
    parent = value["parent_head"]
    if parent is not None:
        _hash(parent, f"snapshot {role}.parent_head")
    _hash(value["update_record_sha256"], f"snapshot {role}.update_record_sha256")
    update_record_sha256 = value["update_record_sha256"]
    checkpoint_sha256 = _hash(value["checkpoint_sha256"], f"snapshot {role}.checkpoint_sha256")
    logical_state_sha256 = _hash(value["logical_state_sha256"], f"snapshot {role}.logical_state_sha256")
    if _hash(value["model_contract_fingerprint"], f"snapshot {role}.model_contract_fingerprint") != model_fingerprint:
        raise ValueError(f"snapshot {role} model fingerprint mismatch")
    if (value["update"] == 0) != (parent is None):
        raise ValueError(f"snapshot {role} parent-head/update invariant mismatch")
    expected_head = compute_head(
        parent_head=parent,
        checkpoint_byte_hash=checkpoint_sha256,
        logical_hash=logical_state_sha256,
        update_hash=update_record_sha256,
    )
    if value["head"] != expected_head:
        raise ValueError(f"snapshot {role} head algebra mismatch")
    return value


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
        "action_selection",
        "seat_schedule",
        "scoring",
        "statistics",
        "files",
        "publication",
    }
    _require_keys(manifest, expected_root, "run.json")
    if manifest["schema"] != RUN_SCHEMA:
        raise ValueError("run.json schema mismatch")
    _require_strict_equal(manifest["package"], {"name": "mtg-kernel-rl", "version": __version__}, "evaluation package identity")
    _require_strict_equal(manifest["algorithm"], ALGORITHM_CONTRACT, "evaluation algorithm identity")
    _require_strict_equal(
        manifest["artifact_schemas"],
        {"game": GAME_SCHEMA, "pair": PAIR_SCHEMA, "run": RUN_SCHEMA},
        "artifact schemas",
    )

    environment = _require_keys(manifest["environment"], {"binary_sha256", "protocol", "protocol_provenance"}, "environment")
    env_sha = _hash(environment["binary_sha256"], "environment.binary_sha256")
    expected_protocol = {"protocol": "kernel_rl_jsonl", "protocol_version": 2, "schema_version": 2}
    _require_strict_equal(environment["protocol"], expected_protocol, "environment protocol contract")
    provenance = _validate_provenance(environment["protocol_provenance"], "environment.protocol_provenance")
    if provenance["protocol_version"] != 2 or provenance["schema_version"] != 2:
        raise ValueError("environment provenance versions differ from protocol contract")

    source = _require_keys(
        manifest["source_training"],
        {
            "run",
            "environment",
            "protocol",
            "protocol_provenance",
            "feature_contract",
            "model_contract",
            "runtime_compatibility",
            "trainer_max_decisions",
        },
        "source_training",
    )
    source_run = _require_keys(source["run"], {"schema", "sha256"}, "source_training.run")
    if source_run["schema"] != SOURCE_RUN_SCHEMA:
        raise ValueError("source training run schema mismatch")
    source_digest = _hash(source_run["sha256"], "source_training.run.sha256")
    source_environment = _require_keys(source["environment"], {"binary_sha256"}, "source_training.environment")
    if _hash(source_environment["binary_sha256"], "source_training.environment.binary_sha256") != env_sha:
        raise ValueError("evaluation environment differs from source training environment")
    _require_strict_equal(source["protocol"], expected_protocol, "source protocol contract")
    _require_strict_equal(source["protocol_provenance"], provenance, "source protocol provenance")
    feature = _require_keys(
        source["feature_contract"],
        {"feature_schema_version", "feature_registry_version", "feature_contract_digest", "feature_encoding_digest"},
        "source_training.feature_contract",
    )
    _str(feature["feature_schema_version"], "feature_schema_version")
    _str(feature["feature_registry_version"], "feature_registry_version")
    _hash(feature["feature_contract_digest"], "feature_contract_digest")
    _hash(feature["feature_encoding_digest"], "feature_encoding_digest")
    model_contract = _require_keys(source["model_contract"], {"config", "contract_fingerprint"}, "source_training.model_contract")
    model_config = ModelConfig.from_dict(model_contract["config"])
    model_fingerprint = _hash(model_contract["contract_fingerprint"], "model contract fingerprint")
    if model_config.contract_fingerprint() != model_fingerprint:
        raise ValueError("model contract fingerprint mismatch")
    if (
        feature["feature_schema_version"] != model_config.feature_schema_version
        or feature["feature_registry_version"] != model_config.feature_registry_version
        or feature["feature_contract_digest"] != model_config.feature_contract_digest
        or feature["feature_encoding_digest"] != model_config.feature_encoding_digest
    ):
        raise ValueError("source feature and model contracts disagree")
    compatibility = _require_keys(
        source["runtime_compatibility"],
        {
            "python_implementation",
            "python_version",
            "python_byteorder",
            "torch_version",
            "torch_config_sha256",
            "os_system",
            "os_release",
            "machine",
            "architecture",
            "cpu_only",
            "default_device",
            "default_dtype",
            "deterministic_algorithms",
            "num_threads",
            "num_interop_threads",
        },
        "source_training.runtime_compatibility",
    )
    for key in (
        "python_implementation",
        "python_version",
        "python_byteorder",
        "torch_version",
        "os_system",
        "os_release",
        "machine",
        "architecture",
        "default_device",
        "default_dtype",
    ):
        _str(compatibility[key], f"runtime_compatibility.{key}", nonempty=False)
    _hash(compatibility["torch_config_sha256"], "runtime_compatibility.torch_config_sha256")
    if compatibility["cpu_only"] is not True or compatibility["deterministic_algorithms"] is not True:
        raise ValueError("runtime compatibility deterministic CPU flags mismatch")
    if compatibility["default_device"] != "cpu" or compatibility["default_dtype"] != "torch.float32":
        raise ValueError("runtime compatibility device/dtype mismatch")
    _int(compatibility["num_threads"], "runtime_compatibility.num_threads", minimum=1, maximum=1)
    _int(compatibility["num_interop_threads"], "runtime_compatibility.num_interop_threads", minimum=1, maximum=1)
    _require_strict_equal(
        manifest["evaluator_runtime_compatibility"],
        compatibility,
        "evaluator runtime compatibility",
    )
    trainer_max_decisions = _int(source["trainer_max_decisions"], "trainer_max_decisions", minimum=1, maximum=MAX_DECISIONS)

    snapshots = manifest["snapshots"]
    if type(snapshots) is not list or len(snapshots) != 2:
        raise ValueError("snapshots must contain candidate then baseline")
    candidate = _validate_snapshot(snapshots[0], role="candidate", source_digest=source_digest, model_fingerprint=model_fingerprint)
    baseline = _validate_snapshot(snapshots[1], role="baseline", source_digest=source_digest, model_fingerprint=model_fingerprint)
    if baseline["update"] != 0 or baseline["parent_head"] is not None:
        raise ValueError("baseline must be update zero")
    if candidate["update"] > 0 and candidate["parent_head"] is None:
        raise ValueError("nonzero candidate update must have a parent head")
    if candidate["update"] == 0:
        _require_strict_equal(candidate, {**baseline, "role": "candidate"}, "update-zero candidate/baseline identity")

    config = _require_keys(
        manifest["configuration"],
        {"base_seed", "bootstrap_replicates", "game_count", "max_decisions", "pair_count", "timeout_ms"},
        "configuration",
    )
    base_seed = validate_uint63(config["base_seed"], "base_seed")
    pair_count = validate_positive_int(config["pair_count"], "pair_count", maximum=MAX_PAIR_COUNT)
    if _int(config["game_count"], "game_count", minimum=1, maximum=2 * MAX_PAIR_COUNT) != 2 * pair_count:
        raise ValueError("game_count must equal twice pair_count")
    bootstrap_replicates = _int(
        config["bootstrap_replicates"],
        "bootstrap_replicates",
        minimum=MIN_BOOTSTRAP_REPLICATES,
        maximum=MAX_BOOTSTRAP_REPLICATES,
    )
    if pair_count * bootstrap_replicates > MAX_BOOTSTRAP_DRAWS:
        raise ValueError("pair_count * bootstrap_replicates exceeds limit")
    max_decisions = _int(config["max_decisions"], "max_decisions", minimum=1, maximum=MAX_DECISIONS)
    if max_decisions != trainer_max_decisions:
        raise ValueError("evaluation max_decisions differs from source training contract")
    _int(config["timeout_ms"], "timeout_ms", minimum=1, maximum=MAX_TIMEOUT_MS)

    seed_derivation = dataclasses.asdict(EvaluatorSeedDerivation())
    seed_derivation["namespaces"] = list(seed_derivation["namespaces"])
    _require_strict_equal(manifest["seed_derivation"], seed_derivation, "evaluator seed derivation contract")
    _require_strict_equal(
        manifest["action_selection"],
        {
            "action_rng": "unused",
            "algorithm": "argmax over finite CPU float32 logits",
            "inference": "torch.inference_mode",
            "mode": "greedy",
            "tie_break": "lowest legal-action index",
        },
        "action selection contract",
    )
    _require_strict_equal(
        manifest["seat_schedule"],
        {
            "candidate_as_p0": "episode 2k",
            "candidate_as_p1": "episode 2k+1",
            "paired_environment_seed": "both games use evaluation-env/base_seed/pair_index",
        },
        "seat schedule contract",
    )
    _require_strict_equal(
        manifest["scoring"],
        {"candidate_loss": 0, "candidate_draw": 1, "candidate_win": 2, "unit": "half_point"},
        "scoring contract",
    )
    _require_strict_equal(
        manifest["publication"],
        {
            "authoritative_file": RUN_FILE_NAME,
            "data_files_published_first": [GAMES_FILE_NAME, PAIRS_FILE_NAME],
            "fresh_only": True,
            "resume": False,
        },
        "publication contract",
    )
    return pair_count, base_seed, bootstrap_replicates, max_decisions, trainer_max_decisions, candidate["head"], baseline["head"]


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
    _require_keys(row, expected_keys, "game row")
    if row["schema"] != GAME_SCHEMA:
        raise ValueError("game row schema mismatch")
    if _int(row["pair_index"], "game pair_index", maximum=MAX_PAIR_COUNT - 1) != pair_index:
        raise ValueError("game pair_index order mismatch")
    if _int(row["game_in_pair"], "game_in_pair", maximum=1) != game_in_pair:
        raise ValueError("game_in_pair order mismatch")
    episode_id = 2 * pair_index + game_in_pair
    if _int(row["episode_id"], "episode_id", maximum=2 * MAX_PAIR_COUNT - 1) != episode_id:
        raise ValueError("episode_id schedule mismatch")
    expected_seed = derive_evaluation_env_seed(base_seed, pair_index)
    if _int(row["env_seed"], "env_seed", maximum=(1 << 63) - 1) != expected_seed:
        raise ValueError("paired environment seed mismatch")
    candidate_seat = _seat(row["candidate_seat"], "candidate_seat")
    expected_candidate_seat = "p0" if game_in_pair == 0 else "p1"
    if candidate_seat != expected_candidate_seat:
        raise ValueError("candidate seat schedule mismatch")
    baseline_seat = _seat(row["baseline_seat"], "baseline_seat")
    if baseline_seat == candidate_seat:
        raise ValueError("baseline seat must oppose candidate seat")
    if row["terminal_classification"] != "natural" or row["terminal_code"] != "natural_game_over":
        raise ValueError("game row is not a natural terminal")
    outcome = _str(row["terminal_outcome"], "terminal_outcome")
    terminal_tuples = {
        "p0_win": ("p0", [1, -1]),
        "p1_win": ("p1", [-1, 1]),
        "draw": (None, [0, 0]),
    }
    if outcome not in terminal_tuples:
        raise ValueError("terminal outcome mismatch")
    expected_winner, expected_reward = terminal_tuples[outcome]
    winner = row["winner"]
    if winner is not None:
        winner = _seat(winner, "winner")
    if winner != expected_winner:
        raise ValueError("terminal winner mismatch")
    _require_strict_equal(row["terminal_reward"], expected_reward, "terminal reward")
    if winner is None:
        result, points = "draw", 1
    elif winner == candidate_seat:
        result, points = "win", 2
    else:
        result, points = "loss", 0
    if row["candidate_result"] != result:
        raise ValueError("candidate result mismatch")
    if _int(row["candidate_half_points"], "candidate_half_points", maximum=2) != points:
        raise ValueError("candidate half-points mismatch")
    decisions = _int(row["decision_count"], "decision_count", minimum=1, maximum=max_decisions)
    candidate_decisions = _int(row["candidate_decisions"], "candidate_decisions", maximum=max_decisions)
    baseline_decisions = _int(row["baseline_decisions"], "baseline_decisions", maximum=max_decisions)
    if candidate_decisions + baseline_decisions != decisions:
        raise ValueError("policy decision counts do not sum to terminal decision_count")
    return points


def _expected_pair_row(pair_index: int, env_seed: int, points: PairedGamePoints) -> dict[str, Any]:
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


def _validate_file_entry(value: Any, captured: CapturedFile, *, rows: int, context: str) -> None:
    value = _require_keys(value, {"row_count", "sha256", "size_bytes"}, context)
    if _int(value["row_count"], f"{context}.row_count", minimum=1, maximum=2 * MAX_PAIR_COUNT) != rows:
        raise ValueError(f"{context} row count mismatch")
    if _hash(value["sha256"], f"{context}.sha256") != captured.sha256:
        raise ValueError(f"{context} hash mismatch")
    if _int(value["size_bytes"], f"{context}.size_bytes", minimum=1, maximum=max(MAX_GAMES_BYTES, MAX_PAIRS_BYTES)) != captured.size:
        raise ValueError(f"{context} size mismatch")


def _validate_captured_payload(
    manifest: dict[str, Any],
    captured_games: CapturedFile,
    captured_pairs: CapturedFile,
) -> _ValidatedContent:
    validate_training_json_privacy(manifest)
    pair_count, base_seed, bootstrap_replicates, max_decisions, _trainer_cap, candidate_head, baseline_head = _validate_manifest(manifest)
    game_rows = _parse_jsonl(
        captured_games,
        row_limit=MAX_GAME_ROW_BYTES,
        row_count_limit=2 * MAX_PAIR_COUNT,
        label=GAMES_FILE_NAME,
    )
    pair_rows = _parse_jsonl(
        captured_pairs,
        row_limit=MAX_PAIR_ROW_BYTES,
        row_count_limit=MAX_PAIR_COUNT,
        label=PAIRS_FILE_NAME,
    )
    if len(game_rows) != 2 * pair_count or len(pair_rows) != pair_count:
        raise ValueError("evaluation row counts do not match configuration")

    paired_points: list[PairedGamePoints] = []
    for pair_index in range(pair_count):
        p0_points = _game_points_from_row(
            game_rows[2 * pair_index],
            pair_index=pair_index,
            game_in_pair=0,
            base_seed=base_seed,
            max_decisions=max_decisions,
        )
        p1_points = _game_points_from_row(
            game_rows[2 * pair_index + 1],
            pair_index=pair_index,
            game_in_pair=1,
            base_seed=base_seed,
            max_decisions=max_decisions,
        )
        points = PairedGamePoints(p0_points, p1_points)
        paired_points.append(points)
        expected_pair = _expected_pair_row(pair_index, derive_evaluation_env_seed(base_seed, pair_index), points)
        _require_strict_equal(pair_rows[pair_index], expected_pair, "pair row derived from game rows")

    score = score_pair_half_points(
        [points.total_half_points for points in paired_points],
        derive_evaluation_bootstrap_seed(base_seed),
        bootstrap_replicates,
    )
    games = summarize_paired_game_points(paired_points)
    expected_statistics = statistics_payload(score, games)
    _require_strict_equal(manifest["statistics"], expected_statistics, "evaluation statistics")
    sign = manifest["statistics"]["paired"]["sign_test"]
    if sign.get("exact_fraction_encoding") != EXACT_FRACTION_ENCODING:
        raise ValueError("sign-test exact fraction encoding mismatch")
    numerator = _parse_magnitude_hex(sign.get("p_value_numerator_hex"), "sign numerator")
    denominator = _parse_magnitude_hex(sign.get("p_value_denominator_hex"), "sign denominator", positive=True)
    if numerator != score.sign_test.p_value_numerator or denominator != score.sign_test.p_value_denominator:
        raise ValueError("sign-test exact fraction mismatch")

    files = _require_keys(manifest["files"], {GAMES_FILE_NAME, PAIRS_FILE_NAME}, "files")
    _validate_file_entry(files[GAMES_FILE_NAME], captured_games, rows=len(game_rows), context=GAMES_FILE_NAME)
    _validate_file_entry(files[PAIRS_FILE_NAME], captured_pairs, rows=len(pair_rows), context=PAIRS_FILE_NAME)
    return _ValidatedContent(
        candidate_head=candidate_head,
        baseline_head=baseline_head,
        pair_count=pair_count,
        score=score,
    )


def validate_evaluation(root: str | Path) -> ValidatedEvaluation:
    """Validate a complete evaluation without writing or launching anything."""

    root = ensure_real_dir(root)
    entries = scandir_no_follow(root)
    expected_names = {OUTPUT_LOCK_FILE_NAME, GAMES_FILE_NAME, PAIRS_FILE_NAME, RUN_FILE_NAME}
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ValueError(f"evaluation root entries mismatch: missing={sorted(expected_names - actual_names)} extra={sorted(actual_names - expected_names)}")
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
    "GAME_SCHEMA",
    "GAMES_FILE_NAME",
    "PAIR_SCHEMA",
    "PAIRS_FILE_NAME",
    "RUN_FILE_NAME",
    "RUN_SCHEMA",
    "ValidatedEvaluation",
    "statistics_payload",
    "validate_evaluation",
]
