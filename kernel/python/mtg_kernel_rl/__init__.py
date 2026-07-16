"""Python client, feature encoder, model, and runner for the kernel RL process."""

__version__ = "0.2.0"

from .evaluation_stats import (
    BootstrapSummary,
    GameOutcomeSummary,
    PairedGamePoints,
    ScoreSummary,
    SignTestResult,
    WilsonInterval,
    bootstrap_pair_half_points,
    exact_two_sided_sign_test,
    score_pair_half_points,
    summarize_paired_game_points,
    wilson_interval,
)
from .evaluation_store import ValidatedEvaluation, validate_evaluation
from .evaluator import EvaluationResult, evaluate
from .sampled_evaluation_store import validate_sampled_evaluation
from .sampled_evaluator import evaluate_sampled
from .training_store import (
    PolicySnapshot,
    ResumeSnapshot,
    SnapshotRef,
    StoreReadCounts,
    TrainingStore,
    ValidatedChain,
)

__all__ = [
    "BootstrapSummary",
    "EvaluationResult",
    "GameOutcomeSummary",
    "PairedGamePoints",
    "PolicySnapshot",
    "ResumeSnapshot",
    "SnapshotRef",
    "StoreReadCounts",
    "TrainingStore",
    "ValidatedChain",
    "ValidatedEvaluation",
    "ScoreSummary",
    "SignTestResult",
    "WilsonInterval",
    "__version__",
    "bootstrap_pair_half_points",
    "exact_two_sided_sign_test",
    "evaluate",
    "evaluate_sampled",
    "score_pair_half_points",
    "summarize_paired_game_points",
    "validate_evaluation",
    "validate_sampled_evaluation",
    "wilson_interval",
]
