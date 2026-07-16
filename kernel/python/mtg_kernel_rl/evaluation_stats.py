"""Deterministic statistics for paired kernel-policy evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Set
from dataclasses import dataclass
from decimal import Context, Decimal, DivisionByZero, InvalidOperation, Overflow, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from typing import Any


_MASK64 = 0xFFFF_FFFF_FFFF_FFFF
_UINT64_CARDINALITY = 1 << 64
_GOLDEN_RATIO_64 = 0x9E37_79B9_7F4A_7C15
_MAX_PAIR_COUNT = 50_000
_MAX_WILSON_TRIALS = 2 * _MAX_PAIR_COUNT
_MIN_BOOTSTRAP_REPLICATES = 1_000
_MAX_BOOTSTRAP_REPLICATES = 100_000
_MAX_BOOTSTRAP_DRAWS = 50_000_000
_WILSON_Z_TEXT = "1.959963984540054"


@dataclass(frozen=True)
class BootstrapSummary:
    """Observed paired score and its deterministic percentile bootstrap."""

    pair_count: int
    total_half_points: int
    estimate: float
    confidence_level: float
    bootstrap_seed: int
    bootstrap_replicates: int
    lower_percentile_index: int
    upper_percentile_index: int
    lower_sum: int
    upper_sum: int
    lower: float
    upper: float
    replicate_sums_sha256: str


@dataclass(frozen=True)
class SignTestResult:
    """Exact two-sided sign test over better, worse, and tied pairs.

    ``p_value_numerator / p_value_denominator`` is authoritative. ``p_value``
    is a lossy convenience float and can underflow for extreme samples.
    """

    wins: int
    losses: int
    ties: int
    non_ties: int
    p_value_numerator: int
    p_value_denominator: int
    p_value: float


@dataclass(frozen=True)
class WilsonInterval:
    """A fixed-95% Wilson interval for independent Bernoulli trials."""

    successes: int
    trials: int
    estimate: float
    lower: float
    upper: float

    @property
    def estimate_hex(self) -> str:
        return self.estimate.hex()

    @property
    def lower_hex(self) -> str:
        return self.lower.hex()

    @property
    def upper_hex(self) -> str:
        return self.upper.hex()


@dataclass(frozen=True)
class ScoreSummary:
    """Primary paired estimate with paired bootstrap and sign-test evidence."""

    pair_count: int
    total_half_points: int
    estimate: float
    bootstrap: BootstrapSummary
    sign_test: SignTestResult


@dataclass(frozen=True)
class PairedGamePoints:
    """Candidate half-points from the two fixed seats in one pair."""

    candidate_as_p0: int
    candidate_as_p1: int

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_as_p0", self.candidate_as_p0),
            ("candidate_as_p1", self.candidate_as_p1),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer and not bool")
            if value < 0 or value > 2:
                raise ValueError(f"{name} must be in [0, 2]")

    @property
    def total_half_points(self) -> int:
        return self.candidate_as_p0 + self.candidate_as_p1


@dataclass(frozen=True)
class GameOutcomeSummary:
    """Game-level W/D/L counts and descriptive fixed-95% Wilson intervals."""

    pair_count: int
    game_count: int
    candidate_wins: int
    draws: int
    baseline_wins: int
    candidate_as_p0_wins: int
    candidate_as_p1_wins: int
    candidate_win: WilsonInterval
    draw: WilsonInterval
    baseline_win: WilsonInterval
    candidate_as_p0_win: WilsonInterval
    candidate_as_p1_win: WilsonInterval


class _SplitMix64:
    def __init__(self, seed: int) -> None:
        self._state = _validate_uint64(seed, "seed")

    def next_u64(self) -> int:
        self._state = (self._state + _GOLDEN_RATIO_64) & _MASK64
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & _MASK64
        return (value ^ (value >> 31)) & _MASK64


def _validate_uint64(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer and not bool")
    if value < 0 or value > _MASK64:
        raise ValueError(f"{name} must be in [0, 2**64 - 1]")
    return value


def _validate_bounded_positive_int(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer and not bool")
    if value <= 0 or value > maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


def _validate_pair_half_points(pair_half_points: Iterable[int]) -> tuple[int, ...]:
    """Materialize one ordered observation stream exactly once."""

    if isinstance(pair_half_points, (Set, Mapping)):
        raise TypeError("pair_half_points must be ordered; sets and mappings are not supported")
    try:
        iterator = iter(pair_half_points)
    except TypeError as exc:
        raise TypeError("pair_half_points must be an iterable of integers") from exc
    values: list[int] = []
    for index, value in enumerate(iterator):
        if index >= _MAX_PAIR_COUNT:
            raise ValueError(f"pair_half_points must contain at most {_MAX_PAIR_COUNT} pairs")
        if type(value) is not int:
            raise TypeError(f"pair_half_points[{index}] must be an integer and not bool")
        if value < 0 or value > 4:
            raise ValueError(f"pair_half_points[{index}] must be in [0, 4]")
        values.append(value)
    if not values:
        raise ValueError("pair_half_points must contain at least one pair")
    return tuple(values)


def _validate_bootstrap_replicates(bootstrap_replicates: Any, pair_count: int) -> int:
    if type(bootstrap_replicates) is not int:
        raise TypeError("bootstrap_replicates must be an integer and not bool")
    if bootstrap_replicates < _MIN_BOOTSTRAP_REPLICATES or bootstrap_replicates > _MAX_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"bootstrap_replicates must be in [{_MIN_BOOTSTRAP_REPLICATES}, {_MAX_BOOTSTRAP_REPLICATES}]"
        )
    if pair_count * bootstrap_replicates > _MAX_BOOTSTRAP_DRAWS:
        raise ValueError(f"pair_count * bootstrap_replicates must be at most {_MAX_BOOTSTRAP_DRAWS}")
    return bootstrap_replicates


def _unbiased_index(population_size: int, next_u64: Callable[[], int]) -> int:
    """Draw an unbiased index, rejecting the incomplete upper modulo bucket."""

    _validate_bounded_positive_int(population_size, "population_size", _MAX_PAIR_COUNT)
    limit = (_UINT64_CARDINALITY // population_size) * population_size
    while True:
        value = next_u64()
        if type(value) is not int or value < 0 or value > _MASK64:
            raise ValueError("next_u64 must return an integer in [0, 2**64 - 1]")
        if value < limit:
            return value % population_size


def _replicate_sums_digest(replicate_sums: tuple[int, ...]) -> str:
    hasher = hashlib.sha256()
    for index, value in enumerate(replicate_sums):
        if index:
            hasher.update(b",")
        hasher.update(str(value).encode("ascii"))
    hasher.update(b"\n")
    return hasher.hexdigest()


def _bootstrap_replicate_sums(
    pair_half_points: tuple[int, ...],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[int, ...]:
    pair_count = len(pair_half_points)
    bootstrap_seed = _validate_uint64(bootstrap_seed, "bootstrap_seed")
    bootstrap_replicates = _validate_bootstrap_replicates(bootstrap_replicates, pair_count)
    rng = _SplitMix64(bootstrap_seed)
    replicate_sums_list: list[int] = []
    for _ in range(bootstrap_replicates):
        replicate_sum = 0
        for _ in range(pair_count):
            replicate_sum += pair_half_points[_unbiased_index(pair_count, rng.next_u64)]
        replicate_sums_list.append(replicate_sum)
    return tuple(replicate_sums_list)


def _bootstrap_validated(
    pair_half_points: tuple[int, ...],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> BootstrapSummary:
    pair_count = len(pair_half_points)
    bootstrap_seed = _validate_uint64(bootstrap_seed, "bootstrap_seed")
    bootstrap_replicates = _validate_bootstrap_replicates(bootstrap_replicates, pair_count)
    replicate_sums = _bootstrap_replicate_sums(pair_half_points, bootstrap_seed, bootstrap_replicates)
    ordered_sums = sorted(replicate_sums)
    lower_index = (bootstrap_replicates - 1) // 40
    upper_index = (39 * (bootstrap_replicates - 1) + 39) // 40
    lower_sum = ordered_sums[lower_index]
    upper_sum = ordered_sums[upper_index]
    denominator = 4 * pair_count
    total_half_points = sum(pair_half_points)
    return BootstrapSummary(
        pair_count=pair_count,
        total_half_points=total_half_points,
        estimate=total_half_points / denominator,
        confidence_level=0.95,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        lower_percentile_index=lower_index,
        upper_percentile_index=upper_index,
        lower_sum=lower_sum,
        upper_sum=upper_sum,
        lower=lower_sum / denominator,
        upper=upper_sum / denominator,
        replicate_sums_sha256=_replicate_sums_digest(replicate_sums),
    )


def bootstrap_pair_half_points(
    pair_half_points: Iterable[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> BootstrapSummary:
    """Bootstrap complete pair totals and return a fixed 95% interval.

    ``bootstrap_seed`` is an explicit raw unsigned-64 SplitMix64 seed. Seeds
    from the versioned evaluator derivation are a compatible unsigned-63 subset.
    The ordered input is materialized exactly once, and that order is part of
    the raw replicate-sum digest contract; sets and mappings are rejected.
    """

    values = _validate_pair_half_points(pair_half_points)
    return _bootstrap_validated(values, bootstrap_seed, bootstrap_replicates)


def _sum_binomial_range(trials: int, start: int, end: int) -> int:
    if start > end:
        return 0
    term = math.comb(trials, start)
    total = term
    for successes in range(start + 1, end + 1):
        term = term * (trials - successes + 1) // successes
        total += term
    return total


def _sign_test_validated(pair_half_points: tuple[int, ...]) -> SignTestResult:
    wins = sum(value > 2 for value in pair_half_points)
    losses = sum(value < 2 for value in pair_half_points)
    ties = len(pair_half_points) - wins - losses
    non_ties = wins + losses
    if non_ties == 0 or abs(wins - losses) <= 1:
        exact_p_value = Fraction(1, 1)
    else:
        smaller_count = min(wins, losses)
        denominator = 1 << non_ties
        central_start = smaller_count + 1
        central_end = non_ties - smaller_count - 1
        lower_tail_terms = smaller_count + 1
        central_terms = central_end - central_start + 1
        if lower_tail_terms <= central_terms:
            numerator = 2 * _sum_binomial_range(non_ties, 0, smaller_count)
        else:
            numerator = denominator - _sum_binomial_range(non_ties, central_start, central_end)
        exact_p_value = Fraction(numerator, denominator)
    return SignTestResult(
        wins=wins,
        losses=losses,
        ties=ties,
        non_ties=non_ties,
        p_value_numerator=exact_p_value.numerator,
        p_value_denominator=exact_p_value.denominator,
        p_value=float(exact_p_value),
    )


def exact_two_sided_sign_test(pair_half_points: Iterable[int]) -> SignTestResult:
    """Test pair totals around two half-points, excluding exact pair ties."""

    return _sign_test_validated(_validate_pair_half_points(pair_half_points))


def wilson_interval(successes: int, trials: int) -> WilsonInterval:
    """Return the fixed-95% Wilson interval using deterministic Decimal arithmetic."""

    trials = _validate_bounded_positive_int(trials, "trials", _MAX_WILSON_TRIALS)
    if type(successes) is not int:
        raise TypeError("successes must be an integer and not bool")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be in [0, trials]")
    decimal_context = Context(
        prec=80,
        rounding=ROUND_HALF_EVEN,
        Emin=-999_999,
        Emax=999_999,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )
    with localcontext(decimal_context):
        decimal_successes = Decimal(successes)
        decimal_trials = Decimal(trials)
        one = Decimal(1)
        two = Decimal(2)
        four = Decimal(4)
        estimate = decimal_successes / decimal_trials
        z = Decimal(_WILSON_Z_TEXT)
        z_squared = z * z
        denominator = one + z_squared / decimal_trials
        center = (estimate + z_squared / (two * decimal_trials)) / denominator
        variance = estimate * (one - estimate) / decimal_trials
        variance += z_squared / (four * decimal_trials * decimal_trials)
        margin = z * variance.sqrt() / denominator
        lower = max(Decimal(0), center - margin)
        upper = min(one, center + margin)
    return WilsonInterval(
        successes=successes,
        trials=trials,
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
    )


def summarize_paired_game_points(points: Iterable[PairedGamePoints]) -> GameOutcomeSummary:
    """Retain seat-level game outcomes while deriving descriptive marginals."""

    if isinstance(points, (Set, Mapping)):
        raise TypeError("points must be ordered; sets and mappings are not supported")
    try:
        iterator = iter(points)
    except TypeError as exc:
        raise TypeError("points must be an iterable of PairedGamePoints") from exc
    values: list[PairedGamePoints] = []
    for index, value in enumerate(iterator):
        if index >= _MAX_PAIR_COUNT:
            raise ValueError(f"points must contain at most {_MAX_PAIR_COUNT} pairs")
        if type(value) is not PairedGamePoints:
            raise TypeError(f"points[{index}] must be a PairedGamePoints")
        values.append(value)
    if not values:
        raise ValueError("points must contain at least one pair")

    pair_count = len(values)
    all_points = [point for pair in values for point in (pair.candidate_as_p0, pair.candidate_as_p1)]
    candidate_wins = sum(point == 2 for point in all_points)
    draws = sum(point == 1 for point in all_points)
    baseline_wins = sum(point == 0 for point in all_points)
    candidate_as_p0_wins = sum(pair.candidate_as_p0 == 2 for pair in values)
    candidate_as_p1_wins = sum(pair.candidate_as_p1 == 2 for pair in values)
    game_count = 2 * pair_count
    return GameOutcomeSummary(
        pair_count=pair_count,
        game_count=game_count,
        candidate_wins=candidate_wins,
        draws=draws,
        baseline_wins=baseline_wins,
        candidate_as_p0_wins=candidate_as_p0_wins,
        candidate_as_p1_wins=candidate_as_p1_wins,
        candidate_win=wilson_interval(candidate_wins, game_count),
        draw=wilson_interval(draws, game_count),
        baseline_win=wilson_interval(baseline_wins, game_count),
        candidate_as_p0_win=wilson_interval(candidate_as_p0_wins, pair_count),
        candidate_as_p1_win=wilson_interval(candidate_as_p1_wins, pair_count),
    )


def score_pair_half_points(
    pair_half_points: Iterable[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> ScoreSummary:
    """Combine the primary paired score, paired bootstrap, and exact sign test.

    ``bootstrap_seed`` has the same raw unsigned-64 contract as
    :func:`bootstrap_pair_half_points`. The ordered input is materialized once
    before both component statistics are computed.
    """

    values = _validate_pair_half_points(pair_half_points)
    bootstrap = _bootstrap_validated(values, bootstrap_seed, bootstrap_replicates)
    sign_test = _sign_test_validated(values)
    return ScoreSummary(
        pair_count=len(values),
        total_half_points=sum(values),
        estimate=sum(values) / (4 * len(values)),
        bootstrap=bootstrap,
        sign_test=sign_test,
    )


__all__ = [
    "BootstrapSummary",
    "GameOutcomeSummary",
    "PairedGamePoints",
    "ScoreSummary",
    "SignTestResult",
    "WilsonInterval",
    "bootstrap_pair_half_points",
    "exact_two_sided_sign_test",
    "score_pair_half_points",
    "summarize_paired_game_points",
    "wilson_interval",
]
