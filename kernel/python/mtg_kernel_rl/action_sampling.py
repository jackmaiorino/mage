"""Versioned, process-global-RNG-free categorical action sampling."""

from __future__ import annotations

from decimal import Context, Decimal, DivisionByZero, InvalidOperation, Overflow, ROUND_HALF_EVEN, localcontext
from typing import Any

import torch


FIXED_CATEGORICAL_SAMPLER_VERSION = "decimal-softmax-hamilton-splitmix64-v1"
SPLITMIX64_VERSION = "splitmix64-v1"
DECIMAL_DELTA_PRECISION_DIGITS = 256
DECIMAL_EXP_PRECISION_DIGITS = 80
DECIMAL_EXP_CUTOFF = "-128"
DECIMAL_CONTEXT_EMIN = -999_999
DECIMAL_CONTEXT_EMAX = 999_999
CATEGORICAL_MASS_BITS = 64
CATEGORICAL_MASS_TOTAL = 1 << CATEGORICAL_MASS_BITS

_MASK64 = CATEGORICAL_MASS_TOTAL - 1
_SPLITMIX64_GAMMA = 0x9E37_79B9_7F4A_7C15
def _decimal_context(precision: int) -> Context:
    # Every field that Context otherwise inherits from mutable DefaultContext is
    # explicit. Import order and process-global Decimal configuration therefore
    # cannot alter the frozen sampler.
    return Context(
        prec=precision,
        rounding=ROUND_HALF_EVEN,
        Emin=DECIMAL_CONTEXT_EMIN,
        Emax=DECIMAL_CONTEXT_EMAX,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


_EXACT_DELTA_CONTEXT = _decimal_context(DECIMAL_DELTA_PRECISION_DIGITS)
_EXP_CONTEXT = _decimal_context(DECIMAL_EXP_PRECISION_DIGITS)
_EXP_CUTOFF = Decimal(DECIMAL_EXP_CUTOFF)


def fixed_categorical_sampler_contract() -> dict[str, Any]:
    """Return one fresh JSON-ready copy of the frozen selector contract."""

    return {
        "action_rng": f"one {SPLITMIX64_VERSION} uint64 output per decision, initialized directly from the action seed",
        "algorithm": "inverse CDF over Hamilton-apportioned 2**64-unit mass in legal-action order",
        "decimal_softmax": {
            "context": {
                "capitals": 1,
                "clamp": 0,
                "emax": DECIMAL_CONTEXT_EMAX,
                "emin": DECIMAL_CONTEXT_EMIN,
                "flags_initially_set": [],
                "traps": ["InvalidOperation", "DivisionByZero", "Overflow"],
            },
            "delta_precision_digits": DECIMAL_DELTA_PRECISION_DIGITS,
            "exp_cutoff": f"strictly below {DECIMAL_EXP_CUTOFF} receives zero mass",
            "exp_precision_digits": DECIMAL_EXP_PRECISION_DIGITS,
            "input": "exact IEEE-754 binary32 logits converted to Decimal",
            "rounding": "ROUND_HALF_EVEN",
        },
        "probability_mass": {
            "apportionment": (
                "floor exact normalized Decimal-exp shares, then residual units by descending exact remainder "
                "and ascending legal-action index"
            ),
            "total": f"2**{CATEGORICAL_MASS_BITS}",
        },
        "sampler_version": FIXED_CATEGORICAL_SAMPLER_VERSION,
    }


def splitmix64_u64(seed: int) -> int:
    """Return the first SplitMix64-v1 output for one exact uint64 seed."""

    if type(seed) is not int:
        raise TypeError("fixed categorical seed must be an integer and not bool")
    if seed < 0 or seed > _MASK64:
        raise ValueError("fixed categorical seed must be a uint64")
    state = (seed + _SPLITMIX64_GAMMA) & _MASK64
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58_476D_1CE4_E5B9) & _MASK64
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D0_49BB_1331_11EB) & _MASK64
    return (mixed ^ (mixed >> 31)) & _MASK64


def _decimal_coefficient(value: Decimal) -> tuple[int, int]:
    sign, digits, exponent = value.as_tuple()
    if sign:
        raise ValueError("fixed categorical Decimal weight must be non-negative")
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    return coefficient, int(exponent)


def fixed_softmax_mass(logits: torch.Tensor) -> tuple[int, ...]:
    """Map finite CPU binary32 logits to exactly ``2**64`` mass units."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError("fixed categorical logits must be a tensor")
    if logits.device.type != "cpu" or logits.dtype != torch.float32 or logits.ndim != 1:
        raise ValueError("fixed categorical logits must be a CPU float32 vector")
    if logits.numel() == 0:
        raise ValueError("fixed categorical sampling requires at least one action")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("fixed categorical logits must be finite")

    exact_logits = tuple(Decimal.from_float(value) for value in logits.detach().tolist())
    maximum = max(exact_logits)
    with localcontext(_EXACT_DELTA_CONTEXT):
        deltas = tuple(value - maximum for value in exact_logits)
    with localcontext(_EXP_CONTEXT):
        decimal_weights = tuple(Decimal(0) if delta < _EXP_CUTOFF else delta.exp() for delta in deltas)

    coefficients_and_exponents = tuple(_decimal_coefficient(weight) for weight in decimal_weights)
    positive_exponents = tuple(exponent for coefficient, exponent in coefficients_and_exponents if coefficient)
    if not positive_exponents:
        raise ValueError("fixed categorical softmax produced no positive mass")
    minimum_exponent = min(positive_exponents)
    exact_weights = tuple(
        coefficient * (10 ** (exponent - minimum_exponent)) if coefficient else 0
        for coefficient, exponent in coefficients_and_exponents
    )
    total = sum(exact_weights)
    if total <= 0:
        raise ValueError("fixed categorical softmax produced invalid total mass")

    apportioned: list[int] = []
    remainders: list[int] = []
    for weight in exact_weights:
        quotient, remainder = divmod(weight * CATEGORICAL_MASS_TOTAL, total)
        apportioned.append(quotient)
        remainders.append(remainder)
    residual = CATEGORICAL_MASS_TOTAL - sum(apportioned)
    if residual < 0 or residual >= len(apportioned):
        raise ValueError("fixed categorical softmax apportionment was invalid")
    for index in sorted(range(len(apportioned)), key=lambda position: (-remainders[position], position))[:residual]:
        apportioned[index] += 1
    if sum(apportioned) != CATEGORICAL_MASS_TOTAL or any(weight < 0 for weight in apportioned):
        raise ValueError("fixed categorical softmax did not preserve 2**64 mass")
    return tuple(apportioned)


def select_categorical_u64(weights: tuple[int, ...], draw: int) -> int:
    """Select by inverse CDF from a uint64 draw and exact ``2**64`` mass."""

    if type(draw) is not int or draw < 0 or draw > _MASK64:
        raise ValueError("fixed categorical draw must be a uint64")
    if not weights or any(type(weight) is not int or weight < 0 for weight in weights):
        raise ValueError("fixed categorical weights must be non-negative integers")
    if sum(weights) != CATEGORICAL_MASS_TOTAL:
        raise ValueError("fixed categorical weights must sum to 2**64")
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    raise ValueError("fixed categorical selection was out of range")


def sample_fixed_categorical(logits: torch.Tensor, seed: int) -> int:
    """Sample one index under the frozen sampler contract."""

    return select_categorical_u64(fixed_softmax_mass(logits), splitmix64_u64(seed))


__all__ = [
    "CATEGORICAL_MASS_BITS",
    "CATEGORICAL_MASS_TOTAL",
    "DECIMAL_DELTA_PRECISION_DIGITS",
    "DECIMAL_CONTEXT_EMAX",
    "DECIMAL_CONTEXT_EMIN",
    "DECIMAL_EXP_CUTOFF",
    "DECIMAL_EXP_PRECISION_DIGITS",
    "FIXED_CATEGORICAL_SAMPLER_VERSION",
    "SPLITMIX64_VERSION",
    "fixed_categorical_sampler_contract",
    "fixed_softmax_mass",
    "sample_fixed_categorical",
    "select_categorical_u64",
    "splitmix64_u64",
]
