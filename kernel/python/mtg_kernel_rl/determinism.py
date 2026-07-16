"""Deterministic seed derivation and Torch runtime setup."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
GOLDEN_RATIO_64 = 0x9E37_79B9_7F4A_7C15
UINT63_MAX = (1 << 63) - 1

SEED_DERIVATION_VERSION = "kernel-python-rl-seed-v1"
ENV_SEED_DOMAIN = 0x4556_5F52_4C5F_7631
UNIFORM_POLICY_DOMAIN = 0x5059_5F55_4E49_7631
SAMPLED_POLICY_DOMAIN = 0x5059_5F53414D_7631
TRAINER_SEED_DERIVATION_VERSION = "kernel-python-rl-trainer-sha256-v1"
EVALUATOR_SEED_DERIVATION_VERSION = "kernel-python-rl-evaluator-sha256-v1"

_TORCH_CONFIGURED = False


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + GOLDEN_RATIO_64) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & MASK64
        return (z ^ (z >> 31)) & MASK64


def _derive(base_seed: int, episode: int, step: int, seat: str, domain: int) -> int:
    seat_tag = 0x5030 if seat == "p0" else 0x5031
    mixed = (
        int(base_seed)
        ^ domain
        ^ ((int(episode) * GOLDEN_RATIO_64) & MASK64)
        ^ ((int(step) * 0xD1B5_4A32_D192_ED03) & MASK64)
        ^ seat_tag
    ) & MASK64
    return SplitMix64(mixed).next()


def derive_env_seed(base_seed: int, episode: int) -> int:
    """Derive the Rust reset seed in a domain separate from policy sampling."""

    return _derive(base_seed, episode, 0, "p0", ENV_SEED_DOMAIN)


def derive_uniform_index(base_seed: int, episode: int, step: int, seat: str, legal_count: int) -> int:
    if legal_count <= 0:
        raise ValueError("legal_count must be positive")
    return _derive(base_seed, episode, step, seat, UNIFORM_POLICY_DOMAIN) % legal_count


def derive_sample_seed(base_seed: int, episode: int, step: int, seat: str) -> int:
    return _derive(base_seed, episode, step, seat, SAMPLED_POLICY_DOMAIN) & 0x7FFF_FFFF


def configure_torch_determinism() -> None:
    global _TORCH_CONFIGURED
    import torch

    if not _TORCH_CONFIGURED:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            if torch.get_num_interop_threads() != 1:
                raise RuntimeError("failed to configure Torch inter-op thread count") from exc
        torch.use_deterministic_algorithms(True)
        _TORCH_CONFIGURED = True
    if torch.get_num_threads() != 1:
        raise RuntimeError("Torch intra-op thread count is not deterministic")
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch inter-op thread count is not deterministic")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Torch deterministic algorithms are not enabled")
    if torch.get_default_dtype() is not torch.float32:
        raise RuntimeError("Torch default dtype is not float32")


def validate_uint63(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer and not bool")
    if value < 0 or value > UINT63_MAX:
        raise ValueError(f"{name} must be in [0, 2**63 - 1]")
    return value


def validate_positive_int(value: Any, name: str, *, maximum: int = UINT63_MAX) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer and not bool")
    if value <= 0 or value > maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


def _atom(hasher: Any, tag: str, payload: bytes) -> None:
    tag_bytes = tag.encode("utf-8")
    hasher.update(len(tag_bytes).to_bytes(4, "big"))
    hasher.update(tag_bytes)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _trainer_seed(namespace: str, fields: list[tuple[str, int | str]]) -> int:
    if type(namespace) is not str or not namespace:
        raise TypeError("namespace must be a nonempty string")
    hasher = hashlib.sha256()
    _atom(hasher, "version", TRAINER_SEED_DERIVATION_VERSION.encode("utf-8"))
    _atom(hasher, "namespace", namespace.encode("utf-8"))
    for name, value in fields:
        if type(name) is not str or not name:
            raise TypeError("field names must be nonempty strings")
        _atom(hasher, "field-name", name.encode("utf-8"))
        if type(value) is int:
            if value < 0 or value > UINT63_MAX:
                raise ValueError(f"{name} out of trainer seed integer domain")
            _atom(hasher, "u63", value.to_bytes(8, "big"))
        elif type(value) is str:
            _atom(hasher, "str", value.encode("utf-8"))
        else:
            raise TypeError(f"{name} has unsupported seed field type {type(value).__name__}")
    return int.from_bytes(hasher.digest()[:8], "big") & UINT63_MAX


def _evaluator_seed(namespace: str, fields: list[tuple[str, int]]) -> int:
    if type(namespace) is not str or not namespace:
        raise TypeError("namespace must be a nonempty string")
    hasher = hashlib.sha256()
    _atom(hasher, "version", EVALUATOR_SEED_DERIVATION_VERSION.encode("utf-8"))
    _atom(hasher, "namespace", namespace.encode("utf-8"))
    for name, value in fields:
        if type(name) is not str or not name:
            raise TypeError("field names must be nonempty strings")
        if type(value) is not int or value < 0 or value > UINT63_MAX:
            raise ValueError(f"{name} out of evaluator seed integer domain")
        _atom(hasher, "field-name", name.encode("utf-8"))
        _atom(hasher, "u63", value.to_bytes(8, "big"))
    return int.from_bytes(hasher.digest()[:8], "big") & UINT63_MAX


def derive_model_init_seed(base_seed: int) -> int:
    return _trainer_seed("model-init", [("base_seed", validate_uint63(base_seed, "base_seed"))])


def derive_evaluation_bootstrap_seed(base_seed: int) -> int:
    return _evaluator_seed(
        "evaluation-bootstrap",
        [("base_seed", validate_uint63(base_seed, "base_seed"))],
    )


def derive_evaluation_env_seed(base_seed: int, pair_index: int) -> int:
    return _evaluator_seed(
        "evaluation-env",
        [
            ("base_seed", validate_uint63(base_seed, "base_seed")),
            ("pair_index", validate_uint63(pair_index, "pair_index")),
        ],
    )


def derive_train_env_seed(base_seed: int, pair_index: int) -> int:
    return _trainer_seed(
        "train-env",
        [
            ("base_seed", validate_uint63(base_seed, "base_seed")),
            ("pair_index", validate_uint63(pair_index, "pair_index")),
        ],
    )


def derive_train_learner_action_seed(base_seed: int, episode_index: int, learner_decision_index: int) -> int:
    return _trainer_seed(
        "train-learner-action",
        [
            ("base_seed", validate_uint63(base_seed, "base_seed")),
            ("episode_index", validate_uint63(episode_index, "episode_index")),
            ("learner_decision_index", validate_uint63(learner_decision_index, "learner_decision_index")),
        ],
    )


def derive_train_opponent_action_seed(base_seed: int, episode_index: int, opponent_decision_index: int) -> int:
    return _trainer_seed(
        "train-opponent-action",
        [
            ("base_seed", validate_uint63(base_seed, "base_seed")),
            ("episode_index", validate_uint63(episode_index, "episode_index")),
            ("opponent_decision_index", validate_uint63(opponent_decision_index, "opponent_decision_index")),
        ],
    )


def deterministic_index_from_seed(seed: int, legal_count: int) -> int:
    validate_uint63(seed, "seed")
    if type(legal_count) is not int or legal_count <= 0:
        raise ValueError("legal_count must be a positive integer")
    return seed % legal_count


@dataclass(frozen=True)
class SeedDerivation:
    version: str = SEED_DERIVATION_VERSION
    env_domain: str = f"0x{ENV_SEED_DOMAIN:016x}"
    uniform_policy_domain: str = f"0x{UNIFORM_POLICY_DOMAIN:016x}"
    sampled_policy_domain: str = f"0x{SAMPLED_POLICY_DOMAIN:016x}"
    algorithm: str = "splitmix64(base_seed, episode, step, actor_seat, domain)"


@dataclass(frozen=True)
class TrainerSeedDerivation:
    version: str = TRAINER_SEED_DERIVATION_VERSION
    algorithm: str = "sha256(type-tagged big-endian length-prefixed fields)[:8] & 0x7fff_ffff_ffff_ffff"
    namespaces: tuple[str, ...] = (
        "model-init/base_seed",
        "train-env/base_seed/pair_index",
        "train-learner-action/base_seed/episode_index/learner_decision_index",
        "train-opponent-action/base_seed/episode_index/opponent_decision_index",
    )


@dataclass(frozen=True)
class EvaluatorSeedDerivation:
    version: str = EVALUATOR_SEED_DERIVATION_VERSION
    algorithm: str = "sha256(type-tagged big-endian length-prefixed fields)[:8] & 0x7fff_ffff_ffff_ffff"
    namespaces: tuple[str, ...] = (
        "evaluation-bootstrap/base_seed",
        "evaluation-env/base_seed/pair_index",
    )
