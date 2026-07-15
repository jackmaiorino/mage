"""Deterministic seed derivation and Torch runtime setup."""

from __future__ import annotations

from dataclasses import dataclass

MASK64 = 0xFFFF_FFFF_FFFF_FFFF
GOLDEN_RATIO_64 = 0x9E37_79B9_7F4A_7C15

SEED_DERIVATION_VERSION = "kernel-python-rl-seed-v1"
ENV_SEED_DOMAIN = 0x4556_5F52_4C5F_7631
UNIFORM_POLICY_DOMAIN = 0x5059_5F55_4E49_7631
SAMPLED_POLICY_DOMAIN = 0x5059_5F53414D_7631


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
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class SeedDerivation:
    version: str = SEED_DERIVATION_VERSION
    env_domain: str = f"0x{ENV_SEED_DOMAIN:016x}"
    uniform_policy_domain: str = f"0x{UNIFORM_POLICY_DOMAIN:016x}"
    sampled_policy_domain: str = f"0x{SAMPLED_POLICY_DOMAIN:016x}"
    algorithm: str = "splitmix64(base_seed, episode, step, actor_seat, domain)"
