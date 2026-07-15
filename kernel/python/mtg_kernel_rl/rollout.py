"""Deterministic Burn-mirror runner over the strict kernel RL client."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .client import Decision, KernelRlClient, Terminal
from .determinism import SeedDerivation, configure_torch_determinism, derive_env_seed, derive_sample_seed, derive_uniform_index
from .features import EncodedDecision, encode_decision
from .model import KernelPolicyValueNet, greedy_action

POLICIES = {"uniform", "greedy", "sampled"}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _policy_for_seat(seat: str, p0: str, p1: str) -> str:
    return p0 if seat == "p0" else p1


class InMemoryModelPolicy:
    def __init__(self) -> None:
        self.model: KernelPolicyValueNet | None = None

    def logits_value(self, encoded: EncodedDecision) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            self.model = KernelPolicyValueNet.from_encoded(encoded)
        self.model.eval()
        with torch.no_grad():
            return self.model(encoded)


def select_action(
    decision: Decision,
    *,
    policy: str,
    base_seed: int,
    episode: int,
    model_policy: InMemoryModelPolicy,
) -> int:
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy {policy}")
    if policy == "uniform":
        return int(derive_uniform_index(base_seed, episode, decision.step, decision.acting_player, len(decision.legal_actions)))
    encoded = encode_decision(decision.observation, decision.legal_actions)
    logits, _value = model_policy.logits_value(encoded)
    if policy == "greedy":
        return greedy_action(logits)
    seed = derive_sample_seed(base_seed, episode, decision.step, decision.acting_player)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    probabilities = torch.softmax(logits, dim=0)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _episode_record(episode: int, env_seed: int, terminal: Terminal, p0_policy: str, p1_policy: str) -> dict[str, Any]:
    return {
        "episode": episode,
        "env_seed": env_seed,
        "terminal_outcome": terminal.terminal_outcome,
        "terminal_classification": terminal.terminal_classification,
        "terminal_code": terminal.terminal_code,
        "winner": terminal.winner,
        "terminal_reward": terminal.terminal_reward,
        "decision_count": terminal.decision_count,
        "p0_policy": p0_policy,
        "p1_policy": p1_policy,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, int]:
    out = {"episodes": len(records), "p0_wins": 0, "p1_wins": 0, "draws": 0, "halted": 0, "truncated": 0}
    for record in records:
        if record["terminal_outcome"] == "p0_win":
            out["p0_wins"] += 1
        elif record["terminal_outcome"] == "p1_win":
            out["p1_wins"] += 1
        elif record["terminal_outcome"] == "draw":
            out["draws"] += 1
        elif record["terminal_outcome"] == "halted":
            out["halted"] += 1
        elif record["terminal_outcome"] == "truncated":
            out["truncated"] += 1
    return out


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_artifacts(out_dir: Path, manifest: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "run.json"
    episodes_path = out_dir / "episodes.jsonl"
    if run_path.exists() or episodes_path.exists():
        raise FileExistsError("run.json or episodes.jsonl already exists")
    run_text = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=False) + "\n"
    episodes_text = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=False) + "\n" for record in episodes)
    _write_text_atomic(run_path, run_text)
    _write_text_atomic(episodes_path, episodes_text)


def run_episodes(
    *,
    env_bin: str | Path,
    out_dir: str | Path,
    episodes: int,
    base_seed: int,
    max_decisions: int,
    p0: str,
    p1: str,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    configure_torch_determinism()
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if p0 not in POLICIES or p1 not in POLICIES:
        raise ValueError("unsupported policy")
    env_path = Path(env_bin)
    env_sha = sha256_file(env_path)
    terminal_records: list[dict[str, Any]] = []
    provenance: dict[str, Any] | None = None
    model_policy = InMemoryModelPolicy()
    with KernelRlClient(env_path, timeout_s=timeout_s) as client:
        for episode in range(episodes):
            env_seed = derive_env_seed(base_seed, episode)
            current: Decision | Terminal = client.reset(episode_id=episode, env_seed=env_seed, max_decisions=max_decisions)
            while isinstance(current, Decision):
                provenance = current.provenance
                policy = _policy_for_seat(current.acting_player, p0, p1)
                index = select_action(current, policy=policy, base_seed=base_seed, episode=episode, model_policy=model_policy)
                action = current.legal_actions[index]
                current = client.step(action["selected_index"], action["stable_id"])
            provenance = current.provenance
            terminal_records.append(_episode_record(episode, env_seed, current, p0, p1))
    aggregate = _aggregate(terminal_records)
    if aggregate["halted"] != 0 or aggregate["truncated"] != 0:
        raise RuntimeError("halted/truncated episodes are not admissible")
    manifest = {
        "artifact_schema_version": 1,
        "package": {"name": "mtg-kernel-rl", "version": __version__},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "environment": {"binary_sha256": env_sha},
        "protocol_provenance": provenance,
        "seed_derivation": SeedDerivation().__dict__,
        "config": {
            "episodes": episodes,
            "base_seed": base_seed,
            "max_decisions": max_decisions,
            "p0_policy": p0,
            "p1_policy": p1,
        },
        "aggregate": aggregate,
        "episodes": terminal_records,
    }
    write_artifacts(Path(out_dir), manifest, terminal_records)
    return manifest
