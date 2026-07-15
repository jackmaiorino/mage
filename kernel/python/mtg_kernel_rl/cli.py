"""Command line interface for mtg-kernel-rl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rollout import POLICIES, run_episodes
from .trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtg-kernel-rl")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--env-bin", required=True, type=Path)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--episodes", required=True, type=int)
    run.add_argument("--base-seed", required=True, type=int)
    run.add_argument("--max-decisions", required=True, type=int)
    run.add_argument("--p0", required=True, choices=sorted(POLICIES))
    run.add_argument("--p1", required=True, choices=sorted(POLICIES))
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--env-bin", required=True, type=Path)
    train_parser.add_argument("--out-dir", required=True, type=Path)
    train_parser.add_argument("--resume", default=None, type=Path)
    train_parser.add_argument("--base-seed", default=None, type=int)
    train_parser.add_argument("--until-update", required=True, type=int)
    train_parser.add_argument("--batch-episodes", default=None, type=int)
    train_parser.add_argument("--learning-rate", default=None, type=float)
    train_parser.add_argument("--value-coef", default=None, type=float)
    train_parser.add_argument("--max-decisions", default=None, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        run_episodes(
            env_bin=args.env_bin,
            out_dir=args.out_dir,
            episodes=args.episodes,
            base_seed=args.base_seed,
            max_decisions=args.max_decisions,
            p0=args.p0,
            p1=args.p1,
        )
        return 0
    if args.command == "train":
        result = train(
            env_bin=args.env_bin,
            out_dir=args.out_dir,
            resume=args.resume,
            base_seed=args.base_seed,
            until_update=args.until_update,
            batch_episodes=args.batch_episodes,
            learning_rate=args.learning_rate,
            value_coef=args.value_coef,
            max_decisions=args.max_decisions,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    parser.error(f"unknown command {args.command}")
    return 2
