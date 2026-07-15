"""Command line interface for mtg-kernel-rl."""

from __future__ import annotations

import argparse
from pathlib import Path

from .rollout import POLICIES, run_episodes


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
    parser.error(f"unknown command {args.command}")
    return 2
