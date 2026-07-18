"""Command line interface for mtg-kernel-rl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import evaluate
from .rollout import POLICIES, run_episodes
from .sampled_evaluator import evaluate_sampled
from .trainer import train


def _add_deck_ids_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--deck-ids",
        nargs=2,
        default=("Burn", "Burn"),
        metavar=("P0_DECK", "P1_DECK"),
        help="ordered physical p0/p1 deck IDs (default: Burn Burn)",
    )


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
    _add_deck_ids_argument(run)
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
    _add_deck_ids_argument(train_parser)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--training-store", required=True, type=Path)
    evaluate_parser.add_argument("--expected-candidate-head", required=True)
    evaluate_parser.add_argument("--env-bin", required=True, type=Path)
    evaluate_parser.add_argument("--out-dir", required=True, type=Path)
    evaluate_parser.add_argument("--pairs", required=True, type=int)
    evaluate_parser.add_argument("--base-seed", required=True, type=int)
    evaluate_parser.add_argument("--bootstrap-replicates", required=True, type=int)
    evaluate_parser.add_argument("--max-decisions", required=True, type=int)
    evaluate_parser.add_argument("--timeout-ms", required=True, type=int)
    _add_deck_ids_argument(evaluate_parser)
    sampled_parser = sub.add_parser("evaluate-sampled")
    sampled_parser.add_argument("--training-store", required=True, type=Path)
    sampled_parser.add_argument("--expected-candidate-head", required=True)
    sampled_parser.add_argument("--env-bin", required=True, type=Path)
    sampled_parser.add_argument("--out-dir", required=True, type=Path)
    sampled_parser.add_argument("--pairs", required=True, type=int)
    sampled_parser.add_argument("--base-seed", required=True, type=int)
    sampled_parser.add_argument("--bootstrap-replicates", required=True, type=int)
    sampled_parser.add_argument("--max-decisions", required=True, type=int)
    sampled_parser.add_argument("--timeout-ms", required=True, type=int)
    _add_deck_ids_argument(sampled_parser)
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
            deck_ids=tuple(args.deck_ids),
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
            deck_ids=tuple(args.deck_ids),
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "evaluate":
        result = evaluate(
            training_store=args.training_store,
            expected_candidate_head=args.expected_candidate_head,
            env_bin=args.env_bin,
            out_dir=args.out_dir,
            pairs=args.pairs,
            base_seed=args.base_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            max_decisions=args.max_decisions,
            timeout_ms=args.timeout_ms,
            deck_ids=tuple(args.deck_ids),
        )
        summary = {
            "baseline_head": result.baseline_head,
            "candidate_head": result.candidate_head,
            "estimate_hex": result.estimate.hex(),
            "game_count": result.game_count,
            "pair_count": result.pair_count,
            "run_sha256": result.run_sha256,
            "total_half_points": result.total_half_points,
        }
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "evaluate-sampled":
        result = evaluate_sampled(
            training_store=args.training_store,
            expected_candidate_head=args.expected_candidate_head,
            env_bin=args.env_bin,
            out_dir=args.out_dir,
            pairs=args.pairs,
            base_seed=args.base_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            max_decisions=args.max_decisions,
            timeout_ms=args.timeout_ms,
            deck_ids=tuple(args.deck_ids),
        )
        summary = {
            "baseline_head": result.baseline_head,
            "candidate_head": result.candidate_head,
            "estimate_hex": result.estimate.hex(),
            "game_count": result.game_count,
            "pair_count": result.pair_count,
            "run_sha256": result.run_sha256,
            "total_half_points": result.total_half_points,
        }
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 0
    parser.error(f"unknown command {args.command}")
    return 2
