#!/usr/bin/env python3
"""Train candidate-Q from captured terminal-line branch-return records.

This is a post-mining driver for runs that already produced
terminal_line_training_data.ser files. It keeps the training recipe
terminal-return-only: score the current profile on the serialized records,
train with signed branch-return targets, then score the same records again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
DEFAULT_ROOT = REPO / "local-training" / "local_pbt" / "terminal_line_return_training"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ_terminal_line_returns")


def powershell_executable() -> str:
    return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell"


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (REPO / path)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: Sequence[str], log_path: Path, dry_run: bool) -> int:
    printable = " ".join(str(part) for part in cmd)
    print(printable, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + printable + "\n")
        log.flush()
        if dry_run:
            return 0
        proc = subprocess.run(
            list(cmd),
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        print(f"command failed rc={proc.returncode}; log={log_path}", flush=True)
    return proc.returncode


def score_command(args: argparse.Namespace, run_id: str) -> List[str]:
    cmd = [
        powershell_executable(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/run_action_counterfactual.ps1",
        "-Profile",
        args.profile,
        "-RunId",
        run_id,
        "-ScoreTrainingDataPath",
        str(args.training_data_root),
        "-ScoreMaxExamples",
        str(args.score_max_examples),
        "-Scenarios",
        "1",
        "-BatchSize",
        str(args.batch_size),
        "-ServiceMode",
        args.service_mode,
        "-ModelDModel",
        str(args.model_d_model),
        "-ModelNumLayers",
        str(args.model_num_layers),
        "-ActionTypes",
        args.action_types,
        "-BranchReturnTargets",
    ]
    if args.skip_compile:
        cmd.append("-SkipCompile")
    return cmd


def train_command(args: argparse.Namespace, run_id: str) -> List[str]:
    cmd = [
        powershell_executable(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/run_action_counterfactual.ps1",
        "-Profile",
        args.profile,
        "-RunId",
        run_id,
        "-ImportTrainingDataPath",
        str(args.training_data_root),
        "-Scenarios",
        "1",
        "-BatchSize",
        str(args.batch_size),
        "-ServiceMode",
        args.service_mode,
        "-ModelDModel",
        str(args.model_d_model),
        "-ModelNumLayers",
        str(args.model_num_layers),
        "-TrainEpochs",
        str(args.train_epochs),
        "-CandidatePermutations",
        "1",
        "-MaxTrainExamples",
        str(args.max_train_examples),
        "-CandidateQLossCoef",
        str(args.candidate_q_loss_coef),
        "-CandidateQHuberBeta",
        str(args.candidate_q_huber_beta),
        "-PolicyLossCoef",
        "0.0",
        "-ValueLossCoef",
        "0.0",
        "-ActionTypes",
        args.action_types,
        "-PostTrainWaitMs",
        str(args.post_train_wait_ms),
        "-CandidateQOnly",
        "-BranchReturnTargets",
        "-BranchReturnBalance",
        "-BranchReturnMaxNegativesPerPositive",
        str(args.branch_return_max_negatives_per_positive),
        "-SkipOptimizerStateLoad",
    ]
    if args.skip_compile:
        cmd.append("-SkipCompile")
    return cmd


def summarize_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/mtgrl/summarize_terminal_line_search.py",
        str(args.training_data_root),
        "--prefer-merged",
    ]
    if args.summary_json:
        cmd.extend(["--out-json", str(args.summary_json)])
    if args.summary_md:
        cmd.extend(["--out-md", str(args.summary_md)])
    return cmd


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and train from terminal-line branch-return TrainingData."
    )
    parser.add_argument("training_data_root", help="Directory containing terminal_line_training_data.ser files.")
    parser.add_argument("--profile", required=True, help="Profile to train in-place.")
    parser.add_argument("--run-id", default=timestamp_id())
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--service-mode", choices=("local", "shared_gpu"), default="local")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--score-max-examples", type=int, default=1024)
    parser.add_argument("--model-d-model", type=int, default=128)
    parser.add_argument("--model-num-layers", type=int, default=2)
    parser.add_argument("--candidate-q-loss-coef", type=float, default=1.0)
    parser.add_argument("--candidate-q-huber-beta", type=float, default=0.25)
    parser.add_argument("--branch-return-max-negatives-per-positive", type=int, default=1)
    parser.add_argument("--post-train-wait-ms", type=int, default=3600000)
    parser.add_argument(
        "--action-types",
        default="ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,DECLARE_ATTACKS,DECLARE_BLOCKS,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
    )
    parser.add_argument("--skip-pre-score", action="store_true")
    parser.add_argument("--skip-post-score", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.training_data_root = resolve_path(args.training_data_root).resolve()
    args.output_root = resolve_path(args.output_root).resolve()
    args.run_dir = args.output_root / args.run_id
    args.summary_json = args.run_dir / "terminal_line_summary.json"
    args.summary_md = args.run_dir / "terminal_line_summary.md"
    if args.max_train_examples < 0:
        parser.error("--max-train-examples must be >= 0")
    if args.score_max_examples <= 0:
        parser.error("--score-max-examples must be > 0")
    if args.branch_return_max_negatives_per_positive < 0:
        parser.error("--branch-return-max-negatives-per-positive must be >= 0")
    return args


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if not args.training_data_root.exists():
        raise FileNotFoundError(args.training_data_root)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    commands: Dict[str, List[str]] = {
        "summary": summarize_command(args),
        "train": train_command(args, args.run_id + "_train"),
    }
    if not args.skip_pre_score:
        commands["pre_score"] = score_command(args, args.run_id + "_pre_score")
    if not args.skip_post_score:
        commands["post_score"] = score_command(args, args.run_id + "_post_score")

    manifest = {
        "created_at": utc_now(),
        "training_data_root": str(args.training_data_root),
        "profile": args.profile,
        "run_id": args.run_id,
        "run_dir": str(args.run_dir),
        "branch_return_max_negatives_per_positive": args.branch_return_max_negatives_per_positive,
        "commands": commands,
        "dry_run": args.dry_run,
    }
    write_json(args.run_dir / "driver_manifest.json", manifest)

    ordered = ["summary", "pre_score", "train", "post_score"]
    for name in ordered:
        cmd = commands.get(name)
        if not cmd:
            continue
        rc = run_command(cmd, args.run_dir / f"{name}.log", args.dry_run)
        if rc != 0:
            return rc
    print(f"terminal-line return training driver output: {args.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
