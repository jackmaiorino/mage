#!/usr/bin/env python3
"""Run LiveCheckpointBranchMiner value-tree mode across local JVM shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


MAIN_CLASS = "mage.player.ai.rl.LiveCheckpointBranchMiner"
MODULE = "Mage.Server.Plugins/Mage.Player.AIRL"

MERGED_FILES = [
    "selected_snapshots.csv",
    "counterfactual_value_tree.csv",
    "counterfactual_value_tree_summary.csv",
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch multiple isolated Maven/JVM value-tree miners with deterministic "
            "selection sharding, then merge their CSV outputs."
        )
    )
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--snapshot-list", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shards", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--max-snapshots", type=int, default=100)
    parser.add_argument("--selection-mode", default="ranked")
    parser.add_argument("--ranked-max-per-game", type=int, default=10)
    parser.add_argument("--action-types", default="ACTIVATE_ABILITY_OR_SPELL")
    parser.add_argument("--tree-rollouts", type=int, default=4)
    parser.add_argument("--tree-max-actions", type=int, default=8)
    parser.add_argument("--tree-include-pass", default="true")
    parser.add_argument("--tree-continuation-policy", default="sample", choices=["stable", "sample"])
    parser.add_argument("--tree-timeout-sec", type=int, default=120)
    parser.add_argument("--tree-seed", type=int, default=2026052301)
    parser.add_argument("--maven", default="mvn")
    parser.add_argument("--online", action="store_true", help="Do not pass -o to Maven.")
    parser.add_argument("--poll-sec", type=float, default=10.0)
    parser.add_argument("--force", action="store_true", help="Allow an existing output directory.")
    return parser.parse_args(argv)


def bool_arg(value: str) -> str:
    return "true" if str(value).strip().lower() in {"1", "true", "yes", "y", "on"} else "false"


def miner_args(args: argparse.Namespace, shard_index: int, shard_dir: Path) -> str:
    parts = [
        "--out", str(shard_dir),
        "--selection-mode", args.selection_mode,
        "--ranked-max-per-game", str(args.ranked_max_per_game),
        "--max-snapshots", str(args.max_snapshots),
        "--action-types", args.action_types,
        "--value-tree", "true",
        "--tree-rollouts", str(args.tree_rollouts),
        "--tree-max-actions", str(args.tree_max_actions),
        "--tree-include-pass", bool_arg(args.tree_include_pass),
        "--tree-continuation-policy", args.tree_continuation_policy,
        "--tree-timeout-sec", str(args.tree_timeout_sec),
        "--tree-seed", str(args.tree_seed),
        "--selection-shards", str(args.shards),
        "--selection-shard-index", str(shard_index),
    ]
    if args.snapshot_list:
        parts[0:0] = ["--snapshot-list", args.snapshot_list]
    else:
        parts[0:0] = ["--checkpoint-root", args.checkpoint_root]
    return " ".join(parts)


def maven_command(args: argparse.Namespace, exec_args: str) -> List[str]:
    cmd = [resolve_maven(args.maven)]
    if not args.online:
        cmd.append("-o")
    cmd.extend(
        [
            "-q",
            "-pl",
            MODULE,
            "-am",
            "-DskipTests",
            f"-Dexec.mainClass={MAIN_CLASS}",
            f"-Dexec.args={exec_args}",
            "exec:java",
        ]
    )
    return cmd


def resolve_maven(value: str) -> str:
    raw = str(value or "").strip() or "mvn"
    resolved = shutil.which(raw)
    if resolved:
        return resolved
    for candidate in (
            r"C:\Program Files\apache-maven-3.9.8\bin\mvn.cmd",
            r"C:\Program Files\Apache\Maven\bin\mvn.cmd",
    ):
        if Path(candidate).is_file():
            return candidate
    return raw


def read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def merge_csv(output_dir: Path, filename: str, shard_dirs: Sequence[Path]) -> int:
    out_path = output_dir / filename
    wrote_header = False
    row_count = 0
    with out_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = None
        for shard_dir in shard_dirs:
            path = shard_dir / filename
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                if writer is None:
                    writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames or [], lineterminator="\n")
                if not wrote_header:
                    writer.writeheader()
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    return row_count


def classification_counts(summary_csv: Path) -> Dict[str, int]:
    if not summary_csv.is_file():
        return {}
    counts: Counter[str] = Counter()
    for row in read_csv(summary_csv):
        counts[(row.get("classification") or "").strip() or "<missing>"] += 1
    return dict(sorted(counts.items()))


def write_readme(output_dir: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Sharded Counterfactual Value Tree Miner",
        "",
        "Generated by `scripts/mtgrl/run_value_tree_shards.py`.",
        "",
        f"- shards: `{summary['shards']}`",
        f"- exit_codes: `{summary['exit_codes']}`",
        f"- selected_rows: `{summary['selected_snapshots.csv_rows']}`",
        f"- action_rows: `{summary['counterfactual_value_tree.csv_rows']}`",
        f"- summary_rows: `{summary['counterfactual_value_tree_summary.csv_rows']}`",
        f"- classification_counts: `{summary['classification_counts']}`",
        "",
        "Each shard is an isolated JVM over a deterministic modulo partition of the same selected ranked checkpoint set.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if args.shards < 1:
        raise ValueError("--shards must be >= 1")
    if not args.checkpoint_root and not args.snapshot_list:
        raise ValueError("--checkpoint-root or --snapshot-list is required")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = [output_dir / f"shard_{i:02d}_of_{args.shards:02d}" for i in range(args.shards)]
    processes = []
    start = time.time()
    for shard_index, shard_dir in enumerate(shard_dirs):
        shard_dir.mkdir(parents=True, exist_ok=True)
        exec_args = miner_args(args, shard_index, shard_dir)
        cmd = maven_command(args, exec_args)
        stdout_path = shard_dir / "stdout.log"
        stderr_path = shard_dir / "stderr.log"
        (shard_dir / "command.json").write_text(
            json.dumps({"cmd": cmd, "exec_args": exec_args}, indent=2) + "\n",
            encoding="utf-8",
        )
        stdout = stdout_path.open("w", encoding="utf-8", newline="")
        stderr = stderr_path.open("w", encoding="utf-8", newline="")
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        processes.append(
            {
                "index": shard_index,
                "dir": str(shard_dir),
                "pid": proc.pid,
                "process": proc,
                "stdout": stdout,
                "stderr": stderr,
                "cmd": cmd,
            }
        )
        print(f"started shard {shard_index}/{args.shards} pid={proc.pid} dir={shard_dir}", flush=True)

    while True:
        running = [entry for entry in processes if entry["process"].poll() is None]
        if not running:
            break
        print("running shards: " + ",".join(str(entry["index"]) for entry in running), flush=True)
        time.sleep(max(1.0, args.poll_sec))

    exit_codes = {}
    for entry in processes:
        proc = entry["process"]
        code = proc.poll()
        exit_codes[str(entry["index"])] = code
        entry["stdout"].close()
        entry["stderr"].close()

    merged_counts = {}
    for filename in MERGED_FILES:
        merged_counts[f"{filename}_rows"] = merge_csv(output_dir, filename, shard_dirs)

    summary = {
        "checkpoint_root": args.checkpoint_root,
        "snapshot_list": args.snapshot_list,
        "output_dir": str(output_dir),
        "shards": args.shards,
        "max_snapshots": args.max_snapshots,
        "ranked_max_per_game": args.ranked_max_per_game,
        "tree_rollouts": args.tree_rollouts,
        "tree_max_actions": args.tree_max_actions,
        "tree_include_pass": bool_arg(args.tree_include_pass),
        "tree_continuation_policy": args.tree_continuation_policy,
        "tree_timeout_sec": args.tree_timeout_sec,
        "tree_seed": args.tree_seed,
        "exit_codes": exit_codes,
        "elapsed_sec": round(time.time() - start, 3),
        "classification_counts": classification_counts(output_dir / "counterfactual_value_tree_summary.csv"),
    }
    summary.update(merged_counts)
    (output_dir / "shard_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0 if all(code == 0 for code in exit_codes.values()) else 3


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
