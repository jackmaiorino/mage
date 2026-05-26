#!/usr/bin/env python3
"""Run LiveCheckpointBranchMiner value-tree mode across local JVM shards."""

from __future__ import annotations

import argparse
import atexit
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

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_cp7_eval_sweep as sweep


MAIN_CLASS = "mage.player.ai.rl.LiveCheckpointBranchMiner"
MODULE = "Mage.Server.Plugins/Mage.Player.AIRL"

MERGED_FILES = [
    "selected_snapshots.csv",
    "counterfactual_value_tree.csv",
    "counterfactual_value_tree_summary.csv",
    "counterfactual_sequence_tree.csv",
    "counterfactual_sequence_tree_summary.csv",
    "terminal_line_search.csv",
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
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional eval manifest for MODEL_PROFILE/RL_ARTIFACTS_ROOT; auto-detected from <run>/live_checkpoints when omitted.",
    )
    parser.add_argument("--profile", default="", help="MODEL_PROFILE override when --manifest is supplied or auto-detected.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="value-tree", choices=["value-tree", "terminal-line"])
    parser.add_argument("--shards", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--max-concurrent-shards",
        type=int,
        default=0,
        help="Limit concurrent shard JVMs while preserving --shards selection partitions; 0 means all shards.",
    )
    parser.add_argument("--max-snapshots", type=int, default=100)
    parser.add_argument("--selection-mode", default="ranked")
    parser.add_argument("--ranked-max-per-game", type=int, default=10)
    parser.add_argument("--action-types", default="ACTIVATE_ABILITY_OR_SPELL")
    parser.add_argument("--tree-rollouts", type=int, default=4)
    parser.add_argument("--tree-max-actions", type=int, default=8)
    parser.add_argument("--tree-include-pass", default="true")
    parser.add_argument("--tree-continuation-policy", default="sample", choices=["stable", "sample", "explore"])
    parser.add_argument("--tree-timeout-sec", type=int, default=120)
    parser.add_argument("--tree-seed", type=int, default=2026052301)
    parser.add_argument("--line-attempts", type=int, default=64)
    parser.add_argument("--line-max-root-actions", type=int, default=0)
    parser.add_argument("--line-timeout-sec", type=int, default=120)
    parser.add_argument("--line-stop-on-win", default="true")
    parser.add_argument("--line-stop-on-win-all", default="false")
    parser.add_argument("--line-common-continuation-seeds", default="false")
    parser.add_argument("--sequence-tree", default="false")
    parser.add_argument("--tree-sequence-depth", type=int, default=2)
    parser.add_argument("--tree-sequence-beam", type=int, default=4)
    parser.add_argument("--tree-sequence-rollouts", type=int, default=1)
    parser.add_argument("--post-branch-autopilot", default="true")
    parser.add_argument(
        "--model-continuation-backend",
        default="single",
        choices=["single", "inherit"],
        help=(
            "When post-branch autopilot is false, default to a single local "
            "Python backend so tiny branch probes do not start learner+inference "
            "worker pools per JVM. Use inherit to keep the caller environment."
        ),
    )
    parser.add_argument(
        "--py4j-port-stride",
        type=int,
        default=32,
        help="Port stride applied per shard for true model-continuation runs.",
    )
    parser.add_argument("--maven", default="mvn")
    parser.add_argument("--online", action="store_true", help="Do not pass -o to Maven.")
    parser.add_argument("--gpu-port", type=int, default=26384)
    parser.add_argument("--gpu-metrics-port", type=int, default=27384)
    parser.add_argument(
        "--shared-gpu-service",
        default="auto",
        choices=["auto", "on", "off"],
        help="Start a shared GPU inference service for manifest-backed snapshots.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip the one-time AIRL compile before shard exec:java launches.",
    )
    parser.add_argument(
        "--skip-shard-compile",
        action="store_true",
        help="Do not include compile in each shard Maven compile exec:java invocation.",
    )
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
        "--tree-rollouts", str(args.tree_rollouts),
        "--tree-max-actions", str(args.tree_max_actions),
        "--tree-include-pass", bool_arg(args.tree_include_pass),
        "--tree-continuation-policy", args.tree_continuation_policy,
        "--tree-timeout-sec", str(args.tree_timeout_sec),
        "--tree-seed", str(args.tree_seed),
        "--sequence-tree", bool_arg(args.sequence_tree),
        "--tree-sequence-depth", str(args.tree_sequence_depth),
        "--tree-sequence-beam", str(args.tree_sequence_beam),
        "--tree-sequence-rollouts", str(args.tree_sequence_rollouts),
        "--post-branch-autopilot", bool_arg(args.post_branch_autopilot),
        "--selection-shards", str(args.shards),
        "--selection-shard-index", str(shard_index),
    ]
    if args.mode == "terminal-line":
        parts.extend(
            [
                "--terminal-line-search", "true",
                "--line-attempts", str(args.line_attempts),
                "--line-max-root-actions", str(args.line_max_root_actions),
                "--line-timeout-sec", str(args.line_timeout_sec),
                "--line-stop-on-win", bool_arg(args.line_stop_on_win),
                "--line-stop-on-win-all", bool_arg(args.line_stop_on_win_all),
                "--line-common-continuation-seeds", bool_arg(args.line_common_continuation_seeds),
            ]
        )
    else:
        parts.extend(["--value-tree", "true"])
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
        ]
    )
    if not args.skip_shard_compile:
        cmd.append("compile")
    cmd.extend(
        [
            "exec:java",
            f"-Dexec.mainClass={MAIN_CLASS}",
            f"-Dexec.args={exec_args}",
        ]
    )
    return cmd


def load_manifest(args: argparse.Namespace) -> Dict[str, object]:
    candidates: List[Path] = []
    if args.manifest:
        candidates.append(Path(args.manifest))
    if args.checkpoint_root:
        root = Path(args.checkpoint_root)
        if root.name == "live_checkpoints":
            candidates.append(root.parent / "manifest.json")
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def profile_from_manifest(manifest: Dict[str, object], requested: str) -> str:
    if requested.strip():
        return requested.strip()
    profiles = manifest.get("profiles") if manifest else None
    if isinstance(profiles, list) and profiles:
        first = profiles[0]
        if isinstance(first, dict) and first.get("profile"):
            return str(first["profile"])
    return ""


def manifest_env(manifest: Dict[str, object], profile: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not manifest:
        return env
    train_env = manifest.get("gpu_service_train_env")
    if isinstance(train_env, dict):
        for key, value in train_env.items():
            env[str(key)] = str(value)
    deterministic_env = manifest.get("deterministic_eval_env")
    if isinstance(deterministic_env, dict):
        for key, value in deterministic_env.items():
            env[str(key)] = str(value)
    snapshot_root = str(manifest.get("snapshot_root") or "").strip()
    if snapshot_root:
        env["RL_ARTIFACTS_ROOT"] = snapshot_root
    if profile:
        env["MODEL_PROFILE"] = profile
    return env


def manifest_train_env(manifest: Dict[str, object]) -> Dict[str, str]:
    train_env = manifest.get("gpu_service_train_env") if manifest else None
    if not isinstance(train_env, dict):
        return {}
    return {str(key): str(value) for key, value in train_env.items()}


def manifest_deterministic_env(manifest: Dict[str, object]) -> Dict[str, str]:
    deterministic_env = manifest.get("deterministic_eval_env") if manifest else None
    if not isinstance(deterministic_env, dict):
        return {}
    return {str(key): str(value) for key, value in deterministic_env.items()}


def compile_command(args: argparse.Namespace) -> List[str]:
    cmd = [resolve_maven(args.maven)]
    if not args.online:
        cmd.append("-o")
    cmd.extend(["-q", "-pl", MODULE, "-am", "-DskipTests", "compile"])
    return cmd


def stop_process(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def shard_env(args: argparse.Namespace, shard_index: int, base_overrides: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(base_overrides)
    if bool_arg(args.post_branch_autopilot) == "false":
        using_shared_gpu = env.get("PY_SERVICE_MODE", "").strip().lower() == "shared_gpu"
        if args.model_continuation_backend == "single" and not using_shared_gpu:
            env.setdefault("PY_BACKEND_MODE", "single")
            env.setdefault("INFER_WORKERS", "0")
            env.setdefault("MODEL_RELOAD_EVERY_MS", "0")
        if not using_shared_gpu:
            env.setdefault("PY_BRIDGE_CONNECT_RETRIES", "60")
            env.setdefault("PY_BRIDGE_CONNECT_RETRY_DELAY_MS", "1000")
            env.setdefault("PY_SCORE_TIMEOUT_MS", "120000")
            base_port = int(env.get("PY4J_BASE_PORT", env.get("PY4J_PORT", "25334")))
            shard_port = base_port + (max(1, args.py4j_port_stride) * shard_index)
            env["PY4J_BASE_PORT"] = str(shard_port)
            env["PY4J_PORT"] = str(shard_port)
    return env


def command_env_summary(env: Dict[str, str]) -> Dict[str, str]:
    keys = [
        "PY_SERVICE_MODE",
        "PY_BACKEND_MODE",
        "INFER_WORKERS",
        "MODEL_RELOAD_EVERY_MS",
        "PY4J_BASE_PORT",
        "PY4J_PORT",
        "PY_BRIDGE_CONNECT_RETRIES",
        "PY_BRIDGE_CONNECT_RETRY_DELAY_MS",
        "PY_SCORE_TIMEOUT_MS",
        "MODEL_PROFILE",
        "RL_ARTIFACTS_ROOT",
        "TORCH_DETERMINISTIC_EVAL",
        "PY_SERVICE_MODE",
        "GPU_SERVICE_ENDPOINT",
        "GPU_SERVICE_NUM_CHANNELS",
    ]
    return {key: env[key] for key in keys if key in env}


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


def sequence_classification_counts(summary_csv: Path) -> Dict[str, int]:
    return classification_counts(summary_csv)


def terminal_outcome_counts(search_csv: Path) -> Dict[str, int]:
    if not search_csv.is_file():
        return {}
    counts: Counter[str] = Counter()
    for row in read_csv(search_csv):
        counts[(row.get("outcome") or "").strip() or "<missing>"] += 1
    return dict(sorted(counts.items()))


def write_readme(output_dir: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Sharded Counterfactual Value Tree Miner",
        "",
        "Generated by `scripts/mtgrl/run_value_tree_shards.py`.",
        "",
        f"- mode: `{summary['mode']}`",
        f"- shards: `{summary['shards']}`",
        f"- max_concurrent_shards: `{summary['max_concurrent_shards']}`",
        f"- exit_codes: `{summary['exit_codes']}`",
        f"- selected_rows: `{summary['selected_snapshots.csv_rows']}`",
        f"- action_rows: `{summary['counterfactual_value_tree.csv_rows']}`",
        f"- summary_rows: `{summary['counterfactual_value_tree_summary.csv_rows']}`",
        f"- terminal_line_rows: `{summary['terminal_line_search.csv_rows']}`",
        f"- classification_counts: `{summary['classification_counts']}`",
        f"- sequence_classification_counts: `{summary['sequence_classification_counts']}`",
        f"- terminal_outcome_counts: `{summary['terminal_outcome_counts']}`",
        f"- sequence_tree: `{summary['sequence_tree']}`",
        f"- tree_sequence_depth: `{summary['tree_sequence_depth']}`",
        f"- tree_sequence_beam: `{summary['tree_sequence_beam']}`",
        f"- tree_sequence_rollouts: `{summary['tree_sequence_rollouts']}`",
        f"- line_attempts: `{summary['line_attempts']}`",
        f"- line_max_root_actions: `{summary['line_max_root_actions']}`",
        f"- line_timeout_sec: `{summary['line_timeout_sec']}`",
        f"- line_stop_on_win: `{summary['line_stop_on_win']}`",
        f"- line_stop_on_win_all: `{summary['line_stop_on_win_all']}`",
        f"- line_common_continuation_seeds: `{summary['line_common_continuation_seeds']}`",
        f"- post_branch_autopilot: `{summary['post_branch_autopilot']}`",
        f"- model_continuation_backend: `{summary['model_continuation_backend']}`",
        f"- py4j_port_stride: `{summary['py4j_port_stride']}`",
        "",
        "Each shard is an isolated JVM over a deterministic modulo partition of the same selected ranked checkpoint set.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if args.shards < 1:
        raise ValueError("--shards must be >= 1")
    if args.max_concurrent_shards < 0:
        raise ValueError("--max-concurrent-shards must be >= 0")
    max_concurrent_shards = args.max_concurrent_shards or args.shards
    max_concurrent_shards = max(1, min(args.shards, max_concurrent_shards))
    if not args.checkpoint_root and not args.snapshot_list:
        raise ValueError("--checkpoint-root or --snapshot-list is required")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args)
    profile = profile_from_manifest(manifest, args.profile)
    env_overrides = manifest_env(manifest, profile)

    if not args.skip_compile:
        cmd = compile_command(args)
        print("compiling AIRL module before shard launch", flush=True)
        subprocess.run(cmd, check=True)

    gpu_proc = None
    shared_gpu_requested = args.shared_gpu_service == "on" or (
        args.shared_gpu_service == "auto" and bool(manifest)
    )
    if shared_gpu_requested:
        service_overrides = manifest_deterministic_env(manifest)
        service_overrides["GPU_SERVICE_NUM_CHANNELS"] = str(max_concurrent_shards)
        print(
            f"starting shared GPU service on port {args.gpu_port} for shard JVMs",
            flush=True,
        )
        gpu_proc = sweep.start_gpu_service(
            args.gpu_port,
            args.gpu_metrics_port,
            output_dir / "gpu_service.log",
            manifest_train_env(manifest),
            service_overrides,
        )
        atexit.register(stop_process, gpu_proc)
        env_overrides.update(
            {
                "PY_SERVICE_MODE": "shared_gpu",
                "GPU_SERVICE_ENDPOINT": f"localhost:{args.gpu_port}",
                "GPU_SERVICE_NUM_GPUS": "1",
                "GPU_SERVICE_NUM_CHANNELS": str(max_concurrent_shards),
            }
        )

    shard_dirs = [output_dir / f"shard_{i:02d}_of_{args.shards:02d}" for i in range(args.shards)]
    processes = []
    start = time.time()
    active = []

    def launch_shard(shard_index: int, shard_dir: Path) -> Dict[str, object]:
        shard_dir.mkdir(parents=True, exist_ok=True)
        exec_args = miner_args(args, shard_index, shard_dir)
        cmd = maven_command(args, exec_args)
        env = shard_env(args, shard_index, env_overrides)
        stdout_path = shard_dir / "stdout.log"
        stderr_path = shard_dir / "stderr.log"
        (shard_dir / "command.json").write_text(
            json.dumps(
                {
                    "cmd": cmd,
                    "exec_args": exec_args,
                    "env": command_env_summary(env),
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        stdout = stdout_path.open("w", encoding="utf-8", newline="")
        stderr = stderr_path.open("w", encoding="utf-8", newline="")
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
        entry = {
            "index": shard_index,
            "dir": str(shard_dir),
            "pid": proc.pid,
            "process": proc,
            "stdout": stdout,
            "stderr": stderr,
            "cmd": cmd,
        }
        print(f"started shard {shard_index}/{args.shards} pid={proc.pid} dir={shard_dir}", flush=True)
        return entry

    next_shard = 0
    while next_shard < len(shard_dirs) or active:
        while next_shard < len(shard_dirs) and len(active) < max_concurrent_shards:
            entry = launch_shard(next_shard, shard_dirs[next_shard])
            processes.append(entry)
            active.append(entry)
            next_shard += 1
        running = [entry for entry in active if entry["process"].poll() is None]
        if running:
            print("running shards: " + ",".join(str(entry["index"]) for entry in running), flush=True)
            time.sleep(max(1.0, args.poll_sec))
        active = [entry for entry in active if entry["process"].poll() is None]

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
        "manifest": args.manifest,
        "manifest_auto_detected": bool(manifest) and not bool(args.manifest),
        "profile": profile,
        "output_dir": str(output_dir),
        "mode": args.mode,
        "shards": args.shards,
        "max_concurrent_shards": max_concurrent_shards,
        "max_snapshots": args.max_snapshots,
        "ranked_max_per_game": args.ranked_max_per_game,
        "tree_rollouts": args.tree_rollouts,
        "tree_max_actions": args.tree_max_actions,
        "tree_include_pass": bool_arg(args.tree_include_pass),
        "tree_continuation_policy": args.tree_continuation_policy,
        "tree_timeout_sec": args.tree_timeout_sec,
        "tree_seed": args.tree_seed,
        "line_attempts": args.line_attempts,
        "line_max_root_actions": args.line_max_root_actions,
        "line_timeout_sec": args.line_timeout_sec,
        "line_stop_on_win": bool_arg(args.line_stop_on_win),
        "line_stop_on_win_all": bool_arg(args.line_stop_on_win_all),
        "line_common_continuation_seeds": bool_arg(args.line_common_continuation_seeds),
        "sequence_tree": bool_arg(args.sequence_tree),
        "tree_sequence_depth": args.tree_sequence_depth,
        "tree_sequence_beam": args.tree_sequence_beam,
        "tree_sequence_rollouts": args.tree_sequence_rollouts,
        "post_branch_autopilot": bool_arg(args.post_branch_autopilot),
        "model_continuation_backend": args.model_continuation_backend,
        "py4j_port_stride": args.py4j_port_stride,
        "skip_compile": bool(args.skip_compile),
        "skip_shard_compile": bool(args.skip_shard_compile),
        "shared_gpu_service": args.shared_gpu_service,
        "shared_gpu_started": gpu_proc is not None,
        "gpu_port": args.gpu_port,
        "gpu_metrics_port": args.gpu_metrics_port,
        "exit_codes": exit_codes,
        "elapsed_sec": round(time.time() - start, 3),
        "classification_counts": classification_counts(output_dir / "counterfactual_value_tree_summary.csv"),
        "sequence_classification_counts": sequence_classification_counts(
            output_dir / "counterfactual_sequence_tree_summary.csv"
        ),
        "terminal_outcome_counts": terminal_outcome_counts(output_dir / "terminal_line_search.csv"),
    }
    summary.update(merged_counts)
    (output_dir / "shard_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if gpu_proc is not None:
        stop_process(gpu_proc)

    return 0 if all(code == 0 for code in exit_codes.values()) else 3


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
