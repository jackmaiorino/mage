#!/usr/bin/env python3
"""Run LiveCheckpointBranchMiner with the shared GPU eval environment.

This is the checkpoint-mining counterpart to run_cp7_eval_sweep.py. It reuses an
eval manifest so branch continuations can use the same profile snapshot and
shared GPU service as the game that produced the checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import run_cp7_eval_sweep as sweep


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def profile_from_manifest(manifest: dict, requested: str) -> str:
    if requested.strip():
        return requested.strip()
    profiles = manifest.get("profiles") or []
    if not profiles:
        raise RuntimeError("Manifest has no profiles")
    return str(profiles[0]["profile"])


def base_env(manifest: dict, profile: str, gpu_port: int) -> Dict[str, str]:
    env = os.environ.copy()
    train_env = manifest.get("gpu_service_train_env") or {}
    for key, value in train_env.items():
        env[str(key)] = str(value)
    snapshot_root = str(manifest.get("snapshot_root") or "").strip()
    if not snapshot_root:
        raise RuntimeError("Manifest is missing snapshot_root")
    env.update(
        {
            "MODEL_PROFILE": profile,
            "RL_ARTIFACTS_ROOT": snapshot_root,
            "PY_SERVICE_MODE": "shared_gpu",
            "GPU_SERVICE_ENDPOINT": f"localhost:{gpu_port}",
            "GPU_SERVICE_NUM_GPUS": "1",
            "MAGE_DB_AUTO_SERVER": "false",
            "RL_HEURISTIC_STEP_REWARDS": "0",
            "MTG_AI_LOG_LEVEL": "WARN",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def quote_exec_args(args: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(args)
    return " ".join(subprocess.list2cmdline([arg]) for arg in args)


def has_branch_arg(args: List[str], name: str) -> bool:
    prefix = f"--{name}="
    return f"--{name}" in args or any(arg.startswith(prefix) for arg in args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Eval manifest that owns the checkpoint/model snapshot.")
    parser.add_argument("--profile", default="", help="MODEL_PROFILE override; defaults to manifest profile[0].")
    parser.add_argument("--run-id", required=True, help="Runner artifact directory name under local-training/local_pbt/live_checkpoint_branch_miner.")
    parser.add_argument("--gpu-port", type=int, default=26384)
    parser.add_argument("--gpu-metrics-port", type=int, default=27384)
    parser.add_argument("--timeout-sec", type=int, default=600, help="Maven process timeout.")
    parser.add_argument("--maven-offline", action="store_true")
    parser.add_argument("--compile-exec", action="store_true")
    parser.add_argument(
        "branch_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to LiveCheckpointBranchMiner. Prefix with --.",
    )
    args = parser.parse_args()

    branch_args = list(args.branch_args)
    if branch_args and branch_args[0] == "--":
        branch_args = branch_args[1:]
    if not branch_args:
        raise RuntimeError("No LiveCheckpointBranchMiner arguments supplied")

    manifest_path = sweep.resolve_repo_path(args.manifest)
    manifest = load_manifest(manifest_path)
    profile = profile_from_manifest(manifest, args.profile)
    deterministic_env = {
        str(key): str(value)
        for key, value in (manifest.get("deterministic_eval_env") or {}).items()
    }
    run_dir = sweep.REPO / "local-training" / "local_pbt" / "live_checkpoint_branch_miner" / args.run_id
    if not has_branch_arg(branch_args, "out"):
        branch_args = ["--out", str(run_dir)] + branch_args
    logs_dir = run_dir / "runner_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    service_env = dict(manifest.get("gpu_service_train_env") or {})
    run_manifest = {
        "run_id": args.run_id,
        "started_utc": sweep.utc_now(),
        "source_manifest": str(manifest_path),
        "profile": profile,
        "gpu_port": args.gpu_port,
        "gpu_metrics_port": args.gpu_metrics_port,
        "deterministic_env": deterministic_env,
        "branch_args": branch_args,
    }
    (run_dir / "runner_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    gpu_proc = None
    try:
        gpu_proc = sweep.start_gpu_service(
            args.gpu_port,
            args.gpu_metrics_port,
            logs_dir / "gpu_service.log",
            service_env,
            deterministic_env,
        )
        env = base_env(manifest, profile, args.gpu_port)
        env.update(deterministic_env)
        goals = ["exec:java"]
        if args.compile_exec:
            goals.insert(0, "compile")
        cmd = sweep.maven_command(args.maven_offline) + [
            "-q",
            "-pl",
            sweep.MODULE,
            "-am",
            "-DskipTests",
        ] + goals + [
            "-Dexec.mainClass=mage.player.ai.rl.LiveCheckpointBranchMiner",
            "-Dexec.args=" + quote_exec_args(branch_args),
        ]
        log_path = logs_dir / "branch_miner.log"
        start = time.time()
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"# started_utc={sweep.utc_now()}\n")
            log.write(f"# cmd={' '.join(cmd)}\n\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(sweep.REPO),
                env=env,
                text=True,
                capture_output=True,
                timeout=args.timeout_sec,
            )
            log.write(proc.stdout or "")
            if proc.stderr:
                log.write("\n--- STDERR ---\n")
                log.write(proc.stderr)
        run_manifest["ended_utc"] = sweep.utc_now()
        run_manifest["duration_sec"] = round(time.time() - start, 3)
        run_manifest["returncode"] = proc.returncode
        (run_dir / "runner_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            print(f"branch miner failed returncode={proc.returncode}; log={log_path}", file=sys.stderr)
        else:
            print(f"branch miner complete; log={log_path}")
        return proc.returncode
    finally:
        if gpu_proc is not None:
            gpu_proc.terminate()
            try:
                gpu_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gpu_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
