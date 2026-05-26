#!/usr/bin/env python3
"""Run online checkpoint capture followed by terminal-line mining.

This is an expert-iteration style harness: let the current model play real
online games, capture branchable checkpoints from those reached states, then
mine terminal win/loss values from those checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
DEFAULT_ROOT = REPO / "local-training" / "local_pbt" / "online_terminal_mining"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_v%H%M%S_online_terminal_mining")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def run_command(cmd: Sequence[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in cmd)
    print(printable, flush=True)
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + printable + "\n")
        log.flush()
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


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = sum(1 for _ in handle)
    return max(0, lines - 1)


def build_eval_command(args: argparse.Namespace, cycle: int, eval_run_id: str, eval_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/run_cp7_eval_sweep.py",
        "--registry",
        args.registry,
        "--run-id",
        eval_run_id,
        "--output-root",
        str(eval_root),
        "--profiles",
        args.profiles,
        "--opponents",
        args.opponents,
        "--games-per-matchup",
        str(args.games_per_matchup),
        "--games-per-job",
        str(args.games_per_job),
        "--skill",
        str(args.skill),
        "--parallel",
        str(args.eval_parallel),
        "--serial-warmup-jobs",
        str(args.serial_warmup_jobs),
        "--ai-threads",
        str(args.ai_threads),
        "--gpu-port",
        str(args.eval_gpu_port + cycle * args.port_stride),
        "--gpu-metrics-port",
        str(args.eval_gpu_metrics_port + cycle * args.port_stride),
        "--timeout-sec",
        str(args.eval_timeout_sec),
        "--live-checkpoints",
        "--live-checkpoint-max-per-game",
        str(args.live_checkpoint_max_per_game),
        "--live-checkpoint-action-types",
        args.live_checkpoint_action_types,
        "--replay-seed-base",
        str(args.replay_seed_base + cycle * 1000),
        "--seed-key-mode",
        args.seed_key_mode,
        "--deterministic-eval",
    ]
    if args.allow_deterministic_parallel:
        cmd.append("--allow-deterministic-parallel")
    if args.chunk_indices:
        cmd.extend(["--chunk-indices", args.chunk_indices])
    if args.eval_game_logging:
        cmd.append("--eval-game-logging")
        cmd.extend(["--game-log-format", args.game_log_format])
    if args.compile_exec:
        cmd.append("--compile-exec")
    if not args.online_maven:
        cmd.append("--maven-offline")
    if args.skip_eval_compile:
        cmd.append("--skip-compile")
    return cmd


def build_mine_command(args: argparse.Namespace, cycle: int, checkpoint_root: Path, mine_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/mtgrl/run_value_tree_shards.py",
        "--checkpoint-root",
        str(checkpoint_root),
        "--output-dir",
        str(mine_dir),
        "--mode",
        "terminal-line",
        "--shards",
        str(args.mine_shards),
        "--max-snapshots",
        str(args.max_snapshots),
        "--selection-mode",
        args.selection_mode,
        "--ranked-max-per-game",
        str(args.ranked_max_per_game),
        "--action-types",
        args.action_types,
        "--line-attempts",
        str(args.line_attempts),
        "--line-max-root-actions",
        str(args.line_max_root_actions),
        "--line-timeout-sec",
        str(args.line_timeout_sec),
        "--tree-timeout-sec",
        str(args.tree_timeout_sec),
        "--tree-seed",
        str(args.tree_seed + cycle * 1000),
        "--tree-continuation-policy",
        args.tree_continuation_policy,
        "--tree-include-pass",
        "true" if args.tree_include_pass else "false",
        "--line-stop-on-win",
        "false" if args.no_stop_on_win else "true",
        "--line-stop-on-win-all",
        "false",
        "--line-common-continuation-seeds",
        "true",
        "--post-branch-autopilot",
        "false",
        "--model-continuation-backend",
        args.model_continuation_backend,
        "--gpu-port",
        str(args.mine_gpu_port + cycle * args.port_stride),
        "--gpu-metrics-port",
        str(args.mine_gpu_metrics_port + cycle * args.port_stride),
        "--poll-sec",
        str(args.poll_sec),
    ]
    if args.online_maven:
        cmd.append("--online")
    if args.skip_mine_compile:
        cmd.append("--skip-compile")
    if args.skip_shard_compile:
        cmd.append("--skip-shard-compile")
    if args.force:
        cmd.append("--force")
    return cmd


def build_export_command(
        args: argparse.Namespace,
        mine_dir: Path,
        output_dir: Path,
        include_pass: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/mtgrl/export_terminal_line_value_targets.py",
        str(mine_dir),
        "--prefer-merged",
        "--output-dir",
        str(output_dir),
        "--require-common-samples",
        "--write-readme",
        "--min-value-delta",
        str(args.min_value_delta),
        "--min-common-samples",
        str(args.min_common_samples),
        "--min-attempts-per-action",
        str(args.min_attempts_per_action),
        "--max-positive-actions",
        str(args.max_positive_actions),
        "--max-positive-fraction",
        str(args.max_positive_fraction),
        "--positive-value-threshold",
        str(args.positive_value_threshold),
        "--min-source-regret",
        str(args.min_source_regret),
        "--min-best-value",
        str(args.min_best_value),
        "--max-group-win-rate",
        str(args.max_group_win_rate),
        "--min-best-over-group-edge",
        str(args.min_best_over_group_edge),
        "--max-targets-per-game",
        str(args.max_targets_per_game),
        "--min-ordinal-gap-per-game",
        str(args.min_ordinal_gap_per_game),
    ]
    if include_pass:
        cmd.append("--include-suspect-pass-best")
    return cmd


def write_readme(run_dir: Path, manifest: Dict[str, object]) -> None:
    lines = [
        "# Online Terminal Mining Loop",
        "",
        "This run captures checkpoints from real online games, then mines terminal win/loss values from those reached states.",
        "",
        "- The teacher signal is terminal outcome only.",
        "- No combo milestone, card-name reward, or combo-ready label is used.",
        "- This harness does not yet override in-game choices; it generates online-distribution training targets.",
        "",
        f"- run_id: `{manifest.get('run_id', '')}`",
        f"- started_utc: `{manifest.get('started_utc', '')}`",
        f"- cycles: `{manifest.get('cycles', 0)}`",
        "",
        "## Cycle Summary",
        "",
        "| Cycle | Eval | Mine | Terminal Rows | Wins | Clean Targets | Pass-Including Targets |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cycle in manifest.get("cycle_summaries", []):
        if not isinstance(cycle, dict):
            continue
        mine = cycle.get("mine_summary", {}) if isinstance(cycle.get("mine_summary"), dict) else {}
        clean = cycle.get("clean_target_summary", {}) if isinstance(cycle.get("clean_target_summary"), dict) else {}
        include = cycle.get("include_pass_target_summary", {}) if isinstance(cycle.get("include_pass_target_summary"), dict) else {}
        lines.append(
            "| {cycle} | `{eval_run_id}` | `{mine_run_id}` | {rows} | {wins} | {clean_targets} | {pass_targets} |".format(
                cycle=cycle.get("cycle", ""),
                eval_run_id=cycle.get("eval_run_id", ""),
                mine_run_id=cycle.get("mine_run_id", ""),
                rows=mine.get("terminal_line_search.csv_rows", cycle.get("terminal_rows", 0)),
                wins=(mine.get("terminal_outcome_counts", {}) or {}).get("terminal_win", ""),
                clean_targets=clean.get("admitted_examples", ""),
                pass_targets=include.get("admitted_examples", ""),
            )
        )
    lines.append("")
    run_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=timestamp_id())
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--registry",
        default="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json",
    )
    parser.add_argument("--profiles", default="Pauper-Spy-Combo-Value")
    parser.add_argument("--opponents", default="grixis")
    parser.add_argument("--games-per-matchup", type=int, default=1)
    parser.add_argument("--games-per-job", type=int, default=1)
    parser.add_argument("--chunk-indices", default="")
    parser.add_argument("--skill", type=int, default=7)
    parser.add_argument("--eval-parallel", type=int, default=1)
    parser.add_argument(
        "--allow-deterministic-parallel",
        action="store_true",
        help=(
            "Pass through to run_cp7_eval_sweep so deterministic online capture can "
            "honor --eval-parallel instead of forcing serial execution."
        ),
    )
    parser.add_argument("--serial-warmup-jobs", type=int, default=1)
    parser.add_argument("--ai-threads", type=int, default=1)
    parser.add_argument("--eval-timeout-sec", type=int, default=1800)
    parser.add_argument("--eval-gpu-port", type=int, default=26100)
    parser.add_argument("--eval-gpu-metrics-port", type=int, default=27100)
    parser.add_argument("--mine-gpu-port", type=int, default=26384)
    parser.add_argument("--mine-gpu-metrics-port", type=int, default=27384)
    parser.add_argument("--port-stride", type=int, default=100)
    parser.add_argument("--live-checkpoint-max-per-game", type=int, default=96)
    parser.add_argument(
        "--live-checkpoint-action-types",
        default="ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,DECLARE_ATTACKS,DECLARE_BLOCKS,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
    )
    parser.add_argument("--eval-game-logging", action="store_true")
    parser.add_argument("--game-log-format", choices=("compact", "full", "both"), default="compact")
    parser.add_argument("--replay-seed-base", type=int, default=6161)
    parser.add_argument("--seed-key-mode", choices=("profile", "matchup"), default="matchup")
    parser.add_argument("--compile-exec", dest="compile_exec", action="store_true", default=True)
    parser.add_argument("--no-compile-exec", dest="compile_exec", action="store_false")
    parser.add_argument("--skip-eval-compile", action="store_true")
    parser.add_argument("--skip-mine-compile", action="store_true")
    parser.add_argument("--skip-shard-compile", action="store_true")
    parser.add_argument(
        "--allow-stale-miner-classpath",
        action="store_true",
        help=(
            "Allow mining to skip both the one-time compile and shard compile. "
            "This is unsafe for fresh online snapshots unless target/classes is "
            "known to match the eval JVM that serialized them."
        ),
    )
    parser.add_argument("--online-maven", action="store_true")
    parser.add_argument("--mine-shards", type=int, default=4)
    parser.add_argument("--max-snapshots", type=int, default=64)
    parser.add_argument("--selection-mode", default="ranked")
    parser.add_argument("--ranked-max-per-game", type=int, default=10)
    parser.add_argument("--action-types", default="ACTIVATE_ABILITY_OR_SPELL")
    parser.add_argument("--line-attempts", type=int, default=32)
    parser.add_argument("--line-max-root-actions", type=int, default=16)
    parser.add_argument("--line-timeout-sec", type=int, default=120)
    parser.add_argument("--tree-timeout-sec", type=int, default=120)
    parser.add_argument("--tree-seed", type=int, default=2026052506)
    parser.add_argument("--tree-continuation-policy", choices=("stable", "sample", "explore"), default="explore")
    parser.add_argument("--tree-include-pass", dest="tree_include_pass", action="store_true", default=True)
    parser.add_argument("--no-tree-include-pass", dest="tree_include_pass", action="store_false")
    parser.add_argument("--no-stop-on-win", dest="no_stop_on_win", action="store_true", default=True)
    parser.add_argument("--stop-on-win", dest="no_stop_on_win", action="store_false")
    parser.add_argument("--model-continuation-backend", choices=("single", "inherit"), default="single")
    parser.add_argument("--min-value-delta", type=float, default=0.1)
    parser.add_argument("--min-common-samples", type=int, default=2)
    parser.add_argument("--min-attempts-per-action", type=int, default=2)
    parser.add_argument("--max-positive-actions", type=int, default=0)
    parser.add_argument("--max-positive-fraction", type=float, default=1.0)
    parser.add_argument("--positive-value-threshold", type=float, default=0.0)
    parser.add_argument("--min-source-regret", type=float, default=0.0)
    parser.add_argument("--min-best-value", type=float, default=0.0)
    parser.add_argument("--max-group-win-rate", type=float, default=1.0)
    parser.add_argument("--min-best-over-group-edge", type=float, default=0.0)
    parser.add_argument("--max-targets-per-game", type=int, default=0)
    parser.add_argument("--min-ordinal-gap-per-game", type=int, default=0)
    parser.add_argument("--poll-sec", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.skip_mine_compile and args.skip_shard_compile and not args.allow_stale_miner_classpath:
        print(
            "refusing unsafe classpath mix: online eval snapshots may be serialized by a freshly "
            "compiled JVM, but mining would skip both the one-time compile and shard compile. "
            "Drop --skip-mine-compile or --skip-shard-compile, or pass "
            "--allow-stale-miner-classpath only when target/classes is known to match.",
            file=sys.stderr,
        )
        return 2
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO / output_root
    run_dir = output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "run_id": args.run_id,
        "started_utc": utc_now(),
        "cycles": args.cycles,
        "profiles": args.profiles,
        "opponents": args.opponents,
        "output_root": rel(output_root),
        "cycle_summaries": [],
    }
    write_json(run_dir / "manifest.json", manifest)

    for cycle in range(max(1, args.cycles)):
        cycle_id = f"cycle{cycle + 1:02d}"
        eval_run_id = f"{args.run_id}_{cycle_id}_eval"
        mine_run_id = f"{args.run_id}_{cycle_id}_mine"
        eval_root = run_dir / "eval_sweeps"
        eval_dir = eval_root / eval_run_id
        checkpoint_root = eval_dir / "live_checkpoints"
        mine_dir = run_dir / "mining" / mine_run_id
        clean_targets_dir = run_dir / "value_targets" / f"{mine_run_id}_softpass"
        include_pass_targets_dir = run_dir / "value_targets" / f"{mine_run_id}_include_pass"

        cycle_summary: Dict[str, object] = {
            "cycle": cycle + 1,
            "eval_run_id": eval_run_id,
            "mine_run_id": mine_run_id,
            "started_utc": utc_now(),
            "eval_dir": rel(eval_dir),
            "checkpoint_root": rel(checkpoint_root),
            "mine_dir": rel(mine_dir),
            "clean_targets_dir": rel(clean_targets_dir),
            "include_pass_targets_dir": rel(include_pass_targets_dir),
        }

        eval_rc = run_command(
            build_eval_command(args, cycle, eval_run_id, eval_root),
            run_dir / "logs" / f"{cycle_id}_eval.log",
            args.dry_run,
        )
        cycle_summary["eval_returncode"] = eval_rc
        cycle_summary["eval_manifest"] = read_json(eval_dir / "manifest.json")
        if eval_rc != 0:
            cycle_summary["ended_utc"] = utc_now()
            manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
            write_json(run_dir / "manifest.json", manifest)
            write_readme(run_dir, manifest)
            return eval_rc

        mine_rc = run_command(
            build_mine_command(args, cycle, checkpoint_root, mine_dir),
            run_dir / "logs" / f"{cycle_id}_mine.log",
            args.dry_run,
        )
        cycle_summary["mine_returncode"] = mine_rc
        cycle_summary["mine_summary"] = read_json(mine_dir / "shard_summary.json")
        cycle_summary["terminal_rows"] = count_rows(mine_dir / "terminal_line_search.csv")
        if mine_rc != 0:
            cycle_summary["ended_utc"] = utc_now()
            manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
            write_json(run_dir / "manifest.json", manifest)
            write_readme(run_dir, manifest)
            return mine_rc

        summarize_rc = run_command(
            [
                sys.executable,
                "scripts/mtgrl/summarize_terminal_line_search.py",
                str(mine_dir),
                "--prefer-merged",
            ],
            run_dir / "logs" / f"{cycle_id}_summarize.log",
            args.dry_run,
        )
        cycle_summary["summarize_returncode"] = summarize_rc

        clean_rc = run_command(
            build_export_command(args, mine_dir, clean_targets_dir, include_pass=False),
            run_dir / "logs" / f"{cycle_id}_export_softpass.log",
            args.dry_run,
        )
        include_pass_rc = run_command(
            build_export_command(args, mine_dir, include_pass_targets_dir, include_pass=True),
            run_dir / "logs" / f"{cycle_id}_export_include_pass.log",
            args.dry_run,
        )
        cycle_summary["clean_export_returncode"] = clean_rc
        cycle_summary["include_pass_export_returncode"] = include_pass_rc
        cycle_summary["clean_target_summary"] = read_json(clean_targets_dir / "terminal_line_value_target_summary.json")
        cycle_summary["include_pass_target_summary"] = read_json(
            include_pass_targets_dir / "terminal_line_value_target_summary.json"
        )
        cycle_summary["ended_utc"] = utc_now()
        manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
        write_json(run_dir / "manifest.json", manifest)
        write_readme(run_dir, manifest)
        if clean_rc != 0:
            return clean_rc
        if include_pass_rc != 0:
            return include_pass_rc

    manifest["ended_utc"] = utc_now()
    write_json(run_dir / "manifest.json", manifest)
    write_readme(run_dir, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
