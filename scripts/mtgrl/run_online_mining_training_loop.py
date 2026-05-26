#!/usr/bin/env python3
"""Run the online terminal-mining -> value-target -> Q-train -> eval loop.

This driver keeps the thesis signal terminal-only:

1. Let the current profile play real CP7 eval games and capture live checkpoints.
2. Mine terminal win/loss continuations from those reached checkpoints.
3. Export admitted terminal-line value targets into AIRL TrainingData.
4. Clone the current profile and train only the candidate-Q head from signed
   branch-return targets.
5. Evaluate the trained candidate, then use it as the next cycle's player.

The loop is fail-closed: if a cycle does not produce enough admitted
terminal-derived examples, it records the blocker and stops before training.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
DEFAULT_ROOT = REPO / "local-training" / "local_pbt" / "online_mining_training_loop"
MODULE = "Mage.Server.Plugins/Mage.Player.AIRL"
DEFAULT_REGISTRY = (
    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/"
    "pauper_spy_pbt_registry.json"
)
PROFILES_ROOT = (
    REPO
    / "Mage.Server.Plugins"
    / "Mage.Player.AIRL"
    / "src"
    / "mage"
    / "player"
    / "ai"
    / "rl"
    / "profiles"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ_online_mining_training")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def safe_name(value: str, limit: int = 96) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return (cleaned or "run")[:limit]


def resolve_repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def python_executable() -> str:
    return sys.executable


def powershell_executable() -> str:
    return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell"


def maven_executable() -> str:
    return shutil.which("mvn") or shutil.which("mvn.cmd") or "mvn"


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: Sequence[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in cmd)
    print(printable, flush=True)
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


def read_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_exported_records(summary_csv: Path) -> int:
    count = 0
    for row in read_csv_rows(summary_csv):
        if (
            (row.get("status") or "").strip() == "exported"
            and (row.get("captured") or "").strip().lower() == "true"
            and (row.get("reentry_matched") or "").strip().lower() == "true"
        ):
            count += 1
    return count


def count_target_rows(target_csv: Path) -> int:
    if not target_csv.is_file():
        return 0
    with target_csv.open("r", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def quote_exec_arg(value: str) -> str:
    if not value:
        return value
    if re.search(r"\s", value):
        return "'" + value.replace("'", "'\\''") + "'"
    return value


def load_registry(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Registry is not a JSON list: {path}")
    return [dict(entry) for entry in data if isinstance(entry, dict)]


def find_registry_entry(registry: List[Dict[str, object]], profile: str) -> Dict[str, object]:
    for entry in registry:
        if str(entry.get("profile", "")) == profile:
            return dict(entry)
    raise KeyError(f"Profile not found in registry: {profile}")


def profile_entry(
        base_entry: Dict[str, object],
        profile: str,
        q_blend: float,
        q_heads: str,
        q_min_margin: float,
        q_min_top_q: Optional[float],
) -> Dict[str, object]:
    entry = dict(base_entry)
    train_env = dict(entry.get("train_env") or {})
    entry["profile"] = profile
    entry["active"] = True
    entry["train_enabled"] = False
    entry["notes"] = "Generated online terminal-mining loop eval entry."
    train_env["CANDIDATE_Q_BLEND"] = str(q_blend)
    if q_heads.strip():
        train_env["CANDIDATE_Q_BLEND_HEADS"] = q_heads.strip()
    if q_min_margin > 0.0:
        train_env["CANDIDATE_Q_BLEND_MIN_MARGIN"] = str(q_min_margin)
    else:
        train_env.pop("CANDIDATE_Q_BLEND_MIN_MARGIN", None)
    if q_min_top_q is not None:
        train_env["CANDIDATE_Q_BLEND_MIN_TOP_Q"] = str(q_min_top_q)
    else:
        train_env.pop("CANDIDATE_Q_BLEND_MIN_TOP_Q", None)
    entry["train_env"] = train_env
    return entry


def write_registry(path: Path, entries: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def clone_profile_models(source_profile: str, target_profile: str, force: bool, dry_run: bool) -> Dict[str, object]:
    source_models = PROFILES_ROOT / source_profile / "models"
    target_root = PROFILES_ROOT / target_profile
    target_models = target_root / "models"
    if source_profile == target_profile:
        raise ValueError("Refusing to clone a profile onto itself")
    if not source_models.is_dir():
        raise FileNotFoundError(f"Missing source profile models: {source_models}")
    if target_root.exists():
        if not force:
            raise FileExistsError(
                f"Target profile already exists: {target_root}; use --force-profile-clone to replace generated candidates"
            )
        resolved = target_root.resolve()
        if resolved.parent != PROFILES_ROOT.resolve():
            raise ValueError(f"Refusing to remove unexpected target path: {resolved}")
        if not dry_run:
            shutil.rmtree(target_root)
    copied: List[str] = []
    if not dry_run:
        target_models.mkdir(parents=True, exist_ok=True)
        (target_root / "logs").mkdir(parents=True, exist_ok=True)
    for item in source_models.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in (".pt", ".txt", ".json"):
            continue
        copied.append(item.name)
        if not dry_run:
            shutil.copy2(item, target_models / item.name)
    latest = source_models / "model_latest.pt"
    if "model_latest.pt" not in copied and latest.is_file() and not dry_run:
        shutil.copy2(latest, target_models / "model_latest.pt")
        copied.append("model_latest.pt")
    if not dry_run and not (target_models / "model.pt").is_file() and (target_models / "model_latest.pt").is_file():
        shutil.copy2(target_models / "model_latest.pt", target_models / "model.pt")
        copied.append("model.pt")
    return {
        "source_profile": source_profile,
        "target_profile": target_profile,
        "source_models": rel(source_models),
        "target_models": rel(target_models),
        "copied_files": sorted(set(copied)),
    }


def build_online_mining_command(
        args: argparse.Namespace,
        cycle: int,
        online_run_id: str,
        online_root: Path,
        registry: Path,
        profile: str,
) -> List[str]:
    cmd = [
        python_executable(),
        "scripts/mtgrl/run_online_terminal_mining_loop.py",
        "--run-id",
        online_run_id,
        "--output-root",
        str(online_root),
        "--cycles",
        "1",
        "--registry",
        str(registry),
        "--profiles",
        profile,
        "--opponents",
        args.opponents,
        "--games-per-matchup",
        str(args.games_per_matchup),
        "--games-per-job",
        str(args.games_per_job),
        "--skill",
        str(args.skill),
        "--eval-parallel",
        str(args.eval_parallel),
        "--serial-warmup-jobs",
        str(args.serial_warmup_jobs),
        "--ai-threads",
        str(args.ai_threads),
        "--eval-timeout-sec",
        str(args.eval_timeout_sec),
        "--eval-gpu-port",
        str(args.eval_gpu_port + cycle * args.port_stride),
        "--eval-gpu-metrics-port",
        str(args.eval_gpu_metrics_port + cycle * args.port_stride),
        "--mine-gpu-port",
        str(args.mine_gpu_port + cycle * args.port_stride),
        "--mine-gpu-metrics-port",
        str(args.mine_gpu_metrics_port + cycle * args.port_stride),
        "--live-checkpoint-max-per-game",
        str(args.live_checkpoint_max_per_game),
        "--live-checkpoint-action-types",
        args.live_checkpoint_action_types,
        "--replay-seed-base",
        str(args.replay_seed_base + cycle * 1000),
        "--seed-key-mode",
        args.seed_key_mode,
        "--mine-shards",
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
        "--model-continuation-backend",
        args.model_continuation_backend,
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
        "--poll-sec",
        str(args.poll_sec),
    ]
    if args.max_concurrent_mine_shards > 0:
        cmd.extend(["--max-concurrent-mine-shards", str(args.max_concurrent_mine_shards)])
    if args.allow_deterministic_parallel:
        cmd.append("--allow-deterministic-parallel")
    if args.chunk_indices:
        cmd.extend(["--chunk-indices", args.chunk_indices])
    if args.eval_game_logging:
        cmd.append("--eval-game-logging")
        cmd.extend(["--game-log-format", args.game_log_format])
    if args.no_compile_exec:
        cmd.append("--no-compile-exec")
    if args.skip_eval_compile:
        cmd.append("--skip-eval-compile")
    if args.skip_mine_compile:
        cmd.append("--skip-mine-compile")
    if args.skip_shard_compile:
        cmd.append("--skip-shard-compile")
    if args.allow_stale_miner_classpath:
        cmd.append("--allow-stale-miner-classpath")
    if args.online_maven:
        cmd.append("--online-maven")
    if not args.tree_include_pass:
        cmd.append("--no-tree-include-pass")
    if not args.no_stop_on_win:
        cmd.append("--stop-on-win")
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def build_training_data_export_command(
        args: argparse.Namespace,
        target_csv: Path,
        out_ser: Path,
        summary_csv: Path,
) -> List[str]:
    goals = ["exec:java"]
    if not args.skip_export_compile:
        goals.insert(0, "compile")
    cmd = [maven_executable()]
    if not args.online_maven:
        cmd.append("-o")
    cmd.extend(["-q", "-pl", MODULE, "-am", "-DskipTests"])
    cmd.extend(goals)
    exec_args = [
        "--value-targets",
        str(target_csv),
        "--out",
        str(out_ser),
        "--summary",
        str(summary_csv),
        "--target-mode",
        args.target_mode,
        "--timeout-sec",
        str(args.training_data_timeout_sec),
    ]
    if args.max_training_records > 0:
        exec_args.extend(["--max-records", str(args.max_training_records)])
    cmd.extend([
        "-Dexec.mainClass=mage.player.ai.rl.TerminalLineValueTargetTrainingDataExporter",
        "-Dexec.args=" + " ".join(quote_exec_arg(part) for part in exec_args),
    ])
    return cmd


def build_train_command(
        args: argparse.Namespace,
        candidate_profile: str,
        train_run_id: str,
        training_data_dir: Path,
) -> List[str]:
    cmd = [
        powershell_executable(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/run_action_counterfactual.ps1",
        "-Profile",
        candidate_profile,
        "-RunId",
        train_run_id,
        "-ImportTrainingDataPath",
        str(training_data_dir),
        "-Scenarios",
        "1",
        "-BatchSize",
        str(args.train_batch_size),
        "-ServiceMode",
        args.train_service_mode,
        "-ModelDModel",
        str(args.model_d_model),
        "-ModelNumLayers",
        str(args.model_num_layers),
        "-TrainEpochs",
        str(args.train_epochs),
        "-CandidatePermutations",
        "1",
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
        "-SkipCompile",
    ]
    if args.skip_optimizer_state_load:
        cmd.append("-SkipOptimizerStateLoad")
    if args.reset_training_state_on_load:
        cmd.append("-ResetTrainingStateOnLoad")
    return cmd


def build_eval_command(
        args: argparse.Namespace,
        eval_run_id: str,
        eval_root: Path,
        registry: Path,
        profile: str,
        gpu_port: int,
        gpu_metrics_port: int,
        seed_base: int,
) -> List[str]:
    cmd = [
        python_executable(),
        "scripts/run_cp7_eval_sweep.py",
        "--registry",
        str(registry),
        "--profiles",
        profile,
        "--opponents",
        args.eval_opponents or args.opponents,
        "--games-per-matchup",
        str(args.post_eval_games_per_matchup),
        "--games-per-job",
        str(args.post_eval_games_per_job),
        "--skill",
        str(args.skill),
        "--parallel",
        str(args.post_eval_parallel),
        "--serial-warmup-jobs",
        str(args.serial_warmup_jobs),
        "--ai-threads",
        str(args.ai_threads),
        "--gpu-port",
        str(gpu_port),
        "--gpu-metrics-port",
        str(gpu_metrics_port),
        "--timeout-sec",
        str(args.eval_timeout_sec),
        "--replay-seed-base",
        str(seed_base),
        "--seed-key-mode",
        "matchup",
        "--run-id",
        eval_run_id,
        "--output-root",
        str(eval_root),
        "--deterministic-eval",
    ]
    if args.post_eval_chunk_indices:
        cmd.extend(["--chunk-indices", args.post_eval_chunk_indices])
    if args.post_eval_game_logging:
        cmd.append("--eval-game-logging")
        cmd.extend(["--game-log-format", args.game_log_format])
    if args.allow_deterministic_parallel:
        cmd.append("--allow-deterministic-parallel")
    if args.no_compile_exec:
        pass
    else:
        cmd.append("--compile-exec")
    if not args.online_maven:
        cmd.append("--maven-offline")
    if args.skip_post_eval_compile:
        cmd.append("--skip-compile")
    return cmd


def cycle_target_paths(online_run_dir: Path) -> Dict[str, Path]:
    manifest = read_json(online_run_dir / "manifest.json")
    cycles = manifest.get("cycle_summaries")
    if not isinstance(cycles, list) or not cycles:
        return {}
    cycle = cycles[-1]
    if not isinstance(cycle, dict):
        return {}
    clean_dir = resolve_repo_path(str(cycle.get("clean_targets_dir") or ""))
    include_dir = resolve_repo_path(str(cycle.get("include_pass_targets_dir") or ""))
    mine_dir = resolve_repo_path(str(cycle.get("mine_dir") or ""))
    return {
        "clean_targets_dir": clean_dir,
        "include_pass_targets_dir": include_dir,
        "mine_dir": mine_dir,
        "target_csv": clean_dir / "terminal_line_value_targets.csv",
        "target_summary": clean_dir / "terminal_line_value_target_summary.json",
    }


def admitted_examples(summary: Dict[str, object]) -> int:
    for key in ("admitted_examples", "trainable_examples"):
        value = summary.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return 0


def profile_summary_text(eval_run_dir: object, returncode: object) -> str:
    if eval_run_dir:
        summary_csv = resolve_repo_path(str(eval_run_dir)) / "profile_summary.csv"
        rows = list(read_csv_rows(summary_csv))
        if rows:
            row = rows[0]
            wins = row.get("wins", "")
            total = row.get("total", "")
            winrate = row.get("winrate", "")
            if wins != "" and total != "":
                return f"{wins}/{total} ({winrate})"
    if returncode != "":
        return f"rc={returncode}"
    return ""


def write_readme(run_dir: Path, manifest: Dict[str, object]) -> None:
    lines = [
        "# Online Mining Training Loop",
        "",
        "Closed loop for online checkpoint capture, terminal-line mining, value-target export, Q-head training, and eval.",
        "",
        "The loop uses terminal outcomes only. It does not add combo-state labels, card-specific rewards, or intermediate combo milestones.",
        "",
        f"- run_id: `{manifest.get('run_id', '')}`",
        f"- started_utc: `{manifest.get('started_utc', '')}`",
        f"- base_profile: `{manifest.get('base_profile', '')}`",
        "",
        "## Cycles",
        "",
        "| Cycle | Play Profile | Targets | TrainingData | Candidate | Train RC | Candidate Eval | Source Eval | Status |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for cycle in manifest.get("cycle_summaries", []):
        if not isinstance(cycle, dict):
            continue
        lines.append(
            "| {cycle} | `{play}` | {targets} | {records} | `{candidate}` | {train} | {candidate_eval} | {source_eval} | {status} |".format(
                cycle=cycle.get("cycle", ""),
                play=cycle.get("play_profile", ""),
                targets=cycle.get("admitted_targets", ""),
                records=cycle.get("exported_training_records", ""),
                candidate=cycle.get("candidate_profile", ""),
                train=cycle.get("train_returncode", ""),
                candidate_eval=profile_summary_text(
                    cycle.get("candidate_eval_run_dir", ""),
                    cycle.get("candidate_eval_returncode", ""),
                ),
                source_eval=profile_summary_text(
                    cycle.get("source_eval_run_dir", ""),
                    cycle.get("source_eval_returncode", ""),
                ),
                status=cycle.get("status", ""),
            )
        )
    run_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=timestamp_id())
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--base-profile", default="Pauper-Spy-Combo-Value")
    parser.add_argument(
        "--play-profile",
        default="",
        help=(
            "Profile to use for cycle-1 online play. When omitted, uses --base-profile. "
            "This lets generated candidates reuse the base profile's registry deck/env entry."
        ),
    )
    parser.add_argument("--candidate-profile", default="")
    parser.add_argument("--candidate-profile-prefix", default="Pauper-Spy-Combo-Value-OnlineLoop")
    parser.add_argument("--force-profile-clone", action="store_true")
    parser.add_argument(
        "--existing-online-run-dir",
        default="",
        help="Reuse an already-started online terminal-mining run for cycle 1 instead of launching one.",
    )
    parser.add_argument(
        "--wait-existing-online-run-sec",
        type=int,
        default=0,
        help="When reusing an online run, wait this long for its value-target summary to appear.",
    )
    parser.add_argument(
        "--existing-target-dir",
        default="",
        help=(
            "When reusing an online run, train from this explicit terminal-line value-target "
            "directory instead of the reused run's clean_targets_dir."
        ),
    )
    parser.add_argument("--wait-poll-sec", type=float, default=30.0)
    parser.add_argument(
        "--initial-q-blend",
        type=float,
        default=None,
        help=(
            "Q blend for the cycle-1 play profile. Defaults to --candidate-q-blend "
            "when --play-profile is supplied, otherwise 0.0 for a base-profile start."
        ),
    )
    parser.add_argument("--candidate-q-blend", type=float, default=1.0)
    parser.add_argument("--candidate-q-blend-heads", default="action,target,card_select")
    parser.add_argument("--candidate-q-blend-min-margin", type=float, default=0.0)
    parser.add_argument("--candidate-q-blend-min-top-q", type=float, default=None)

    parser.add_argument("--opponents", default="grixis")
    parser.add_argument("--eval-opponents", default="")
    parser.add_argument("--games-per-matchup", type=int, default=2)
    parser.add_argument("--games-per-job", type=int, default=1)
    parser.add_argument("--chunk-indices", default="")
    parser.add_argument("--skill", type=int, default=7)
    parser.add_argument("--eval-parallel", type=int, default=1)
    parser.add_argument(
        "--post-eval-parallel",
        type=int,
        default=0,
        help="Post-train eval parallelism. Defaults to --eval-parallel when unset or <= 0.",
    )
    parser.add_argument(
        "--allow-deterministic-parallel",
        action="store_true",
        help=(
            "Allow deterministic online/post evals to honor parallelism. Without this, "
            "run_cp7_eval_sweep keeps deterministic evals serial for strict replay comparability."
        ),
    )
    parser.add_argument("--serial-warmup-jobs", type=int, default=1)
    parser.add_argument("--ai-threads", type=int, default=1)
    parser.add_argument("--eval-timeout-sec", type=int, default=1800)
    parser.add_argument("--eval-gpu-port", type=int, default=29100)
    parser.add_argument("--eval-gpu-metrics-port", type=int, default=30100)
    parser.add_argument("--mine-gpu-port", type=int, default=29300)
    parser.add_argument("--mine-gpu-metrics-port", type=int, default=30300)
    parser.add_argument("--post-eval-gpu-port", type=int, default=29500)
    parser.add_argument("--post-eval-gpu-metrics-port", type=int, default=30500)
    parser.add_argument("--port-stride", type=int, default=100)
    parser.add_argument("--live-checkpoint-max-per-game", type=int, default=96)
    parser.add_argument(
        "--live-checkpoint-action-types",
        default="ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,DECLARE_ATTACKS,DECLARE_BLOCKS,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
    )
    parser.add_argument("--eval-game-logging", action="store_true")
    parser.add_argument("--post-eval-game-logging", action="store_true")
    parser.add_argument("--game-log-format", choices=("compact", "full", "both"), default="compact")
    parser.add_argument("--replay-seed-base", type=int, default=6161)
    parser.add_argument("--post-eval-seed-base", type=int, default=7161)
    parser.add_argument("--seed-key-mode", choices=("profile", "matchup"), default="matchup")
    parser.add_argument("--no-compile-exec", action="store_true")
    parser.add_argument("--skip-eval-compile", action="store_true")
    parser.add_argument("--skip-mine-compile", action="store_true")
    parser.add_argument("--skip-shard-compile", action="store_true")
    parser.add_argument("--allow-stale-miner-classpath", action="store_true")
    parser.add_argument("--skip-export-compile", action="store_true")
    parser.add_argument("--skip-post-eval-compile", action="store_true")
    parser.add_argument("--online-maven", action="store_true")

    parser.add_argument("--mine-shards", type=int, default=2)
    parser.add_argument(
        "--max-concurrent-mine-shards",
        type=int,
        default=0,
        help="Limit concurrent mining shard JVMs while preserving --mine-shards selection partitions.",
    )
    parser.add_argument("--max-snapshots", type=int, default=32)
    parser.add_argument("--selection-mode", default="ranked")
    parser.add_argument("--ranked-max-per-game", type=int, default=16)
    parser.add_argument("--action-types", default="ACTIVATE_ABILITY_OR_SPELL")
    parser.add_argument("--line-attempts", type=int, default=16)
    parser.add_argument("--line-max-root-actions", type=int, default=16)
    parser.add_argument("--line-timeout-sec", type=int, default=120)
    parser.add_argument("--tree-timeout-sec", type=int, default=120)
    parser.add_argument("--tree-seed", type=int, default=2026052511)
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

    parser.add_argument("--target-mode", choices=("distribution", "signed-values", "advantage-values"), default="advantage-values")
    parser.add_argument("--training-data-timeout-sec", type=int, default=45)
    parser.add_argument("--max-training-records", type=int, default=0)
    parser.add_argument("--min-trainable-targets", type=int, default=4)
    parser.add_argument("--fail-on-insufficient-targets", action="store_true")

    parser.add_argument("--train-service-mode", choices=("local", "shared_gpu"), default="local")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--model-d-model", type=int, default=128)
    parser.add_argument("--model-num-layers", type=int, default=2)
    parser.add_argument("--candidate-q-loss-coef", type=float, default=1.0)
    parser.add_argument("--candidate-q-huber-beta", type=float, default=0.25)
    parser.add_argument("--post-train-wait-ms", type=int, default=3600000)
    parser.add_argument("--skip-optimizer-state-load", action="store_true", default=True)
    parser.add_argument("--load-optimizer-state", dest="skip_optimizer_state_load", action="store_false")
    parser.add_argument("--reset-training-state-on-load", action="store_true")

    parser.add_argument("--post-train-eval", dest="post_train_eval", action="store_true", default=True)
    parser.add_argument("--no-post-train-eval", dest="post_train_eval", action="store_false")
    parser.add_argument("--post-eval-source-comparator", action="store_true")
    parser.add_argument("--post-eval-games-per-matchup", type=int, default=4)
    parser.add_argument("--post-eval-games-per-job", type=int, default=1)
    parser.add_argument("--post-eval-chunk-indices", default="")

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.post_eval_parallel <= 0:
        args.post_eval_parallel = args.eval_parallel
    if args.initial_q_blend is None:
        args.initial_q_blend = args.candidate_q_blend if args.play_profile else 0.0
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO / output_root
    run_dir = output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    base_registry_path = resolve_repo_path(args.registry)
    registry = load_registry(base_registry_path)
    base_entry = find_registry_entry(registry, args.base_profile)

    manifest: Dict[str, object] = {
        "run_id": args.run_id,
        "started_utc": utc_now(),
        "base_profile": args.base_profile,
        "initial_play_profile": args.play_profile.strip() or args.base_profile,
        "base_registry": rel(base_registry_path),
        "output_root": rel(output_root),
        "cycle_summaries": [],
    }
    write_json(run_dir / "manifest.json", manifest)

    current_profile = args.play_profile.strip() or args.base_profile
    current_q_blend = args.initial_q_blend

    for cycle_index in range(max(1, args.cycles)):
        cycle_id = f"cycle{cycle_index + 1:02d}"
        cycle_summary: Dict[str, object] = {
            "cycle": cycle_index + 1,
            "started_utc": utc_now(),
            "play_profile": current_profile,
            "play_q_blend": current_q_blend,
        }

        play_registry = run_dir / "registries" / f"{cycle_id}_play_registry.json"
        play_entry = profile_entry(
            base_entry,
            current_profile,
            current_q_blend,
            args.candidate_q_blend_heads,
            args.candidate_q_blend_min_margin,
            args.candidate_q_blend_min_top_q,
        )
        write_registry(play_registry, [play_entry])
        cycle_summary["play_registry"] = rel(play_registry)

        if cycle_index == 0 and args.existing_online_run_dir.strip():
            online_run_dir = resolve_repo_path(args.existing_online_run_dir)
            online_rc = 0
            cycle_summary["online_reused"] = True
            cycle_summary["online_returncode"] = online_rc
            cycle_summary["online_run_dir"] = rel(online_run_dir)
            if args.wait_existing_online_run_sec > 0 and not args.dry_run:
                deadline = dt.datetime.now(dt.timezone.utc).timestamp() + args.wait_existing_online_run_sec
                while dt.datetime.now(dt.timezone.utc).timestamp() < deadline:
                    target_paths = cycle_target_paths(online_run_dir)
                    summary_path = target_paths.get("target_summary")
                    if summary_path is not None and summary_path.is_file():
                        break
                    print(f"waiting for reused online run targets: {online_run_dir}", flush=True)
                    time.sleep(max(1.0, float(args.wait_poll_sec)))
        else:
            online_run_id = f"{safe_name(args.run_id, 64)}_{cycle_id}_online_{safe_name(current_profile, 48)}"
            online_root = run_dir / "online_mining"
            online_run_dir = online_root / online_run_id
            online_rc = run_command(
                build_online_mining_command(args, cycle_index, online_run_id, online_root, play_registry, current_profile),
                run_dir / "logs" / f"{cycle_id}_online_mining.log",
                args.dry_run,
            )
            cycle_summary["online_returncode"] = online_rc
            cycle_summary["online_run_dir"] = rel(online_run_dir)
            if online_rc != 0:
                cycle_summary["status"] = "online_mining_failed"
                cycle_summary["ended_utc"] = utc_now()
                manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
                manifest["status"] = cycle_summary["status"]
                write_json(run_dir / "manifest.json", manifest)
                write_readme(run_dir, manifest)
                return online_rc

        target_paths = cycle_target_paths(online_run_dir)
        if cycle_index == 0 and args.existing_target_dir.strip():
            override_dir = resolve_repo_path(args.existing_target_dir)
            target_paths["clean_targets_dir"] = override_dir
            target_paths["target_csv"] = override_dir / "terminal_line_value_targets.csv"
            target_paths["target_summary"] = override_dir / "terminal_line_value_target_summary.json"
            cycle_summary["target_override_dir"] = rel(override_dir)
        target_csv = target_paths.get("target_csv", online_run_dir / "missing.csv")
        target_summary = read_json(target_paths.get("target_summary", online_run_dir / "missing.json"))
        admitted = admitted_examples(target_summary)
        cycle_summary["target_csv"] = rel(target_csv)
        cycle_summary["target_rows"] = count_target_rows(target_csv)
        cycle_summary["target_summary"] = target_summary
        cycle_summary["admitted_targets"] = admitted
        if admitted < args.min_trainable_targets:
            cycle_summary["status"] = "insufficient_targets"
            cycle_summary["ended_utc"] = utc_now()
            manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
            manifest["status"] = cycle_summary["status"]
            write_json(run_dir / "manifest.json", manifest)
            write_readme(run_dir, manifest)
            return 3 if args.fail_on_insufficient_targets else 0

        training_data_dir = run_dir / "training_data" / cycle_id
        out_ser = training_data_dir / f"{cycle_id}_{args.target_mode}_training_data.ser"
        export_summary_csv = training_data_dir / f"{cycle_id}_{args.target_mode}_training_data_summary.csv"
        export_rc = run_command(
            build_training_data_export_command(args, target_csv, out_ser, export_summary_csv),
            run_dir / "logs" / f"{cycle_id}_training_data_export.log",
            args.dry_run,
        )
        exported_records = count_exported_records(export_summary_csv) if not args.dry_run else admitted
        cycle_summary["training_data_file"] = rel(out_ser)
        cycle_summary["training_data_summary_csv"] = rel(export_summary_csv)
        cycle_summary["training_data_export_returncode"] = export_rc
        cycle_summary["exported_training_records"] = exported_records
        if export_rc != 0:
            cycle_summary["status"] = "training_data_export_failed"
            cycle_summary["ended_utc"] = utc_now()
            manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
            manifest["status"] = cycle_summary["status"]
            write_json(run_dir / "manifest.json", manifest)
            write_readme(run_dir, manifest)
            return export_rc
        if exported_records < args.min_trainable_targets:
            cycle_summary["status"] = "insufficient_training_data"
            cycle_summary["ended_utc"] = utc_now()
            manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
            manifest["status"] = cycle_summary["status"]
            write_json(run_dir / "manifest.json", manifest)
            write_readme(run_dir, manifest)
            return 3 if args.fail_on_insufficient_targets else 0

        if args.candidate_profile and args.cycles == 1:
            candidate_profile = args.candidate_profile
        else:
            candidate_profile = (
                f"{safe_name(args.candidate_profile_prefix, 64)}-"
                f"{safe_name(args.run_id, 32)}-{cycle_id}"
            )
        cycle_summary["candidate_profile"] = candidate_profile
        clone_summary = clone_profile_models(current_profile, candidate_profile, args.force_profile_clone, args.dry_run)
        cycle_summary["profile_clone"] = clone_summary

        train_run_id = f"{safe_name(args.run_id, 64)}_{cycle_id}_train_{safe_name(candidate_profile, 48)}"
        train_rc = run_command(
            build_train_command(args, candidate_profile, train_run_id, training_data_dir),
            run_dir / "logs" / f"{cycle_id}_train_candidate_q.log",
            args.dry_run,
        )
        cycle_summary["train_run_id"] = train_run_id
        cycle_summary["train_returncode"] = train_rc
        if train_rc != 0:
            cycle_summary["status"] = "train_failed"
            cycle_summary["ended_utc"] = utc_now()
            manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
            manifest["status"] = cycle_summary["status"]
            write_json(run_dir / "manifest.json", manifest)
            write_readme(run_dir, manifest)
            return train_rc

        if args.post_train_eval:
            eval_root = run_dir / "post_train_eval"
            candidate_registry = run_dir / "registries" / f"{cycle_id}_candidate_eval_registry.json"
            candidate_entry = profile_entry(
                base_entry,
                candidate_profile,
                args.candidate_q_blend,
                args.candidate_q_blend_heads,
                args.candidate_q_blend_min_margin,
                args.candidate_q_blend_min_top_q,
            )
            write_registry(candidate_registry, [candidate_entry])
            candidate_eval_id = f"{safe_name(args.run_id, 64)}_{cycle_id}_candidate_eval"
            candidate_eval_rc = run_command(
                build_eval_command(
                    args,
                    candidate_eval_id,
                    eval_root,
                    candidate_registry,
                    candidate_profile,
                    args.post_eval_gpu_port + cycle_index * args.port_stride,
                    args.post_eval_gpu_metrics_port + cycle_index * args.port_stride,
                    args.post_eval_seed_base + cycle_index * 1000,
                ),
                run_dir / "logs" / f"{cycle_id}_candidate_eval.log",
                args.dry_run,
            )
            cycle_summary["candidate_eval_registry"] = rel(candidate_registry)
            cycle_summary["candidate_eval_run_dir"] = rel(eval_root / candidate_eval_id)
            cycle_summary["candidate_eval_returncode"] = candidate_eval_rc
            cycle_summary["candidate_eval_profile_summary"] = read_json(eval_root / candidate_eval_id / "manifest.json")
            if candidate_eval_rc != 0:
                cycle_summary["status"] = "candidate_eval_failed"
                cycle_summary["ended_utc"] = utc_now()
                manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
                manifest["status"] = cycle_summary["status"]
                write_json(run_dir / "manifest.json", manifest)
                write_readme(run_dir, manifest)
                return candidate_eval_rc

            if args.post_eval_source_comparator:
                source_registry = run_dir / "registries" / f"{cycle_id}_source_eval_registry.json"
                source_entry = profile_entry(
                    base_entry,
                    current_profile,
                    current_q_blend,
                    args.candidate_q_blend_heads,
                    args.candidate_q_blend_min_margin,
                    args.candidate_q_blend_min_top_q,
                )
                write_registry(source_registry, [source_entry])
                source_eval_id = f"{safe_name(args.run_id, 64)}_{cycle_id}_source_eval"
                source_eval_rc = run_command(
                    build_eval_command(
                        args,
                        source_eval_id,
                        eval_root,
                        source_registry,
                        current_profile,
                        args.post_eval_gpu_port + cycle_index * args.port_stride + 10,
                        args.post_eval_gpu_metrics_port + cycle_index * args.port_stride + 10,
                        args.post_eval_seed_base + cycle_index * 1000,
                    ),
                    run_dir / "logs" / f"{cycle_id}_source_eval.log",
                    args.dry_run,
                )
                cycle_summary["source_eval_registry"] = rel(source_registry)
                cycle_summary["source_eval_run_dir"] = rel(eval_root / source_eval_id)
                cycle_summary["source_eval_returncode"] = source_eval_rc
                if source_eval_rc != 0:
                    cycle_summary["status"] = "source_eval_failed"
                    cycle_summary["ended_utc"] = utc_now()
                    manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
                    manifest["status"] = cycle_summary["status"]
                    write_json(run_dir / "manifest.json", manifest)
                    write_readme(run_dir, manifest)
                    return source_eval_rc

        cycle_summary["status"] = "completed"
        cycle_summary["ended_utc"] = utc_now()
        manifest["cycle_summaries"].append(cycle_summary)  # type: ignore[index]
        manifest["status"] = "running" if cycle_index + 1 < args.cycles else "completed"
        write_json(run_dir / "manifest.json", manifest)
        write_readme(run_dir, manifest)

        current_profile = candidate_profile
        current_q_blend = args.candidate_q_blend

    manifest["ended_utc"] = utc_now()
    manifest["status"] = "completed"
    write_json(run_dir / "manifest.json", manifest)
    write_readme(run_dir, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
