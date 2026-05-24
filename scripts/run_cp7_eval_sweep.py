#!/usr/bin/env python3
"""Run a balanced CP7 evaluation sweep for active MTGRL profiles.

The runner snapshots the active profile checkpoints, starts an inference-only
GPU service, then executes one `league_bench` JVM per profile/opponent matchup.
Results are written under local-training/local_pbt/cp7_eval_sweeps/<run_id>/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.request import urlopen


REPO = Path(__file__).resolve().parents[1]
MODULE = "Mage.Server.Plugins/Mage.Player.AIRL"
DEFAULT_REGISTRY = (
    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/"
    "pauper_spy_pbt_registry.json"
)
ML_CODE = (
    REPO
    / "Mage.Server.Plugins"
    / "Mage.Player.AIRL"
    / "src"
    / "mage"
    / "player"
    / "ai"
    / "rl"
    / "MLPythonCode"
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


def maven_executable() -> str:
    return shutil.which("mvn") or shutil.which("mvn.cmd") or "mvn"


def python_executable() -> str:
    if os.name == "nt":
        candidate = REPO / ".mtgrl_venv" / "Scripts" / "python.exe"
    else:
        candidate = REPO / ".mtgrl_venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def read_deck_pool(deck_path: str) -> List[Path]:
    path = resolve_repo_path(deck_path)
    if path.suffix.lower() != ".txt":
        return [path]
    decks: List[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        decks.append(candidate.resolve())
    return decks


def load_active_entries(registry_path: Path) -> List[dict]:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [e for e in data if e.get("active", False)]
    if not entries:
        raise RuntimeError(f"No active profiles in {registry_path}")
    return entries


def filter_entries(entries: List[dict], profiles: str) -> List[dict]:
    if not profiles.strip():
        return entries
    wanted = {p.strip() for p in profiles.split(",") if p.strip()}
    filtered = [e for e in entries if str(e.get("profile", "")) in wanted]
    missing = wanted - {str(e.get("profile", "")) for e in filtered}
    if missing:
        raise RuntimeError(f"Requested profile(s) not active in registry: {sorted(missing)}")
    return filtered


def filter_decks(decks: List[Path], opponents: str) -> List[Path]:
    if not opponents.strip():
        return decks
    tokens = [t.strip().lower() for t in opponents.split(",") if t.strip()]
    filtered: List[Path] = []
    for deck in decks:
        haystack = f"{deck.stem.lower()} {deck.name.lower()} {str(deck).lower()}"
        if any(token in haystack for token in tokens):
            filtered.append(deck)
    if not filtered:
        raise RuntimeError(f"Opponent filter matched no decks: {opponents}")
    return filtered


def common_train_env(entries: List[dict]) -> Dict[str, str]:
    common: Dict[str, str] = {}
    keys = {
        str(key)
        for entry in entries
        for key in (entry.get("train_env") or {}).keys()
    }
    for key in sorted(keys):
        values: List[str] = []
        missing = False
        for entry in entries:
            raw = str((entry.get("train_env") or {}).get(key, "")).strip()
            if not raw:
                missing = True
                break
            values.append(raw)
        unique = set(values)
        if not missing and len(unique) == 1:
            common[key] = values[0]
    return common


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def read_episode(profile: str) -> Optional[int]:
    stats = PROFILES_ROOT / profile / "logs" / "stats" / "training_stats.csv"
    if not stats.exists():
        return None
    max_episode: Optional[int] = None
    try:
        with stats.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for key in ("episode", "episode_id", "episodes", "game_index", "episode_index"):
                    val = (row.get(key) or "").strip()
                    if val:
                        try:
                            episode = int(float(val))
                            max_episode = episode if max_episode is None else max(max_episode, episode)
                            break
                        except ValueError:
                            pass
    except Exception:
        return None
    return max_episode


def snapshot_profiles(entries: List[dict], snapshot_root: Path) -> List[dict]:
    manifest: List[dict] = []
    for entry in entries:
        profile = str(entry["profile"])
        src_models = PROFILES_ROOT / profile / "models"
        latest = src_models / "model_latest.pt"
        if not latest.exists():
            raise FileNotFoundError(f"Missing model_latest.pt for {profile}: {latest}")
        dst_models = snapshot_root / "profiles" / profile / "models"
        method_latest = link_or_copy(latest, dst_models / "model_latest.pt")
        method_model = link_or_copy(latest, dst_models / "model.pt")
        for optional_name in ("mulligan_model.pt", "episodes.txt", "mulligan_episodes.txt"):
            optional_src = src_models / optional_name
            if optional_src.exists():
                link_or_copy(optional_src, dst_models / optional_name)
        manifest.append(
            {
                "profile": profile,
                "source_model_latest": str(latest),
                "snapshot_model_latest": str(dst_models / "model_latest.pt"),
                "source_model_latest_mtime_utc": dt.datetime.fromtimestamp(
                    latest.stat().st_mtime, dt.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bytes": latest.stat().st_size,
                "episode": read_episode(profile),
                "snapshot_method_latest": method_latest,
                "snapshot_method_model": method_model,
            }
        )
    return manifest


def start_gpu_service(
    port: int,
    metrics_port: int,
    log_path: Path,
    train_env: Dict[str, str],
) -> subprocess.Popen:
    env = os.environ.copy()
    for key, value in train_env.items():
        env[str(key)] = str(value)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "GPU_SERVICE_PORT": str(port),
            "GPU_SERVICE_METRICS_PORT": str(metrics_port),
            "GPU_SERVICE_ROLE": "infer",
            "PY_BATCH_TIMEOUT_MS": env.get("PY_BATCH_TIMEOUT_MS", "25"),
            "PY_BATCH_MAX_SIZE": env.get("PY_BATCH_MAX_SIZE", "256"),
            "SCORE_WORKER_THREADS": env.get("SCORE_WORKER_THREADS", "0"),
            "TRAIN_WORKER_THREADS": "1",
            "TRAIN_GPU_MAX_CONCURRENT": "1",
            "INFER_CUDA_DEVICE": env.get("INFER_CUDA_DEVICE", "cuda:0"),
            "TRAIN_CUDA_DEVICE": env.get("TRAIN_CUDA_DEVICE", "cpu"),
            "CUDA_MEM_FRACTION": env.get("CUDA_MEM_FRACTION", "0.60"),
            "USE_TRT_INFERENCE": env.get("USE_TRT_INFERENCE", "0"),
            "MULLIGAN_DEVICE": env.get("MULLIGAN_DEVICE", "cpu"),
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [python_executable(), str(ML_CODE / "gpu_service_host.py")],
        cwd=str(REPO),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 45
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"GPU service exited early with code {proc.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{metrics_port}/metrics", timeout=2) as resp:
                if resp.status == 200:
                    time.sleep(float(os.environ.get("CP7_GPU_READY_GRACE_SEC", "20")))
                    return proc
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"GPU service did not become ready: {last_error}")


def parse_eval_result(output: str, result_file: Path) -> Tuple[int, int, float, int]:
    if result_file.exists():
        text = result_file.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            parts = text.split(",")
            if len(parts) >= 3:
                mcts_activations = int(parts[4]) if len(parts) >= 5 and parts[4].strip() else 0
                return int(parts[0]), int(parts[1]), float(parts[2]), mcts_activations
    match = re.search(
        r"EVAL_RESULT:\s+wins=(\d+)\s+total=(\d+)\s+winrate=([0-9.]+)"
        r"(?:\s+profile=\S+)?(?:\s+mcts_activations=(\d+))?",
        output,
    )
    if not match:
        return 0, 0, 0.0, 0
    return int(match.group(1)), int(match.group(2)), float(match.group(3)), int(match.group(4) or 0)


def job_env(
    base_env: Dict[str, str],
    entry: dict,
    opponent_deck: Path,
    snapshot_root: Path,
    run_dir: Path,
    result_file: Path,
    game_log_dir: Path,
    db_dir: Path,
    games: int,
    skill: int,
    gpu_port: int,
    ai_threads: int,
    mcts_enabled: bool,
    mcts_env: Dict[str, str],
    eval_game_logging: bool,
    game_log_format: str,
    replay_metadata: bool,
    replay_seed_base: int,
    live_checkpoints: bool,
    live_checkpoint_max_per_game: int,
    live_checkpoint_action_types: str,
) -> Dict[str, str]:
    env = base_env.copy()
    train_env = entry.get("train_env") or {}
    for key, value in train_env.items():
        env[str(key)] = str(value)
    env.update(
        {
            "MODE": "league_bench",
            "MODEL_PROFILE": str(entry["profile"]),
            "RL_ARTIFACTS_ROOT": str(snapshot_root),
            "RL_AGENT_DECK_LIST": str(resolve_repo_path(train_env["RL_AGENT_DECK_LIST"])),
            "DECK_LIST_FILE": str(resolve_repo_path(entry["deck_path"])),
            "EVAL_OPPONENT_DECK": str(opponent_deck),
            "EVAL_OPPONENT_SKILL": str(skill),
            "EVAL_NUM_GAMES": str(games),
            "EVAL_RESULTS_FILE": str(result_file),
            "GAME_LOG_DIR": str(game_log_dir),
            "EVAL_GAME_LOGGING": "1" if eval_game_logging else "0",
            "GAME_LOGGING": "1" if eval_game_logging else "0",
            "GAME_LOG_FREQUENCY": "1" if eval_game_logging else "0",
            "GAME_LOG_FORMAT": game_log_format if eval_game_logging else env.get("GAME_LOG_FORMAT", "full"),
            "EVAL_REPLAY_METADATA": "1" if replay_metadata else "0",
            "EVAL_REPLAY_PREGAME_DECISION_LOG": "1" if replay_metadata else env.get("EVAL_REPLAY_PREGAME_DECISION_LOG", "0"),
            "EVAL_REPLAY_DECISION_LOG": "1" if replay_metadata else env.get("EVAL_REPLAY_DECISION_LOG", "0"),
            "EVAL_AGENT_SEARCH_TRACE_JSON": "1" if replay_metadata else env.get("EVAL_AGENT_SEARCH_TRACE_JSON", "0"),
            "EVAL_AGENT_ACTUAL_SEARCH_TRACE_JSON": "1" if replay_metadata else env.get("EVAL_AGENT_ACTUAL_SEARCH_TRACE_JSON", "0"),
            "EVAL_REPLAY_SEED_BASE": str(replay_seed_base),
            "EVAL_LIVE_CHECKPOINTS": "1" if live_checkpoints else "0",
            "EVAL_LIVE_CHECKPOINT_DIR": str((run_dir / "live_checkpoints" / game_log_dir.name).resolve()) if live_checkpoints else env.get("EVAL_LIVE_CHECKPOINT_DIR", ""),
            "EVAL_LIVE_CHECKPOINT_MAX_PER_GAME": str(live_checkpoint_max_per_game),
            "EVAL_LIVE_CHECKPOINT_ACTION_TYPES": live_checkpoint_action_types,
            "METRICS_PORT": "0",
            "PY_SERVICE_MODE": "shared_gpu",
            "GPU_SERVICE_ENDPOINT": f"localhost:{gpu_port}",
            "GPU_SERVICE_NUM_GPUS": "1",
            "GPU_SERVICE_NUM_CHANNELS": "4",
            "MAGE_DB_DIR": str(db_dir),
            "MAGE_DB_AUTO_SERVER": "false",
            "AI_MAX_THREADS_FOR_SIMULATIONS": str(ai_threads),
            "RL_HEURISTIC_STEP_REWARDS": "0",
            "USE_GAE": "0",
            "MCTS_TRAINING_ENABLE": "0",
            "ISMCTS_ENABLE": "1" if mcts_enabled else "0",
            "ISMCTS_ROLLOUTS_PER_TURN": "0",
            "MTG_AI_LOG_LEVEL": "WARN",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if mcts_enabled:
        env.update(mcts_env)
    return env


def stable_seed_offset(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 1_000_000_000


def parse_chunk_indices(raw: str) -> Optional[Set[int]]:
    text = (raw or "").strip()
    if not text:
        return None
    selected: Set[int] = set()
    for part in text.replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid chunk range: {token}")
            selected.update(range(start, end + 1))
        else:
            value = int(token)
            if value <= 0:
                raise ValueError(f"Invalid chunk index: {token}")
            selected.add(value)
    return selected


def prepare_db_copy(slot_dir: Path) -> Path:
    slot_dir.mkdir(parents=True, exist_ok=True)
    override = os.environ.get("CP7_EVAL_DB_SOURCE", "").strip()
    if override:
        src = resolve_repo_path(override)
        if src.is_dir():
            src = src / "cards.h2.mv.db"
    else:
        src = REPO / "db_eval" / "cards.h2.mv.db"
    if not src.exists():
        src = REPO / "db" / "cards.h2.mv.db"
    if not src.exists():
        raise FileNotFoundError("Could not find cards.h2.mv.db in CP7_EVAL_DB_SOURCE, db_eval, or db")
    dst = slot_dir / "cards.h2.mv.db"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        if dst.exists():
            dst.unlink()
        if os.environ.get("CP7_EVAL_DB_HARDLINK", "0").strip() in ("1", "true", "TRUE", "yes"):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
    return slot_dir


def maven_command(offline: bool) -> List[str]:
    cmd = [maven_executable()]
    if offline:
        cmd.append("-o")
    return cmd


def run_job(job: dict) -> dict:
    start = time.time()
    started_utc = utc_now()
    log_file: Path = job["log_file"]
    result_file: Path = job["result_file"]
    db_dir: Path = job["db_dir"]
    result_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    prepare_db_copy(db_dir)
    goals = ["exec:java"]
    if bool(job.get("compile_exec", False)):
        goals.insert(0, "compile")
    cmd = maven_command(bool(job.get("maven_offline", False))) + [
        "-q",
        "-pl",
        MODULE,
        "-am",
        "-DskipTests",
    ] + goals + [
        "-Dexec.mainClass=mage.player.ai.rl.RLTrainer",
        "-Dexec.args=league_bench",
    ]
    stdout = ""
    stderr = ""
    returncode = -1
    timed_out = False
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"# started_utc={started_utc}\n")
        log.write(f"# profile={job['profile']} opponent={job['opponent_label']}\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                env=job["env"],
                text=True,
                capture_output=True,
                timeout=job["timeout_sec"],
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = proc.returncode
            log.write(stdout)
            if stderr:
                log.write("\n--- STDERR ---\n")
                log.write(stderr)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            log.write(stdout)
            if stderr:
                log.write("\n--- STDERR ---\n")
                log.write(stderr)
            log.write(f"\nTIMEOUT after {job['timeout_sec']} seconds\n")
    wins, total, winrate, mcts_activations = parse_eval_result(stdout + "\n" + stderr, job["result_file"])
    if job["env"].get("CP7_EVAL_CLEAN_DB_AFTER_JOB", "0") == "1":
        run_db_root = (job["run_dir"] / "db").resolve()
        resolved_db_dir = db_dir.resolve()
        if str(resolved_db_dir).startswith(str(run_db_root)):
            shutil.rmtree(resolved_db_dir, ignore_errors=True)
    ended_utc = utc_now()
    return {
        "profile": job["profile"],
        "agent_deck": job["agent_label"],
        "opponent_deck": job["opponent_label"],
        "opponent_path": str(job["opponent_path"]),
        "skill": job["skill"],
        "games_requested": job["games"],
        "wins": wins,
        "total": total,
        "losses": max(0, total - wins),
        "winrate": winrate,
        "mcts_enabled": bool(job["mcts_enabled"]),
        "mcts_activations": mcts_activations,
        "returncode": returncode,
        "timed_out": timed_out,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "duration_sec": round(time.time() - start, 3),
        "result_file": str(job["result_file"]),
        "log_file": str(log_file),
        "chunk_index": int(job.get("chunk_index", 1)),
        "chunk_count": int(job.get("chunk_count", 1)),
    }


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_by_profile(rows: Iterable[dict]) -> List[dict]:
    by_profile: Dict[str, Dict[str, float]] = {}
    for row in rows:
        item = by_profile.setdefault(row["profile"], {"wins": 0, "total": 0})
        item["wins"] += int(row["wins"])
        item["total"] += int(row["total"])
    out = []
    for profile, vals in sorted(by_profile.items()):
        total = int(vals["total"])
        wins = int(vals["wins"])
        out.append(
            {
                "profile": profile,
                "wins": wins,
                "total": total,
                "losses": max(0, total - wins),
                "winrate": round(wins / total, 4) if total else 0.0,
            }
        )
    return out


def aggregate_by_matchup(rows: Iterable[dict]) -> List[dict]:
    buckets: Dict[Tuple[str, str, str, int, bool], dict] = {}
    for row in rows:
        key = (
            str(row["profile"]),
            str(row["agent_deck"]),
            str(row["opponent_deck"]),
            int(row["skill"]),
            bool(row["mcts_enabled"]),
        )
        bucket = buckets.setdefault(
            key,
            {
                "profile": row["profile"],
                "agent_deck": row["agent_deck"],
                "opponent_deck": row["opponent_deck"],
                "skill": row["skill"],
                "games_requested": 0,
                "chunks": 0,
                "wins": 0,
                "total": 0,
                "losses": 0,
                "winrate": 0.0,
                "mcts_enabled": bool(row["mcts_enabled"]),
                "mcts_activations": 0,
                "duration_sec": 0.0,
            },
        )
        bucket["games_requested"] += int(row.get("games_requested", 0) or 0)
        bucket["chunks"] += 1
        bucket["wins"] += int(row.get("wins", 0) or 0)
        bucket["total"] += int(row.get("total", 0) or 0)
        bucket["mcts_activations"] += int(row.get("mcts_activations", 0) or 0)
        bucket["duration_sec"] += float(row.get("duration_sec", 0.0) or 0.0)
    out: List[dict] = []
    for bucket in buckets.values():
        total = int(bucket["total"])
        wins = int(bucket["wins"])
        bucket["losses"] = max(0, total - wins)
        bucket["winrate"] = round(wins / total, 4) if total else 0.0
        bucket["duration_sec"] = round(float(bucket["duration_sec"]), 3)
        out.append(bucket)
    out.sort(key=lambda r: (str(r["profile"]), str(r["agent_deck"]), str(r["opponent_deck"])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--games-per-matchup", type=int, default=20)
    parser.add_argument(
        "--games-per-job",
        type=int,
        default=0,
        help="Split each matchup into smaller JVM jobs; 0 means one job per matchup.",
    )
    parser.add_argument("--skill", type=int, default=7)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--ai-threads", type=int, default=4)
    parser.add_argument("--gpu-port", type=int, default=26100)
    parser.add_argument("--gpu-metrics-port", type=int, default=27100)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--run-id", default=timestamp_id())
    parser.add_argument(
        "--output-root",
        default=os.environ.get("CP7_EVAL_SWEEP_ROOT", ""),
        help="Directory that receives <run-id>; defaults to local-training/local_pbt/cp7_eval_sweeps.",
    )
    parser.add_argument("--limit-matchups", type=int, default=0)
    parser.add_argument(
        "--chunk-indices",
        default="",
        help=(
            "Optional comma-separated 1-based chunk indices or ranges to run after "
            "chunking, e.g. '3,7,8,11,14'. Chunk numbers and replay seeds remain "
            "the same as the full sweep."
        ),
    )
    parser.add_argument("--profiles", default="")
    parser.add_argument("--opponents", default="")
    parser.add_argument("--split-agent-decks", action="store_true")
    parser.add_argument("--mcts", action="store_true")
    parser.add_argument("--mcts-iterations", type=int, default=8)
    parser.add_argument("--mcts-determinizations", type=int, default=4)
    parser.add_argument("--mcts-rollout-depth", type=int, default=0)
    parser.add_argument("--mcts-parallel-rollouts", type=int, default=4)
    parser.add_argument("--mcts-skip-top-prob", type=float, default=0.80)
    parser.add_argument("--multi-ply-mcts", action="store_true")
    parser.add_argument("--mcts-selective-enable", action="store_true")
    parser.add_argument("--eval-game-logging", action="store_true")
    parser.add_argument(
        "--game-log-format",
        choices=("compact", "full", "both"),
        default="compact",
        help="Game log format when --eval-game-logging is enabled.",
    )
    parser.add_argument(
        "--replay-metadata",
        action="store_true",
        help="With --eval-game-logging, use replay-compatible seeded deck order and log scenario/seed metadata.",
    )
    parser.add_argument(
        "--replay-seed-base",
        type=int,
        default=7777,
        help="Base seed for replay metadata; each matchup chunk gets a stable offset.",
    )
    parser.add_argument(
        "--seed-key-mode",
        choices=("profile", "matchup"),
        default="profile",
        help=(
            "Controls the stable offset key for replay seeds. "
            "'profile' preserves historical behavior; 'matchup' shares exact "
            "per-chunk seeds across profiles for paired candidate/baseline comparisons."
        ),
    )
    parser.add_argument(
        "--live-checkpoints",
        action="store_true",
        help="Capture durable branchable engine checkpoints during replay-logged live eval decisions.",
    )
    parser.add_argument(
        "--live-checkpoint-max-per-game",
        type=int,
        default=96,
        help="Maximum live checkpoints to write per game when --live-checkpoints is enabled.",
    )
    parser.add_argument(
        "--live-checkpoint-action-types",
        default="ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,DECLARE_ATTACKS,DECLARE_BLOCKS,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X",
        help="Comma-separated action types to checkpoint; use '*' for all replay-logged decisions.",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--compile-exec",
        action="store_true",
        help="Run each Maven job with compile exec:java in one invocation to avoid stale exec classpaths.",
    )
    parser.add_argument(
        "--maven-offline",
        action="store_true",
        help="Run Maven compile and exec:java with -o using the local repository cache.",
    )
    parser.add_argument("--reuse-gpu-service", action="store_true")
    parser.add_argument(
        "--serial-warmup-jobs",
        type=int,
        default=1,
        help="Run this many eval jobs serially before launching the parallel pool. This warms the shared GPU service.",
    )
    args = parser.parse_args()
    selected_chunk_indices = parse_chunk_indices(args.chunk_indices)

    registry = resolve_repo_path(args.registry)
    entries = filter_entries(load_active_entries(registry), args.profiles)
    output_root = Path(args.output_root) if args.output_root else (
        REPO / "local-training" / "local_pbt" / "cp7_eval_sweeps"
    )
    if not output_root.is_absolute():
        output_root = REPO / output_root
    run_dir = output_root / args.run_id
    snapshot_root = run_dir / "snapshot" / "rl"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    game_logs_dir = run_dir / "game_logs"
    db_root = run_dir / "db"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_compile:
        subprocess.run(
            maven_command(args.maven_offline) + ["-q", "-pl", MODULE, "-am", "-DskipTests", "compile"],
            cwd=str(REPO),
            check=True,
        )

    manifest_models = snapshot_profiles(entries, snapshot_root)
    gpu_service_train_env = common_train_env(entries)
    opponent_decks = filter_decks(read_deck_pool(entries[0]["deck_path"]), args.opponents)
    mcts_env = {
        "MCTS_ITERATIONS": str(args.mcts_iterations),
        "MCTS_DETERMINIZATIONS": str(args.mcts_determinizations),
        "MCTS_ROLLOUT_DEPTH": str(args.mcts_rollout_depth),
        "MCTS_PARALLEL_ROLLOUTS": str(args.mcts_parallel_rollouts),
        "MCTS_SKIP_TOP_PROB": str(args.mcts_skip_top_prob),
        "MULTI_PLY_MCTS": "1" if args.multi_ply_mcts else "0",
        "MCTS_SELECTIVE_ENABLE": "1" if args.mcts_selective_enable else "0",
    }

    manifest = {
        "run_id": args.run_id,
        "started_utc": utc_now(),
        "registry": str(registry),
        "games_per_matchup": args.games_per_matchup,
        "games_per_job": args.games_per_job,
        "chunk_indices_filter": args.chunk_indices,
        "skill": args.skill,
        "parallel": args.parallel,
        "serial_warmup_jobs": args.serial_warmup_jobs,
        "ai_threads": args.ai_threads,
        "profiles_filter": args.profiles,
        "opponents_filter": args.opponents,
        "split_agent_decks": bool(args.split_agent_decks),
        "mcts_enabled": bool(args.mcts),
        "mcts_env": mcts_env if args.mcts else {},
        "game_log_format": args.game_log_format,
        "replay_metadata": bool(args.replay_metadata),
        "replay_seed_base": args.replay_seed_base,
        "seed_key_mode": args.seed_key_mode,
        "live_checkpoints": bool(args.live_checkpoints),
        "live_checkpoint_max_per_game": args.live_checkpoint_max_per_game,
        "live_checkpoint_action_types": args.live_checkpoint_action_types,
        "compile_exec": bool(args.compile_exec),
        "snapshot_root": str(snapshot_root),
        "gpu_service_train_env": gpu_service_train_env,
        "profiles": manifest_models,
        "opponent_decks": [str(p) for p in opponent_decks],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    gpu_proc: Optional[subprocess.Popen] = None
    if not args.reuse_gpu_service:
        gpu_proc = start_gpu_service(
            args.gpu_port,
            args.gpu_metrics_port,
            logs_dir / "gpu_service.log",
            gpu_service_train_env,
        )

    rows: List[dict] = []
    rows_lock = threading.Lock()
    fields = [
        "profile",
        "agent_deck",
        "opponent_deck",
        "skill",
        "games_requested",
        "wins",
        "total",
        "losses",
        "winrate",
        "mcts_enabled",
        "mcts_activations",
        "returncode",
        "timed_out",
        "started_utc",
        "ended_utc",
        "duration_sec",
        "result_file",
        "log_file",
        "chunk_index",
        "chunk_count",
    ]
    matchup_fields = [
        "profile",
        "agent_deck",
        "opponent_deck",
        "skill",
        "games_requested",
        "chunks",
        "wins",
        "total",
        "losses",
        "winrate",
        "mcts_enabled",
        "mcts_activations",
        "duration_sec",
    ]

    try:
        jobs = []
        base_env = os.environ.copy()
        for entry in entries:
            train_env = entry.get("train_env") or {}
            agent_path = resolve_repo_path(train_env["RL_AGENT_DECK_LIST"])
            agent_decks = read_deck_pool(str(agent_path)) if args.split_agent_decks else [agent_path]
            for agent_deck in agent_decks:
                agent_label = agent_deck.stem
                entry_for_job = dict(entry)
                train_env_for_job = dict(train_env)
                train_env_for_job["RL_AGENT_DECK_LIST"] = str(agent_deck)
                entry_for_job["train_env"] = train_env_for_job
                for opponent in opponent_decks:
                    opponent_label = opponent.stem
                    base_key = f"{entry['profile']}__{agent_label}__vs__{opponent_label}".replace(" ", "_")
                    matchup_seed_base_key = f"{agent_label}__vs__{opponent_label}".replace(" ", "_")
                    games_per_job = args.games_per_matchup
                    if args.games_per_job > 0:
                        games_per_job = min(args.games_per_job, args.games_per_matchup)
                    chunk_count = max(1, (args.games_per_matchup + games_per_job - 1) // games_per_job)
                    remaining = args.games_per_matchup
                    chunk_index = 1
                    while remaining > 0:
                        chunk_games = min(games_per_job, remaining)
                        current_chunk_index = chunk_index
                        remaining -= chunk_games
                        chunk_index += 1
                        if (
                            selected_chunk_indices is not None
                            and current_chunk_index not in selected_chunk_indices
                        ):
                            continue
                        key = base_key if chunk_count == 1 else (
                            f"{base_key}__chunk_{current_chunk_index:03d}"
                        )
                        if args.seed_key_mode == "matchup":
                            seed_key = matchup_seed_base_key if chunk_count == 1 else (
                                f"{matchup_seed_base_key}__chunk_{current_chunk_index:03d}"
                            )
                        else:
                            seed_key = key
                        result_file = results_dir / f"{key}.csv"
                        db_dir = db_root / key
                        jobs.append(
                            {
                                "profile": str(entry["profile"]),
                                "agent_label": agent_label,
                                "opponent_label": opponent_label,
                                "opponent_path": opponent,
                                "skill": args.skill,
                                "games": chunk_games,
                                "result_file": result_file,
                                "log_file": logs_dir / f"{key}.log",
                                "db_dir": db_dir,
                                "run_dir": run_dir,
                                "timeout_sec": args.timeout_sec,
                                "mcts_enabled": args.mcts,
                                "maven_offline": args.maven_offline,
                                "compile_exec": args.compile_exec,
                                "chunk_index": current_chunk_index,
                                "chunk_count": chunk_count,
                                "env": job_env(
                                    base_env,
                                    entry_for_job,
                                    opponent,
                                    snapshot_root,
                                    run_dir,
                                    result_file,
                                    game_logs_dir / key,
                                    db_dir,
                                    chunk_games,
                                    args.skill,
                                    args.gpu_port,
                                    args.ai_threads,
                                    args.mcts,
                                    mcts_env,
                                    args.eval_game_logging,
                                    args.game_log_format,
                                    args.replay_metadata,
                                    args.replay_seed_base + stable_seed_offset(seed_key),
                                    args.live_checkpoints,
                                    args.live_checkpoint_max_per_game,
                                    args.live_checkpoint_action_types,
                                ),
                            }
                        )
        if args.limit_matchups > 0:
            jobs = jobs[: args.limit_matchups]
        if not jobs:
            raise RuntimeError("No eval jobs selected; check --profiles, --opponents, and --chunk-indices.")

        print(
            f"CP7 eval sweep {args.run_id}: {len(jobs)} jobs, "
            f"{args.games_per_matchup} games/matchup, games/job="
            f"{args.games_per_job if args.games_per_job > 0 else args.games_per_matchup}, "
            f"parallel={args.parallel}",
            flush=True,
        )

        def record_row(row: dict) -> None:
            with rows_lock:
                rows.append(row)
                write_csv(run_dir / "matchups.csv", rows, fields)
                write_csv(
                    run_dir / "profile_summary.csv",
                    aggregate_by_profile(rows),
                    ["profile", "wins", "total", "losses", "winrate"],
                )
                write_csv(run_dir / "matchup_summary.csv", aggregate_by_matchup(rows), matchup_fields)
            chunk_label = ""
            if int(row.get("chunk_count", 1) or 1) > 1:
                chunk_label = f" chunk={row['chunk_index']}/{row['chunk_count']}"
            print(
                f"{row['profile']} {row['agent_deck']} vs {row['opponent_deck']}{chunk_label}: "
                f"{row['wins']}/{row['total']} wr={row['winrate']:.3f} "
                f"mcts={row['mcts_activations']} ({row['duration_sec']:.1f}s)",
                flush=True,
            )

        warmup_count = max(0, min(args.serial_warmup_jobs, len(jobs)))
        for job in jobs[:warmup_count]:
            record_row(run_job(job))

        remaining_jobs = jobs[warmup_count:]
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            future_map = {pool.submit(run_job, job): job for job in remaining_jobs}
            for future in as_completed(future_map):
                record_row(future.result())

        summary = aggregate_by_profile(rows)
        write_csv(run_dir / "matchups.csv", rows, fields)
        write_csv(run_dir / "matchup_summary.csv", aggregate_by_matchup(rows), matchup_fields)
        write_csv(run_dir / "profile_summary.csv", summary, ["profile", "wins", "total", "losses", "winrate"])
        manifest["ended_utc"] = utc_now()
        manifest["completed_matchups"] = len(rows)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Done. Results: {run_dir}", flush=True)
        return 0
    finally:
        if gpu_proc is not None and gpu_proc.poll() is None:
            gpu_proc.terminate()
            try:
                gpu_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                gpu_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
