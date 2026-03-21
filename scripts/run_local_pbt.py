#!/usr/bin/env python3
"""Local PBT orchestrator -- runs multi-profile training with Population-Based Training
on a single machine, no Slurm required.

Usage:
    py -3.12 scripts/run_local_pbt.py

Env vars:
    TRAIN_PROFILES       Number of profiles to train (default: 4)
    NUM_GAME_RUNNERS     Runners per profile (default: 8)
    TOTAL_EPISODES       Episodes before exit (default: 1000000)
    WINRATE_WINDOW       Rolling winrate window (default: 200)
    PBT_EXPLOIT_INTERVAL Minutes between exploit attempts (default: 10)
    PBT_MIN_EPISODES     Min episodes before first exploit (default: 200)
    PBT_EPISODE_DELTA    Min episodes between exploits (default: 100)
    PBT_MIN_WINNER_GAP   Min winrate gap for exploitation (default: 0.02)
    PBT_MIN_WINNER_WR    Min winner winrate (default: 0.03)
    PBT_MUTATION_PCT     Perturbation range (default: 0.20)
    GPU_SERVICE_PORT     GPU service port (default: 26100)
    REGISTRY_PATH        Path to PBT registry JSON
"""
import csv
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MLCODE = REPO_ROOT / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "MLPythonCode"
PROFILES_ROOT = REPO_ROOT / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "profiles"
DEFAULT_REGISTRY = REPO_ROOT / "Mage.Server.Plugins" / "Mage.Player.AIRL" / "src" / "mage" / "player" / "ai" / "rl" / "league" / "pauper_spy_pbt_registry.json"


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


PBT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ENTROPY_START": (0.01, 1.0),
    "ENTROPY_END": (0.001, 0.5),
    "RL_ACTION_EPS_START": (0.0, 1.0),
    "RL_FULL_TURN_RANDOM_START": (0.0, 1.0),
    "TEMPERATURE_FLOOR": (0.05, 2.0),
    "ACTOR_LR": (1e-5, 1e-3),
}


def mutate_env_value(key: str, value: str, mutation_pct: float) -> str:
    try:
        numeric = float(str(value).strip())
    except Exception:
        return str(value)
    factor = 1.0 + ((2.0 * random.random() - 1.0) * mutation_pct)
    if factor < 0.01:
        factor = 0.01
    mutated = numeric * factor
    k = key.upper()
    bounds = PBT_BOUNDS.get(k)
    if bounds is not None:
        mutated = max(bounds[0], min(bounds[1], mutated))
    return f"{mutated:g}"


def read_winrate(profile: str, window: int) -> Tuple[int, Optional[float]]:
    """Read episode count and rolling winrate from training_stats.csv."""
    stats_path = PROFILES_ROOT / profile / "logs" / "stats" / "training_stats.csv"
    episode = 0
    values: deque = deque(maxlen=max(50, window))
    if not stats_path.exists():
        return 0, None
    try:
        with stats_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(row.get("episode", 0))
                    if ep > episode:
                        episode = ep
                except (ValueError, TypeError):
                    pass
                try:
                    wr = float(row.get("winrate", 0))
                    values.append(wr)
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    rolling = (sum(values) / len(values)) if values else None
    return episode, rolling


def copy_model_weights(src_profile: str, dst_profile: str) -> bool:
    """Copy model weights from winner to loser."""
    src_models = PROFILES_ROOT / src_profile / "models"
    dst_models = PROFILES_ROOT / dst_profile / "models"
    dst_models.mkdir(parents=True, exist_ok=True)
    copied = False
    for name in ["model_latest.pt", "model.pt", "mulligan_model.pt"]:
        src = src_models / name
        dst = dst_models / name
        if src.exists():
            shutil.copy2(str(src), str(dst))
            copied = True
    return copied


class LocalPBT:
    def __init__(self):
        self.port = env_int("GPU_SERVICE_PORT", 26100)
        self.metrics_port = self.port + 1000
        self.train_profiles = env_int("TRAIN_PROFILES", 5)
        self.num_runners = env_int("NUM_GAME_RUNNERS", 48)
        self.total_episodes = env_int("TOTAL_EPISODES", 1000000)
        self.winrate_window = env_int("WINRATE_WINDOW", 200)
        self.exploit_interval_min = env_float("PBT_EXPLOIT_INTERVAL", 30.0)
        self.min_episodes = env_int("PBT_MIN_EPISODES", 200)
        self.episode_delta = env_int("PBT_EPISODE_DELTA", 100)
        self.min_winner_gap = env_float("PBT_MIN_WINNER_GAP", 0.02)
        self.min_winner_wr = env_float("PBT_MIN_WINNER_WR", 0.03)
        self.mutation_pct = env_float("PBT_MUTATION_PCT", 0.20)
        self.registry_path = Path(os.getenv("REGISTRY_PATH", str(DEFAULT_REGISTRY)))
        self.stop_requested = False
        self.gpu_process: Optional[subprocess.Popen] = None
        self.trainer_process: Optional[subprocess.Popen] = None
        self.selected_profiles: List[Dict[str, Any]] = []
        self.last_exploit_at: Dict[str, float] = {}
        self.last_exploit_episode: Dict[str, int] = {}
        self.exploit_count: Dict[str, int] = {}
        self.eval_results: Dict[str, float] = {}

    def load_registry(self) -> List[Dict[str, Any]]:
        with self.registry_path.open("r") as f:
            entries = json.load(f)
        active = [e for e in entries if e.get("active") and e.get("train_enabled")]
        active.sort(key=lambda e: (int(e.get("priority", 1000)), str(e.get("profile", ""))))
        return active[:self.train_profiles]

    def start_gpu_service(self) -> None:
        python = sys.executable
        script = str(MLCODE / "gpu_service_host.py")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["GPU_SERVICE_PORT"] = str(self.port)
        env["GPU_SERVICE_METRICS_PORT"] = str(self.metrics_port)
        env["PY_BATCH_TIMEOUT_MS"] = os.getenv("PY_BATCH_TIMEOUT_MS", "25")
        env["PY_BATCH_MAX_SIZE"] = os.getenv("PY_BATCH_MAX_SIZE", "256")
        env["TRAIN_WORKER_THREADS"] = os.getenv("TRAIN_WORKER_THREADS", "3")
        env["SCORE_WORKER_THREADS"] = os.getenv("SCORE_WORKER_THREADS", "3")
        env["LEARNER_BATCH_MAX_EPISODES"] = os.getenv("LEARNER_BATCH_MAX_EPISODES", "64")
        env["LEARNER_BATCH_MAX_STEPS"] = os.getenv("LEARNER_BATCH_MAX_STEPS", "16384")
        log_path = REPO_ROOT / "local-training" / "local_pbt" / "gpu_service.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w")
        self.gpu_process = subprocess.Popen(
            [python, script], env=env, cwd=str(REPO_ROOT),
            stdout=log_handle, stderr=subprocess.STDOUT,
        )
        log(f"GPU service started pid={self.gpu_process.pid} port={self.port}")
        # Wait for ready
        import urllib.request
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{self.metrics_port}/metrics", timeout=2)
                log("GPU service ready")
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("GPU service failed to start")

    def start_trainer(self) -> None:
        profile_names = [str(e["profile"]) for e in self.selected_profiles]
        profiles_str = ",".join(profile_names)
        deck_paths = list({str(e["deck_path"]) for e in self.selected_profiles})
        env = dict(os.environ)
        env["PY_SERVICE_MODE"] = "shared_gpu"
        env["GPU_SERVICE_ENDPOINT"] = f"localhost:{self.port}"
        env["GPU_SERVICE_NUM_GPUS"] = "1"
        env["GPU_SERVICE_NUM_CHANNELS"] = "4"
        env["TRAIN_PROFILES_LIST"] = profiles_str
        env["MODE"] = "trainAll"
        env["NUM_GAME_RUNNERS"] = str(self.num_runners)
        env["TOTAL_EPISODES"] = str(self.total_episodes)
        env["WINRATE_WINDOW"] = str(self.winrate_window)
        env["OPPONENT_SAMPLER"] = os.getenv("OPPONENT_SAMPLER", "meta")
        env["MULLIGAN_DEVICE"] = "cpu"
        env["GAME_LOG_FREQUENCY"] = os.getenv("GAME_LOG_FREQUENCY", "100")
        env["LEAGUE_REGISTRY_PATH"] = str(self.registry_path)
        env["ORCHESTRATED_RUN"] = "1"
        if deck_paths:
            env["DECK_LIST_FILE"] = deck_paths[0]
        # Apply per-profile train_env from registry
        for entry in self.selected_profiles:
            train_env = entry.get("train_env", {})
            for k, v in train_env.items():
                env.setdefault(str(k), str(v))
        args_str = "trainAll " + " ".join(profile_names)
        cmd = (
            f'mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL '
            f'-am -DskipTests exec:java '
            f'-Dexec.mainClass=mage.player.ai.rl.RLTrainer '
            f'"-Dexec.args={args_str}"'
        )
        log_path = REPO_ROOT / "local-training" / "local_pbt" / "trainer.log"
        log_handle = log_path.open("w")
        self.trainer_process = subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=log_handle, stderr=subprocess.STDOUT, shell=True,
        )
        log(f"Trainer started pid={self.trainer_process.pid} profiles={profiles_str} runners={self.num_runners}")

    def stop_trainer(self) -> None:
        if self.trainer_process and self.trainer_process.poll() is None:
            self.trainer_process.terminate()
            try:
                self.trainer_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.trainer_process.kill()
            log("Trainer stopped")

    def restart_trainer(self, reason: str) -> None:
        log(f"Restarting trainer: {reason}")
        self.stop_trainer()
        time.sleep(2)
        self.start_trainer()

    def run_eval(self) -> Dict[str, float]:
        """Stop trainer, run eval games for each profile vs heuristic bot, return winrates."""
        log("[EVAL] Starting heuristic evaluation")
        self.stop_trainer()
        time.sleep(2)

        eval_games = env_int("EVAL_NUM_GAMES", 20)
        eval_skill = os.getenv("EVAL_OPPONENT_SKILL", "7")
        results: Dict[str, float] = {}

        for entry in self.selected_profiles:
            profile = str(entry["profile"])
            deck_path = str(entry.get("deck_path", ""))
            if not deck_path:
                continue

            env = dict(os.environ)
            env["MODE"] = "league_bench"
            env["MODEL_PROFILE"] = profile
            env["EVAL_OPPONENT_DECK"] = deck_path
            env["EVAL_OPPONENT_SKILL"] = eval_skill
            env["EVAL_NUM_GAMES"] = str(eval_games)
            env["GAME_LOG_FREQUENCY"] = "1"
            env["PY_SERVICE_MODE"] = "shared_gpu"
            env["GPU_SERVICE_ENDPOINT"] = f"localhost:{self.port}"
            env["GPU_SERVICE_NUM_GPUS"] = "1"
            env["GPU_SERVICE_NUM_CHANNELS"] = "4"
            env["MULLIGAN_DEVICE"] = "cpu"
            env["DECK_LIST_FILE"] = deck_path
            env["LEAGUE_REGISTRY_PATH"] = str(self.registry_path)

            cmd = (
                f'mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL '
                f'-am -DskipTests exec:java '
                f'-Dexec.mainClass=mage.player.ai.rl.RLTrainer '
                f'"-Dexec.args=league_bench"'
            )

            eval_timeout = env_int("EVAL_TIMEOUT_SEC", 180)
            log(f"[EVAL] {profile} vs CP7-Skill{eval_skill} ({eval_games} games, {eval_timeout}s timeout)")
            try:
                proc = subprocess.Popen(
                    cmd, env=env, cwd=str(REPO_ROOT), shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                try:
                    stdout, _ = proc.communicate(timeout=eval_timeout)
                except subprocess.TimeoutExpired:
                    # Kill entire process tree on Windows
                    import signal as _sig
                    try:
                        os.kill(proc.pid, _sig.SIGTERM)
                    except Exception:
                        pass
                    try:
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                      capture_output=True, timeout=10)
                    except Exception:
                        pass
                    proc.kill()
                    proc.wait(timeout=5)
                    log(f"[EVAL] TIMEOUT: {profile} (killed after {eval_timeout}s)")
                    results[profile] = 0.0
                    continue
                # Parse EVAL_RESULT from stdout
                wins = 0
                total = 0
                for line in (stdout or "").splitlines():
                    if "EVAL_RESULT:" in line:
                        for tok in line.split():
                            if tok.startswith("wins="):
                                wins = int(tok.split("=")[1])
                            elif tok.startswith("total="):
                                total = int(tok.split("=")[1])
                if total > 0:
                    wr = wins / total
                else:
                    wr = 0.0
                results[profile] = wr
                log(f"[EVAL] {profile}: {wins}/{total} = {wr:.3f}")
            except Exception as exc:
                log(f"[EVAL] ERROR: {profile}: {exc}")
                results[profile] = 0.0

        self.eval_results = results
        self.start_trainer()
        return results

    def check_exploit(self) -> None:
        """Run eval, then check each population group for PBT exploitation."""
        # Check timing -- only eval periodically
        now = time.time()
        any_group_due = False
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self.selected_profiles:
            group = str(entry.get("population_group", entry.get("profile", "")))
            groups.setdefault(group, []).append(entry)

        for group, entries in groups.items():
            if len(entries) < 2:
                continue
            last = self.last_exploit_at.get(group, 0)
            if (now - last) < self.exploit_interval_min * 60:
                continue
            # Check episode gate
            min_ep = min(read_winrate(str(e["profile"]), self.winrate_window)[0] for e in entries)
            last_ep = self.last_exploit_episode.get(group, 0)
            count = self.exploit_count.get(group, 0)
            if count == 0 and min_ep < self.min_episodes:
                continue
            if count > 0 and (min_ep - last_ep) < self.episode_delta:
                continue
            any_group_due = True

        if not any_group_due:
            return

        # Run eval to get heuristic winrates
        eval_wr = self.run_eval()
        if not eval_wr:
            return

        for group, entries in groups.items():
            if len(entries) < 2:
                continue

            now = time.time()
            last = self.last_exploit_at.get(group, 0)
            if (now - last) < self.exploit_interval_min * 60:
                continue

            # Gather candidates with eval winrates
            candidates = []
            for entry in entries:
                profile = str(entry["profile"])
                episode, _ = read_winrate(profile, self.winrate_window)
                wr = eval_wr.get(profile, 0.0)
                candidates.append({
                    "entry": entry,
                    "profile": profile,
                    "episode": episode,
                    "winrate": wr,
                })

            if len(candidates) < 2:
                continue

            min_ep = min(c["episode"] for c in candidates)

            # Sort by eval winrate (best first)
            candidates.sort(key=lambda c: (-c["winrate"], c["profile"]))
            winner = candidates[0]
            losers = candidates[len(candidates) // 2:]

            exploited = False
            for loser in losers:
                if loser["profile"] == winner["profile"]:
                    continue

                gap = winner["winrate"] - loser["winrate"]
                if gap < self.min_winner_gap:
                    log(f"PBT {group}: gap {gap:.3f} < {self.min_winner_gap} "
                        f"({winner['profile']}={winner['winrate']:.3f} vs {loser['profile']}={loser['winrate']:.3f}), skipping")
                    continue
                if winner["winrate"] < self.min_winner_wr:
                    log(f"PBT {group}: winner wr {winner['winrate']:.3f} < {self.min_winner_wr}, skipping")
                    continue

                # Exploit: copy weights + mutate hyperparams
                log(f"PBT EXPLOIT {group}: {winner['profile']} (wr={winner['winrate']:.3f}) -> {loser['profile']} (wr={loser['winrate']:.3f})")
                self.stop_trainer()
                time.sleep(1)

                copy_model_weights(winner["profile"], loser["profile"])

                # Mutate loser's hyperparams from winner's values
                winner_entry = winner["entry"]
                loser_entry = loser["entry"]
                mutable_keys = loser_entry.get("pbt_mutable_env", [])
                winner_env = winner_entry.get("train_env", {})
                new_env = dict(winner_env)
                for key in mutable_keys:
                    if key in new_env:
                        old_val = new_env[key]
                        new_val = mutate_env_value(key, old_val, self.mutation_pct)
                        new_env[key] = new_val
                        log(f"  {key}: {old_val} -> {new_val}")
                loser_entry["train_env"] = new_env

                # Update seed
                seed = loser_entry.get("seed", 1)
                loser_entry["seed"] = int(seed) + random.randint(1, 99999)

                exploited = True

            if exploited:
                self.last_exploit_at[group] = now
                self.last_exploit_episode[group] = min_ep
                self.exploit_count[group] = count + 1
                self.start_trainer()

    def monitor_loop(self) -> None:
        tick = 0
        while not self.stop_requested:
            time.sleep(30)
            tick += 1

            # Check trainer alive
            if self.trainer_process and self.trainer_process.poll() is not None:
                rc = self.trainer_process.returncode
                log(f"Trainer exited with rc={rc}")
                if not self.stop_requested:
                    self.restart_trainer(f"exit rc={rc}")
                continue

            # Print episode counts and eval winrates
            parts = []
            for entry in self.selected_profiles:
                profile = str(entry["profile"])
                episode, _ = read_winrate(profile, self.winrate_window)
                eval_wr = self.eval_results.get(profile)
                eval_str = f" eval:{eval_wr:.3f}" if eval_wr is not None else ""
                parts.append(f"{profile}=ep:{episode}{eval_str}")
            log(f"Progress: {', '.join(parts)}")

            # PBT exploitation check every other tick
            if tick % 2 == 0:
                self.check_exploit()

    def run(self) -> int:
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop_requested', True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop_requested', True))

        self.selected_profiles = self.load_registry()
        if not self.selected_profiles:
            log("ERROR: No active profiles in registry")
            return 1

        profile_names = [str(e["profile"]) for e in self.selected_profiles]
        groups = {}
        for e in self.selected_profiles:
            g = e.get("population_group", e.get("profile"))
            groups.setdefault(g, []).append(str(e["profile"]))
        log(f"Profiles: {profile_names}")
        log(f"Population groups: {dict(groups)}")
        log(f"Runners per profile: {self.num_runners}")
        log(f"PBT: exploit_interval={self.exploit_interval_min}min min_episodes={self.min_episodes} "
            f"episode_delta={self.episode_delta} min_gap={self.min_winner_gap} mutation={self.mutation_pct}")

        try:
            self.start_gpu_service()
            self.start_trainer()
            self.monitor_loop()
        except KeyboardInterrupt:
            log("Interrupted")
        finally:
            self.stop_trainer()
            if self.gpu_process:
                self.gpu_process.terminate()
                log("GPU service stopped")
        return 0


if __name__ == "__main__":
    sys.exit(LocalPBT().run())
