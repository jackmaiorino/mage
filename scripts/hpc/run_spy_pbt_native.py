#!/usr/bin/env python3
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[ORCH] {message}", flush=True)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def normalize_truthy(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


class TrainerState:
    def __init__(
        self,
        entry: dict,
        slot: int,
        profile: str,
        metrics_port: int,
        py4j_base_port: int,
        runners_per_profile: int,
        opponent_decklist: Path,
        command: List[str],
        env: Dict[str, str],
        stdout_handle: object,
        stderr_handle: object,
        process: subprocess.Popen,
    ) -> None:
        self.entry = entry
        self.slot = slot
        self.profile = profile
        self.metrics_port = metrics_port
        self.py4j_base_port = py4j_base_port
        self.runners_per_profile = runners_per_profile
        self.opponent_decklist = opponent_decklist
        self.command = command
        self.env = env
        self.stdout_handle = stdout_handle
        self.stderr_handle = stderr_handle
        self.process = process
        self.restart_count = 0
        self.consecutive_failures = 0
        self.completed = False


class NativeOrchestrator:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.registry_path = resolve_path(
            self.repo_root,
            os.getenv(
                "REGISTRY_PATH",
                "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json",
            ),
        )
        self.reports_root = self.repo_root / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator"
        self.trainer_logs_dir = self.reports_root / "trainers"
        self.generated_decklists_dir = self.reports_root / "generated_decklists"
        self.pbt_state_path = self.reports_root / "pbt_state.json"
        self.total_episodes = env_int("TOTAL_EPISODES", 1_000_000)
        self.train_profiles = env_int("TRAIN_PROFILES", 3)
        self.cpu_headroom = env_int("CPU_HEADROOM", 4)
        self.metrics_port_base = env_int("METRICS_PORT_BASE", 9100)
        self.py4j_base_port = env_int("PY4J_BASE_PORT", 25334)
        self.py4j_port_stride = max(1, env_int("PY4J_PORT_STRIDE", 50))
        self.poll_seconds = max(2, env_int("POLL_SECONDS", 30))
        self.restart_backoff = max(1, env_int("RESTART_BACKOFF_SECONDS", 5))
        self.restart_backoff_max = max(self.restart_backoff, env_int("RESTART_BACKOFF_MAX_SECONDS", 60))
        self.max_restart_attempts = max(1, env_int("MAX_RESTART_ATTEMPTS_PER_PROFILE", 8))
        self.trainer_stop_grace_seconds = max(0, env_int("TRAINER_STOP_GRACE_SECONDS", 10))
        self.game_log_frequency = env_int("GAME_LOG_FREQUENCY", 500)
        self.eval_every_minutes = env_int("EVAL_EVERY_MINUTES", 180)
        self.stall_restart_minutes = env_int("STALL_RESTART_MINUTES", 25)
        self.pbt_interval_minutes = env_int("PBT_EXPLOIT_INTERVAL_MINUTES", 240)
        self.pbt_first_exploit_min_ep = env_int("PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT", 12000)
        self.pbt_episode_delta = env_int("PBT_MIN_EPISODE_DELTA_PER_PROFILE", 10000)
        self.pbt_time_fallback_episode_delta = env_int("PBT_TIME_FALLBACK_MIN_EPISODE_DELTA", 3000)
        self.pbt_min_winner_gap = env_float("PBT_MIN_WINNER_GAP", 0.03)
        self.pbt_min_winner_wr = env_float("PBT_MIN_WINNER_WINRATE", 0.06)
        self.main_class = os.getenv("MAIN_CLASS", "mage.player.ai.rl.RLTrainer").strip() or "mage.player.ai.rl.RLTrainer"
        self.train_log_level = os.getenv("TRAIN_LOG_LEVEL", "").strip()
        self.league_promote_wr = os.getenv("LEAGUE_PROMOTE_WR", "0.55").strip() or "0.55"
        self.runtime_dir = self.resolve_runtime_dir()
        self.launch_mode = ""
        self.artifact_prefix: List[str] = []
        self.maven_prefix: List[str] = []
        if self.runtime_dir is not None:
            self.artifact_prefix = self.resolve_artifact_prefix()
            self.launch_mode = "artifact"
        else:
            self.maven_prefix = self.resolve_maven_prefix()
            self.launch_mode = "maven"
        self.stop_requested = False
        self.trainers: Dict[str, TrainerState] = {}

    def resolve_runtime_dir(self) -> Optional[Path]:
        raw = os.getenv("MAGE_RL_RUNTIME_DIR", "").strip()
        if not raw:
            return None
        runtime_dir = resolve_path(self.repo_root, raw)
        if not runtime_dir.exists():
            raise RuntimeError(f"MAGE_RL_RUNTIME_DIR does not exist: {runtime_dir}")
        app_dir = runtime_dir / "app"
        lib_dir = runtime_dir / "lib"
        if not app_dir.exists():
            raise RuntimeError(f"Artifact runtime is missing app directory: {app_dir}")
        if not any(app_dir.glob("*.jar")):
            raise RuntimeError(f"Artifact runtime has no app jars in: {app_dir}")
        if not lib_dir.exists():
            raise RuntimeError(f"Artifact runtime is missing lib directory: {lib_dir}")
        return runtime_dir

    def resolve_artifact_prefix(self) -> List[str]:
        if self.runtime_dir is None:
            raise RuntimeError("Artifact runtime directory is not configured")
        java_path = shutil.which("java")
        if not java_path:
            raise RuntimeError("Java executable not found on PATH for artifact launch mode")
        app_jars = sorted((self.runtime_dir / "app").glob("*.jar"))
        lib_jars = sorted((self.runtime_dir / "lib").glob("*.jar"))
        classpath_entries = [str(p) for p in app_jars + lib_jars]
        if not classpath_entries:
            raise RuntimeError(f"No jars found under artifact runtime: {self.runtime_dir}")
        classpath = os.pathsep.join(classpath_entries)
        return [java_path, "-cp", classpath, self.main_class]

    def resolve_maven_prefix(self) -> List[str]:
        raw_cmd = os.getenv("MAGE_MVN_CMD", "").strip()
        if raw_cmd:
            try:
                parsed = shlex.split(raw_cmd)
            except Exception:
                parsed = []
            if parsed:
                return parsed
        mvn_path = shutil.which("mvn")
        if mvn_path:
            return [mvn_path]
        raise RuntimeError(
            "Maven executable not found on PATH. "
            "Load a Maven module in the Slurm job (for example: module load maven)."
        )

    def load_profiles(self) -> List[dict]:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")
        with self.registry_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list):
            raise RuntimeError(f"Registry has unexpected shape: {self.registry_path}")
        active = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            profile = str(entry.get("profile", "")).strip()
            if not profile:
                continue
            if not normalize_truthy(entry.get("active", True), True):
                continue
            active.append(entry)
        active.sort(key=lambda e: (int(e.get("priority", 1000)), str(e.get("profile", ""))))
        return active

    def compute_runners_per_profile(self, profile_count: int) -> int:
        cpu_total = env_int("SLURM_CPUS_ON_NODE", os.cpu_count() or 8)
        usable = cpu_total - self.cpu_headroom
        min_usable = max(2, profile_count * 2)
        if usable < min_usable:
            usable = min_usable
        runners = usable // max(1, profile_count)
        return max(2, runners)

    def build_meta_opponent_decklist(self, active_profiles: List[dict]) -> Path:
        self.generated_decklists_dir.mkdir(parents=True, exist_ok=True)
        output = self.generated_decklists_dir / "_league_meta_opponents.decklist.txt"
        deck_paths: List[str] = []
        seen = set()
        for entry in active_profiles:
            raw = str(entry.get("deck_path", "")).strip()
            if not raw:
                continue
            resolved = resolve_path(self.repo_root, raw)
            text = str(resolved)
            if text in seen:
                continue
            seen.add(text)
            deck_paths.append(text)
        output.write_text("\n".join(deck_paths) + ("\n" if deck_paths else ""), encoding="utf-8")
        return output

    def build_command(self) -> List[str]:
        if self.launch_mode == "artifact":
            return self.artifact_prefix + ["train"]
        return self.maven_prefix + [
            "-q",
            "-pl",
            "Mage.Server.Plugins/Mage.Player.AIRL",
            "-am",
            "-DskipTests",
            "compile",
            "exec:java",
            f"-Dexec.mainClass={self.main_class}",
            "-Dexec.args=train",
        ]

    def start_trainer(self, entry: dict, slot: int, runners_per_profile: int, opponent_decklist: Path) -> TrainerState:
        profile = str(entry["profile"]).strip()
        metrics_port = self.metrics_port_base + slot
        py4j_base_port = self.py4j_base_port + (slot * self.py4j_port_stride)
        trainer_db_dir = self.repo_root / "local-training/rl-db" / profile
        trainer_db_dir.mkdir(parents=True, exist_ok=True)
        self.trainer_logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.trainer_logs_dir / f"{profile}.stdout.log"
        stderr_path = self.trainer_logs_dir / f"{profile}.stderr.log"
        stdout_handle = stdout_path.open("ab", buffering=0)
        stderr_handle = stderr_path.open("ab", buffering=0)

        env = dict(os.environ)
        env["MODE"] = "train"
        env["TOTAL_EPISODES"] = str(self.total_episodes)
        env["NUM_GAME_RUNNERS"] = str(runners_per_profile)
        env["DECK_LIST_FILE"] = str(opponent_decklist)
        env["MODEL_PROFILE"] = profile
        env["METRICS_PORT"] = str(metrics_port)
        env["OPPONENT_SAMPLER"] = "league"
        env["ORCHESTRATED_RUN"] = "1"
        env["LEAGUE_REGISTRY_PATH"] = str(self.registry_path)
        env["LEAGUE_PROMOTE_WR"] = self.league_promote_wr
        env["PY4J_BASE_PORT"] = str(py4j_base_port)
        env["PY_BRIDGE_CLEANUP"] = "0"
        env["MAGE_DB_DIR"] = str(trainer_db_dir)
        env["MAGE_DB_AUTO_SERVER"] = "false"
        if self.train_log_level:
            env["MTG_AI_LOG_LEVEL"] = self.train_log_level
        if self.game_log_frequency > 0:
            env["GAME_LOG_FREQUENCY"] = str(self.game_log_frequency)

        deck_path_raw = str(entry.get("deck_path", "")).strip()
        if deck_path_raw:
            env["RL_AGENT_DECK_LIST"] = str(resolve_path(self.repo_root, deck_path_raw))

        seed_value = entry.get("seed")
        if seed_value is not None and str(seed_value).strip():
            seed_text = str(seed_value).strip()
            env["RL_BASE_SEED"] = seed_text
            env["PY_GLOBAL_SEED"] = seed_text
            env["MULLIGAN_REPLAY_SEED"] = seed_text

        train_env = entry.get("train_env", {})
        if isinstance(train_env, dict):
            for key, value in train_env.items():
                name = str(key).strip()
                if not name:
                    continue
                env[name] = str(value)

        command = self.build_command()
        process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        log(f"Started trainer profile={profile} pid={process.pid} metricsPort={metrics_port} py4jBasePort={py4j_base_port}")
        return TrainerState(
            entry=entry,
            slot=slot,
            profile=profile,
            metrics_port=metrics_port,
            py4j_base_port=py4j_base_port,
            runners_per_profile=runners_per_profile,
            opponent_decklist=opponent_decklist,
            command=command,
            env=env,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            process=process,
        )

    def stop_trainer(self, state: TrainerState) -> None:
        proc = state.process
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception:
            return
        deadline = time.time() + self.trainer_stop_grace_seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.2)
        try:
            proc.kill()
        except Exception:
            pass

    def close_logs(self, state: TrainerState) -> None:
        for handle in (state.stdout_handle, state.stderr_handle):
            try:
                handle.flush()
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass

    def write_pbt_state_stub(self) -> None:
        self.reports_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": now_utc(),
            "mode": "native_no_pbt",
            "events": [],
        }
        self.pbt_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run(self) -> int:
        stop_script = self.repo_root / "scripts/rl-stop.sh"
        if stop_script.exists():
            log(f"Running cleanup via {stop_script} (reason=orchestrator_start)")
            try:
                subprocess.run(
                    ["bash", str(stop_script), "-q"],
                    cwd=str(self.repo_root),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                log(f"WARNING: cleanup failed: {exc}")

        active_profiles = self.load_profiles()
        trainable = [e for e in active_profiles if normalize_truthy(e.get("train_enabled", True), True)]
        if not trainable:
            log("No train_enabled profiles found in registry.")
            return 2
        selected = trainable[: max(1, self.train_profiles)]
        runners_per_profile = self.compute_runners_per_profile(len(selected))
        opponent_decklist = self.build_meta_opponent_decklist(active_profiles)
        self.write_pbt_state_stub()

        log(f"League orchestrator starting. profiles={len(active_profiles)} evalEveryMinutes={self.eval_every_minutes} runOnce=False dryRun=False")
        log(f"Registry: {self.registry_path}")
        log(
            f"PBT gating: firstExploitMinEp={self.pbt_first_exploit_min_ep} "
            f"deltaEpPerProfile={self.pbt_episode_delta} maxTimeFallbackMin={self.pbt_interval_minutes} "
            f"minGap={self.pbt_min_winner_gap:.3f} minWinnerWr={self.pbt_min_winner_wr:.3f} "
            f"timeFallbackMinEpisodeDelta={self.pbt_time_fallback_episode_delta}"
        )
        log(f"Trainer launch mode: {self.launch_mode}")
        if self.runtime_dir is not None:
            log(f"Artifact runtime dir: {self.runtime_dir}")
        log("Native orchestrator mode active; PBT exploit/copy loop is disabled in this mode.")
        log(f"Meta opponent decklist: {opponent_decklist}")
        log(f"Configured NumGameRunners per profile={runners_per_profile}")

        for idx, entry in enumerate(selected):
            state = self.start_trainer(entry, idx, runners_per_profile, opponent_decklist)
            self.trainers[state.profile] = state

        last_status = 0.0
        while not self.stop_requested:
            all_completed = True
            for profile, state in list(self.trainers.items()):
                rc = state.process.poll()
                if rc is None:
                    all_completed = False
                    continue

                if state.completed:
                    continue

                if rc == 0:
                    state.completed = True
                    log(f"Trainer completed profile={profile} pid={state.process.pid} exit=0")
                    self.close_logs(state)
                    continue

                all_completed = False
                state.restart_count += 1
                state.consecutive_failures += 1
                if state.restart_count > self.max_restart_attempts:
                    log(
                        f"CRITICAL: Trainer profile={profile} exceeded restart limit "
                        f"({state.restart_count}>{self.max_restart_attempts}); aborting orchestrator"
                    )
                    self.stop_requested = True
                    break
                backoff = min(self.restart_backoff_max, self.restart_backoff * state.restart_count)
                log(
                    f"WARNING: Trainer exited profile={profile} pid={state.process.pid}; "
                    f"restarting reason=exit code={rc} count={state.restart_count} backoff={backoff}s"
                )
                self.close_logs(state)
                time.sleep(backoff)
                replacement = self.start_trainer(
                    state.entry,
                    state.slot,
                    state.runners_per_profile,
                    state.opponent_decklist,
                )
                replacement.restart_count = state.restart_count
                replacement.consecutive_failures = state.consecutive_failures
                replacement.completed = False
                self.trainers[profile] = replacement

            if self.stop_requested:
                break

            now = time.time()
            if now - last_status >= 60:
                running = [name for name, t in self.trainers.items() if t.process.poll() is None]
                log(f"Concurrent training profiles running ({len(running)}): {', '.join(running) if running else '(none)'}")
                last_status = now

            if all_completed and self.trainers:
                log("All trainers completed successfully.")
                break
            time.sleep(self.poll_seconds)

        for state in self.trainers.values():
            self.stop_trainer(state)
            self.close_logs(state)
        return 130 if self.stop_requested else 0


def main() -> int:
    try:
        orchestrator = NativeOrchestrator()
    except Exception as exc:
        log(f"FATAL: {exc}")
        return 1

    def _signal_handler(signum, _frame):
        log(f"Signal received: {signum}; shutting down")
        orchestrator.stop_requested = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        return orchestrator.run()
    except Exception as exc:
        log(f"FATAL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
