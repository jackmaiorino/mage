#!/usr/bin/env python3
import json
import math
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
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


def parse_iso_utc(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def parse_int64(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def to_string_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    out: List[str] = []
    try:
        for item in value:  # type: ignore[assignment]
            text = str(item).strip()
            if text:
                out.append(text)
    except Exception:
        pass
    return out


class TrainerState:
    def __init__(
        self,
        entry: Dict[str, Any],
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
        effective_train_env: Dict[str, str],
        effective_seed: Optional[int],
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
        self.effective_train_env = dict(effective_train_env)
        self.effective_seed = effective_seed
        self.restart_count = 0
        self.consecutive_failures = 0
        self.completed = False
        self.last_restart_reason = ""
        self.last_progress_episode = 0
        self.last_progress_at = time.time()
        self.launched_at_utc = now_utc()


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
        self.shared_gpu_logs_dir = self.reports_root / "gpu_hosts"
        self.generated_decklists_dir = self.reports_root / "generated_decklists"
        self.pbt_state_path = self.reports_root / "pbt_state.json"
        self.orchestrator_status_path = self.reports_root / "orchestrator_status.json"
        self.total_episodes = env_int("TOTAL_EPISODES", 1_000_000)
        self.train_profiles = env_int("TRAIN_PROFILES", 3)
        self.cpu_headroom = env_int("CPU_HEADROOM", 4)
        self.runner_oversubscription_factor = max(0.1, env_float("RUNNER_OVERSUBSCRIPTION_FACTOR", 1.0))
        self.metrics_port_base = env_int("METRICS_PORT_BASE", 9100)
        self.py_service_mode = os.getenv("PY_SERVICE_MODE", "local").strip().lower() or "local"
        self.gpu_service_bind_host = os.getenv("GPU_SERVICE_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"
        self.gpu_service_port_base = env_int("GPU_SERVICE_PORT_BASE", 26100)
        self.gpu_service_metrics_port_base = env_int("GPU_SERVICE_METRICS_PORT_BASE", 27100)
        self.py4j_base_port = env_int("PY4J_BASE_PORT", 25334)
        self.py4j_port_stride = max(1, env_int("PY4J_PORT_STRIDE", 50))
        self.py4j_kill_stale_on_start = env_bool("PY4J_KILL_STALE_ON_START", True)
        self.py4j_stale_kill_grace_seconds = max(1, env_int("PY4J_STALE_KILL_GRACE_SECONDS", 5))
        self.poll_seconds = max(2, env_int("POLL_SECONDS", 30))
        self.restart_backoff = max(1, env_int("RESTART_BACKOFF_SECONDS", 5))
        self.restart_backoff_max = max(self.restart_backoff, env_int("RESTART_BACKOFF_MAX_SECONDS", 60))
        self.max_restart_attempts = max(1, env_int("MAX_RESTART_ATTEMPTS_PER_PROFILE", 8))
        self.trainer_stop_grace_seconds = max(0, env_int("TRAINER_STOP_GRACE_SECONDS", 10))
        self.trainer_start_stagger_seconds = max(0, env_int("TRAINER_START_STAGGER_SECONDS", 20))
        self.visible_gpu_list = self.detect_visible_gpu_list()
        self.visible_gpu_count = max(1, len(self.visible_gpu_list))
        self.trainer_start_wave_size = max(1, env_int("TRAINER_START_WAVE_SIZE", self.visible_gpu_count))
        self.trainer_start_intra_wave_delay_ms = max(0, env_int("TRAINER_START_INTRA_WAVE_DELAY_MS", 250))
        self.game_log_frequency = env_int("GAME_LOG_FREQUENCY", 0 if self.py_service_mode == "shared_gpu" else 500)
        self.eval_every_minutes = env_int("EVAL_EVERY_MINUTES", 180)
        self.stall_restart_minutes = env_int("STALL_RESTART_MINUTES", 25)
        self.pbt_interval_minutes = env_int("PBT_EXPLOIT_INTERVAL_MINUTES", 240)
        self.pbt_first_exploit_min_ep = env_int("PBT_MIN_EPISODES_BEFORE_FIRST_EXPLOIT", 12000)
        self.pbt_episode_delta = env_int("PBT_MIN_EPISODE_DELTA_PER_PROFILE", 10000)
        self.pbt_time_fallback_episode_delta = env_int("PBT_TIME_FALLBACK_MIN_EPISODE_DELTA", 3000)
        self.pbt_min_population = max(2, env_int("PBT_MIN_POPULATION_SIZE", 3))
        self.pbt_mutation_pct = max(0.0, env_float("PBT_MUTATION_PCT", 0.20))
        self.pbt_min_winner_gap = env_float("PBT_MIN_WINNER_GAP", 0.03)
        self.pbt_min_winner_wr = env_float("PBT_MIN_WINNER_WINRATE", 0.06)
        self.winrate_window = max(10, env_int("WINRATE_WINDOW", 200))
        self.enable_pbt = env_bool("ENABLE_PBT", True)
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
        self.shared_gpu_hosts: Dict[int, Dict[str, Any]] = {}
        self.pbt_events: List[Dict[str, Any]] = []
        self.pbt_group_state: Dict[str, Dict[str, Any]] = {}
        self.last_pbt_at: Optional[datetime] = None
        self.selected_profiles: List[Dict[str, Any]] = []
        self.active_profiles: List[Dict[str, Any]] = []

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

    def load_profiles(self) -> List[Dict[str, Any]]:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")
        with self.registry_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list):
            raise RuntimeError(f"Registry has unexpected shape: {self.registry_path}")

        active: List[Dict[str, Any]] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            profile = str(item.get("profile", "")).strip()
            if not profile:
                continue
            if not normalize_truthy(item.get("active", True), True):
                continue
            train_enabled = normalize_truthy(item.get("train_enabled", True), True)
            population_group = str(item.get("population_group", "")).strip() or profile
            priority = parse_int64(item.get("priority"))
            if priority is None:
                priority = 1000
            target_wr = item.get("target_winrate", 0.60)
            try:
                target_wr_num = float(target_wr)
            except Exception:
                target_wr_num = 0.60
            entry = dict(item)
            entry["profile"] = profile
            entry["train_enabled"] = train_enabled
            entry["population_group"] = population_group
            entry["priority"] = int(priority)
            entry["target_winrate"] = target_wr_num
            if not isinstance(entry.get("train_env"), dict):
                entry["train_env"] = {}
            entry["pbt_mutable_env"] = to_string_list(entry.get("pbt_mutable_env"))
            active.append(entry)
        active.sort(key=lambda e: (int(e.get("priority", 1000)), str(e.get("profile", ""))))
        return active

    def compute_runners_per_profile(self, profile_count: int) -> int:
        cpu_total = env_int("SLURM_CPUS_ON_NODE", os.cpu_count() or 8)
        usable = max(0, cpu_total - self.cpu_headroom)
        min_total_runners = max(2, profile_count * 2)
        target_total_runners = int(math.floor(float(usable) * self.runner_oversubscription_factor))
        if target_total_runners < min_total_runners:
            target_total_runners = min_total_runners
        runners = target_total_runners // max(1, profile_count)
        return max(2, runners)

    def detect_visible_gpu_list(self) -> List[str]:
        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_vis:
            items = [g.strip() for g in cuda_vis.split(",") if g.strip()]
            return items or ["0"]
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return ["0"]
        count = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
        if count <= 0:
            return ["0"]
        return [str(i) for i in range(count)]

    def detect_visible_gpu_count(self) -> int:
        return max(1, len(self.detect_visible_gpu_list()))

    def metrics_port_ready(self, metrics_port: int) -> bool:
        url = f"http://127.0.0.1:{metrics_port}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                return int(getattr(response, "status", 200)) == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            return False

    def wait_for_startup_readiness(self, states: List[TrainerState]) -> None:
        timeout_seconds = float(self.trainer_start_stagger_seconds)
        if timeout_seconds <= 0 or not states:
            return

        pending: Dict[str, TrainerState] = {state.profile: state for state in states}
        deadline = time.time() + timeout_seconds
        while pending and time.time() < deadline and not self.stop_requested:
            for profile, state in list(pending.items()):
                rc = state.process.poll()
                if rc is not None:
                    pending.pop(profile, None)
                    continue
                if self.metrics_port_ready(state.metrics_port):
                    pending.pop(profile, None)
            if pending:
                time.sleep(0.5)

        if pending:
            names = ", ".join(sorted(pending.keys()))
            log(
                f"WARNING: Startup readiness wait timed out after {timeout_seconds:g}s "
                f"for profiles={names}"
            )

    def resolve_profile_deck_path(self, entry: Dict[str, Any]) -> Optional[Path]:
        raw = str(entry.get("deck_path", "")).strip()
        if not raw:
            return None
        candidate = resolve_path(self.repo_root, raw).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    def build_meta_opponent_decklist(self, active_profiles: List[Dict[str, Any]]) -> Path:
        self.generated_decklists_dir.mkdir(parents=True, exist_ok=True)
        output = self.generated_decklists_dir / "_league_meta_opponents.decklist.txt"
        deck_paths: List[str] = []
        seen = set()
        for entry in active_profiles:
            resolved = self.resolve_profile_deck_path(entry)
            if resolved is None:
                log(f"WARNING: Profile {entry.get('profile','')} has missing deck_path; skipping for meta opponent list")
                continue
            text = str(resolved)
            if text in seen:
                continue
            seen.add(text)
            deck_paths.append(text)
        if not deck_paths:
            raise RuntimeError("No usable deck paths found to build meta opponent decklist")
        output.write_text("\n".join(deck_paths) + "\n", encoding="utf-8")
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

    def resolve_python_executable(self) -> str:
        venv_path = os.getenv("MTG_VENV_PATH", "").strip()
        if venv_path:
            candidate = Path(venv_path) / "bin" / "python"
            if candidate.exists():
                return str(candidate)
        return shutil.which("python3") or "python3"

    def shared_gpu_host_ready(self, port: int) -> bool:
        try:
            with socket.create_connection((self.gpu_service_bind_host, port), timeout=1.0):
                return True
        except OSError:
            return False

    def launch_shared_gpu_hosts(self) -> None:
        if self.py_service_mode != "shared_gpu":
            return
        if self.shared_gpu_hosts:
            return
        python_bin = self.resolve_python_executable()
        host_script = self.repo_root / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_host.py"
        if not host_script.exists():
            raise RuntimeError(f"shared GPU host script missing: {host_script}")
        self.shared_gpu_logs_dir.mkdir(parents=True, exist_ok=True)
        for gpu_slot, gpu_id in enumerate(self.visible_gpu_list):
            port = self.gpu_service_port_base + gpu_slot
            metrics_port = self.gpu_service_metrics_port_base + gpu_slot
            stdout_path = self.shared_gpu_logs_dir / f"gpu_{gpu_slot}.stdout.log"
            stderr_path = self.shared_gpu_logs_dir / f"gpu_{gpu_slot}.stderr.log"
            stdout_handle = stdout_path.open("ab", buffering=0)
            stderr_handle = stderr_path.open("ab", buffering=0)
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["PY_SERVICE_MODE"] = "shared_gpu"
            env["GPU_SERVICE_PORT"] = str(port)
            env["GPU_SERVICE_METRICS_PORT"] = str(metrics_port)
            env["GPU_SERVICE_BIND_HOST"] = self.gpu_service_bind_host
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env.setdefault("MULLIGAN_DEVICE", "cpu")
            process = subprocess.Popen(
                [python_bin, str(host_script)],
                cwd=str(self.repo_root),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            self.shared_gpu_hosts[gpu_slot] = {
                "gpu_id": str(gpu_id),
                "port": port,
                "metrics_port": metrics_port,
                "process": process,
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
            }
            log(
                f"Started shared GPU host slot={gpu_slot} gpu={gpu_id} "
                f"pid={process.pid} port={port} metricsPort={metrics_port}"
            )

        deadline = time.time() + max(10, self.trainer_start_stagger_seconds)
        pending = set(self.shared_gpu_hosts.keys())
        while pending and time.time() < deadline and not self.stop_requested:
            for gpu_slot in list(pending):
                host = self.shared_gpu_hosts[gpu_slot]
                process = host["process"]
                if process.poll() is not None:
                    raise RuntimeError(
                        f"shared GPU host slot={gpu_slot} exited early with rc={process.returncode}"
                    )
                if self.shared_gpu_host_ready(int(host["port"])):
                    pending.discard(gpu_slot)
            if pending:
                time.sleep(0.25)
        if pending:
            raise RuntimeError(f"shared GPU hosts not ready before timeout: slots={sorted(pending)}")

    def stop_shared_gpu_hosts(self) -> None:
        for host in self.shared_gpu_hosts.values():
            process = host.get("process")
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
        deadline = time.time() + max(3, self.trainer_stop_grace_seconds)
        while time.time() < deadline:
            if all(host.get("process") is None or host["process"].poll() is not None for host in self.shared_gpu_hosts.values()):
                break
            time.sleep(0.2)
        for host in self.shared_gpu_hosts.values():
            process = host.get("process")
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass
            for key in ("stdout_handle", "stderr_handle"):
                handle = host.get(key)
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
        self.shared_gpu_hosts.clear()

    def start_trainer(
        self,
        entry: Dict[str, Any],
        slot: int,
        runners_per_profile: int,
        opponent_decklist: Path,
        effective_train_env: Optional[Dict[str, str]] = None,
        effective_seed: Optional[int] = None,
    ) -> TrainerState:
        profile = str(entry["profile"]).strip()
        metrics_port = self.metrics_port_base + slot
        py4j_base_port = self.py4j_base_port + (slot * self.py4j_port_stride)
        if self.py_service_mode != "shared_gpu":
            self.cleanup_stale_py4j_port(profile, py4j_base_port)
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

        resolved_deck = self.resolve_profile_deck_path(entry)
        if resolved_deck is not None:
            env["RL_AGENT_DECK_LIST"] = str(resolved_deck)

        train_env: Dict[str, str] = {}
        if isinstance(entry.get("train_env"), dict):
            for key, value in entry.get("train_env", {}).items():
                name = str(key).strip()
                if not name:
                    continue
                train_env[name] = str(value)
        if effective_train_env:
            for key, value in effective_train_env.items():
                name = str(key).strip()
                if not name:
                    continue
                train_env[name] = str(value)
        for key, value in train_env.items():
            env[key] = value

        seed_now = effective_seed
        if seed_now is None:
            seed_now = parse_int64(entry.get("seed"))
        if seed_now is not None:
            seed_text = str(seed_now)
            env["RL_BASE_SEED"] = seed_text
            env["PY_GLOBAL_SEED"] = seed_text
            env["MULLIGAN_REPLAY_SEED"] = seed_text

        gpu_list = list(self.visible_gpu_list)
        gpu_slot = slot % max(1, len(gpu_list))
        if self.py_service_mode == "shared_gpu":
            env["PY_SERVICE_MODE"] = "shared_gpu"
            env["GPU_SERVICE_ENDPOINT"] = f"{self.gpu_service_bind_host}:{self.gpu_service_port_base + gpu_slot}"
            env["GPU_SERVICE_METRICS_ENDPOINT"] = (
                f"http://{self.gpu_service_bind_host}:{self.gpu_service_metrics_port_base + gpu_slot}/metrics"
            )
            env["MULLIGAN_DEVICE"] = env.get("MULLIGAN_DEVICE", "cpu")
            if not env.get("LEAGUE_TICK_EPISODES", "").strip():
                env["LEAGUE_TICK_EPISODES"] = "0"
            if not env.get("LADDER_TICK_EPISODES", "").strip():
                env["LADDER_TICK_EPISODES"] = "0"
            if gpu_list:
                env["CUDA_VISIBLE_DEVICES"] = gpu_list[gpu_slot]
        elif len(gpu_list) > 1:
            env["CUDA_VISIBLE_DEVICES"] = gpu_list[gpu_slot]

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
            effective_train_env=train_env,
            effective_seed=seed_now,
        )

    @staticmethod
    def is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    def find_stale_py4j_pids(self, py4j_port: int) -> List[int]:
        pattern = f"py4j_entry_point.py --port {py4j_port}"
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", pattern],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return []

        pids: List[int] = []
        for raw in out.splitlines():
            text = raw.strip()
            if not text:
                continue
            try:
                pid = int(text)
            except Exception:
                continue
            if pid == os.getpid() or pid <= 0:
                continue
            pids.append(pid)
        return sorted(set(pids))

    def cleanup_stale_py4j_port(self, profile: str, py4j_port: int) -> None:
        if not self.py4j_kill_stale_on_start:
            return

        stale_pids = self.find_stale_py4j_pids(py4j_port)
        if not stale_pids:
            return

        pid_text = ",".join(str(pid) for pid in stale_pids)
        log(
            f"WARNING: Found stale py4j_entry_point process(es) "
            f"profile={profile} port={py4j_port} pids={pid_text}; terminating before restart"
        )

        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        deadline = time.time() + float(self.py4j_stale_kill_grace_seconds)
        while time.time() < deadline:
            remaining = [pid for pid in stale_pids if self.is_pid_alive(pid)]
            if not remaining:
                return
            time.sleep(0.2)

        remaining = [pid for pid in stale_pids if self.is_pid_alive(pid)]
        if not remaining:
            return

        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        remaining_after = [pid for pid in remaining if self.is_pid_alive(pid)]
        if remaining_after:
            log(
                f"WARNING: Could not kill all stale py4j processes "
                f"profile={profile} port={py4j_port} pids={','.join(str(pid) for pid in remaining_after)}"
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

    def profile_models_dir(self, profile: str) -> Path:
        return self.repo_root / f"Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{profile}/models"

    def profile_model_latest_path(self, profile: str) -> Path:
        return self.profile_models_dir(profile) / "model_latest.pt"

    def profile_model_path(self, profile: str) -> Path:
        return self.profile_models_dir(profile) / "model.pt"

    def copy_profile_latest_model(self, source_profile: str, target_profile: str) -> bool:
        source_latest = self.profile_model_latest_path(source_profile)
        source_model = self.profile_model_path(source_profile)
        source = source_latest if source_latest.exists() else source_model
        if not source.exists():
            return False
        target_latest = self.profile_model_latest_path(target_profile)
        target_model = self.profile_model_path(target_profile)
        target_latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target_latest))
        shutil.copy2(str(target_latest), str(target_model))
        return True

    def mutate_env_value(self, key: str, value: str) -> str:
        try:
            numeric = float(str(value).strip())
        except Exception:
            return str(value)
        pct = max(0.0, self.pbt_mutation_pct)
        factor = 1.0 + ((2.0 * random.random() - 1.0) * pct)
        if factor < 0.01:
            factor = 0.01
        mutated = numeric * factor
        k = key.upper() if key else ""
        if ("EPS" in k) or ("PROB" in k) or ("RATE" in k) or ("FRAC" in k) or k.endswith("_P") or k.startswith("P_"):
            mutated = max(0.0, min(1.0, mutated))
        return f"{mutated:g}"

    def get_profile_training_snapshot(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        profile = str(entry.get("profile", "")).strip()
        stats_path = self.repo_root / f"Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{profile}/logs/stats/training_stats.csv"
        status_path = self.repo_root / f"Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/{profile}/logs/league/agent_status.json"

        episode = 0
        baseline_wr = 0.0
        promoted = False

        if status_path.exists():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                ep = parse_int64(payload.get("episode"))
                if ep is not None:
                    episode = max(episode, int(ep))
                baseline_wr = float(payload.get("baseline_wr", baseline_wr))
                promoted = bool(payload.get("promoted", promoted))
            except Exception:
                pass

        values: Deque[float] = deque(maxlen=max(50, self.winrate_window))
        if stats_path.exists():
            try:
                with stats_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for raw in handle:
                        line = raw.strip()
                        if not line or line.startswith("episode,"):
                            continue
                        parts = line.split(",")
                        if len(parts) < 5:
                            continue
                        ep = parse_int64(parts[0].strip())
                        if ep is not None and int(ep) > episode:
                            episode = int(ep)
                        try:
                            values.append(float(parts[4].strip()))
                        except Exception:
                            continue
            except Exception:
                pass

        rolling_current = values[-1] if values else None
        rolling_avg = (sum(values) / len(values)) if values else None
        return {
            "profile": profile,
            "episode": int(episode),
            "rolling_current": rolling_current,
            "rolling_avg": rolling_avg,
            "sample_count": len(values),
            "baseline_wr": baseline_wr,
            "promoted": promoted,
            "target_winrate": float(entry.get("target_winrate", 0.60)),
            "train_enabled": bool(entry.get("train_enabled", True)),
        }

    def update_stall_progress(self, state: TrainerState, snapshots: Dict[str, Dict[str, Any]]) -> None:
        snap = snapshots.get(state.profile)
        if snap is None:
            return
        episode = int(snap.get("episode", 0))
        if episode > state.last_progress_episode:
            state.last_progress_episode = episode
            state.last_progress_at = time.time()

    def restart_trainer(self, state: TrainerState, reason: str, backoff_seconds: int) -> Optional[TrainerState]:
        profile = state.profile
        entry = state.entry
        if reason == "exit":
            log(
                f"WARNING: Trainer exited profile={profile} pid={state.process.pid}; "
                f"restarting reason=exit count={state.restart_count} backoff={backoff_seconds}s"
            )
        elif reason == "stall":
            log(
                f"WARNING: Trainer stalled profile={profile} pid={state.process.pid}; "
                f"restarting reason=stall count={state.restart_count} backoff={backoff_seconds}s"
            )
        self.stop_trainer(state)
        self.close_logs(state)
        if backoff_seconds > 0:
            time.sleep(backoff_seconds)
        replacement = self.start_trainer(
            entry,
            state.slot,
            state.runners_per_profile,
            state.opponent_decklist,
            effective_train_env=state.effective_train_env,
            effective_seed=state.effective_seed,
        )
        replacement.restart_count = state.restart_count
        replacement.consecutive_failures = state.consecutive_failures
        replacement.last_restart_reason = reason
        replacement.last_progress_episode = state.last_progress_episode
        replacement.last_progress_at = time.time()
        return replacement

    def _append_pbt_event(self, event: Dict[str, Any]) -> None:
        self.pbt_events.append(event)
        if len(self.pbt_events) > 5000:
            self.pbt_events = self.pbt_events[-2000:]

    def invoke_pbt_exploit(self, snapshots: Dict[str, Dict[str, Any]], now_dt: datetime) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not self.enable_pbt:
            return events

        group_map: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self.selected_profiles:
            if not bool(entry.get("train_enabled", True)):
                continue
            profile = str(entry.get("profile", "")).strip()
            if profile not in self.trainers:
                continue
            group = str(entry.get("population_group", "")).strip() or profile
            group_map.setdefault(group, []).append(entry)

        for group, entries in group_map.items():
            if len(entries) < max(2, self.pbt_min_population):
                continue

            group_episodes: List[int] = []
            episodes_ready = True
            for entry in entries:
                snap = snapshots.get(str(entry.get("profile", "")))
                if snap is None or snap.get("episode") is None:
                    episodes_ready = False
                    break
                group_episodes.append(int(snap.get("episode", 0)))
            if not episodes_ready or len(group_episodes) < len(entries):
                continue
            group_min_episode = min(group_episodes)

            group_state = self.pbt_group_state.setdefault(
                group,
                {
                    "last_exploit_utc": "",
                    "last_exploit_min_episode": 0,
                    "exploit_count": 0,
                },
            )
            exploit_count = int(group_state.get("exploit_count", 0))
            last_exploit_min_episode = int(group_state.get("last_exploit_min_episode", 0))
            last_exploit_at = parse_iso_utc(str(group_state.get("last_exploit_utc", "")))

            first_exploit_gate = max(0, int(self.pbt_first_exploit_min_ep))
            delta_gate = max(1, int(self.pbt_episode_delta))
            time_delta_gate = max(1, int(self.pbt_time_fallback_episode_delta))
            episode_delta_since_last = group_min_episode - last_exploit_min_episode

            if exploit_count <= 0:
                due_by_episode = group_min_episode >= first_exploit_gate
            else:
                due_by_episode = episode_delta_since_last >= delta_gate

            due_by_time = False
            time_delta_gate_passed = False
            if last_exploit_at is None:
                elapsed_minutes = float("inf")
            else:
                elapsed_minutes = (now_dt - last_exploit_at).total_seconds() / 60.0
            if self.pbt_interval_minutes > 0:
                time_delta_gate_passed = episode_delta_since_last >= time_delta_gate
                due_by_time = (
                    group_min_episode >= first_exploit_gate
                    and elapsed_minutes >= max(1, self.pbt_interval_minutes)
                    and time_delta_gate_passed
                )

            if not (due_by_episode or due_by_time):
                continue
            trigger = "episode" if due_by_episode else "time"

            candidates: List[Dict[str, Any]] = []
            for entry in entries:
                profile = str(entry.get("profile", "")).strip()
                snap = snapshots.get(profile)
                if snap is None:
                    continue
                wr = snap.get("rolling_current")
                if wr is None:
                    continue
                if profile not in self.trainers:
                    continue
                candidates.append({"entry": entry, "snapshot": snap})
            if len(candidates) < max(2, self.pbt_min_population):
                continue

            ordered = sorted(
                candidates,
                key=lambda c: (-float(c["snapshot"]["rolling_current"]), str(c["entry"]["profile"])),
            )
            if len(ordered) < 2:
                continue

            winner = ordered[0]
            loser_count = max(1, len(ordered) // 2)
            losers = ordered[-loser_count:]
            did_exploit_group = False

            for loser in losers:
                winner_profile = str(winner["entry"]["profile"])
                loser_profile = str(loser["entry"]["profile"])
                if winner_profile == loser_profile:
                    continue
                state = self.trainers.get(loser_profile)
                if state is None:
                    continue

                winner_wr = float(winner["snapshot"]["rolling_current"])
                loser_wr = float(loser["snapshot"]["rolling_current"])
                winner_gap = winner_wr - loser_wr
                gap_gate_passed = winner_gap >= self.pbt_min_winner_gap
                winner_wr_gate_passed = winner_wr >= self.pbt_min_winner_wr

                if not (gap_gate_passed and winner_wr_gate_passed):
                    if not gap_gate_passed and not winner_wr_gate_passed:
                        skip_reason = "gap_and_winner_wr_below_threshold"
                    elif not gap_gate_passed:
                        skip_reason = "gap_below_threshold"
                    else:
                        skip_reason = "winner_wr_below_threshold"
                    event = {
                        "timestamp_utc": now_utc(),
                        "population_group": group,
                        "trigger": trigger,
                        "group_min_episode": group_min_episode,
                        "episode_delta": episode_delta_since_last,
                        "elapsed_minutes": -1.0 if math.isinf(elapsed_minutes) else float(elapsed_minutes),
                        "winner": winner_profile,
                        "loser": loser_profile,
                        "winner_wr": winner_wr,
                        "loser_wr": loser_wr,
                        "winner_gap": winner_gap,
                        "winner_gap_min": float(self.pbt_min_winner_gap),
                        "winner_wr_min_gate": float(self.pbt_min_winner_wr),
                        "gap_gate_passed": gap_gate_passed,
                        "winner_wr_gate_passed": winner_wr_gate_passed,
                        "time_delta_gate": int(time_delta_gate),
                        "time_delta_gate_passed": bool(time_delta_gate_passed),
                        "skip_reason": skip_reason,
                        "copied": False,
                        "new_seed": "",
                        "mutated_keys": "",
                    }
                    events.append(event)
                    self._append_pbt_event(event)
                    continue

                copied = False
                try:
                    copied = self.copy_profile_latest_model(winner_profile, loser_profile)
                except Exception as exc:
                    copied = False
                    log(f"WARNING: PBT copy failed winner={winner_profile} loser={loser_profile}: {exc}")

                effective_env = dict(state.effective_train_env)
                if not effective_env and isinstance(loser["entry"].get("train_env"), dict):
                    for key, value in loser["entry"].get("train_env", {}).items():
                        name = str(key).strip()
                        if name:
                            effective_env[name] = str(value)

                mutable_keys = to_string_list(loser["entry"].get("pbt_mutable_env"))
                for env_key in mutable_keys:
                    if env_key in effective_env:
                        effective_env[env_key] = self.mutate_env_value(env_key, effective_env[env_key])

                seed_now = state.effective_seed
                if seed_now is None:
                    seed_now = parse_int64(loser["entry"].get("seed"))
                if seed_now is None:
                    seed_now = 1
                seed_now = int(seed_now) + random.randint(1, 99999)
                state.effective_seed = seed_now
                state.effective_train_env = dict(effective_env)

                try:
                    self.stop_trainer(state)
                    self.close_logs(state)
                    time.sleep(0.25)
                    replacement = self.start_trainer(
                        loser["entry"],
                        state.slot,
                        state.runners_per_profile,
                        state.opponent_decklist,
                        effective_train_env=effective_env,
                        effective_seed=seed_now,
                    )
                    replacement.restart_count = state.restart_count
                    replacement.consecutive_failures = state.consecutive_failures
                    replacement.last_restart_reason = "pbt_replace"
                    replacement.last_progress_episode = state.last_progress_episode
                    replacement.last_progress_at = time.time()
                    self.trainers[loser_profile] = replacement
                except Exception as exc:
                    log(f"WARNING: PBT restart failed loser={loser_profile}: {exc}")

                event = {
                    "timestamp_utc": now_utc(),
                    "population_group": group,
                    "trigger": trigger,
                    "group_min_episode": group_min_episode,
                    "episode_delta": episode_delta_since_last,
                    "elapsed_minutes": -1.0 if math.isinf(elapsed_minutes) else float(elapsed_minutes),
                    "winner": winner_profile,
                    "loser": loser_profile,
                    "winner_wr": winner_wr,
                    "loser_wr": loser_wr,
                    "winner_gap": winner_gap,
                    "winner_gap_min": float(self.pbt_min_winner_gap),
                    "winner_wr_min_gate": float(self.pbt_min_winner_wr),
                    "gap_gate_passed": gap_gate_passed,
                    "winner_wr_gate_passed": winner_wr_gate_passed,
                    "time_delta_gate": int(time_delta_gate),
                    "time_delta_gate_passed": bool(time_delta_gate_passed),
                    "skip_reason": "",
                    "copied": bool(copied),
                    "new_seed": str(seed_now),
                    "mutated_keys": ";".join(mutable_keys),
                }
                events.append(event)
                self._append_pbt_event(event)
                did_exploit_group = True

            if did_exploit_group:
                group_state["last_exploit_utc"] = now_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                group_state["last_exploit_min_episode"] = int(group_min_episode)
                group_state["exploit_count"] = int(exploit_count) + 1
                self.pbt_group_state[group] = group_state

        return events

    def write_pbt_state(self) -> None:
        self.reports_root.mkdir(parents=True, exist_ok=True)

        profiles_payload: List[Dict[str, Any]] = []
        for entry in self.selected_profiles:
            profile = str(entry.get("profile", "")).strip()
            state = self.trainers.get(profile)
            seed_now: Optional[int] = None
            train_env_now: Dict[str, str] = {}
            if state is not None:
                seed_now = state.effective_seed
                train_env_now = dict(state.effective_train_env)
            if seed_now is None:
                seed_now = parse_int64(entry.get("seed"))
            if not train_env_now and isinstance(entry.get("train_env"), dict):
                for key, value in entry.get("train_env", {}).items():
                    name = str(key).strip()
                    if name:
                        train_env_now[name] = str(value)
            profiles_payload.append(
                {
                    "profile": profile,
                    "population_group": str(entry.get("population_group", "")).strip() or profile,
                    "seed": seed_now,
                    "pbt_mutable_env": to_string_list(entry.get("pbt_mutable_env")),
                    "train_env": train_env_now,
                }
            )

        group_payload: List[Dict[str, Any]] = []
        for key, st in self.pbt_group_state.items():
            group_payload.append(
                {
                    "population_group": str(key),
                    "last_exploit_utc": str(st.get("last_exploit_utc", "")),
                    "last_exploit_min_episode": int(st.get("last_exploit_min_episode", 0)),
                    "exploit_count": int(st.get("exploit_count", 0)),
                }
            )

        payload = {
            "updated_at_utc": now_utc(),
            "mode": "native_full_pbt",
            "enable_pbt": bool(self.enable_pbt),
            "pbt_exploit_max_interval_minutes": int(self.pbt_interval_minutes),
            "pbt_min_episodes_before_first_exploit": int(self.pbt_first_exploit_min_ep),
            "pbt_min_episode_delta_per_profile": int(self.pbt_episode_delta),
            "pbt_time_fallback_min_episode_delta": int(self.pbt_time_fallback_episode_delta),
            "pbt_mutation_pct": float(self.pbt_mutation_pct),
            "pbt_min_winner_gap": float(self.pbt_min_winner_gap),
            "pbt_min_winner_winrate": float(self.pbt_min_winner_wr),
            "last_exploit_utc": "" if self.last_pbt_at is None else self.last_pbt_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "profiles": profiles_payload,
            "group_state": group_payload,
            "events": self.pbt_events[-200:],
        }
        self.pbt_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_orchestrator_status(self, snapshots: Dict[str, Dict[str, Any]], note: str) -> None:
        self.reports_root.mkdir(parents=True, exist_ok=True)

        trainer_rows: List[Dict[str, Any]] = []
        for entry in self.active_profiles:
            profile = str(entry.get("profile", "")).strip()
            if not profile:
                continue
            state = self.trainers.get(profile)
            proc = state.process if state is not None else None
            running = False
            pid = 0
            exit_code: Optional[int] = None
            if proc is not None:
                try:
                    rc = proc.poll()
                    running = rc is None
                    pid = int(proc.pid)
                    if rc is not None:
                        exit_code = int(rc)
                except Exception:
                    pass
            stdout_log = ""
            stderr_log = ""
            if state is not None:
                stdout_log = str(self.trainer_logs_dir / f"{profile}.stdout.log")
                stderr_log = str(self.trainer_logs_dir / f"{profile}.stderr.log")
            trainer_rows.append(
                {
                    "profile": profile,
                    "running": running,
                    "pid": pid,
                    "restart_count": 0 if state is None else int(state.restart_count),
                    "consecutive_failures": 0 if state is None else int(state.consecutive_failures),
                    "last_restart_reason": "" if state is None else str(state.last_restart_reason),
                    "launched_at_utc": "" if state is None else str(state.launched_at_utc),
                    "metrics_port": 0 if state is None else int(state.metrics_port),
                    "py4j_base_port": 0 if state is None else int(state.py4j_base_port),
                    "train_decklist": "" if state is None else str(state.env.get("RL_AGENT_DECK_LIST", "")),
                    "opponent_decklist": "" if state is None else str(state.opponent_decklist),
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                    "exit_code": exit_code,
                }
            )

        snapshot_rows: List[Dict[str, Any]] = []
        for entry in self.active_profiles:
            profile = str(entry.get("profile", "")).strip()
            if not profile:
                continue
            snap = snapshots.get(profile)
            if snap is None:
                continue
            state = self.trainers.get(profile)
            completed = False
            if state is not None:
                completed = bool(state.completed)
            snapshot_rows.append(
                {
                    "profile": profile,
                    "episode": int(snap.get("episode", 0)),
                    "rolling_winrate": snap.get("rolling_current"),
                    "rolling_avg": snap.get("rolling_avg"),
                    "baseline_wr": float(snap.get("baseline_wr", 0.0)),
                    "target_winrate": float(snap.get("target_winrate", 0.60)),
                    "promoted": bool(snap.get("promoted", False)),
                    "train_enabled": bool(snap.get("train_enabled", True)),
                    "completed": completed,
                }
            )

        payload = {
            "updated_at_utc": now_utc(),
            "registry_path": str(self.registry_path),
            "eval_every_minutes": int(self.eval_every_minutes),
            "last_eval_utc": "",
            "next_eval_utc": "",
            "dry_run": False,
            "sequential_training": False,
            "current_training_profile": "",
            "note": note,
            "trainers": trainer_rows,
            "profile_snapshots": snapshot_rows,
        }
        self.orchestrator_status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_status(self, snapshots: Dict[str, Dict[str, Any]]) -> None:
        running = [name for name, t in self.trainers.items() if t.process.poll() is None]
        log(f"Concurrent training profiles running ({len(running)}): {', '.join(running) if running else '(none)'}")
        rows: List[str] = []
        for entry in self.selected_profiles:
            profile = str(entry.get("profile", "")).strip()
            snap = snapshots.get(profile)
            if snap is None:
                continue
            wr = snap.get("rolling_current")
            ep = int(snap.get("episode", 0))
            if wr is None:
                rows.append(f"{profile}=ep:{ep} wr:n/a")
            else:
                rows.append(f"{profile}=ep:{ep} wr:{float(wr):.3f}")
        if rows:
            log("Rolling winrates: " + ", ".join(rows))

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
        self.active_profiles = list(active_profiles)
        trainable = [e for e in active_profiles if normalize_truthy(e.get("train_enabled", True), True)]
        if not trainable:
            log("No train_enabled profiles found in registry.")
            return 2
        self.selected_profiles = trainable[: max(1, self.train_profiles)]
        runners_per_profile = self.compute_runners_per_profile(len(self.selected_profiles))
        opponent_decklist = self.build_meta_opponent_decklist(active_profiles)

        if len(self.selected_profiles) < 2:
            self.enable_pbt = False
        if len(self.selected_profiles) < self.pbt_min_population:
            self.enable_pbt = False

        log(f"League orchestrator starting. profiles={len(active_profiles)} evalEveryMinutes={self.eval_every_minutes} runOnce=False dryRun=False")
        log(f"Registry: {self.registry_path}")
        log(
            f"PBT gating: firstExploitMinEp={self.pbt_first_exploit_min_ep} "
            f"deltaEpPerProfile={self.pbt_episode_delta} maxTimeFallbackMin={self.pbt_interval_minutes} "
            f"minGap={self.pbt_min_winner_gap:.3f} minWinnerWr={self.pbt_min_winner_wr:.3f} "
            f"timeFallbackMinEpisodeDelta={self.pbt_time_fallback_episode_delta} "
            f"enabled={self.enable_pbt}"
        )
        log(f"Trainer launch mode: {self.launch_mode}")
        if self.runtime_dir is not None:
            log(f"Artifact runtime dir: {self.runtime_dir}")
        log(f"Python service mode: {self.py_service_mode}")
        log(f"Meta opponent decklist: {opponent_decklist}")
        total_runner_target = runners_per_profile * len(self.selected_profiles)
        log(
            f"Runner sizing: cpuTotal={env_int('SLURM_CPUS_ON_NODE', os.cpu_count() or 8)} "
            f"cpuHeadroom={self.cpu_headroom} runnerOversubscriptionFactor={self.runner_oversubscription_factor:g} "
            f"targetTotalRunners={total_runner_target}"
        )
        log(f"Configured NumGameRunners per profile={runners_per_profile}")
        log(f"Trainer startup max wait seconds={self.trainer_start_stagger_seconds}")
        log(
            f"Trainer startup waves: visibleGpus={self.visible_gpu_count} "
            f"waveSize={self.trainer_start_wave_size} intraWaveDelayMs={self.trainer_start_intra_wave_delay_ms}"
        )

        if self.py_service_mode == "shared_gpu":
            self.launch_shared_gpu_hosts()

        startup_wave: List[TrainerState] = []
        for idx, entry in enumerate(self.selected_profiles):
            state = self.start_trainer(entry, idx, runners_per_profile, opponent_decklist)
            self.trainers[state.profile] = state
            startup_wave.append(state)

            is_last = idx >= (len(self.selected_profiles) - 1)
            wave_full = len(startup_wave) >= self.trainer_start_wave_size
            if wave_full or is_last:
                self.wait_for_startup_readiness(startup_wave)
                startup_wave = []
            elif self.trainer_start_intra_wave_delay_ms > 0:
                time.sleep(self.trainer_start_intra_wave_delay_ms / 1000.0)

        initial_snapshots: Dict[str, Dict[str, Any]] = {}
        for entry in self.active_profiles:
            profile = str(entry.get("profile", "")).strip()
            if profile:
                initial_snapshots[profile] = self.get_profile_training_snapshot(entry)
        self.write_pbt_state()
        self.write_orchestrator_status(initial_snapshots, note=f"training_concurrent_{len(self.selected_profiles)}")

        last_status = 0.0
        last_snapshots: Dict[str, Dict[str, Any]] = dict(initial_snapshots)
        while not self.stop_requested:
            snapshots: Dict[str, Dict[str, Any]] = {}
            for entry in self.active_profiles:
                profile = str(entry.get("profile", "")).strip()
                snapshots[profile] = self.get_profile_training_snapshot(entry)
            last_snapshots = snapshots

            if self.py_service_mode == "shared_gpu":
                for gpu_slot, host in list(self.shared_gpu_hosts.items()):
                    process = host.get("process")
                    if process is not None and process.poll() is not None:
                        log(
                            f"CRITICAL: shared GPU host slot={gpu_slot} gpu={host.get('gpu_id')} "
                            f"exited rc={process.returncode}; aborting orchestrator"
                        )
                        self.stop_requested = True
                        break
                if self.stop_requested:
                    break

            all_completed = True
            now_ts = time.time()
            for profile, state in list(self.trainers.items()):
                rc = state.process.poll()
                if rc is None:
                    all_completed = False
                    self.update_stall_progress(state, snapshots)
                    if self.stall_restart_minutes > 0:
                        stalled_minutes = (now_ts - state.last_progress_at) / 60.0
                        if stalled_minutes >= float(self.stall_restart_minutes):
                            state.restart_count += 1
                            state.consecutive_failures += 1
                            if state.restart_count > self.max_restart_attempts:
                                log(
                                    f"CRITICAL: Trainer profile={profile} exceeded restart limit "
                                    f"({state.restart_count}>{self.max_restart_attempts}); aborting orchestrator"
                                )
                                self.stop_requested = True
                                break
                            backoff = int(min(self.restart_backoff_max, self.restart_backoff * max(1, state.restart_count)))
                            replacement = self.restart_trainer(state, "stall", backoff)
                            if replacement is not None:
                                self.trainers[profile] = replacement
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
                backoff = int(min(self.restart_backoff_max, self.restart_backoff * max(1, state.restart_count)))
                replacement = self.restart_trainer(state, "exit", backoff)
                if replacement is not None:
                    self.trainers[profile] = replacement

            if self.stop_requested:
                break

            now_dt = datetime.now(timezone.utc)
            events = self.invoke_pbt_exploit(snapshots, now_dt)
            if events:
                self.last_pbt_at = now_dt
                for ev in events:
                    skip_reason = str(ev.get("skip_reason", "")).strip()
                    if skip_reason:
                        log(
                            "PBT exploit skipped "
                            f"group={ev.get('population_group')} winner={ev.get('winner')} loser={ev.get('loser')} "
                            f"reason={skip_reason} winner_wr={float(ev.get('winner_wr', 0.0)):.3f} "
                            f"loser_wr={float(ev.get('loser_wr', 0.0)):.3f} gap={float(ev.get('winner_gap', 0.0)):.3f} "
                            f"trigger={ev.get('trigger')} groupMinEp={int(ev.get('group_min_episode', 0))} "
                            f"deltaEp={int(ev.get('episode_delta', 0))}"
                        )
                    else:
                        log(
                            "PBT exploit "
                            f"group={ev.get('population_group')} winner={ev.get('winner')} loser={ev.get('loser')} "
                            f"winner_wr={float(ev.get('winner_wr', 0.0)):.3f} loser_wr={float(ev.get('loser_wr', 0.0)):.3f} "
                            f"gap={float(ev.get('winner_gap', 0.0)):.3f} seed={ev.get('new_seed')} trigger={ev.get('trigger')} "
                            f"groupMinEp={int(ev.get('group_min_episode', 0))} deltaEp={int(ev.get('episode_delta', 0))}"
                        )
                self.write_pbt_state()
                self.write_orchestrator_status(snapshots, note=f"training_concurrent_{len(self.selected_profiles)}")

            now = time.time()
            if now - last_status >= 60:
                self.log_status(snapshots)
                self.write_pbt_state()
                self.write_orchestrator_status(snapshots, note=f"training_concurrent_{len(self.selected_profiles)}")
                last_status = now

            if all_completed and self.trainers:
                log("All trainers completed successfully.")
                break
            time.sleep(self.poll_seconds)

        for state in self.trainers.values():
            self.stop_trainer(state)
            self.close_logs(state)
        self.stop_shared_gpu_hosts()
        self.write_pbt_state()
        self.write_orchestrator_status(last_snapshots, note="stopping")
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
    finally:
        try:
            orchestrator.stop_shared_gpu_hosts()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
