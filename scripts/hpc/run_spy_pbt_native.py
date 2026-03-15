#!/usr/bin/env python3
import csv
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
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


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


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def normalize_mode(value: object) -> str:
    return str(value or "").strip().lower()


def sanitize_run_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = now_utc().replace(":", "").replace("-", "")
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("._")
    return safe or "run"


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}")
    try:
        temp_path.write_text(text, encoding=encoding)
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2), encoding="utf-8")


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
        # Multi-node: auxiliary JVM processes on CPU nodes
        self.aux_processes: List[Dict[str, Any]] = []


class NativeOrchestrator:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.source_repo_root = resolve_path(
            self.repo_root,
            os.getenv("SOURCE_REPO_ROOT", "").strip() or str(self.repo_root),
        )
        self.mode = "native_full_pbt"
        default_run_id = os.getenv("SLURM_JOB_ID", "").strip() or now_utc().replace(":", "").replace("-", "")
        self.run_id = sanitize_run_id(os.getenv("ORCH_RUN_ID", "").strip() or default_run_id)
        self.registry_path = resolve_path(
            self.repo_root,
            os.getenv(
                "REGISTRY_PATH",
                "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json",
            ),
        )
        self.compat_reports_root = resolve_path(
            self.repo_root,
            os.getenv("ORCHESTRATOR_COMPAT_REPORTS_ROOT", "").strip()
            or "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator",
        )
        self.runs_root = self.compat_reports_root / "runs"
        self.reports_root = self.runs_root / self.run_id
        self.trainer_logs_dir = self.reports_root / "trainers"
        self.shared_gpu_logs_dir = self.reports_root / "gpu_hosts"
        self.generated_decklists_dir = self.reports_root / "generated_decklists"
        self.pbt_state_path = self.reports_root / "pbt_state.json"
        self.orchestrator_status_path = self.reports_root / "orchestrator_status.json"
        self.compat_pbt_state_path = self.compat_reports_root / "pbt_state.json"
        self.compat_orchestrator_status_path = self.compat_reports_root / "orchestrator_status.json"
        self.latest_run_path = self.compat_reports_root / "latest_run.json"
        self.rl_artifacts_root = resolve_path(
            self.repo_root,
            os.getenv("RL_ARTIFACTS_ROOT", "").strip()
            or "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl",
        )
        self.db_root = resolve_path(
            self.repo_root,
            os.getenv("MAGE_DB_DIR", "").strip() or "local-training/rl-db",
        )
        self.total_episodes = env_int("TOTAL_EPISODES", 1_000_000)
        self.train_profiles = env_int("TRAIN_PROFILES", 3)
        self.cpu_headroom = env_int("CPU_HEADROOM", 4)
        self.runner_oversubscription_factor = max(0.1, env_float("RUNNER_OVERSUBSCRIPTION_FACTOR", 1.0))
        self.metrics_port_base = env_int("METRICS_PORT_BASE", 9100)
        self.py_service_mode = os.getenv("PY_SERVICE_MODE", "local").strip().lower() or "local"
        self.gpu_service_bind_host = os.getenv("GPU_SERVICE_BIND_HOST", "0.0.0.0").strip() or "0.0.0.0"
        self.league_mode = os.getenv("LEAGUE_MODE", "").strip().lower()
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
        self.gpu_service_startup_timeout_seconds = max(
            10,
            env_int("GPU_SERVICE_STARTUP_TIMEOUT_SECONDS", max(60, self.trainer_start_stagger_seconds)),
        )
        self.visible_gpu_list = self.detect_visible_gpu_list()

        # Eval benchmark config
        self.eval_results: Dict[str, float] = {}
        self.eval_num_games = env_int("EVAL_NUM_GAMES_PER_OPPONENT", 50)
        self.eval_opponents: List[Dict[str, str]] = [
            {
                "deck": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono Red Rally.dek",
                "skill": "1",
                "label": "Rally-CP7",
            },
            {
                "deck": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono-Blue Terror.dek",
                "skill": "1",
                "label": "Terror-CP7",
            },
        ]

        # Multi-node discovery
        self.gpu_node = ""
        self.cpu_nodes: List[str] = []
        # Het-group jobs expose per-component nodelists
        het0 = os.getenv("SLURM_JOB_NODELIST_HET_GROUP_0", "").strip()
        het1 = os.getenv("SLURM_JOB_NODELIST_HET_GROUP_1", "").strip()
        if het0 and het1:
            gpu_nodes = self._expand_slurm_nodelist(het0)
            cpu_nodes = self._expand_slurm_nodelist(het1)
            if gpu_nodes:
                self.gpu_node = gpu_nodes[0]
            self.cpu_nodes = cpu_nodes
            log(f"Het-group multi-node: gpu_node={self.gpu_node} cpu_nodes={self.cpu_nodes}")
        else:
            nodelist = os.getenv("SLURM_JOB_NODELIST", "").strip()
            if nodelist:
                expanded = self._expand_slurm_nodelist(nodelist)
                if len(expanded) >= 2:
                    self.gpu_node = expanded[0]
                    self.cpu_nodes = expanded[1:]
                    log(f"Multi-node: gpu_node={self.gpu_node} cpu_nodes={self.cpu_nodes}")
                elif len(expanded) == 1:
                    self.gpu_node = expanded[0]
                    log(f"Single-node: gpu_node={self.gpu_node}")
        self.visible_gpu_count = max(1, len(self.visible_gpu_list))
        self.trainer_start_wave_size = max(1, env_int("TRAINER_START_WAVE_SIZE", self.visible_gpu_count))
        self.trainer_start_intra_wave_delay_ms = max(0, env_int("TRAINER_START_INTRA_WAVE_DELAY_MS", 250))
        self.game_log_frequency = env_int("GAME_LOG_FREQUENCY", 0 if self.py_service_mode == "shared_gpu" else 500)
        self.slurm_mem_per_node_mb = max(0, env_int("SLURM_MEM_PER_NODE", 0))
        self.trainer_jvm_xms_mb = max(0, env_int("TRAINER_JVM_XMS_MB", 512))
        self.trainer_jvm_xmx_mb = max(0, env_int("TRAINER_JVM_XMX_MB", 0))
        self.trainer_jvm_heap_reserve_mb = max(
            0,
            env_int("TRAINER_JVM_HEAP_RESERVE_MB", 16384 if self.py_service_mode == "shared_gpu" else 8192),
        )
        self.trainer_jvm_heap_fraction = min(
            0.95,
            max(0.10, env_float("TRAINER_JVM_HEAP_FRACTION", 0.65 if self.py_service_mode == "shared_gpu" else 0.75)),
        )
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
        self.multi_profile_jvm = env_bool("MULTI_PROFILE_JVM", False)
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
        self.last_eval_at: Optional[datetime] = datetime.now(timezone.utc)  # defer first eval by pbt_interval_minutes
        self.selected_profiles: List[Dict[str, Any]] = []
        self.active_profiles: List[Dict[str, Any]] = []
        self.snapshot_warning_keys: Set[str] = set()

    def resolve_runtime_dir(self) -> Optional[Path]:
        raw = os.getenv("MAGE_RL_RUNTIME_DIR", "").strip()
        if not raw:
            return None
        runtime_dir = resolve_path(self.source_repo_root, raw)
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
        if not app_jars and not lib_jars:
            raise RuntimeError(f"No jars found under artifact runtime: {self.runtime_dir}")
        classpath_entries = [str(p) for p in app_jars + lib_jars]
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

    def write_latest_run_pointer(self) -> None:
        payload = {
            "updated_at_utc": now_utc(),
            "run_id": self.run_id,
            "mode": self.mode,
            "reports_root": str(self.reports_root),
            "orchestrator_status_path": str(self.orchestrator_status_path),
            "pbt_state_path": str(self.pbt_state_path),
            "compat_reports_root": str(self.compat_reports_root),
            "artifacts_root": str(self.rl_artifacts_root),
        }
        atomic_write_json(self.latest_run_path, payload)

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
            mode = normalize_mode(item.get("mode"))
            train_enabled = normalize_truthy(item.get("train_enabled", True), True)
            if mode in {"frozen", "eval_only"}:
                train_enabled = False
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
            entry["mode"] = mode
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
        cpu_total = env_int("GAME_CPU_CORES", env_int("SLURM_CPUS_ON_NODE", os.cpu_count() or 8))
        usable = max(0, cpu_total - self.cpu_headroom)
        min_total_runners = max(2, profile_count * 2)
        target_total_runners = int(math.floor(float(usable) * self.runner_oversubscription_factor))
        if target_total_runners < min_total_runners:
            target_total_runners = min_total_runners
        runners = target_total_runners // max(1, profile_count)
        return max(2, runners)

    @staticmethod
    def java_opts_include_heap(text: str) -> bool:
        value = str(text or "")
        return any(
            marker in value
            for marker in (
                "-Xmx",
                "-Xms",
                "-XX:MaxRAMPercentage",
                "-XX:InitialRAMPercentage",
                "-XX:MaxRAMFraction",
            )
        )

    def compute_trainer_jvm_xmx_mb(self) -> int:
        if self.trainer_jvm_xmx_mb > 0:
            return self.trainer_jvm_xmx_mb
        if self.slurm_mem_per_node_mb <= 0:
            return 0
        available_mb = max(2048, self.slurm_mem_per_node_mb - self.trainer_jvm_heap_reserve_mb)
        heap_pool_mb = max(2048, int(math.floor(float(available_mb) * self.trainer_jvm_heap_fraction)))
        return max(2048, heap_pool_mb // max(1, self.train_profiles))

    def apply_trainer_java_tool_options(self, env: Dict[str, str]) -> int:
        existing = str(env.get("JAVA_TOOL_OPTIONS", "")).strip()
        if self.java_opts_include_heap(existing):
            return 0
        xmx_mb = self.compute_trainer_jvm_xmx_mb()
        if xmx_mb <= 0:
            return 0
        xms_mb = min(max(0, self.trainer_jvm_xms_mb), xmx_mb)
        auto_parts: List[str] = []
        if xms_mb > 0:
            auto_parts.append(f"-Xms{xms_mb}m")
        auto_parts.append(f"-Xmx{xmx_mb}m")
        env["JAVA_TOOL_OPTIONS"] = " ".join(auto_parts + ([existing] if existing else []))
        env["TRAINER_JVM_XMX_MB"] = str(xmx_mb)
        return xmx_mb

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

    @staticmethod
    def _expand_slurm_nodelist(nodelist: str) -> List[str]:
        """Expand Slurm compact nodelist like 'gpu-a6-[5,7]' into individual hostnames."""
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()
        except Exception:
            pass
        return [h.strip() for h in nodelist.split(",") if h.strip()]

    def run_eval_benchmark(self) -> Dict[str, float]:
        """Stop trainers, run eval games for each profile, return {profile: winrate}."""
        log("[EVAL] Starting eval benchmark")

        # Stop all trainers
        for profile in list(self.trainers.keys()):
            state = self.trainers.get(profile)
            if state:
                log(f"[EVAL] Stopping trainer {profile} for eval benchmark")
                self.stop_trainer(state)
        self.trainers.clear()
        time.sleep(2)

        results: Dict[str, float] = {}
        for entry in self.load_profiles():
            profile = str(entry.get("profile", "")).strip()
            if not profile:
                continue

            total_wins = 0
            total_games = 0
            for opp in self.eval_opponents:
                eval_results_file = str(
                    self.profile_models_dir(profile) / f"eval_{opp['label']}.csv"
                )
                env = dict(os.environ)
                env["MODE"] = "league_bench"
                env["MODEL_PROFILE"] = profile
                env["EVAL_OPPONENT_DECK"] = opp["deck"]
                env["EVAL_OPPONENT_SKILL"] = opp["skill"]
                env["EVAL_NUM_GAMES"] = str(self.eval_num_games)
                env["EVAL_RESULTS_FILE"] = eval_results_file
                env["RL_ARTIFACTS_ROOT"] = str(self.rl_artifacts_root)

                if self.py_service_mode == "shared_gpu":
                    gpu_slot = 0
                    endpoint_host = self.gpu_node if self.gpu_node else self.gpu_service_bind_host
                    env["PY_SERVICE_MODE"] = "shared_gpu"
                    env["GPU_SERVICE_ENDPOINT"] = (
                        f"{endpoint_host}:{self.gpu_service_port_base + gpu_slot}"
                    )

                command = self.build_command()
                # If multi-node, run eval on first CPU node
                if self.cpu_nodes:
                    command = [
                        "srun", "--overlap", f"--nodelist={self.cpu_nodes[0]}",
                        "--ntasks=1", "--cpus-per-task=1",
                    ] + command

                log(f"[EVAL] Running {profile} vs {opp['label']} ({self.eval_num_games} games)")

                try:
                    proc = subprocess.run(
                        command,
                        cwd=str(self.source_repo_root),
                        env=env,
                        timeout=600,
                        capture_output=True,
                        text=True,
                    )
                    if os.path.isfile(eval_results_file):
                        with open(eval_results_file) as f:
                            parts = f.read().strip().split(",")
                            if len(parts) >= 3:
                                total_wins += int(parts[0])
                                total_games += int(parts[1])
                    else:
                        for line in (proc.stdout or "").splitlines():
                            if line.startswith("EVAL_RESULT:"):
                                for tok in line.split():
                                    if tok.startswith("wins="):
                                        total_wins += int(tok.split("=")[1])
                                    elif tok.startswith("total="):
                                        total_games += int(tok.split("=")[1])
                except subprocess.TimeoutExpired:
                    log(f"[EVAL] TIMEOUT: {profile} vs {opp['label']}")
                except Exception as exc:
                    log(f"[EVAL] ERROR: {profile} vs {opp['label']}: {exc}")

            winrate = total_wins / max(1, total_games)
            results[profile] = winrate
            log(f"[EVAL] {profile}: {total_wins}/{total_games} = {winrate:.3f}")

        self.eval_results = results

        # Append to eval CSV
        eval_csv = self.run_dir / "eval_results.csv"
        write_header = not eval_csv.exists()
        try:
            with open(eval_csv, "a") as f:
                if write_header:
                    f.write("timestamp," + ",".join(sorted(results.keys())) + "\n")
                ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                vals = ",".join(f"{results.get(k, 0.0):.4f}" for k in sorted(results.keys()))
                f.write(f"{ts},{vals}\n")
        except Exception as exc:
            log(f"[EVAL] Failed to write eval CSV: {exc}")

        log(f"[EVAL] Benchmark complete: {results}")
        return results

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
        candidate = resolve_path(self.source_repo_root, raw).resolve()
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
        atomic_write_text(output, "\n".join(deck_paths) + "\n", encoding="utf-8")
        return output

    def build_command(self, mode: str = "train") -> List[str]:
        if self.launch_mode == "artifact":
            return self.artifact_prefix + [mode]
        return self.maven_prefix + [
            "-q",
            "-pl",
            "Mage.Server.Plugins/Mage.Player.AIRL",
            "-am",
            "-DskipTests",
            "compile",
            "exec:java",
            f"-Dexec.mainClass={self.main_class}",
            f"-Dexec.args={mode}",
        ]

    def build_multi_profile_command(self, profile_names: List[str]) -> List[str]:
        args = "trainAll " + " ".join(profile_names)
        if self.launch_mode == "artifact":
            return self.artifact_prefix + args.split()
        return self.maven_prefix + [
            "-q",
            "-pl",
            "Mage.Server.Plugins/Mage.Player.AIRL",
            "-am",
            "-DskipTests",
            "compile",
            "exec:java",
            f"-Dexec.mainClass={self.main_class}",
            f'-Dexec.args={args}',
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
        host_script = self.source_repo_root / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_host.py"
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
            env["PYTHON_LOGS_DIR"] = str(self.shared_gpu_python_logs_dir(gpu_slot))
            env["MTG_AI_LOG_FILE"] = str(self.shared_gpu_python_logs_dir(gpu_slot) / "mtg_ai.log")
            env["MULLIGAN_TRAINING_LOG_FILE"] = str(
                self.shared_gpu_python_logs_dir(gpu_slot) / "mulligan_training.log"
            )
            env["MULLIGAN_TRACE_JSONL_FILE"] = str(
                self.shared_gpu_python_logs_dir(gpu_slot) / "mulligan_trace.jsonl"
            )
            env["VRAM_DIAGNOSTICS_LOG_FILE"] = str(
                self.shared_gpu_python_logs_dir(gpu_slot) / "VRAM_diagnostics.log"
            )
            env.setdefault("MULLIGAN_DEVICE", "cpu")
            process = subprocess.Popen(
                [python_bin, str(host_script)],
                cwd=str(self.source_repo_root),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            self.shared_gpu_hosts[gpu_slot] = {
                "gpu_id": str(gpu_id),
                "port": port,
                "metrics_port": metrics_port,
                "process": process,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
            }
            log(
                f"Started shared GPU host slot={gpu_slot} gpu={gpu_id} "
                f"pid={process.pid} port={port} metricsPort={metrics_port}"
            )

        deadline = time.time() + self.gpu_service_startup_timeout_seconds
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
        trainer_db_dir = self.db_root / profile
        trainer_db_dir.mkdir(parents=True, exist_ok=True)
        self.profile_models_dir(profile).mkdir(parents=True, exist_ok=True)
        self.profile_logs_dir(profile).mkdir(parents=True, exist_ok=True)
        self.python_logs_dir(profile).mkdir(parents=True, exist_ok=True)
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
        env["RL_ARTIFACTS_ROOT"] = str(self.rl_artifacts_root)
        env["RL_MODELS_DIR"] = str(self.profile_models_dir(profile))
        env["RL_LOGS_DIR"] = str(self.profile_logs_dir(profile))
        env["PYTHON_LOGS_DIR"] = str(self.python_logs_dir(profile))
        env["MTG_AI_LOG_FILE"] = str(self.python_logs_dir(profile) / "mtg_ai.log")
        env["MULLIGAN_TRAINING_LOG_FILE"] = str(self.python_logs_dir(profile) / "mulligan_training.log")
        env["MULLIGAN_TRACE_JSONL_FILE"] = str(self.python_logs_dir(profile) / "mulligan_trace.jsonl")
        env["VRAM_DIAGNOSTICS_LOG_FILE"] = str(self.python_logs_dir(profile) / "VRAM_diagnostics.log")
        env["CANDIDATE_EXPLOSIONS_LOG_FILE"] = str(self.profile_logs_dir(profile) / "health" / "candidate_explosions.log")
        env["METRICS_PORT"] = str(metrics_port)
        env["OPPONENT_SAMPLER"] = "league"
        if self.league_mode:
            env["LEAGUE_MODE"] = self.league_mode
        env["ORCHESTRATED_RUN"] = "1"
        env["ORCH_RUN_ID"] = self.run_id
        env["LEAGUE_REGISTRY_PATH"] = str(self.registry_path)
        env["LEAGUE_PROMOTE_WR"] = self.league_promote_wr
        env["PY4J_BASE_PORT"] = str(py4j_base_port)
        env["PY_BRIDGE_CLEANUP"] = "0"
        env["MAGE_DB_DIR"] = str(trainer_db_dir)
        env["MAGE_DB_AUTO_SERVER"] = "false"
        env["ORCHESTRATOR_REPORTS_ROOT"] = str(self.reports_root)
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

        trainer_jvm_xmx_mb = self.apply_trainer_java_tool_options(env)

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
            endpoint_host = self.gpu_node if self.gpu_node else self.gpu_service_bind_host
            env["GPU_SERVICE_ENDPOINT"] = f"{endpoint_host}:{self.gpu_service_port_base + gpu_slot}"
            env["GPU_SERVICE_METRICS_ENDPOINT"] = (
                f"http://{endpoint_host}:{self.gpu_service_metrics_port_base + gpu_slot}/metrics"
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
        if self.cpu_nodes:
            cpu_alloc = env_int("GAME_CPU_CORES", 128)
            command = [
                "srun", "--het-group=1", "--overlap", f"--nodelist={self.cpu_nodes[0]}",
                "--ntasks=1", f"--cpus-per-task={cpu_alloc}",
            ] + command
        process = subprocess.Popen(
            command,
            cwd=str(self.source_repo_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        if trainer_jvm_xmx_mb > 0:
            log(
                f"Started trainer profile={profile} pid={process.pid} metricsPort={metrics_port} "
                f"py4jBasePort={py4j_base_port} jvmXmxMb={trainer_jvm_xmx_mb}"
            )
        else:
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

    def start_multi_profile_trainer(
        self,
        entries: List[Dict[str, Any]],
        runners_per_profile: int,
        opponent_decklist: Path,
    ) -> TrainerState:
        """Launch a single JVM that trains all profiles (card DB loaded once)."""
        profile_names = [str(e.get("profile", "")).strip() for e in entries]
        profiles_csv = ",".join(profile_names)
        log(f"Multi-profile JVM: profiles={profiles_csv} runners_per_profile={runners_per_profile}")

        # Multi-node: primary JVM uses GPU node CPUs; auxiliaries use compute node CPUs
        if self.cpu_nodes:
            gpu_cpu_total = env_int("SLURM_CPUS_ON_NODE", os.cpu_count() or 8)
            gpu_usable = max(0, gpu_cpu_total - self.cpu_headroom)
            primary_total = max(len(entries) * 2, int(math.floor(gpu_usable * self.runner_oversubscription_factor)))
            aux_total = runners_per_profile * len(entries)
            log(f"Multi-node runners: primary={primary_total} (from {gpu_cpu_total} GPU-node CPUs) aux={aux_total} (from GAME_CPU_CORES/default)")
        else:
            primary_total = runners_per_profile * len(entries)
            aux_total = 0
        metrics_port = self.metrics_port_base
        self.trainer_logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.trainer_logs_dir / "multi_profile.stdout.log"
        stderr_path = self.trainer_logs_dir / "multi_profile.stderr.log"
        stdout_handle = stdout_path.open("ab", buffering=0)
        stderr_handle = stderr_path.open("ab", buffering=0)

        env = dict(os.environ)
        env["MODE"] = "trainAll"
        env["TRAIN_PROFILES_LIST"] = profiles_csv
        env["TOTAL_EPISODES"] = str(self.total_episodes)
        env["NUM_GAME_RUNNERS"] = str(primary_total)
        env["DECK_LIST_FILE"] = str(opponent_decklist)
        env["RL_ARTIFACTS_ROOT"] = str(self.rl_artifacts_root)
        env["METRICS_PORT"] = str(metrics_port)
        env["OPPONENT_SAMPLER"] = "league"
        if self.league_mode:
            env["LEAGUE_MODE"] = self.league_mode
        env["ORCHESTRATED_RUN"] = "1"
        env["ORCH_RUN_ID"] = self.run_id
        env["LEAGUE_REGISTRY_PATH"] = str(self.registry_path)
        env["LEAGUE_PROMOTE_WR"] = self.league_promote_wr
        env["MAGE_DB_DIR"] = str(self.db_root / "shared")
        env["MAGE_DB_AUTO_SERVER"] = "false"
        if self.train_log_level:
            env["MTG_AI_LOG_LEVEL"] = self.train_log_level
        if self.game_log_frequency > 0:
            env["GAME_LOG_FREQUENCY"] = str(self.game_log_frequency)

        if self.py_service_mode == "shared_gpu":
            env["PY_SERVICE_MODE"] = "shared_gpu"
            endpoint_host = self.gpu_node if self.gpu_node else self.gpu_service_bind_host
            env["GPU_SERVICE_ENDPOINT"] = f"{endpoint_host}:{self.gpu_service_port_base}"
            num_gpus = len(self.visible_gpu_list)
            env["GPU_SERVICE_NUM_GPUS"] = str(num_gpus)
            env["GPU_SERVICE_NUM_CHANNELS"] = str(max(4, num_gpus))
            env["GPU_SERVICE_METRICS_ENDPOINT"] = (
                f"http://{endpoint_host}:{self.gpu_service_metrics_port_base}/metrics"
            )
            env["MULLIGAN_DEVICE"] = env.get("MULLIGAN_DEVICE", "cpu")

        # Ensure each profile's directories exist
        for entry in entries:
            profile = str(entry.get("profile", "")).strip()
            self.profile_models_dir(profile).mkdir(parents=True, exist_ok=True)
            self.profile_logs_dir(profile).mkdir(parents=True, exist_ok=True)
            self.python_logs_dir(profile).mkdir(parents=True, exist_ok=True)

        self.apply_trainer_java_tool_options(env)

        # Primary JVM runs locally on GPU node (no srun wrapping)
        command = self.build_multi_profile_command(profile_names)
        process = subprocess.Popen(
            command,
            cwd=str(self.source_repo_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        log(f"Started primary multi-profile trainer pid={process.pid} profiles={profiles_csv} totalRunners={primary_total}")

        # Return a single TrainerState using first profile as representative
        state = TrainerState(
            entry=entries[0],
            slot=0,
            profile="multi:" + profiles_csv,
            metrics_port=metrics_port,
            py4j_base_port=0,
            runners_per_profile=runners_per_profile,
            opponent_decklist=opponent_decklist,
            command=command,
            env=env,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            process=process,
            effective_train_env={},
            effective_seed=None,
        )

        # Launch auxiliary JVMs on CPU nodes (game simulation only, no stats writing)
        for idx, cpu_node in enumerate(self.cpu_nodes):
            aux_metrics_port = metrics_port + 1 + idx
            aux_stdout_path = self.trainer_logs_dir / f"multi_profile_aux_{cpu_node}.stdout.log"
            aux_stderr_path = self.trainer_logs_dir / f"multi_profile_aux_{cpu_node}.stderr.log"
            aux_stdout = aux_stdout_path.open("ab", buffering=0)
            aux_stderr = aux_stderr_path.open("ab", buffering=0)

            aux_env = dict(env)
            aux_env["GAME_STATS_WRITER"] = "0"
            aux_env["METRICS_PORT"] = str(aux_metrics_port)
            aux_env["NUM_GAME_RUNNERS"] = str(aux_total)

            aux_cpus = env_int("GAME_CPU_CORES", 128)
            srun_args = ["srun", "--overlap", f"--nodelist={cpu_node}", "--ntasks=1", f"--cpus-per-task={aux_cpus}"]
            het1 = os.getenv("SLURM_JOB_NODELIST_HET_GROUP_1", "").strip()
            if het1:
                srun_args.insert(1, "--het-group=1")
            aux_command = srun_args + self.build_multi_profile_command(profile_names)

            aux_proc = subprocess.Popen(
                aux_command,
                cwd=str(self.source_repo_root),
                env=aux_env,
                stdout=aux_stdout,
                stderr=aux_stderr,
            )
            state.aux_processes.append({
                "process": aux_proc,
                "stdout": aux_stdout,
                "stderr": aux_stderr,
                "node": cpu_node,
                "metrics_port": aux_metrics_port,
            })
            log(f"Started auxiliary JVM on {cpu_node} pid={aux_proc.pid} metricsPort={aux_metrics_port}")

        return state

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
        # Terminate auxiliary JVMs first
        for aux in state.aux_processes:
            aux_proc = aux["process"]
            if aux_proc.poll() is None:
                try:
                    aux_proc.terminate()
                except Exception:
                    pass
        # Terminate primary
        proc = state.process
        if proc.poll() is not None:
            # Still wait for auxiliaries
            self._wait_and_kill_auxiliaries(state)
            return
        try:
            proc.terminate()
        except Exception:
            self._wait_and_kill_auxiliaries(state)
            return
        deadline = time.time() + self.trainer_stop_grace_seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        self._wait_and_kill_auxiliaries(state)

    def _wait_and_kill_auxiliaries(self, state: TrainerState) -> None:
        deadline = time.time() + min(5.0, self.trainer_stop_grace_seconds)
        for aux in state.aux_processes:
            aux_proc = aux["process"]
            while time.time() < deadline:
                if aux_proc.poll() is not None:
                    break
                time.sleep(0.2)
            if aux_proc.poll() is None:
                try:
                    aux_proc.kill()
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
        for aux in state.aux_processes:
            for key in ("stdout", "stderr"):
                handle = aux.get(key)
                if handle is None:
                    continue
                try:
                    handle.flush()
                except Exception:
                    pass
                try:
                    handle.close()
                except Exception:
                    pass

    def profile_models_dir(self, profile: str) -> Path:
        return self.rl_artifacts_root / "profiles" / profile / "models"

    def profile_logs_dir(self, profile: str) -> Path:
        return self.rl_artifacts_root / "profiles" / profile / "logs"

    def python_logs_dir(self, profile: str) -> Path:
        return self.profile_logs_dir(profile) / "python"

    def shared_gpu_python_logs_dir(self, gpu_slot: int) -> Path:
        return self.shared_gpu_logs_dir / f"gpu_{gpu_slot}" / "python"

    def profile_model_latest_path(self, profile: str) -> Path:
        return self.profile_models_dir(profile) / "model_latest.pt"

    def profile_model_path(self, profile: str) -> Path:
        return self.profile_models_dir(profile) / "model.pt"

    def profile_mulligan_model_path(self, profile: str) -> Path:
        return self.profile_models_dir(profile) / "mulligan_model.pt"

    def resolve_stable_checkpoint_path(self, profile: str) -> Optional[Path]:
        source_latest = self.profile_model_latest_path(profile)
        if source_latest.exists():
            return source_latest
        source_model = self.profile_model_path(profile)
        if source_model.exists():
            return source_model
        return None

    def _copy_file_atomic(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}")
        try:
            shutil.copy2(str(source), str(temp_path))
            os.replace(str(temp_path), str(target))
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    def restore_profile_model_files(self, restore_info: Dict[str, Any]) -> None:
        for target_key, backup_key in (("target_latest", "backup_latest"), ("target_model", "backup_model")):
            target = restore_info.get(target_key)
            backup = restore_info.get(backup_key)
            created = bool(restore_info.get(f"{target_key}_created", False))
            if not isinstance(target, Path):
                continue
            if isinstance(backup, Path) and backup.exists():
                self._copy_file_atomic(backup, target)
                try:
                    backup.unlink()
                except Exception:
                    pass
            elif created and target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass

    def stage_profile_model_replacement(self, source_profile: str, target_profile: str) -> Dict[str, Any]:
        source = self.resolve_stable_checkpoint_path(source_profile)
        if source is None:
            raise RuntimeError(f"stable checkpoint not found for winner={source_profile}")

        target_latest = self.profile_model_latest_path(target_profile)
        target_model = self.profile_model_path(target_profile)
        target_latest.parent.mkdir(parents=True, exist_ok=True)

        restore_info: Dict[str, Any] = {
            "source": source,
            "target_latest": target_latest,
            "target_model": target_model,
            "backup_latest": None,
            "backup_model": None,
            "target_latest_created": not target_latest.exists(),
            "target_model_created": not target_model.exists(),
            "mulligan_copied": False,
        }

        for target_key in ("target_latest", "target_model"):
            target = restore_info[target_key]
            if target.exists():
                backup = target.with_name(f".{target.name}.bak.{os.getpid()}.{int(time.time() * 1000)}")
                shutil.copy2(str(target), str(backup))
                restore_info[f"backup_{target_key.split('_', 1)[1]}"] = backup

        self._copy_file_atomic(source, target_latest)
        self._copy_file_atomic(source, target_model)

        source_mulligan = self.profile_mulligan_model_path(source_profile)
        if source_mulligan.exists():
            target_mulligan = self.profile_mulligan_model_path(target_profile)
            if target_mulligan.exists():
                backup = target_mulligan.with_name(
                    f".{target_mulligan.name}.bak.{os.getpid()}.{int(time.time() * 1000)}"
                )
                shutil.copy2(str(target_mulligan), str(backup))
                restore_info["backup_mulligan"] = backup
            self._copy_file_atomic(source_mulligan, target_mulligan)
            restore_info["mulligan_copied"] = True

        return restore_info

    @staticmethod
    def cleanup_restore_info_backups(restore_info: Dict[str, Any]) -> None:
        for key in ("backup_latest", "backup_model", "backup_mulligan"):
            backup = restore_info.get(key)
            if isinstance(backup, Path) and backup.exists():
                try:
                    backup.unlink()
                except Exception:
                    pass

    @staticmethod
    def new_pbt_event(
        group: str,
        trigger: str,
        group_min_episode: int,
        episode_delta_since_last: int,
        elapsed_minutes: float,
        winner_profile: str,
        loser_profile: str,
        winner_wr: float,
        loser_wr: float,
        winner_gap: float,
        winner_gap_min: float,
        winner_wr_min_gate: float,
        time_delta_gate: int,
        time_delta_gate_passed: bool,
    ) -> Dict[str, Any]:
        elapsed_value = -1.0 if math.isinf(elapsed_minutes) else float(elapsed_minutes)
        return {
            "timestamp_utc": now_utc(),
            "run_id": "",
            "population_group": group,
            "trigger": trigger,
            "group_min_episode": int(group_min_episode),
            "episode_delta": int(episode_delta_since_last),
            "elapsed_minutes": elapsed_value,
            "winner": winner_profile,
            "loser": loser_profile,
            "winner_wr": float(winner_wr),
            "loser_wr": float(loser_wr),
            "winner_gap": float(winner_gap),
            "winner_gap_min": float(winner_gap_min),
            "winner_wr_min_gate": float(winner_wr_min_gate),
            "time_delta_gate": int(time_delta_gate),
            "time_delta_gate_passed": bool(time_delta_gate_passed),
            "skip_reason": "",
            "result": "",
            "failure_reason": "",
            "state": "candidate",
            "state_transitions": ["candidate"],
            "copied": False,
            "copy_source_path": "",
            "new_seed": "",
            "mutated_keys": "",
        }

    @staticmethod
    def advance_event_state(event: Dict[str, Any], state: str) -> None:
        event["state"] = state
        transitions = event.setdefault("state_transitions", [])
        if not transitions or transitions[-1] != state:
            transitions.append(state)

    PBT_BOUNDS: Dict[str, Tuple[float, float]] = {
        "ENTROPY_START": (0.01, 1.0),
        "ENTROPY_END": (0.001, 0.5),
        "RL_ACTION_EPS_START": (0.0, 1.0),
        "RL_FULL_TURN_RANDOM_START": (0.0, 1.0),
        "TEMPERATURE_FLOOR": (0.05, 2.0),
        "ACTOR_LR": (1e-5, 1e-3),
    }

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
        bounds = self.PBT_BOUNDS.get(k)
        if bounds is not None:
            mutated = max(bounds[0], min(bounds[1], mutated))
        elif ("EPS" in k) or ("PROB" in k) or ("RATE" in k) or ("FRAC" in k) or k.endswith("_P") or k.startswith("P_"):
            mutated = max(0.0, min(1.0, mutated))
        return f"{mutated:g}"

    def get_profile_training_snapshot(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        profile = str(entry.get("profile", "")).strip()
        logs_dir = self.profile_logs_dir(profile)
        stats_path = logs_dir / "stats" / "training_stats.csv"
        status_path = logs_dir / "league" / "agent_status.json"

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
                with stats_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames and "episode" in reader.fieldnames and "winrate" in reader.fieldnames:
                        for row in reader:
                            ep = parse_int64(row.get("episode"))
                            if ep is not None and int(ep) > episode:
                                episode = int(ep)
                            wr = parse_float(row.get("winrate"))
                            if wr is not None:
                                values.append(wr)
                    else:
                        handle.seek(0)
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
                            wr = parse_float(parts[4].strip())
                            if wr is not None:
                                values.append(wr)
            except Exception as exc:
                warn_key = f"snapshot:{stats_path}"
                if warn_key not in self.snapshot_warning_keys:
                    self.snapshot_warning_keys.add(warn_key)
                    log(f"WARNING: Failed parsing training stats for profile={profile}: {stats_path} ({exc})")

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
            "mode": str(entry.get("mode", "")),
        }

    def update_stall_progress(self, state: TrainerState, snapshots: Dict[str, Dict[str, Any]]) -> None:
        # For multi-profile trainers, check any constituent profile for progress
        if state.profile.startswith("multi:"):
            sub_profiles = state.profile[len("multi:"):].split(",")
            total_ep = 0
            for sp in sub_profiles:
                snap = snapshots.get(sp.strip())
                if snap:
                    total_ep += int(snap.get("episode", 0))
            if total_ep > state.last_progress_episode:
                state.last_progress_episode = total_ep
                state.last_progress_at = time.time()
            return
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
        # In multi-profile mode, restart the multi-profile JVM
        if self.multi_profile_jvm and profile.startswith("multi:"):
            replacement = self.start_multi_profile_trainer(
                self.selected_profiles, state.runners_per_profile, state.opponent_decklist,
            )
        else:
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
                if self.league_mode == "rl_only" and self.eval_results:
                    wr = self.eval_results.get(profile)
                else:
                    wr = snap.get("rolling_current")
                if wr is None:
                    continue
                if profile not in self.trainers:
                    continue
                candidates.append({"entry": entry, "snapshot": snap})
            if len(candidates) < max(2, self.pbt_min_population):
                continue

            def _pbt_sort_key(c):
                p = str(c["entry"]["profile"])
                if self.league_mode == "rl_only" and self.eval_results:
                    w = self.eval_results.get(p, 0.0)
                else:
                    w = float(c["snapshot"].get("rolling_current", 0.0))
                return (-w, p)

            ordered = sorted(candidates, key=_pbt_sort_key)
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

                if self.league_mode == "rl_only" and self.eval_results:
                    winner_wr = self.eval_results.get(winner_profile, 0.0)
                    loser_wr = self.eval_results.get(loser_profile, 0.0)
                else:
                    winner_wr = float(winner["snapshot"]["rolling_current"])
                    loser_wr = float(loser["snapshot"]["rolling_current"])
                winner_gap = winner_wr - loser_wr
                gap_gate_passed = winner_gap >= self.pbt_min_winner_gap
                winner_wr_gate_passed = winner_wr >= self.pbt_min_winner_wr
                event = self.new_pbt_event(
                    group=group,
                    trigger=trigger,
                    group_min_episode=group_min_episode,
                    episode_delta_since_last=episode_delta_since_last,
                    elapsed_minutes=elapsed_minutes,
                    winner_profile=winner_profile,
                    loser_profile=loser_profile,
                    winner_wr=winner_wr,
                    loser_wr=loser_wr,
                    winner_gap=winner_gap,
                    winner_gap_min=self.pbt_min_winner_gap,
                    winner_wr_min_gate=self.pbt_min_winner_wr,
                    time_delta_gate=time_delta_gate,
                    time_delta_gate_passed=time_delta_gate_passed,
                )
                event["run_id"] = self.run_id
                event["gap_gate_passed"] = gap_gate_passed
                event["winner_wr_gate_passed"] = winner_wr_gate_passed

                if not (gap_gate_passed and winner_wr_gate_passed):
                    if not gap_gate_passed and not winner_wr_gate_passed:
                        skip_reason = "gap_and_winner_wr_below_threshold"
                    elif not gap_gate_passed:
                        skip_reason = "gap_below_threshold"
                    else:
                        skip_reason = "winner_wr_below_threshold"
                    self.advance_event_state(event, "failed")
                    event["skip_reason"] = skip_reason
                    event["result"] = "skipped"
                    event["failure_reason"] = skip_reason
                    events.append(event)
                    self._append_pbt_event(event)
                    continue

                winner_state = self.trainers.get(winner_profile)
                if winner_state is not None and winner_state.effective_train_env:
                    effective_env = dict(winner_state.effective_train_env)
                elif isinstance(winner["entry"].get("train_env"), dict):
                    effective_env = {
                        str(k).strip(): str(v)
                        for k, v in winner["entry"].get("train_env", {}).items()
                        if str(k).strip()
                    }
                else:
                    effective_env = dict(state.effective_train_env)
                    if not effective_env and isinstance(loser["entry"].get("train_env"), dict):
                        effective_env = {
                            str(k).strip(): str(v)
                            for k, v in loser["entry"].get("train_env", {}).items()
                            if str(k).strip()
                        }

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
                restore_info: Optional[Dict[str, Any]] = None
                replacement: Optional[TrainerState] = None
                recovery_state: Optional[TrainerState] = None
                try:
                    self.advance_event_state(event, "copy_started")
                    self.stop_trainer(state)
                    self.close_logs(state)
                    restore_info = self.stage_profile_model_replacement(winner_profile, loser_profile)
                    event["copied"] = True
                    event["copy_source_path"] = str(restore_info["source"])
                    self.advance_event_state(event, "copy_succeeded")
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
                    self.advance_event_state(event, "restart_succeeded")
                except Exception as exc:
                    failure_prefix = "restart_failed"
                    if str(event.get("state", "")) == "copy_started" and not bool(event.get("copied", False)):
                        failure_prefix = "copy_failed"
                    failure_reason = f"{failure_prefix}: {exc}"
                    log(
                        f"WARNING: PBT {failure_prefix} winner={winner_profile} "
                        f"loser={loser_profile}: {exc}"
                    )
                    if restore_info is not None:
                        try:
                            self.restore_profile_model_files(restore_info)
                        except Exception as rollback_exc:
                            failure_reason = f"{failure_reason}; rollback_failed: {rollback_exc}"
                    try:
                        recovery_state = self.start_trainer(
                            loser["entry"],
                            state.slot,
                            state.runners_per_profile,
                            state.opponent_decklist,
                            effective_train_env=state.effective_train_env,
                            effective_seed=state.effective_seed,
                        )
                        recovery_state.restart_count = state.restart_count
                        recovery_state.consecutive_failures = state.consecutive_failures
                        recovery_state.last_restart_reason = "pbt_recover"
                        recovery_state.last_progress_episode = state.last_progress_episode
                        recovery_state.last_progress_at = time.time()
                        self.trainers[loser_profile] = recovery_state
                    except Exception as recovery_exc:
                        failure_reason = f"{failure_reason}; recovery_restart_failed: {recovery_exc}"
                        log(
                            f"WARNING: PBT recovery restart failed winner={winner_profile} "
                            f"loser={loser_profile}: {recovery_exc}"
                        )
                    self.advance_event_state(event, "failed")
                    event["result"] = "failed"
                    event["failure_reason"] = failure_reason
                    event["new_seed"] = str(seed_now)
                    event["mutated_keys"] = ";".join(mutable_keys)
                    events.append(event)
                    self._append_pbt_event(event)
                    continue

                if replacement is None:
                    self.advance_event_state(event, "failed")
                    event["result"] = "failed"
                    event["failure_reason"] = "restart_failed: replacement_missing"
                    event["new_seed"] = str(seed_now)
                    event["mutated_keys"] = ";".join(mutable_keys)
                    events.append(event)
                    self._append_pbt_event(event)
                    continue

                if restore_info is not None:
                    self.cleanup_restore_info_backups(restore_info)
                self.trainers[loser_profile] = replacement
                self.advance_event_state(event, "committed")
                event["result"] = "committed"
                event["new_seed"] = str(seed_now)
                event["mutated_keys"] = ";".join(mutable_keys)
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
        selected_lookup = {str(entry.get("profile", "")).strip() for entry in self.selected_profiles}

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
                    "mode": str(entry.get("mode", "")),
                    "selected": True,
                    "population_group": str(entry.get("population_group", "")).strip() or profile,
                    "target_winrate": float(entry.get("target_winrate", 0.60)),
                    "seed": seed_now,
                    "pbt_mutable_env": to_string_list(entry.get("pbt_mutable_env")),
                    "train_env": train_env_now,
                }
            )

        all_active_profiles: List[Dict[str, Any]] = []
        for entry in self.active_profiles:
            profile = str(entry.get("profile", "")).strip()
            if not profile:
                continue
            all_active_profiles.append(
                {
                    "profile": profile,
                    "mode": str(entry.get("mode", "")),
                    "selected": profile in selected_lookup,
                    "train_enabled": bool(entry.get("train_enabled", True)),
                    "population_group": str(entry.get("population_group", "")).strip() or profile,
                    "priority": int(entry.get("priority", 1000)),
                    "target_winrate": float(entry.get("target_winrate", 0.60)),
                    "deck_path": str(entry.get("deck_path", "")),
                }
            )

        group_payload: List[Dict[str, Any]] = []
        for key, st in self.pbt_group_state.items():
            group_profiles = [row["profile"] for row in all_active_profiles if row["population_group"] == str(key)]
            selected_profiles = [row["profile"] for row in all_active_profiles if row["population_group"] == str(key) and row["selected"]]
            group_payload.append(
                {
                    "population_group": str(key),
                    "profiles": group_profiles,
                    "selected_profiles": selected_profiles,
                    "last_exploit_utc": str(st.get("last_exploit_utc", "")),
                    "last_exploit_min_episode": int(st.get("last_exploit_min_episode", 0)),
                    "exploit_count": int(st.get("exploit_count", 0)),
                }
            )

        payload = {
            "updated_at_utc": now_utc(),
            "run_id": self.run_id,
            "mode": self.mode,
            "registry_path": str(self.registry_path),
            "reports_root": str(self.reports_root),
            "compat_reports_root": str(self.compat_reports_root),
            "enable_pbt": bool(self.enable_pbt),
            "pbt_exploit_max_interval_minutes": int(self.pbt_interval_minutes),
            "pbt_min_episodes_before_first_exploit": int(self.pbt_first_exploit_min_ep),
            "pbt_min_episode_delta_per_profile": int(self.pbt_episode_delta),
            "pbt_time_fallback_min_episode_delta": int(self.pbt_time_fallback_episode_delta),
            "pbt_mutation_pct": float(self.pbt_mutation_pct),
            "pbt_min_winner_gap": float(self.pbt_min_winner_gap),
            "pbt_min_winner_winrate": float(self.pbt_min_winner_wr),
            "last_exploit_utc": "" if self.last_pbt_at is None else self.last_pbt_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "selected_profiles": [row["profile"] for row in profiles_payload],
            "all_active_profiles": all_active_profiles,
            "profiles": profiles_payload,
            "group_state": group_payload,
            "events": self.pbt_events[-200:],
        }
        atomic_write_json(self.pbt_state_path, payload)
        atomic_write_json(self.compat_pbt_state_path, payload)
        self.write_latest_run_pointer()

    def write_orchestrator_status(self, snapshots: Dict[str, Dict[str, Any]], note: str) -> None:
        self.reports_root.mkdir(parents=True, exist_ok=True)
        selected_lookup = {str(entry.get("profile", "")).strip() for entry in self.selected_profiles}

        trainer_rows: List[Dict[str, Any]] = []
        for entry in self.selected_profiles:
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
                    "run_id": self.run_id,
                    "train_decklist": "" if state is None else str(state.env.get("RL_AGENT_DECK_LIST", "")),
                    "opponent_decklist": "" if state is None else str(state.opponent_decklist),
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                    "exit_code": exit_code,
                }
            )

        all_snapshot_rows: List[Dict[str, Any]] = []
        selected_snapshot_rows: List[Dict[str, Any]] = []
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
            row = {
                "profile": profile,
                "mode": str(snap.get("mode", entry.get("mode", ""))),
                "selected": profile in selected_lookup,
                "episode": int(snap.get("episode", 0)),
                "rolling_winrate": snap.get("rolling_current"),
                "rolling_avg": snap.get("rolling_avg"),
                "sample_count": int(snap.get("sample_count", 0)),
                "baseline_wr": float(snap.get("baseline_wr", 0.0)),
                "target_winrate": float(snap.get("target_winrate", 0.60)),
                "promoted": bool(snap.get("promoted", False)),
                "train_enabled": bool(snap.get("train_enabled", True)),
                "completed": completed,
                "population_group": str(entry.get("population_group", "")).strip() or profile,
            }
            all_snapshot_rows.append(row)
            if profile in selected_lookup:
                selected_snapshot_rows.append(row)

        all_active_profiles: List[Dict[str, Any]] = []
        for entry in self.active_profiles:
            profile = str(entry.get("profile", "")).strip()
            if not profile:
                continue
            all_active_profiles.append(
                {
                    "profile": profile,
                    "mode": str(entry.get("mode", "")),
                    "selected": profile in selected_lookup,
                    "train_enabled": bool(entry.get("train_enabled", True)),
                    "population_group": str(entry.get("population_group", "")).strip() or profile,
                    "priority": int(entry.get("priority", 1000)),
                    "target_winrate": float(entry.get("target_winrate", 0.60)),
                    "deck_path": str(entry.get("deck_path", "")),
                }
            )

        population_groups: List[Dict[str, Any]] = []
        for group in sorted({row["population_group"] for row in all_active_profiles}):
            group_all = [row["profile"] for row in all_active_profiles if row["population_group"] == group]
            group_selected = [row["profile"] for row in all_active_profiles if row["population_group"] == group and row["selected"]]
            population_groups.append(
                {
                    "population_group": group,
                    "all_active_profiles": group_all,
                    "selected_profiles": group_selected,
                    "selected_count": len(group_selected),
                }
            )

        shared_gpu_host_rows: List[Dict[str, Any]] = []
        for slot, host in sorted(self.shared_gpu_hosts.items()):
            process = host.get("process")
            running = False
            pid = 0
            exit_code: Optional[int] = None
            if process is not None:
                try:
                    rc = process.poll()
                    running = rc is None
                    pid = int(process.pid)
                    if rc is not None:
                        exit_code = int(rc)
                except Exception:
                    pass
            shared_gpu_host_rows.append(
                {
                    "slot": int(slot),
                    "gpu_id": str(host.get("gpu_id", "")),
                    "port": int(host.get("port", 0)),
                    "metrics_port": int(host.get("metrics_port", 0)),
                    "running": running,
                    "pid": pid,
                    "exit_code": exit_code,
                    "stdout_log": str(host.get("stdout_log", "")),
                    "stderr_log": str(host.get("stderr_log", "")),
                }
            )

        payload = {
            "updated_at_utc": now_utc(),
            "run_id": self.run_id,
            "mode": self.mode,
            "registry_path": str(self.registry_path),
            "reports_root": str(self.reports_root),
            "compat_reports_root": str(self.compat_reports_root),
            "eval_every_minutes": int(self.eval_every_minutes),
            "last_eval_utc": "",
            "next_eval_utc": "",
            "dry_run": False,
            "sequential_training": False,
            "current_training_profile": "",
            "note": note,
            "paths": {
                "reports_root": str(self.reports_root),
                "trainers_dir": str(self.trainer_logs_dir),
                "gpu_hosts_dir": str(self.shared_gpu_logs_dir),
                "generated_decklists_dir": str(self.generated_decklists_dir),
                "artifacts_root": str(self.rl_artifacts_root),
                "orchestrator_status_path": str(self.orchestrator_status_path),
                "pbt_state_path": str(self.pbt_state_path),
            },
            "all_active_profiles": all_active_profiles,
            "selected_profiles": [row["profile"] for row in all_active_profiles if row["selected"]],
            "population_groups": population_groups,
            "trainers": trainer_rows,
            "shared_gpu_hosts": shared_gpu_host_rows,
            "all_active_profile_snapshots": all_snapshot_rows,
            "selected_profile_snapshots": selected_snapshot_rows,
            "profile_snapshots": selected_snapshot_rows,
        }
        atomic_write_json(self.orchestrator_status_path, payload)
        atomic_write_json(self.compat_orchestrator_status_path, payload)
        self.write_latest_run_pointer()

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
        stop_script = self.source_repo_root / "scripts/rl-stop.sh"
        if stop_script.exists():
            log(f"Running cleanup via {stop_script} (reason=orchestrator_start)")
            try:
                subprocess.run(
                    ["bash", str(stop_script), "-q"],
                    cwd=str(self.source_repo_root),
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

        log(
            f"League orchestrator starting. runId={self.run_id} profiles={len(active_profiles)} "
            f"selected={len(self.selected_profiles)} evalEveryMinutes={self.eval_every_minutes} runOnce=False dryRun=False"
        )
        log(f"Registry: {self.registry_path}")
        log(f"Reports root: {self.reports_root}")
        log(f"Compatibility reports root: {self.compat_reports_root}")
        log(f"RL artifacts root: {self.rl_artifacts_root}")
        log(f"DB root: {self.db_root}")
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
        if self.py_service_mode == "shared_gpu":
            log(f"Shared GPU host startup timeout seconds={self.gpu_service_startup_timeout_seconds}")
        log(
            f"Trainer startup waves: visibleGpus={self.visible_gpu_count} "
            f"waveSize={self.trainer_start_wave_size} intraWaveDelayMs={self.trainer_start_intra_wave_delay_ms}"
        )

        if self.py_service_mode == "shared_gpu":
            self.launch_shared_gpu_hosts()

        if self.multi_profile_jvm:
            log("Multi-profile JVM mode: launching single JVM for all profiles")
            state = self.start_multi_profile_trainer(
                self.selected_profiles, runners_per_profile, opponent_decklist,
            )
            self.trainers[state.profile] = state
        else:
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
            # Run eval benchmark before PBT exploitation in rl_only mode
            if self.league_mode == "rl_only" and self.pbt_interval_minutes > 0:
                eval_elapsed = float("inf") if self.last_eval_at is None else (now_dt - self.last_eval_at).total_seconds() / 60.0
                if eval_elapsed >= self.pbt_interval_minutes:
                    self.run_eval_benchmark()
                    self.last_eval_at = datetime.now(timezone.utc)
                    # Restart trainers after eval
                    for entry in self.load_profiles():
                        p = str(entry.get("profile", "")).strip()
                        if p and p not in self.trainers:
                            self.start_trainer(entry)
            events = self.invoke_pbt_exploit(snapshots, now_dt)
            if events:
                if any(str(ev.get("result", "")) == "committed" for ev in events):
                    self.last_pbt_at = now_dt
                for ev in events:
                    skip_reason = str(ev.get("skip_reason", "")).strip()
                    result = str(ev.get("result", "")).strip() or "unknown"
                    if result == "committed":
                        log(
                            "PBT exploit "
                            f"group={ev.get('population_group')} winner={ev.get('winner')} loser={ev.get('loser')} "
                            f"winner_wr={float(ev.get('winner_wr', 0.0)):.3f} loser_wr={float(ev.get('loser_wr', 0.0)):.3f} "
                            f"gap={float(ev.get('winner_gap', 0.0)):.3f} seed={ev.get('new_seed')} trigger={ev.get('trigger')} "
                            f"groupMinEp={int(ev.get('group_min_episode', 0))} deltaEp={int(ev.get('episode_delta', 0))}"
                        )
                    elif result == "skipped":
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
                            "PBT exploit failed "
                            f"group={ev.get('population_group')} winner={ev.get('winner')} loser={ev.get('loser')} "
                            f"reason={ev.get('failure_reason')} winner_wr={float(ev.get('winner_wr', 0.0)):.3f} "
                            f"loser_wr={float(ev.get('loser_wr', 0.0)):.3f} gap={float(ev.get('winner_gap', 0.0)):.3f} "
                            f"trigger={ev.get('trigger')} groupMinEp={int(ev.get('group_min_episode', 0))} "
                            f"deltaEp={int(ev.get('episode_delta', 0))}"
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
