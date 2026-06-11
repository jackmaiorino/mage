#!/usr/bin/env python3
"""Local PBT orchestrator -- runs multi-profile training with Population-Based Training
on a single machine, no Slurm required.

Usage:
    py -3.12 scripts/run_local_pbt.py

Env vars:
    TRAIN_PROFILES       Number of profiles to train (default: 4)
    NUM_GAME_RUNNERS     Total game runners; Java splits them across active profiles (default: 64)
    TOTAL_EPISODES       Episodes before exit (default: 1000000)
    TOTAL_EPISODES_DELTA Additional episodes from current profile counters.
                         If >0, overrides TOTAL_EPISODES by setting the absolute
                         target to max(selected profile current episodes) + delta.
    MAX_WALL_SECONDS     Wall-clock seconds before graceful exit (default: 0 = disabled)
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
import re
import shutil
import signal
import subprocess
import sys
import threading
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


def _find_cudnn_bin() -> Optional[str]:
    """Find cuDNN bin directory for ONNX Runtime GPU on Windows."""
    if sys.platform != "win32":
        return None
    cudnn_root = Path(os.environ.get("CUDNN_PATH", r"C:\Program Files\NVIDIA\CUDNN"))
    if not cudnn_root.exists():
        return None
    # Scan for versioned dirs: v9.20/bin/12.x/x64/
    for ver_dir in sorted(cudnn_root.iterdir(), reverse=True):
        bin_dir = ver_dir / "bin"
        if not bin_dir.exists():
            continue
        # Prefer CUDA 12.x (ORT 1.19.2 built against CUDA 12), then fall back
        cuda_subs = sorted(bin_dir.iterdir(), reverse=True)
        for cuda_sub in sorted(cuda_subs, key=lambda p: (not p.name.startswith("12"), p.name), reverse=False):
            candidate = cuda_sub / "x64"
            if candidate.exists() and any(candidate.glob("cudnn*.dll")):
                return str(candidate)
    return None


def _prepend_cuda_paths(env: dict) -> None:
    """Prepend cuDNN and CUDA toolkit to PATH so ORT GPU provider can load."""
    additions = []
    cudnn = _find_cudnn_bin()
    if cudnn:
        additions.append(cudnn)
    # Also ensure a CUDA toolkit bin is on PATH
    cuda_root = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"))
    cuda_bin = cuda_root / "bin"
    if cuda_bin.exists():
        additions.append(str(cuda_bin))
    if additions:
        sep = ";" if sys.platform == "win32" else ":"
        env["PATH"] = sep.join(additions) + sep + env.get("PATH", "")


def _visible_nvidia_gpu_count() -> Optional[int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        cp = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    return sum(1 for line in cp.stdout.splitlines() if line.strip().startswith("GPU "))


def _validate_cuda_device_visible(env: dict, key: str) -> None:
    raw = str(env.get(key, "")).strip().lower()
    match = re.fullmatch(r"cuda:(\d+)", raw)
    if not match:
        return
    requested = int(match.group(1))
    count = _visible_nvidia_gpu_count()
    if count is None:
        log(f"WARNING: unable to validate {key}={raw}; nvidia-smi is unavailable")
        return
    if requested >= count:
        raise RuntimeError(
            f"{key}={raw} requested GPU index {requested}, but nvidia-smi only exposes "
            f"{count} CUDA-visible NVIDIA GPU(s). Refusing to start to avoid silent CPU fallback."
        )


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
    """Read episode count and rolling winrate from tail of training_stats.csv."""
    stats_path = PROFILES_ROOT / profile / "logs" / "stats" / "training_stats.csv"
    if not stats_path.exists():
        return 0, None
    episode = 0
    values: deque = deque(maxlen=max(50, window))
    try:
        # Read only the last N lines (avoid scanning entire file)
        read_lines = max(window, 500)
        with stats_path.open("rb") as f:
            # Seek to end, scan backwards for enough newlines
            f.seek(0, 2)
            fsize = f.tell()
            # Estimate: ~60 bytes per line
            seek_back = min(fsize, read_lines * 80)
            f.seek(max(0, fsize - seek_back))
            tail = f.read().decode("utf-8", errors="replace")
        lines = tail.strip().split("\n")
        # Find header to parse columns (might be in the chunk or use known order)
        # CSV format: episode,turns,final_reward,opponent_type,winrate,episode_seconds
        for line in lines:
            parts = line.split(",")
            if len(parts) < 5 or parts[0] == "episode":
                continue
            try:
                ep = int(parts[0])
                if ep > episode:
                    episode = ep
            except (ValueError, TypeError):
                continue
            try:
                values.append(float(parts[4]))
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


def _stop_process_tree(process: subprocess.Popen, label: str, timeout: int) -> None:
    """Stop a process and its children without touching unrelated Java/Python jobs."""
    if process.poll() is not None:
        return
    pid = process.pid
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            log(f"WARNING: graceful {label} tree stop failed for pid={pid}: {exc}")
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
            process.wait(timeout=10)
            return
        except Exception as exc:
            log(f"WARNING: forced {label} tree stop failed for pid={pid}: {exc}")
            process.kill()
            return

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _resolve_active_onnx_dir(onnx_dir: Path) -> Path:
    pointer = onnx_dir / ".active_dir"
    if pointer.exists():
        try:
            raw = pointer.read_text(encoding="utf-8").strip()
            if raw:
                active = Path(raw)
                if not active.is_absolute():
                    active = onnx_dir / active
                if active.is_dir():
                    return active
        except Exception:
            pass
    return onnx_dir


def _prune_versioned_onnx_dirs(onnx_dir: Path, keep: int = 3) -> None:
    try:
        versions = sorted(
            [p for p in onnx_dir.iterdir() if p.is_dir() and p.name.startswith("v")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return
    for old in versions[keep:]:
        try:
            shutil.rmtree(old)
            log(f"[TRT] Removed stale ONNX export {old}")
        except Exception:
            # ONNX Runtime can still hold the previous directory briefly on Windows.
            pass


class LocalPBT:
    def __init__(self):
        self.port = env_int("GPU_SERVICE_PORT", 26100)
        self.metrics_port = self.port + 1000
        self.train_profiles = env_int("TRAIN_PROFILES", 4)
        self.num_runners = env_int("NUM_GAME_RUNNERS", 64)
        self.total_episodes = env_int("TOTAL_EPISODES", 500000)
        self.total_episodes_delta = env_int("TOTAL_EPISODES_DELTA", 0)
        self.max_wall_seconds = env_int("MAX_WALL_SECONDS", 0)
        self.winrate_window = env_int("WINRATE_WINDOW", 200)
        self.exploit_interval_min = env_float("PBT_EXPLOIT_INTERVAL", 60.0)
        self.onnx_export_interval_ticks = env_int("ONNX_EXPORT_INTERVAL_TICKS", 20)
        self.min_episodes = env_int("PBT_MIN_EPISODES", 200)
        self.episode_delta = env_int("PBT_EPISODE_DELTA", 100)
        self.min_winner_gap = env_float("PBT_MIN_WINNER_GAP", 0.02)
        self.min_winner_wr = env_float("PBT_MIN_WINNER_WR", 0.03)
        self.mutation_pct = env_float("PBT_MUTATION_PCT", 0.20)
        self.registry_path = Path(os.getenv("REGISTRY_PATH", str(DEFAULT_REGISTRY)))
        self.stop_requested = False
        self._prev_ep_count: Dict[str, int] = {}
        self._prev_ep_time: Dict[str, float] = {}
        self.gpu_process: Optional[subprocess.Popen] = None
        self.trainer_process: Optional[subprocess.Popen] = None
        self.selected_profiles: List[Dict[str, Any]] = []
        self.last_exploit_at: Dict[str, float] = {}
        self.last_exploit_episode: Dict[str, int] = {}
        self.exploit_count: Dict[str, int] = {}
        self.eval_results: Dict[str, float] = {}
        self.start_episodes: Dict[str, int] = {}
        self._last_eval_time: float = time.time()  # skip immediate eval on startup
        self.trainer_log_path = REPO_ROOT / "local-training" / "local_pbt" / "trainer.log"

    def _cleanup_temp_files(self) -> None:
        """Remove leaked ORT temp dirs and oversized log files on startup."""
        import glob, tempfile
        # ORT temp dirs leak on every JVM restart
        pattern = os.path.join(tempfile.gettempdir(), "onnxruntime-java*")
        ort_dirs = glob.glob(pattern)
        if ort_dirs:
            for d in ort_dirs:
                shutil.rmtree(d, ignore_errors=True)
            log(f"Cleaned {len(ort_dirs)} leaked ORT temp dirs")
        # Mulligan trace grows unbounded
        for profile_dir in PROFILES_ROOT.iterdir():
            trace = profile_dir / "logs" / "python" / "mulligan_trace.jsonl"
            if trace.exists() and trace.stat().st_size > 50 * 1024 * 1024:  # > 50MB
                trace.unlink(missing_ok=True)
                log(f"Cleaned oversized mulligan trace for {profile_dir.name}")
        # Copy card DB for eval before training JVM locks it
        src_db = REPO_ROOT / "db" / "cards.h2.mv.db"
        eval_db_dir = REPO_ROOT / "db_eval"
        eval_db_dir.mkdir(parents=True, exist_ok=True)
        dst_db = eval_db_dir / "cards.h2.mv.db"
        if src_db.exists() and (not dst_db.exists()
                                or src_db.stat().st_mtime > dst_db.stat().st_mtime):
            try:
                shutil.copy2(str(src_db), str(dst_db))
                log("Copied card DB for eval")
            except PermissionError as exc:
                log(f"Could not refresh eval card DB because it is locked: {exc}")

    def load_registry(self) -> List[Dict[str, Any]]:
        with self.registry_path.open("r") as f:
            entries = json.load(f)
        active = [e for e in entries if e.get("active") and e.get("train_enabled")]
        active.sort(key=lambda e: (int(e.get("priority", 1000)), str(e.get("profile", ""))))
        return active[:self.train_profiles]

    def _apply_episode_delta_target(self) -> None:
        """Translate TOTAL_EPISODES_DELTA into Java's absolute TOTAL_EPISODES target."""
        if self.total_episodes_delta <= 0:
            return
        starts: Dict[str, int] = {}
        for entry in self.selected_profiles:
            profile = str(entry["profile"])
            episode, _ = read_winrate(profile, 1)
            starts[profile] = episode
        self.start_episodes = starts
        max_start = max(starts.values()) if starts else 0
        previous_target = self.total_episodes
        self.total_episodes = max_start + self.total_episodes_delta
        log(
            "TOTAL_EPISODES_DELTA="
            f"{self.total_episodes_delta} from selected profile counters; "
            f"absolute TOTAL_EPISODES target set to {self.total_episodes} "
            f"(previous TOTAL_EPISODES={previous_target})"
        )
        for profile, start in sorted(starts.items()):
            planned = max(0, self.total_episodes - start)
            log(f"[TARGET] {profile}: start_episode={start} planned_delta_at_least={planned}")

    def _common_train_env(self, key: str) -> Optional[str]:
        values = {
            str(e.get("train_env", {}).get(key, "")).strip()
            for e in self.selected_profiles
            if str(e.get("train_env", {}).get(key, "")).strip()
        }
        if len(values) == 1:
            return next(iter(values))
        return None

    def _auto_export_onnx(self) -> None:
        """Check if any profile needs ONNX export and do it live."""
        self.export_onnx_models()

    def export_onnx_models(self) -> None:
        """Pre-export ONNX models for all profiles (must run before GPU service starts)."""
        if os.getenv("USE_TRT_INFERENCE", "1") != "1":
            return
        # fp32 exports: fp16 conversion flattens probs ~2x and poisons PPO
        # ratios (root cause of the 2026-06 training collapses).
        os.environ.setdefault("ONNX_EXPORT_FP16", "0")
        sys.path.insert(0, str(MLCODE))
        try:
            from onnx_export import export_all_heads
        except ImportError as exc:
            log(f"ONNX export not available, skipping TRT: {exc}")
            return
        for entry in self.selected_profiles:
            profile = str(entry["profile"])
            models_dir = PROFILES_ROOT / profile / "models"
            model_path = models_dir / "model_latest.pt"
            onnx_dir = models_dir / "onnx"
            # Apply per-profile train_env for the ENTIRE fresh-init + ONNX-export
            # span. Without this the ONNX exporter builds its MTGTransformerModel
            # with env-var defaults (d_model=128, num_layers=2) and fails to
            # load a state_dict from a profile saved with MODEL_D_MODEL=256 etc.
            # Observed 2026-04-23: `size mismatch for cls_token [1,1,256] vs
            # [1,1,128]` blocked training on Pauper-Standard-MCTS-fresh.
            old_env = {}
            for k, v in entry.get("train_env", {}).items():
                old_env[k] = os.environ.get(k)
                os.environ[str(k)] = str(v)
            old_profile = os.environ.get("MODEL_PROFILE")
            os.environ["MODEL_PROFILE"] = profile
            try:
                if not model_path.exists():
                    # Fresh-start: initialize a random-weights PyTorch model and save
                    # it so ONNX export can run immediately. Without this, the trainer
                    # spends ~10 min on slow PyTorch inference before the first
                    # checkpoint save triggers ONNX export organically.
                    log(f"[TRT] No model for {profile}, initializing fresh random-weights model...")
                    try:
                        models_dir.mkdir(parents=True, exist_ok=True)
                        from py4j_entry_point import PythonEntryPoint
                        ep = PythonEntryPoint()
                        ep.initializeModel()
                        ep.saveLatestModelAtomic(str(model_path))
                        del ep
                        log(f"[TRT] Wrote fresh random model for {profile} -> {model_path}")
                    except Exception as e:
                        log(f"[TRT] Failed to initialize fresh model for {profile}: {e}; skipping export")
                        continue
                # Skip if the active ONNX export is newer than the model and
                # matches the current export version. Active exports are
                # versioned directories selected by .active_dir; this avoids
                # overwriting files held open by ONNX Runtime on Windows.
                ONNX_EXPORT_VERSION = "3"  # v3 = adds belief head for archetype classification
                pointer = onnx_dir / ".active_dir"
                active_onnx_dir = _resolve_active_onnx_dir(onnx_dir)
                onnx_action = active_onnx_dir / "model_action.onnx"
                onnx_ver_file = active_onnx_dir / ".export_version"
                current_ver = onnx_ver_file.read_text().strip() if onnx_ver_file.exists() else ""
                if (pointer.exists()
                        and onnx_action.exists()
                        and onnx_action.stat().st_mtime >= model_path.stat().st_mtime
                        and current_ver == ONNX_EXPORT_VERSION):
                    log(f"[TRT] ONNX up-to-date for {profile}")
                    continue
                log(f"[TRT] Exporting ONNX for {profile}...")
                try:
                    stamp = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S_%f")
                    stage_dir = onnx_dir / (stamp + "_staging")
                    final_dir = onnx_dir / stamp
                    if stage_dir.exists():
                        shutil.rmtree(stage_dir, ignore_errors=True)
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    export_all_heads(str(model_path), str(stage_dir))
                    (stage_dir / ".export_version").write_text(ONNX_EXPORT_VERSION, encoding="utf-8")
                    last_rename_error = None
                    for attempt in range(6):
                        try:
                            stage_dir.rename(final_dir)
                            last_rename_error = None
                            break
                        except PermissionError as e:
                            last_rename_error = e
                            time.sleep(0.5 * (attempt + 1))
                    if last_rename_error is not None:
                        raise last_rename_error
                    pointer.write_text(final_dir.name, encoding="utf-8")
                    log(f"[TRT] Exported {profile} -> {final_dir.name}")
                    _prune_versioned_onnx_dirs(onnx_dir, keep=3)
                except Exception as e:
                    log(f"[TRT] Export failed for {profile}: {e}")
                    try:
                        if 'stage_dir' in locals() and stage_dir.exists():
                            shutil.rmtree(stage_dir, ignore_errors=True)
                    except Exception:
                        pass
            finally:
                if old_profile is None:
                    os.environ.pop("MODEL_PROFILE", None)
                else:
                    os.environ["MODEL_PROFILE"] = old_profile
                for k, prev in old_env.items():
                    if prev is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = prev
            # Also export mulligan model if it exists
            mulligan_path = models_dir / "mulligan_model.pt"
            mulligan_onnx = onnx_dir / "mulligan_model.onnx"
            if mulligan_path.exists():
                if mulligan_onnx.exists() and mulligan_onnx.stat().st_mtime >= mulligan_path.stat().st_mtime:
                    pass  # up to date
                else:
                    try:
                        from onnx_export_mulligan import export_mulligan
                        export_mulligan(str(mulligan_path), str(onnx_dir))
                        log(f"[TRT] Exported mulligan for {profile}")
                    except Exception as e:
                        log(f"[TRT] Mulligan export failed for {profile}: {e}")

    def start_gpu_service(self) -> None:
        # If GPU_SERVICE_ENDPOINT is set externally, skip local GPU service
        ext_endpoint = os.getenv("GPU_SERVICE_ENDPOINT", "")
        if ext_endpoint and ext_endpoint != f"localhost:{self.port}":
            log(f"Using external GPU service at {ext_endpoint}, skipping local start")
            self._external_gpu = True
            # Wait for external service readiness
            host = ext_endpoint.split(":")[0]
            metrics_port = int(os.getenv("GPU_SERVICE_METRICS_PORT", str(self.metrics_port)))
            import urllib.request
            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    urllib.request.urlopen(f"http://{host}:{metrics_port}/metrics", timeout=2)
                    log(f"External GPU service ready at {host}:{metrics_port}")
                    return
                except Exception:
                    time.sleep(0.5)
            raise RuntimeError(f"External GPU service at {ext_endpoint} not reachable")

        python = sys.executable
        script = str(MLCODE / "gpu_service_host.py")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["GPU_SERVICE_PORT"] = str(self.port)
        env["GPU_SERVICE_METRICS_PORT"] = str(self.metrics_port)
        # The shared learner must instantiate the same model shape as the
        # Java/ONNX side. Apply any train_env values that are common to all
        # selected profiles before Python loads checkpoints.
        common_keys = {
            str(k)
            for entry in self.selected_profiles
            for k in (entry.get("train_env") or {}).keys()
        }
        for key in sorted(common_keys):
            value = self._common_train_env(key)
            if value is not None:
                env.setdefault(key, value)
        env["PY_BATCH_TIMEOUT_MS"] = os.getenv("PY_BATCH_TIMEOUT_MS", "25")
        env["PY_BATCH_MAX_SIZE"] = os.getenv("PY_BATCH_MAX_SIZE", "256")
        env["GPU_SERVICE_TRAIN_BATCH_TIMEOUT_MS"] = os.getenv(
            "GPU_SERVICE_TRAIN_BATCH_TIMEOUT_MS",
            os.getenv("GPU_SERVICE_LOCAL_TRAIN_BATCH_TIMEOUT_MS", "250"),
        )
        env["TRAIN_WORKER_THREADS"] = os.getenv("TRAIN_WORKER_THREADS", "2")
        env["TRAIN_GPU_MAX_CONCURRENT"] = os.getenv("TRAIN_GPU_MAX_CONCURRENT", "1")
        # Cap pending train queue so memory stays bounded when JVMs generate
        # episodes faster than the GPU can train. Default policy blocks
        # producers instead of silently dropping completed self-play games.
        env.setdefault("PENDING_TRAIN_MAX", "32")
        env["PENDING_TRAIN_BACKPRESSURE"] = os.getenv("PENDING_TRAIN_BACKPRESSURE", "block")
        env["PENDING_TRAIN_OFFER_TIMEOUT_MS"] = os.getenv("PENDING_TRAIN_OFFER_TIMEOUT_MS", "30000")
        env["SCORE_WORKER_THREADS"] = os.getenv("SCORE_WORKER_THREADS", "0")
        env["USE_TRT_INFERENCE"] = os.getenv("USE_TRT_INFERENCE", "0")
        # Smaller batches reduce PyTorch peak VRAM during forward/backward pass,
        # critical when sharing GPU with ONNX inference.
        env["LEARNER_BATCH_MAX_EPISODES"] = os.getenv("LEARNER_BATCH_MAX_EPISODES", "2")
        env["LEARNER_BATCH_MAX_STEPS"] = os.getenv("LEARNER_BATCH_MAX_STEPS", "1024")
        env["TRAIN_MULTI_MAX_STEPS"] = os.getenv("TRAIN_MULTI_MAX_STEPS", "1024")
        env["TRAIN_CHUNK_SIZE"] = os.getenv("TRAIN_CHUNK_SIZE", "128")
        env["TRAIN_VRAM_GUARD_ENABLE"] = os.getenv("TRAIN_VRAM_GUARD_ENABLE", "1")
        env["TRAIN_MIN_FREE_VRAM_MB"] = os.getenv("TRAIN_MIN_FREE_VRAM_MB", "2048")
        env["TRAIN_MAX_USED_VRAM_FRAC"] = os.getenv("TRAIN_MAX_USED_VRAM_FRAC", "0.84")
        env["AUTO_TRAIN_MB_PER_STEP_INIT"] = os.getenv("AUTO_TRAIN_MB_PER_STEP_INIT", "2.5")
        env["CUDA_EMPTY_CACHE_AFTER_TRAIN"] = os.getenv("CUDA_EMPTY_CACHE_AFTER_TRAIN", "1")
        env["TRAIN_CUDA_DEVICE"] = os.getenv("TRAIN_CUDA_DEVICE", "cuda:0")
        # Disable PPO value clipping -- vf_clip=0.2 traps the value head
        # at whatever constant it first converges to
        env.setdefault("PPO_VF_CLIP", "0")
        # Higher critic LR so value head can learn faster
        env.setdefault("CRITIC_LR", "5e-4")
        # Entropy schedule: decay over 500K train steps (~1M episodes)
        # to match expected 2M+ episode saturation horizon.
        # Default 50K steps decays too fast, locking policy at 10% of training.
        env.setdefault("ENTROPY_START", "0.10")
        env.setdefault("ENTROPY_END", "0.005")
        env.setdefault("ENTROPY_DECAY_STEPS", "530000")
        env.setdefault("TEMPERATURE_FLOOR", "0.05")
        # PyTorch training shares GPU with ONNX inference.
        # ONNX capped at 2GB in JVM (via ONNX_GPU_MEM_LIMIT_MB in start_trainer).
        # PyTorch gets 55% of 12GB = 6.6GB. Total ~8.6GB, leaving 3.7GB headroom.
        env.setdefault("TRAIN_CUDA_DEVICE", "cuda:0")
        env.setdefault("INFER_CUDA_DEVICE", "cpu")
        env.setdefault("CUDA_MEM_FRACTION", "0.55")
        _validate_cuda_device_visible(env, "TRAIN_CUDA_DEVICE")
        _validate_cuda_device_visible(env, "INFER_CUDA_DEVICE")
        log("GPU service config: "
            f"train_device={env.get('TRAIN_CUDA_DEVICE')} "
            f"infer_device={env.get('INFER_CUDA_DEVICE')} "
            f"train_concurrency={env.get('TRAIN_GPU_MAX_CONCURRENT')} "
            f"train_workers={env.get('TRAIN_WORKER_THREADS')} "
            f"learner_batch_episodes={env.get('LEARNER_BATCH_MAX_EPISODES')} "
            f"learner_batch_steps={env.get('LEARNER_BATCH_MAX_STEPS')}")
        log_path = REPO_ROOT / "local-training" / "local_pbt" / "gpu_service.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8", errors="replace")
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
        _prepend_cuda_paths(env)
        # In hybrid mode Java ONNX handles inference directly and the shared
        # Python GPU service is used only for training. Keep that fast path by
        # default, but let INFER_CUDA_DEVICE=cuda:N steer ONNX to a secondary
        # GPU. Set PY_SERVICE_MODE=shared_gpu explicitly to route scoring
        # through the Python service instead.
        infer_device = os.getenv("INFER_CUDA_DEVICE", "").strip().lower()
        if infer_device.startswith("cuda:") and not os.getenv("ONNX_CUDA_DEVICE_ID", "").strip():
            env["ONNX_CUDA_DEVICE_ID"] = infer_device.split(":", 1)[1]
        env["PY_SERVICE_MODE"] = os.getenv("PY_SERVICE_MODE", "hybrid")
        _validate_cuda_device_visible(env, "TRAIN_CUDA_DEVICE")
        _validate_cuda_device_visible(env, "INFER_CUDA_DEVICE")
        env["ONNX_FORCE_CPU"] = os.getenv("ONNX_FORCE_CPU", "0")
        ext_endpoint = os.getenv("GPU_SERVICE_ENDPOINT", "")
        if ext_endpoint and ext_endpoint != f"localhost:{self.port}":
            env["GPU_SERVICE_ENDPOINT"] = ext_endpoint
        else:
            env["GPU_SERVICE_ENDPOINT"] = f"localhost:{self.port}"
        env["GPU_SERVICE_NUM_GPUS"] = "1"
        env["GPU_SERVICE_NUM_CHANNELS"] = "4"
        # GPU training is fast (~300ms), keep reasonable timeout
        env.setdefault("GPU_SERVICE_LOCAL_TRAIN_BATCH_TIMEOUT_MS", "15000")
        env["TRAIN_PROFILES_LIST"] = profiles_str
        env["MODE"] = "trainAll"
        # For hybrid/onnx mode, set MODEL_PROFILE so ONNX path resolves.
        # In multi-profile mode, set to first profile (training target)
        # so MultiProfileOnnxRouter gets constructed.
        if profile_names:
            env["MODEL_PROFILE"] = profile_names[0]
        env["NUM_GAME_RUNNERS"] = str(self.num_runners)
        env["TOTAL_EPISODES"] = str(self.total_episodes)
        env["WINRATE_WINDOW"] = str(self.winrate_window)
        # Opponent sampler default:
        #   multi-profile -> "meta" (routes to other profiles' models)
        #   single-profile -> "adaptive" (CP7 curriculum WEAK/MEDIUM/STRONG/SELFPLAY)
        # Previously defaulted single-profile to "self" which silently disabled the
        # entire curriculum system — any ADAPTIVE_CURRICULUM and THRESHOLD_* env
        # vars had no effect because createTrainingOpponent dispatched to
        # straight self-play. Discovered 2026-04-22: wasted ~4 hours of
        # "curriculum" experiments that were actually pure self-play.
        registry_sampler = self._common_train_env("OPPONENT_SAMPLER")
        default_sampler = registry_sampler or ("meta" if len(self.selected_profiles) > 1 else "adaptive")
        env["OPPONENT_SAMPLER"] = os.getenv("OPPONENT_SAMPLER", default_sampler)
        env["MULLIGAN_DEVICE"] = "cpu"
        # Cap ONNX VRAM in the JVM. Models are 170MB total (FP16) but ORT's
        # default arena grabs 5GB. 2048MB is generous for actual inference needs
        # (~500MB model weights + workspace) and leaves room for PyTorch training.
        env.setdefault("ONNX_GPU_MEM_LIMIT_MB", "2048")
        # Cap JVM heap to force aggressive GC -- without this, the JVM grows
        # to 19GB+ over an hour as game objects accumulate in old gen.
        env["MAVEN_OPTS"] = env.get("MAVEN_OPTS", "") + " -Xmx10g -Xms4g -XX:+UseG1GC -XX:MaxGCPauseMillis=50"
        env["GAME_LOG_FREQUENCY"] = os.getenv("GAME_LOG_FREQUENCY", "500")
        env["LEAGUE_REGISTRY_PATH"] = str(self.registry_path)
        env["ORCHESTRATED_RUN"] = "1"
        # applyEffects dirty-flag skip-gate — ~45% of applyEffects calls skipped
        # on Pauper workloads. Regresses 5 complex cards (morph/Yixlid/Shadowbane)
        # but none appear in our 4 RL decks. Verified 2026-04-23 via full
        # Mage.Tests suite parity + grep across all Pauper .dek files.
        env.setdefault("MAGE_DIRTY_APPLY", "1")
        if deck_paths:
            env["DECK_LIST_FILE"] = deck_paths[0]
        # Apply per-profile train_env from registry
        for entry in self.selected_profiles:
            train_env = entry.get("train_env", {})
            for k, v in train_env.items():
                env.setdefault(str(k), str(v))
        log("Trainer config: "
            f"service_mode={env.get('PY_SERVICE_MODE')} "
            f"infer_device={env.get('INFER_CUDA_DEVICE', 'cpu')} "
            f"onnx_device_id={env.get('ONNX_CUDA_DEVICE_ID', '')} "
            f"opponent_sampler={env.get('OPPONENT_SAMPLER')} "
            f"onnx_mem_limit_mb={env.get('ONNX_GPU_MEM_LIMIT_MB')} "
            f"total_episodes={env.get('TOTAL_EPISODES')}")
        args_str = "trainAll " + " ".join(profile_names)
        mvn_exe = shutil.which("mvn") or shutil.which("mvn.cmd") or "mvn.cmd"
        cmd = [
            mvn_exe,
            "-q",
            "-pl", "Mage.Server.Plugins/Mage.Player.AIRL",
            "-am",
            "-DskipTests",
            "exec:java",
            "-Dexec.mainClass=mage.player.ai.rl.RLTrainer",
            "-Dexec.args=" + args_str,
        ]
        log_handle = self.trainer_log_path.open("w", encoding="utf-8", errors="replace")
        self.trainer_process = subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=log_handle, stderr=subprocess.STDOUT,
        )
        log(f"Trainer started pid={self.trainer_process.pid} profiles={profiles_str} runners={self.num_runners}")

    def _archive_trainer_log(self, reason: str) -> None:
        """Preserve trainer diagnostics that would otherwise be overwritten on restart."""
        log_path = self.trainer_log_path
        if not log_path.exists():
            return
        try:
            safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason).strip("_") or "trainer"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_dir = REPO_ROOT / "local-training" / "local_pbt" / "trainer_logs"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{stamp}_{safe_reason}.log"
            shutil.copy2(str(log_path), str(archive_path))
            log(f"Archived trainer log: {archive_path}")
            interesting = (
                "MCTS_GATE", "EVAL_SUMMARY", "GENERIC_CHOOSE_DIAG",
                "ISMCTS-INIT", "MCTS_STATS", "mcts_activations", "PolicyValueMCTS"
            )
            emitted = 0
            with log_path.open("r", encoding="utf-8", errors="replace") as fh:
                recent = deque(fh, maxlen=400)
            for line in recent:
                if any(token in line for token in interesting):
                    log("[TRAINER] " + line.rstrip())
                    emitted += 1
            if emitted == 0:
                log("[TRAINER] no MCTS/eval diagnostics found in trainer log tail")
        except Exception as exc:
            log(f"Failed to archive trainer log: {exc}")

    def stop_trainer(self) -> None:
        if self.trainer_process and self.trainer_process.poll() is None:
            _stop_process_tree(self.trainer_process, "trainer", timeout=15)
        self.trainer_process = None
        time.sleep(2)
        log("Trainer stopped")

    def restart_trainer(self, reason: str) -> None:
        log(f"Restarting trainer: {reason}")
        self.stop_trainer()
        time.sleep(2)
        self.start_trainer()

    def _restart_gpu_service(self) -> None:
        """Restart GPU service to flush leaked connections and reclaim memory."""
        if self.gpu_process and self.gpu_process.poll() is None:
            _stop_process_tree(self.gpu_process, "GPU service", timeout=10)
        self.gpu_process = None
        time.sleep(2)
        self.start_gpu_service()

    def _resolve_deck_pool(self, deck_path: str) -> List[str]:
        """Resolve a .txt pool file to individual .dek paths."""
        decks = []
        dp = Path(deck_path)
        if dp.suffix == '.txt' and dp.exists():
            try:
                for line in dp.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        resolved = dp.parent / line
                        if resolved.exists():
                            decks.append(str(resolved))
            except Exception:
                pass
        if not decks:
            decks = [deck_path]
        return decks

    def _start_eval_background(self) -> None:
        """Launch eval in a background thread so training continues."""
        if getattr(self, '_eval_thread', None) and self._eval_thread.is_alive():
            log("[EVAL] Skipping -- previous eval still running")
            return
        self._eval_thread = threading.Thread(target=self._run_eval_sync, daemon=True,
                                              name="EvalThread")
        self._eval_thread.start()
        log("[EVAL] Started background evaluation")

    def _run_eval_sync(self) -> None:
        """Run all matchups synchronously (called from background thread)."""
        try:
            results = self._run_eval_inner()
            # Update results on main thread's state
            matchup_data = {}
            for k in list(results.keys()):
                if k.startswith("_matchups_"):
                    matchup_data[k.replace("_matchups_", "")] = results.pop(k)
            self.eval_results = results
            # Persist
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for profile, wr in results.items():
                try:
                    ep, _ = read_winrate(profile, 1)
                    eval_path = PROFILES_ROOT / profile / "logs" / "eval_history.csv"
                    eval_path.parent.mkdir(parents=True, exist_ok=True)
                    matchups = matchup_data.get(profile, [])
                    matchup_str = "  ".join(
                        f"{m[0]}_vs_{m[1]}={m[2]}/{m[3]}" for m in matchups)
                    write_header = not eval_path.exists()
                    with eval_path.open("a") as f:
                        if write_header:
                            f.write("timestamp,episode,winrate,matchups\n")
                        f.write(f"{ts},{ep},{wr:.4f},{matchup_str}\n")
                except Exception as exc:
                    log(f"[EVAL] Failed to write eval history for {profile}: {exc}")
            log("[EVAL] Background evaluation complete")
        except Exception as exc:
            log(f"[EVAL] Background evaluation failed: {exc}")

    def _run_eval_inner(self) -> Dict[str, float]:
        """Run full NxN matchup eval for all profiles, parallelized by agent deck."""
        log("[EVAL] Starting full matchup evaluation")
        eval_skill = os.getenv("EVAL_OPPONENT_SKILL", "7")
        results: Dict[str, float] = {}

        for entry in self.selected_profiles:
            profile = str(entry["profile"])
            deck_path = str(entry.get("deck_path", ""))
            if not deck_path:
                continue

            agent_decks = self._resolve_deck_pool(deck_path)
            opp_decks = list(agent_decks)

            # Clear eval logs dir
            eval_logs_dir = PROFILES_ROOT / profile / "logs" / "eval"
            if eval_logs_dir.exists():
                shutil.rmtree(eval_logs_dir, ignore_errors=True)

            # Ensure per-worker DB copies exist (H2 file lock prevents sharing)
            for i in range(len(agent_decks)):
                db_dir = REPO_ROOT / f"db_eval_{i}"
                db_dir.mkdir(parents=True, exist_ok=True)
                dst = db_dir / "cards.h2.mv.db"
                src = REPO_ROOT / "db_eval" / "cards.h2.mv.db"
                if src.exists() and not dst.exists():
                    shutil.copy2(str(src), str(dst))

            # Launch one JVM per agent deck in parallel, each plays all opponents
            pending: List[Tuple[str, List[str], subprocess.Popen]] = []
            for idx, agent_deck in enumerate(agent_decks):
                agent_name = Path(agent_deck).stem
                opp_names = []
                for opp_deck in opp_decks:
                    opp_name = Path(opp_deck).stem
                    opp_names.append(opp_name)
                    deck_log_dir = eval_logs_dir / f"{agent_name}_vs_{opp_name}"
                    deck_log_dir.mkdir(parents=True, exist_ok=True)

                # Run all opponents sequentially within one JVM by launching
                # separate processes per opponent but in parallel across agents
                for opp_idx, opp_deck in enumerate(opp_decks):
                    opp_name = Path(opp_deck).stem
                    env = dict(os.environ)
                    _prepend_cuda_paths(env)
                    env["MODE"] = "league_bench"
                    env["MODEL_PROFILE"] = profile
                    env["RL_AGENT_DECK_LIST"] = agent_deck
                    env["EVAL_OPPONENT_DECK"] = opp_deck
                    env["EVAL_OPPONENT_SKILL"] = eval_skill
                    env["EVAL_NUM_GAMES"] = "2"
                    env["GAME_LOG_FREQUENCY"] = "1"
                    deck_log_dir = eval_logs_dir / f"{agent_name}_vs_{opp_name}"
                    env["GAME_LOG_DIR"] = str(deck_log_dir)
                    env["PY_SERVICE_MODE"] = "hybrid"
                    env["GPU_SERVICE_ENDPOINT"] = f"localhost:{self.port}"
                    env["GPU_SERVICE_NUM_GPUS"] = "1"
                    # Force eval ONNX to CPU so it doesn't contend with
                    # training's ONNX GPU sessions
                    env["CUDA_VISIBLE_DEVICES"] = ""
                    env["GPU_SERVICE_NUM_CHANNELS"] = "4"
                    db_dir = REPO_ROOT / f"db_eval_{idx}"
                    env["MAGE_DB_DIR"] = str(db_dir.resolve())
                    env["MAGE_DB_AUTO_SERVER"] = "false"
                    env["DECK_LIST_FILE"] = deck_path
                    env["LEAGUE_REGISTRY_PATH"] = str(self.registry_path)

                    cmd = (
                        f'mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL '
                        f'-am -DskipTests exec:java '
                        f'-Dexec.mainClass=mage.player.ai.rl.RLTrainer '
                        f'"-Dexec.args=league_bench"'
                    )
                    # Queue matchup -- will be dispatched in batches
                    pending.append((
                        f"{agent_name}_vs_{opp_name}",
                        [agent_name, opp_name],
                        cmd, env, idx,
                    ))

            # Run in parallel batches: one matchup per DB slot at a time
            # (each DB copy supports one JVM). N agent decks = N parallel slots.
            num_slots = len(agent_decks)
            total_wins = 0
            total_games = 0
            matchup_results: List[Tuple[str, str, int, int]] = []

            # Group by DB slot (idx) so same-slot jobs run sequentially
            from collections import defaultdict
            slot_queues: Dict[int, list] = defaultdict(list)
            for key, names, cmd, env, slot_idx in pending:
                slot_queues[slot_idx].append((key, names, cmd, env))

            # Process all slots in parallel using threads
            import concurrent.futures
            def run_slot(jobs):
                slot_results = []
                for key, names, cmd, env in jobs:
                    agent_name, opp_name = names
                    deck_wins = 0
                    deck_total = 0
                    try:
                        proc = subprocess.run(
                            cmd, env=env, cwd=str(REPO_ROOT), shell=True,
                            capture_output=True, text=True, timeout=600,
                        )
                        for line in (proc.stdout or "").splitlines():
                            if "EVAL_RESULT:" in line:
                                for tok in line.split():
                                    if tok.startswith("wins="):
                                        deck_wins = int(tok.split("=")[1])
                                    elif tok.startswith("total="):
                                        deck_total = int(tok.split("=")[1])
                    except subprocess.TimeoutExpired:
                        log(f"[EVAL] TIMEOUT: {agent_name} vs {opp_name}")
                    except Exception as exc:
                        log(f"[EVAL] ERROR: {agent_name} vs {opp_name}: {exc}")
                    result_char = "W" if deck_wins > 0 else "L"
                    log(f"[EVAL] {agent_name} vs {opp_name}: {result_char}")
                    slot_results.append((agent_name, opp_name, deck_wins, deck_total))
                return slot_results

            # Limit to 1 parallel eval JVM -- 2 caused throughput collapse
            # from CPU contention with 96 training runners
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                futures = [pool.submit(run_slot, jobs) for jobs in slot_queues.values()]
                for f in concurrent.futures.as_completed(futures):
                    for agent_name, opp_name, dw, dt in f.result():
                        total_wins += dw
                        total_games += dt
                        matchup_results.append((agent_name, opp_name, dw, dt))

            wr = total_wins / max(1, total_games)
            results[profile] = wr
            log(f"[EVAL] {profile} overall: {total_wins}/{total_games} = {wr:.3f}")
            results["_matchups_" + profile] = matchup_results

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
        eval_wr = self.eval_results
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
        started_at = time.time()
        while not self.stop_requested:
            time.sleep(30)
            tick += 1
            if self.max_wall_seconds > 0 and (time.time() - started_at) >= self.max_wall_seconds:
                log(f"MAX_WALL_SECONDS={self.max_wall_seconds} reached; stopping orchestrator")
                self.stop_requested = True
                break

            # Check trainer alive
            if self.trainer_process and self.trainer_process.poll() is not None:
                rc = self.trainer_process.returncode
                log(f"Trainer exited with rc={rc}")
                self._archive_trainer_log(f"trainer_exit_rc_{rc}")
                if rc == 0 and self._all_profiles_reached_target():
                    log(f"All selected profiles reached TOTAL_EPISODES={self.total_episodes}; stopping orchestrator")
                    self.stop_requested = True
                    break
                if rc == 0 and self._all_profiles_within_target_tolerance():
                    log(f"Selected profiles reached TOTAL_EPISODES={self.total_episodes} within tolerance; stopping orchestrator")
                    self.stop_requested = True
                    break
                if not self.stop_requested:
                    self.restart_trainer(f"exit rc={rc}")
                continue

            # Print human-readable status
            now = time.time()
            for entry in self.selected_profiles:
                profile = str(entry["profile"])
                stats_path = PROFILES_ROOT / profile / "logs" / "stats" / "training_stats.csv"
                ep_total, _ = read_winrate(profile, 1)  # max episode number, not line count
                # Compute eps/sec from last check
                prev_ep = getattr(self, '_prev_ep_count', {}).get(profile, ep_total)
                prev_time = getattr(self, '_prev_ep_time', {}).get(profile, now)
                dt = now - prev_time
                eps_sec = (ep_total - prev_ep) / dt if dt > 0 else 0
                if not hasattr(self, '_prev_ep_count'):
                    self._prev_ep_count = {}
                    self._prev_ep_time = {}
                self._prev_ep_count[profile] = ep_total
                self._prev_ep_time[profile] = now
                eval_wr = self.eval_results.get(profile)
                eval_str = f"  eval_wr: {eval_wr:.1%}" if eval_wr is not None else ""
                status_line = f"[{profile}] episodes: {ep_total:,}  eps/s: {eps_sec:.1f}{eval_str}"
                log(status_line)
                # Write status file under profile dir for easy monitoring
                try:
                    status_path = PROFILES_ROOT / profile / "logs" / "status.txt"
                    status_path.parent.mkdir(parents=True, exist_ok=True)
                    with status_path.open("w") as f:
                        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        f.write(f"updated: {ts}\n")
                        f.write(f"episodes: {ep_total:,}\n")
                        f.write(f"eps/sec: {eps_sec:.1f}\n")
                        if eval_wr is not None:
                            f.write(f"eval_winrate: {eval_wr:.1%}\n")
                        # Append recent eval history
                        eval_path = PROFILES_ROOT / profile / "logs" / "eval_history.csv"
                        if eval_path.exists():
                            f.write("\n-- eval history --\n")
                            lines = eval_path.read_text().strip().split("\n")
                            for line in lines[-11:]:  # header + last 10
                                f.write(line + "\n")
                except Exception:
                    pass

            # League NxN eval disabled -- wasteful, evaluates model on decks it doesn't train on.
            # Per-checkpoint eval (runEvalCheckpoint in RLTrainer.java) handles eval vs CP7.
            # if elapsed_since_last_eval >= self.exploit_interval_min * 60:
            #     self._last_eval_time = time.time()
            #     self._start_eval_background()

            # Auto-export ONNX for profiles whose model has updated since last export.
            # Default is every 20 ticks = ~10 min. Export+reload disrupts inference
            # briefly, so long autonomous runs can raise this interval for steadier
            # throughput at the cost of slightly staler inference weights.
            if self.onnx_export_interval_ticks > 0 and tick % self.onnx_export_interval_ticks == 0:
                self._auto_export_onnx()

            # PBT exploitation check every other tick
            if tick % 2 == 0:
                self.check_exploit()

    def _all_profiles_reached_target(self) -> bool:
        if self.total_episodes <= 0:
            return False
        for entry in self.selected_profiles:
            profile = str(entry["profile"])
            ep_total, _ = read_winrate(profile, 1)
            if ep_total < self.total_episodes:
                return False
        return True

    def _all_profiles_within_target_tolerance(self) -> bool:
        if self.total_episodes <= 0:
            return False
        tolerance = env_int("PBT_TARGET_EXIT_TOLERANCE", max(1, self.num_runners))
        if tolerance <= 0:
            return False
        threshold = max(0, self.total_episodes - tolerance)
        for entry in self.selected_profiles:
            profile = str(entry["profile"])
            ep_total, _ = read_winrate(profile, 1)
            if ep_total < threshold:
                return False
        return True

    def _validate_startup_cuda_devices(self) -> None:
        env = {
            "TRAIN_CUDA_DEVICE": os.getenv("TRAIN_CUDA_DEVICE", "cuda:0"),
            "INFER_CUDA_DEVICE": os.getenv("INFER_CUDA_DEVICE", "cpu"),
        }
        _validate_cuda_device_visible(env, "TRAIN_CUDA_DEVICE")
        _validate_cuda_device_visible(env, "INFER_CUDA_DEVICE")

    def run(self) -> int:
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop_requested', True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop_requested', True))

        # Cleanup temp files from previous runs
        self._cleanup_temp_files()

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
        log(f"Requested total game runners: {self.num_runners} (Java trainAll splits across profiles)")
        self._apply_episode_delta_target()
        if self.max_wall_seconds > 0:
            log(f"Wall-clock stop: MAX_WALL_SECONDS={self.max_wall_seconds}")
        if self.total_episodes_delta > 0:
            log(f"Episode-delta stop: TOTAL_EPISODES_DELTA={self.total_episodes_delta}")
        log(f"PBT: exploit_interval={self.exploit_interval_min}min min_episodes={self.min_episodes} "
            f"episode_delta={self.episode_delta} min_gap={self.min_winner_gap} mutation={self.mutation_pct}")
        self._validate_startup_cuda_devices()

        try:
            self.export_onnx_models()
            self.start_gpu_service()
            self.start_trainer()
            self.monitor_loop()
        except KeyboardInterrupt:
            log("Interrupted")
        finally:
            self.stop_trainer()
            if self.gpu_process:
                _stop_process_tree(self.gpu_process, "GPU service", timeout=10)
                log("GPU service stopped")
        return 0


if __name__ == "__main__":
    sys.exit(LocalPBT().run())
