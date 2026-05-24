try:
    from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
except ImportError:
    ClientServer = JavaParameters = PythonParameters = None
from mtg_transformer import MTGTransformerModel
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import signal
import time
import threading
import queue
import argparse
from collections import deque
import struct
import random
import hashlib
import csv
from contextlib import nullcontext

# Import new modules
from logging_utils import logger, mulligan_logger, vram_logger, LogCategory, TEMP_DIR, script_dir, log_file, mulligan_log_file, vram_diag_log_file
from cuda_manager import CUDAManager
from snapshot_manager import SnapshotManager
from metrics_collector import MetricsCollector
from model_persistence import ModelPersistence
from gpu_lock import GPULock
from profile_paths import profile_models_dir, profile_logs_dir

HEAD_NAMES = ["action", "target", "card_select", "attack", "block", "mulligan"]

# Now we can safely log initialization
logger.info(LogCategory.SYSTEM_INIT, f"Logging to file: {log_file}")
logger.info(LogCategory.SYSTEM_INIT,
            f"VRAM diagnostics log file: {vram_diag_log_file}")
logger.info(LogCategory.SYSTEM_INIT,
            f"Created temporary directory for shared memory: {TEMP_DIR}")


def _maybe_set_cuda_memory_fraction():
    """
    Hard-cap CUDA allocator to avoid silent paging/spill into shared memory.
    Default: 0.90 (override with CUDA_MEM_FRACTION env var).
    """
    try:
        if not torch.cuda.is_available():
            return
        frac_s = os.getenv("CUDA_MEM_FRACTION", "0.90").strip()
        try:
            frac = float(frac_s)
        except Exception:
            frac = 0.90
        frac = max(0.05, min(1.0, frac))
        dev = int(os.getenv("CUDA_MEM_FRACTION_DEVICE", "0"))
        torch.cuda.set_per_process_memory_fraction(frac, device=dev)
        try:
            vram_logger.info(
                "VRAM",
                "CudaMemCap set_per_process_memory_fraction=%.3f device=%d",
                float(frac),
                int(dev),
            )
        except Exception:
            pass
    except Exception as e:
        try:
            logger.warning(LogCategory.SYSTEM_INIT,
                           "Failed to set CUDA per-process memory fraction: %s", str(e))
        except Exception:
            pass


def _env_truthy(name):
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _maybe_configure_deterministic_eval():
    if not _env_truthy("TORCH_DETERMINISTIC_EVAL"):
        return
    try:
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        logger.info(LogCategory.SYSTEM_INIT,
                    "Enabled deterministic eval torch settings")
    except Exception as e:
        logger.warning(LogCategory.SYSTEM_INIT,
                       "Failed to configure deterministic eval torch settings: %s", str(e))


_maybe_set_cuda_memory_fraction()
_maybe_configure_deterministic_eval()

# ------------------------------
# Runtime tuning via environment
# ------------------------------
try:
    BATCH_SAMPLE_MB = int(os.getenv("BATCH_SAMPLE_MB", "16"))
except Exception:
    BATCH_SAMPLE_MB = 16
try:
    BATCH_SAFETY_FRACTION = float(os.getenv("BATCH_SAFETY_FRACTION", "0.5"))
except Exception:
    BATCH_SAFETY_FRACTION = 0.5
try:
    BATCH_MAX_SAMPLES = int(os.getenv("BATCH_MAX_SAMPLES", "200000"))
except Exception:
    BATCH_MAX_SAMPLES = 200000

# Global variable to store the gateway
gateway = None


def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        for file in os.listdir(TEMP_DIR):
            try:
                os.remove(os.path.join(TEMP_DIR, file))
            except Exception as e:
                logger.warning(LogCategory.SYSTEM_CLEANUP,
                               f"Failed to remove temporary file {file}: {e}")
    except Exception as e:
        logger.warning(LogCategory.SYSTEM_CLEANUP,
                       f"Failed to clean up temporary directory: {e}")


def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(LogCategory.SYSTEM_CLEANUP,
                "Received termination signal, shutting down...")
    cleanup_temp_files()
    if gateway is not None:
        gateway.shutdown()
    sys.exit(0)


def register_signal_handlers():
    """Register signal handlers when running on the main interpreter thread."""
    if threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


register_signal_handlers()


def log_gpu_memory():
    """Log GPU memory usage if CUDA is available"""
    if torch.cuda.is_available():
        # Get current device
        device = torch.cuda.current_device()
        # Get memory info
        allocated = torch.cuda.memory_allocated(
            device) / 1024**2  # Convert to MB
        reserved = torch.cuda.memory_reserved(
            device) / 1024**2    # Convert to MB
        total = torch.cuda.get_device_properties(
            device).total_memory / 1024**2  # Total GPU memory in MB
        free = total - allocated  # Approximate free memory
        logger.info(LogCategory.GPU_MEMORY, "GPU Memory - Device %d:", device)
        logger.info(LogCategory.GPU_MEMORY, "  Allocated: %.2f MB", allocated)
        logger.info(LogCategory.GPU_MEMORY, "  Reserved:  %.2f MB", reserved)
        logger.info(LogCategory.GPU_MEMORY, "  Free:      %.2f MB", free)
        logger.info(LogCategory.GPU_MEMORY, "  Total:     %.2f MB", total)
        return free
    return None


def calculate_optimal_batch_size():
    """Calculate optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        logger.info(LogCategory.GPU_MEMORY,
                    "CUDA not available, using CPU default batch size: 1000")
        return 1000  # CPU fallback

    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        total = torch.cuda.get_device_properties(
            device).total_memory / 1024**2  # MB

        # Calculate truly available memory (accounting for fragmentation)
        available = total - reserved

        # Memory requirements per sample (MUCH higher than expected):
        # - Input data: 256 × 128 × 4 bytes = ~128KB
        # - Transformer attention: O(seq_len²) = 256² × 4 = ~256KB
        # - Model activations during forward: ~1-2MB per sample
        # - Gradient storage during backward: ~2-4MB per sample
        # - PyTorch memory fragmentation: ~2x overhead
        # Total observed: ~5-10MB per sample in practice!

        # Tunable parameters for GPU utilization
        sample_memory_kb = float(BATCH_SAMPLE_MB) * 1024.0
        pytorch_multiplier = 1.0
        safety_margin = float(BATCH_SAFETY_FRACTION)

        usable_memory_mb = available * (1.0 - safety_margin)
        memory_per_sample_mb = (sample_memory_kb * pytorch_multiplier) / 1024.0

        optimal_samples = int(usable_memory_mb / memory_per_sample_mb)

        # Sanity bounds
        optimal_samples = max(1, min(optimal_samples, int(BATCH_MAX_SAMPLES)))

        logger.info(LogCategory.GPU_MEMORY,
                    "GPU Memory Analysis:")
        logger.info(LogCategory.GPU_MEMORY,
                    "  Total VRAM: %.0f MB", total)
        logger.info(LogCategory.GPU_MEMORY,
                    "  Available: %.0f MB", available)
        logger.info(LogCategory.GPU_MEMORY,
                    "  Usable (after %.0f%% safety): %.0f MB",
                    safety_margin * 100, usable_memory_mb)
        logger.info(LogCategory.GPU_MEMORY,
                    "  Memory per sample: %.2f MB", memory_per_sample_mb)
        logger.info(LogCategory.GPU_MEMORY,
                    "  Optimal batch size: %d samples", optimal_samples)

        return optimal_samples

    except Exception as e:
        logger.error(LogCategory.GPU_MEMORY,
                     "Error calculating optimal batch size: %s", str(e))
        return 10000  # Conservative fallback


class PythonEntryPoint:
    def __init__(self):
        # Core model state
        self.model = None
        self.optimizer = None
        cuda_device_override = os.getenv("CUDA_DEVICE", "").strip()
        if cuda_device_override:
            self.device = torch.device(cuda_device_override)
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        seed_raw = os.getenv("PY_GLOBAL_SEED", "").strip()
        if not seed_raw:
            seed_raw = os.getenv("RL_BASE_SEED", "").strip()
        self.global_seed = None
        if seed_raw:
            try:
                self.global_seed = int(seed_raw)
            except Exception:
                self.global_seed = None
        if self.global_seed is not None:
            random.seed(self.global_seed)
            np.random.seed(self.global_seed & 0xFFFFFFFF)
            torch.manual_seed(self.global_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.global_seed)
            logger.info(LogCategory.SYSTEM_INIT,
                        "Applied global seed PY_GLOBAL_SEED=%d", int(self.global_seed))
        self.py_role = os.getenv("PY_ROLE", "learner").strip().lower()
        self.backend_mode = os.getenv(
            "PY_BACKEND_MODE", "multi").strip().lower()
        # Single-backend runs inference+training in one process; ensure training pauses inference.
        self._gpu_mutex = threading.Lock()
        self._model_init_lock = threading.Lock()

        # Training losses CSV path (resolved relative to RL logs dir)
        self.training_losses_csv_path = os.getenv(
            'TRAINING_LOSSES_PATH',
            'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/logs/stats/training_losses.csv'
        )
        self._training_losses_header_written = False

        # GPU coordination lock (inter-process)
        self.gpu_lock = GPULock()
        self.process_name = f"{self.py_role}_{os.getpid()}"

        # Initialize helper modules
        self.cuda_mgr = CUDAManager(self.py_role)
        self.snapshot_mgr = SnapshotManager(self.device)
        self.metrics = MetricsCollector()
        self.persistence = ModelPersistence()
        self.model_load_determinism_gate = self._env_flag(
            "RL_MODEL_LOAD_DETERMINISM_GATE",
            "EVAL_REPLAY_MODEL_LOAD_DETERMINISM_GATE",
        )
        self.model_load_determinism_gate_path = (
            os.getenv("RL_MODEL_LOAD_DETERMINISM_GATE_FILE", "").strip()
            or os.getenv("EVAL_REPLAY_MODEL_LOAD_DETERMINISM_GATE_FILE", "").strip()
        )
        if self.model_load_determinism_gate and not self.model_load_determinism_gate_path:
            self.model_load_determinism_gate_path = os.path.join(
                profile_logs_dir(), "python_model_load_determinism_gate.csv")
        self._model_load_determinism_gate_lock = threading.Lock()
        self._model_load_determinism_snapshot = {}
        self.fail_on_skipped_incompatible = self._env_flag(
            "RL_FAIL_ON_SKIPPED_INCOMPATIBLE",
            "EVAL_REPLAY_FAIL_ON_SKIPPED_INCOMPATIBLE",
        )
        if self.model_load_determinism_gate:
            self._initialize_model_load_determinism_gate()
        self.python_inference_duplicate_probe = self._env_flag(
            "RL_PYTHON_INFERENCE_DUPLICATE_PROBE",
            "EVAL_REPLAY_PYTHON_INFERENCE_DUPLICATE_PROBE",
        )
        self.python_inference_duplicate_probe_path = (
            os.getenv("RL_PYTHON_INFERENCE_DUPLICATE_PROBE_FILE", "").strip()
            or os.getenv("EVAL_REPLAY_PYTHON_INFERENCE_DUPLICATE_PROBE_FILE", "").strip()
        )
        if self.python_inference_duplicate_probe and not self.python_inference_duplicate_probe_path:
            self.python_inference_duplicate_probe_path = os.path.join(
                profile_logs_dir(), "python_inference_duplicate_probe.csv")
        self.python_inference_duplicate_probe_max_rows = self._env_int(
            "RL_PYTHON_INFERENCE_DUPLICATE_PROBE_MAX_ROWS",
            "EVAL_REPLAY_PYTHON_INFERENCE_DUPLICATE_PROBE_MAX_ROWS",
            default=256,
        )
        self._python_inference_duplicate_probe_rows = 0
        self._python_inference_duplicate_probe_lock = threading.Lock()
        if self.python_inference_duplicate_probe:
            self._initialize_python_inference_duplicate_probe()

        # PPO configuration
        self.ppo_epsilon = float(os.getenv('PPO_EPSILON', '0.2'))
        self.use_ppo = bool(int(os.getenv('USE_PPO', '1')))
        self.gamma = float(os.getenv('PPO_GAMMA', '0.99'))
        self._ppo_stats_every = int(os.getenv("PPO_STATS_EVERY", "50"))

        # Loss scheduling
        self.loss_schedule_enable = bool(
            int(os.getenv('LOSS_SCHEDULE_ENABLE', '1')))
        self.critic_warmup_steps = int(os.getenv('CRITIC_WARMUP_STEPS', '200'))
        self.freeze_encoder_in_warmup = bool(
            int(os.getenv('FREEZE_ENCODER_IN_WARMUP', '1')))
        self._encoder_frozen = False
        self.distill_head_only = bool(int(os.getenv('DISTILL_HEAD_ONLY', '0')))
        self.distill_policy_path_only = bool(int(os.getenv('DISTILL_POLICY_PATH_ONLY', '0')))
        self.candidate_q_only = bool(int(os.getenv('CANDIDATE_Q_ONLY', '0')))
        self.value_pair_rank_critic_only = bool(int(os.getenv('VALUE_PAIR_RANK_CRITIC_ONLY', '0')))
        self.belief_head_only = bool(int(os.getenv('BELIEF_HEAD_ONLY', '0')))
        self.card_belief_head_only = bool(int(os.getenv('CARD_BELIEF_HEAD_ONLY', '0')))

        # Loss coefficients
        self.policy_loss_coef_warmup = float(
            os.getenv('POLICY_LOSS_COEF_WARMUP', '0.0'))
        self.value_loss_coef_warmup = float(
            os.getenv('VALUE_LOSS_COEF_WARMUP', '20.0'))
        self.entropy_loss_mult_warmup = float(
            os.getenv('ENTROPY_LOSS_MULT_WARMUP', '0.0'))
        self.policy_loss_coef_main = float(
            os.getenv('POLICY_LOSS_COEF', '1.0'))
        self.value_loss_coef_main = float(os.getenv('VALUE_LOSS_COEF', '5.0'))
        self.entropy_loss_mult_main = float(
            os.getenv('ENTROPY_LOSS_MULT', '1.0'))
        # Phase 1 belief auxiliary loss coefficient. 0 disables the loss.
        self.belief_loss_coef = float(os.getenv('BELIEF_LOSS_COEF', '0.3'))
        # Generic hidden-card belief auxiliary loss coefficient. 0 disables.
        self.card_belief_loss_coef = float(os.getenv('CARD_BELIEF_LOSS_COEF', '0.0'))
        # AlphaZero policy distillation loss: cross-entropy between the model's
        # policy and the MCTS visit distribution at steps where MCTS was run.
        # 0 disables the loss (use when MCTS-generated targets aren't present).
        self.mcts_kl_loss_coef = float(os.getenv('MCTS_KL_LOSS_COEF', '1.0'))
        # Optional frozen-reference policy anchor. This is generic: it keeps
        # candidate distributions close to a previous checkpoint without using
        # action text, card names, or deck-specific labels.
        self.reference_policy_kl_coef = float(os.getenv('REFERENCE_POLICY_KL_COEF', '0.0'))
        self.mcts_reference_model_path = os.getenv('MCTS_REFERENCE_MODEL_PATH', '').strip()
        self.mcts_reference_model = None
        self.policy_ensemble_model_paths = self._parse_env_paths(
            os.getenv('POLICY_ENSEMBLE_MODEL_PATHS', ''))
        self.policy_ensemble_weight_spec = os.getenv(
            'POLICY_ENSEMBLE_WEIGHTS', '').strip()
        self.policy_ensemble_models = []
        self.policy_ensemble_weights = []
        self._policy_ensemble_loaded = False
        self._policy_ensemble_runtime_failures = set()

        # Auto-batching state (kept for backward compatibility)
        self._infer_safe_max = None
        self._train_safe_max_episodes = None
        self._autobatch_counts = {
            "infer_splits_cap": 0, "infer_splits_paging": 0, "infer_splits_oom": 0,
            "train_splits_cap": 0, "train_splits_paging": 0, "train_splits_oom": 0,
        }

        # Running advantage statistics for cross-batch normalization
        # Per-batch normalization destroys policy signal when all samples have similar outcomes
        self._adv_running_mean = 0.0
        self._adv_running_var = 1.0
        self._adv_ema_alpha = float(
            os.getenv('ADV_EMA_ALPHA', '0.01'))  # Smooth update
        self._adv_use_running = bool(
            int(os.getenv('ADV_USE_RUNNING_STATS', '1')))  # Enable by default

        # ---------------------------------------------------------
        # Microbatching + AMP (VRAM pressure control)
        # ---------------------------------------------------------
        self.infer_chunk = int(os.getenv("INFER_CHUNK", "256"))
        self.train_chunk = int(os.getenv("TRAIN_CHUNK", "256"))
        self.train_multi_chunk = int(os.getenv("TRAIN_MULTI_CHUNK", "256"))

        self.amp_enable = bool(int(os.getenv("AMP_ENABLE", "1")))
        self.amp_dtype_name = os.getenv("AMP_DTYPE", "bf16").strip().lower()
        self.amp_use_scaler = False
        self.amp_scaler = None
        if torch.cuda.is_available() and str(self.device).startswith("cuda") and self.amp_enable:
            if self.amp_dtype_name in ("bf16", "bfloat16"):
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
                self.amp_use_scaler = True
            if self.amp_use_scaler:
                try:
                    self.amp_scaler = torch.cuda.amp.GradScaler()
                except Exception:
                    self.amp_scaler = None
                    self.amp_use_scaler = False

        logger.info(LogCategory.GPU_MEMORY, "Using device: %s", self.device)
        logger.info(
            LogCategory.MODEL_INIT,
            "RL horizon config: gamma=%.5f use_gae=%s gae_lambda_high=%.3f gae_lambda_low=%.3f gae_lambda_decay_steps=%d",
            float(self.gamma),
            bool(self.use_gae),
            float(self.metrics.gae_lambda_high),
            float(self.metrics.gae_lambda_low),
            int(self.metrics.gae_lambda_decay_steps),
        )

    def _log_cuda_mem(self, where: str):
        """Lightweight CUDA memory snapshot for VRAM creep diagnostics."""
        try:
            if not torch.cuda.is_available():
                return
            if not str(self.device).startswith("cuda"):
                return
            dev = torch.cuda.current_device()
            alloc = int(torch.cuda.memory_allocated(dev))
            reserved = int(torch.cuda.memory_reserved(dev))
            max_alloc = int(torch.cuda.max_memory_allocated(dev))
            vram_logger.info(
                "VRAM",
                "CudaMem %s dev=%d alloc_mb=%.1f reserved_mb=%.1f max_alloc_mb=%.1f",
                str(where),
                int(dev),
                float(alloc) / (1024.0 * 1024.0),
                float(reserved) / (1024.0 * 1024.0),
                float(max_alloc) / (1024.0 * 1024.0),
            )
        except Exception:
            pass

    # CUDA/profiler methods and properties - delegate to cuda_mgr
    @property
    def auto_batch_enable(self):
        return self.cuda_mgr.auto_batch_enable

    @property
    def auto_avoid_paging(self):
        return self.cuda_mgr.auto_avoid_paging

    @property
    def auto_target_used_frac(self):
        return self.cuda_mgr.auto_target_used_frac

    @property
    def auto_min_free_mb(self):
        return self.cuda_mgr.auto_min_free_mb

    @property
    def auto_mem_ema_alpha(self):
        return self.cuda_mgr.auto_mem_ema_alpha

    @property
    def _infer_mb_per_sample(self):
        return self.cuda_mgr._infer_mb_per_sample

    @_infer_mb_per_sample.setter
    def _infer_mb_per_sample(self, value):
        self.cuda_mgr._infer_mb_per_sample = value

    @property
    def _train_mb_per_step(self):
        return self.cuda_mgr._train_mb_per_step

    @_train_mb_per_step.setter
    def _train_mb_per_step(self, value):
        self.cuda_mgr._train_mb_per_step = value

    @property
    def _autobatch_last_free_mb(self):
        return self.cuda_mgr._autobatch_last_free_mb

    @property
    def _autobatch_last_total_mb(self):
        return self.cuda_mgr._autobatch_last_total_mb

    @property
    def _autobatch_last_desired_free_mb(self):
        return self.cuda_mgr._autobatch_last_desired_free_mb

    def _is_cuda_oom(self, e: Exception) -> bool:
        return self.cuda_mgr.is_cuda_oom(e)

    def _cuda_cleanup_after_oom(self):
        self.cuda_mgr.cuda_cleanup_after_oom()

    def _cuda_mem_info_mb(self):
        return self.cuda_mgr.cuda_mem_info_mb()

    def _desired_free_mb(self):
        return self.cuda_mgr.desired_free_mb()

    def _should_split_for_paging(self, estimated_extra_mb: float):
        return self.cuda_mgr.should_split_for_paging(estimated_extra_mb)

    def _train_has_vram_headroom(self, estimated_extra_mb: float, wait: bool = False, tag: str = "train"):
        return self.cuda_mgr.train_has_headroom(estimated_extra_mb, wait=wait, tag=tag)

    def _update_mem_ema(self, kind: str, extra_mb: float, n: int):
        self.cuda_mgr.update_mem_ema(kind, extra_mb, n)

    @staticmethod
    def _env_flag(*keys) -> bool:
        for key in keys:
            value = os.getenv(key, "").strip().lower()
            if value in ("1", "true", "yes", "y", "on"):
                return True
        return False

    @staticmethod
    def _env_int(*keys, default: int = 0) -> int:
        for key in keys:
            value = os.getenv(key, "").strip()
            if not value:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return int(default)

    def _initialize_python_inference_duplicate_probe(self):
        path = str(self.python_inference_duplicate_probe_path or "").strip()
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "call_index",
                    "batch_row",
                    "policy_key",
                    "head_id",
                    "batch_size",
                    "seq_len",
                    "d_model",
                    "max_candidates",
                    "cand_feat_dim",
                    "candidate_count",
                    "pid",
                    "thread_name",
                    "thread_id",
                    "device",
                    "py_role",
                    "backend_mode",
                    "global_seed",
                    "torch_initial_seed",
                    "torch_rng_digest",
                    "numpy_rng_digest",
                    "python_rng_digest",
                    "cuda_rng_digest",
                    "torch_deterministic_algorithms",
                    "cudnn_enabled",
                    "cudnn_deterministic",
                    "cudnn_benchmark",
                    "cuda_matmul_allow_tf32",
                    "cudnn_allow_tf32",
                    "amp_enable",
                    "amp_dtype",
                    "model_class",
                    "model_object_id",
                    "model_training_before",
                    "model_training_after",
                    "dropout_summary",
                    "model_path",
                    "model_path_exists",
                    "model_path_mtime_ns",
                    "model_latest_path",
                    "model_latest_exists",
                    "model_latest_mtime_ns",
                    "model_load_snapshot_digest",
                    "model_load_path",
                    "model_load_path_mtime_ns",
                    "model_load_global_seed",
                    "model_load_torch_initial_seed",
                    "model_load_skipped_incompatible_count",
                    "model_load_skipped_incompatible_sample",
                    "model_load_skipped_incompatible_digest",
                    "model_load_state_digest",
                    "model_load_state_sample",
                    "input_sha256",
                    "sequence_sha256",
                    "mask_sha256",
                    "token_ids_sha256",
                    "candidate_features_sha256",
                    "candidate_ids_sha256",
                    "candidate_mask_sha256",
                    "pass0_probs",
                    "pass1_probs",
                    "pass0_logits",
                    "pass1_logits",
                    "pass0_value",
                    "pass1_value",
                    "max_abs_prob_diff",
                    "max_abs_logit_diff",
                    "value_abs_diff",
                    "duplicate_match",
                ])
        except Exception as e:
            try:
                logger.warning(LogCategory.MODEL_TRAIN,
                               "Failed to initialize python inference duplicate probe: %s", str(e))
            except Exception:
                pass

    def _initialize_model_load_determinism_gate(self):
        path = str(self.model_load_determinism_gate_path or "").strip()
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "event",
                    "pid",
                    "thread_name",
                    "thread_id",
                    "device",
                    "py_role",
                    "backend_mode",
                    "global_seed",
                    "torch_initial_seed",
                    "torch_rng_digest",
                    "numpy_rng_digest",
                    "python_rng_digest",
                    "cuda_rng_digest",
                    "model_path",
                    "model_path_exists",
                    "model_path_mtime_ns",
                    "model_latest_path",
                    "model_latest_exists",
                    "model_latest_mtime_ns",
                    "model_class",
                    "model_object_id",
                    "model_arch",
                    "skipped_incompatible_count",
                    "skipped_incompatible_sample",
                    "skipped_incompatible_digest",
                    "healed_params",
                    "healed_digest",
                    "model_state_digest",
                    "model_state_sample",
                    "snapshot_digest",
                ])
        except Exception as e:
            try:
                logger.warning(LogCategory.MODEL_LOAD,
                               "Failed to initialize model-load determinism gate: %s", str(e))
            except Exception:
                pass

    @staticmethod
    def _short_digest_bytes(data: bytes) -> str:
        try:
            return hashlib.sha256(data).hexdigest()[:16]
        except Exception:
            return ""

    @staticmethod
    def _short_digest_text(text: str) -> str:
        try:
            return PythonEntryPoint._short_digest_bytes(
                str(text or "").encode("utf-8", "replace"))
        except Exception:
            return ""

    @staticmethod
    def _short_digest_array(arr) -> str:
        try:
            return PythonEntryPoint._short_digest_bytes(np.ascontiguousarray(arr).tobytes())
        except Exception:
            return ""

    @staticmethod
    def _tensor_digest_payload(tensor):
        try:
            cpu = tensor.detach().cpu().contiguous()
            dtype_name = str(cpu.dtype)
            shape_name = "x".join(str(int(x)) for x in tuple(cpu.shape))
            try:
                raw = cpu.numpy().tobytes()
            except Exception:
                raw = cpu.float().numpy().tobytes()
            return dtype_name, shape_name, raw
        except Exception as e:
            return "unavailable", str(getattr(tensor, "shape", "")), repr(e).encode("utf-8", "replace")

    @staticmethod
    def _model_state_digest(model, sample_limit: int = 16):
        try:
            state = model.state_dict() if model is not None else {}
            digest = hashlib.sha256()
            samples = []
            for name in sorted(state.keys()):
                tensor = state[name]
                dtype_name, shape_name, raw = PythonEntryPoint._tensor_digest_payload(tensor)
                digest.update(str(name).encode("utf-8", "replace"))
                digest.update(b"\0")
                digest.update(dtype_name.encode("utf-8", "replace"))
                digest.update(b"\0")
                digest.update(shape_name.encode("utf-8", "replace"))
                digest.update(b"\0")
                digest.update(raw)
                digest.update(b"\0")
                if len(samples) < int(sample_limit):
                    tensor_digest = hashlib.sha256(raw).hexdigest()[:16]
                    samples.append(f"{name}:{dtype_name}:{shape_name}:{tensor_digest}")
            return digest.hexdigest()[:16], "|".join(samples)
        except Exception as e:
            return "", f"unavailable:{e.__class__.__name__}"

    @staticmethod
    def _model_arch_summary(model) -> str:
        try:
            return ";".join([
                f"class={model.__class__.__name__}",
                f"d_model={getattr(model, 'd_model', '')}",
                f"input_dim={getattr(model, 'input_dim', '')}",
                f"num_actions={getattr(model, 'num_actions', '')}",
                f"token_vocab={getattr(model, 'token_vocab', '')}",
                f"action_vocab={getattr(model, 'action_vocab', '')}",
                f"cand_feat_dim={getattr(model, 'cand_feat_dim', '')}",
                f"layers={len(getattr(model, 'transformer_layers', []))}",
            ])
        except Exception as e:
            return f"unavailable:{e.__class__.__name__}"

    def _record_model_load_determinism_gate(self, path, extra, skipped_incompatible, healed):
        if not self.model_load_determinism_gate:
            return
        gate_path = str(self.model_load_determinism_gate_path or "").strip()
        if not gate_path:
            return
        try:
            skipped = list(skipped_incompatible or [])
            skipped_sample = "|".join(str(x) for x in skipped[:24])
            skipped_digest = self._short_digest_text("|".join(str(x) for x in skipped))
            healed_params = "|".join(str(x) for x in list(healed or [])[:24])
            healed_digest = self._short_digest_text("|".join(str(x) for x in list(healed or [])))
            model_state_digest, model_state_sample = self._model_state_digest(self.model)
            model_path = str(path or getattr(self, "model_path", "") or "")
            latest_path = str(getattr(self, "model_latest_path", "") or "")
            arch = self._model_arch_summary(self.model)
            stable_snapshot = "|".join([
                f"global_seed={'' if self.global_seed is None else int(self.global_seed)}",
                f"torch_initial_seed={int(torch.initial_seed())}",
                f"model_path={model_path}",
                f"model_path_mtime_ns={self._file_mtime_ns(model_path)}",
                f"latest_path={latest_path}",
                f"latest_mtime_ns={self._file_mtime_ns(latest_path)}",
                f"arch={arch}",
                f"skipped_count={len(skipped)}",
                f"skipped_digest={skipped_digest}",
                f"healed_digest={healed_digest}",
                f"model_state_digest={model_state_digest}",
            ])
            snapshot_digest = self._short_digest_text(stable_snapshot)
            snapshot = {
                "snapshot_digest": snapshot_digest,
                "path": model_path,
                "path_mtime_ns": self._file_mtime_ns(model_path),
                "global_seed": "" if self.global_seed is None else int(self.global_seed),
                "torch_initial_seed": int(torch.initial_seed()),
                "skipped_incompatible_count": len(skipped),
                "skipped_incompatible_sample": skipped_sample,
                "skipped_incompatible_digest": skipped_digest,
                "model_state_digest": model_state_digest,
                "model_state_sample": model_state_sample,
            }
            self._model_load_determinism_snapshot = snapshot
            row = [
                "loadModel",
                int(os.getpid()),
                threading.current_thread().name,
                int(threading.get_ident()),
                str(self.device),
                str(getattr(self, "py_role", "")),
                str(getattr(self, "backend_mode", "")),
                "" if self.global_seed is None else int(self.global_seed),
                int(torch.initial_seed()),
                self._rng_state_digest(torch.random.get_rng_state),
                self._short_digest_array(np.random.get_state()[1]),
                self._short_digest_bytes(repr(random.getstate()).encode("utf-8", "replace")),
                self._rng_state_digest(lambda: torch.cuda.get_rng_state(self.device)) if torch.cuda.is_available() and str(self.device).startswith("cuda") else "",
                model_path,
                int(bool(model_path and os.path.exists(model_path))),
                self._file_mtime_ns(model_path),
                latest_path,
                int(bool(latest_path and os.path.exists(latest_path))),
                self._file_mtime_ns(latest_path),
                self.model.__class__.__name__ if self.model is not None else "",
                int(id(self.model)) if self.model is not None else "",
                arch,
                len(skipped),
                skipped_sample,
                skipped_digest,
                healed_params,
                healed_digest,
                model_state_digest,
                model_state_sample,
                snapshot_digest,
            ]
            with self._model_load_determinism_gate_lock:
                with open(gate_path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(row)
            logger.info(
                LogCategory.MODEL_LOAD,
                "Model-load determinism snapshot digest=%s skipped_incompatible=%d model_state=%s",
                snapshot_digest,
                len(skipped),
                model_state_digest,
            )
        except Exception as e:
            try:
                logger.warning(LogCategory.MODEL_LOAD,
                               "Failed to record model-load determinism gate: %s", str(e))
            except Exception:
                pass

    @staticmethod
    def _file_mtime_ns(path: str) -> int:
        try:
            return int(os.stat(path).st_mtime_ns)
        except Exception:
            return 0

    @staticmethod
    def _rng_state_digest(getter) -> str:
        try:
            state = getter()
            if hasattr(state, "detach"):
                state = state.detach().cpu().numpy()
            return PythonEntryPoint._short_digest_array(state)
        except Exception:
            return ""

    @staticmethod
    def _format_indexed_values(values, mask=None) -> str:
        try:
            arr = values.detach().float().cpu().numpy() if hasattr(values, "detach") else np.asarray(values, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            mask_arr = None
            if mask is not None:
                mask_arr = mask.detach().bool().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask).astype(bool)
                mask_arr = mask_arr.reshape(-1)
            parts = []
            for idx in range(int(arr.shape[0])):
                if mask_arr is not None and idx < int(mask_arr.shape[0]) and not bool(mask_arr[idx]):
                    continue
                value = float(arr[idx])
                parts.append(f"{idx}:{value:.9g}")
            return ";".join(parts)
        except Exception:
            return ""

    @staticmethod
    def _max_abs_diff(a, b, mask=None) -> float:
        try:
            aa = a.detach().float().cpu().numpy() if hasattr(a, "detach") else np.asarray(a, dtype=np.float32)
            bb = b.detach().float().cpu().numpy() if hasattr(b, "detach") else np.asarray(b, dtype=np.float32)
            aa = aa.reshape(-1)
            bb = bb.reshape(-1)
            n = min(int(aa.shape[0]), int(bb.shape[0]))
            if n <= 0:
                return 0.0
            aa = aa[:n]
            bb = bb[:n]
            if mask is not None:
                mm = mask.detach().bool().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask).astype(bool)
                mm = mm.reshape(-1)[:n]
                if mm.shape[0] == n:
                    aa = aa[mm]
                    bb = bb[mm]
            if aa.size == 0:
                return 0.0
            return float(np.max(np.abs(aa - bb)))
        except Exception:
            return float("nan")

    @staticmethod
    def _dropout_summary(model) -> str:
        try:
            items = []
            total = 0
            training = 0
            for name, module in model.named_modules():
                cls_name = module.__class__.__name__
                if "Dropout" not in cls_name:
                    continue
                total += 1
                if bool(getattr(module, "training", False)):
                    training += 1
                if len(items) < 12:
                    p = getattr(module, "p", "")
                    items.append(f"{name}:{cls_name}:training={int(bool(getattr(module, 'training', False)))}:p={p}")
            return f"count={total};training={training};" + "|".join(items)
        except Exception as e:
            return f"unavailable:{e.__class__.__name__}"

    def _append_python_inference_duplicate_probe(
            self,
            call_index: int,
            start: int,
            policy_key,
            head_id,
            batch_size: int,
            seq_len: int,
            d_model: int,
            max_candidates: int,
            cand_feat_dim: int,
            model,
            model_training_before: bool,
            model_training_after: bool,
            probe_inputs,
            probs0,
            value0,
            logits0,
            probs1,
            value1,
            logits1,
            cand_mask_t):
        if not self.python_inference_duplicate_probe:
            return
        path = str(self.python_inference_duplicate_probe_path or "").strip()
        if not path:
            return
        if self._python_inference_duplicate_probe_rows >= int(self.python_inference_duplicate_probe_max_rows):
            return
        try:
            seq_np = probe_inputs.get("seq")
            mask_np = probe_inputs.get("mask")
            tok_np = probe_inputs.get("token_ids")
            cand_feat_np = probe_inputs.get("candidate_features")
            cand_ids_np = probe_inputs.get("candidate_ids")
            cand_mask_np = probe_inputs.get("candidate_mask")
            input_digest = self._short_digest_bytes(
                np.ascontiguousarray(seq_np).tobytes()
                + np.ascontiguousarray(mask_np).tobytes()
                + np.ascontiguousarray(tok_np).tobytes()
                + np.ascontiguousarray(cand_feat_np).tobytes()
                + np.ascontiguousarray(cand_ids_np).tobytes()
                + np.ascontiguousarray(cand_mask_np).tobytes())
            row_count = int(probs0.shape[0]) if hasattr(probs0, "shape") else 0
            rows = []
            for local_row in range(row_count):
                if self._python_inference_duplicate_probe_rows + len(rows) >= int(self.python_inference_duplicate_probe_max_rows):
                    break
                mask_row = cand_mask_t[local_row] if cand_mask_t is not None else None
                candidate_count = 0
                try:
                    candidate_count = int(mask_row.detach().bool().sum().item()) if mask_row is not None else 0
                except Exception:
                    candidate_count = 0
                prob_diff = self._max_abs_diff(probs0[local_row], probs1[local_row], mask_row)
                logit_diff = self._max_abs_diff(logits0[local_row], logits1[local_row], mask_row)
                try:
                    v0 = float(value0[local_row].detach().float().cpu().reshape(-1)[0].item())
                    v1 = float(value1[local_row].detach().float().cpu().reshape(-1)[0].item())
                except Exception:
                    v0 = 0.0
                    v1 = 0.0
                duplicate_match = (prob_diff <= 1.0e-7) and (logit_diff <= 1.0e-6) and (abs(v0 - v1) <= 1.0e-7)
                model_path = str(getattr(self, "model_path", "") or "")
                latest_path = str(getattr(self, "model_latest_path", "") or "")
                load_snapshot = getattr(self, "_model_load_determinism_snapshot", {}) or {}
                rows.append([
                    int(call_index),
                    int(start + local_row),
                    str(policy_key),
                    str(head_id),
                    int(batch_size),
                    int(seq_len),
                    int(d_model),
                    int(max_candidates),
                    int(cand_feat_dim),
                    candidate_count,
                    int(os.getpid()),
                    threading.current_thread().name,
                    int(threading.get_ident()),
                    str(self.device),
                    str(getattr(self, "py_role", "")),
                    str(getattr(self, "backend_mode", "")),
                    "" if self.global_seed is None else int(self.global_seed),
                    int(torch.initial_seed()),
                    self._rng_state_digest(torch.random.get_rng_state),
                    self._short_digest_array(np.random.get_state()[1]),
                    self._short_digest_bytes(repr(random.getstate()).encode("utf-8", "replace")),
                    self._rng_state_digest(lambda: torch.cuda.get_rng_state(self.device)) if torch.cuda.is_available() and str(self.device).startswith("cuda") else "",
                    int(bool(torch.are_deterministic_algorithms_enabled())),
                    int(bool(torch.backends.cudnn.enabled)),
                    int(bool(torch.backends.cudnn.deterministic)),
                    int(bool(torch.backends.cudnn.benchmark)),
                    int(bool(torch.backends.cuda.matmul.allow_tf32)) if hasattr(torch.backends, "cuda") else "",
                    int(bool(torch.backends.cudnn.allow_tf32)),
                    int(bool(getattr(self, "amp_enable", False))),
                    str(getattr(self, "amp_dtype_name", "")),
                    model.__class__.__name__,
                    int(id(model)),
                    int(bool(model_training_before)),
                    int(bool(model_training_after)),
                    self._dropout_summary(model),
                    model_path,
                    int(bool(model_path and os.path.exists(model_path))),
                    self._file_mtime_ns(model_path),
                    latest_path,
                    int(bool(latest_path and os.path.exists(latest_path))),
                    self._file_mtime_ns(latest_path),
                    load_snapshot.get("snapshot_digest", ""),
                    load_snapshot.get("path", ""),
                    load_snapshot.get("path_mtime_ns", ""),
                    load_snapshot.get("global_seed", ""),
                    load_snapshot.get("torch_initial_seed", ""),
                    load_snapshot.get("skipped_incompatible_count", ""),
                    load_snapshot.get("skipped_incompatible_sample", ""),
                    load_snapshot.get("skipped_incompatible_digest", ""),
                    load_snapshot.get("model_state_digest", ""),
                    load_snapshot.get("model_state_sample", ""),
                    input_digest,
                    self._short_digest_array(seq_np[local_row]),
                    self._short_digest_array(mask_np[local_row]),
                    self._short_digest_array(tok_np[local_row]),
                    self._short_digest_array(cand_feat_np[local_row]),
                    self._short_digest_array(cand_ids_np[local_row]),
                    self._short_digest_array(cand_mask_np[local_row]),
                    self._format_indexed_values(probs0[local_row], mask_row),
                    self._format_indexed_values(probs1[local_row], mask_row),
                    self._format_indexed_values(logits0[local_row], mask_row),
                    self._format_indexed_values(logits1[local_row], mask_row),
                    f"{v0:.9g}",
                    f"{v1:.9g}",
                    f"{prob_diff:.9g}",
                    f"{logit_diff:.9g}",
                    f"{abs(v0 - v1):.9g}",
                    int(bool(duplicate_match)),
                ])
            if not rows:
                return
            with self._python_inference_duplicate_probe_lock:
                if self._python_inference_duplicate_probe_rows >= int(self.python_inference_duplicate_probe_max_rows):
                    return
                allowed = int(self.python_inference_duplicate_probe_max_rows) - self._python_inference_duplicate_probe_rows
                rows = rows[:max(0, allowed)]
                with open(path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerows(rows)
                self._python_inference_duplicate_probe_rows += len(rows)
        except Exception as e:
            try:
                logger.warning(LogCategory.MODEL_TRAIN,
                               "Failed to append python inference duplicate probe: %s", str(e))
            except Exception:
                pass

    def _measure_peak_extra_mb(self, fn):
        return self.cuda_mgr.measure_peak_extra_mb(fn)

    def _set_encoder_requires_grad(self, requires_grad: bool):
        """
        Toggle grads for the shared state encoder trunk.
        Encoder params:
        - input_* (proj/norm/scale)
        - token_id_emb
        - cls_token
        - transformer_layers.*
        """
        if self.model is None:
            return
        prefixes = (
            'input_',
            'token_id_emb',
            'cls_token',
            'transformer_layers',
        )
        for name, p in self.model.named_parameters():
            if name.startswith(prefixes):
                p.requires_grad = requires_grad

    def _set_distill_head_only_requires_grad(self):
        """
        Restrict terminal-prefix distillation to policy scorer heads.
        This keeps mulligan state routing, shared encoder features, candidate
        embeddings, and critic/belief parameters fixed while allowing per-head
        candidate ranking to absorb searched labels.
        """
        if self.model is None:
            return
        trainable_prefixes = (
            'policy_scorer.',
            'policy_scorer_target.',
            'policy_scorer_card_select.',
            'policy_scorer_attack.',
            'policy_scorer_block.',
            'policy_scorer_mulligan.',
        )
        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith(trainable_prefixes)

    def _set_distill_policy_path_only_requires_grad(self):
        """
        Restrict distillation to candidate-policy routing while freezing the
        state encoder and critic. This lets BC teach candidate distinctions
        such as KEEP vs MULLIGAN without rewriting the learned state features.
        """
        if self.model is None:
            return
        trainable_prefixes = (
            'action_id_emb.',
            'cand_feat_proj.',
            'cross_attn.',
            'cross_attn_norm.',
            'cand_self_attn.',
            'cand_self_attn_norm.',
            'policy_scorer.',
            'policy_scorer_target.',
            'policy_scorer_card_select.',
            'policy_scorer_attack.',
            'policy_scorer_block.',
            'policy_scorer_mulligan.',
        )
        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith(trainable_prefixes)

    def _set_candidate_q_only_requires_grad(self):
        """Train only the action-conditioned terminal value scorer."""
        if self.model is None:
            return
        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith('candidate_q_scorer.')

    def _set_value_pair_rank_critic_only_requires_grad(self):
        """Train only the scalar value path for branch-pair ranking imports."""
        if self.model is None:
            return
        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith('critic_') or name == 'value_scale'

    def _set_belief_head_only_requires_grad(self):
        """Train only the opponent-archetype belief classifier."""
        if self.model is None:
            return
        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith('belief_head.')

    def _set_card_belief_head_only_requires_grad(self):
        """Train only the generic hidden-card belief regressor."""
        if self.model is None:
            return
        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith('card_belief_head.')

    def _ensure_main_model_initialized(self):
        if self.model is not None:
            return
        with self._model_init_lock:
            if self.model is None:
                self.initializeModel()

    def initializeModel(self):
        """Initialize the model and optimizer"""
        logger.info(LogCategory.GPU_MEMORY, "Initializing model")
        if self.model is None:
            # Model v2 (small): 2-layer, d_model=128, ~2M params, saturates in 50k-200k episodes
            # Model v1 (large): 6-layer, d_model=512, ~55M params, saturates in 20M+ episodes
            self.model = MTGTransformerModel(
                d_model=int(os.getenv('MODEL_D_MODEL', '128')),
                nhead=int(os.getenv('MODEL_NHEAD', '4')),
                num_layers=int(os.getenv('MODEL_NUM_LAYERS', '2')),
                dim_feedforward=int(os.getenv('MODEL_DIM_FEEDFORWARD', '512')),
                cand_feat_dim=48,
            ).to(self.device)
            if self.distill_head_only:
                self._set_distill_head_only_requires_grad()
            elif self.distill_policy_path_only:
                self._set_distill_policy_path_only_requires_grad()
            elif self.candidate_q_only:
                self._set_candidate_q_only_requires_grad()
            elif self.value_pair_rank_critic_only:
                self._set_value_pair_rank_critic_only_requires_grad()
            elif self.belief_head_only:
                self._set_belief_head_only_requires_grad()
            elif self.card_belief_head_only:
                self._set_card_belief_head_only_requires_grad()

            # Separate LR groups: actor head, critic head, everything else
            actor_param_names = [
                'actor_proj1', 'actor_proj2', 'actor_norm', 'actor_norm1'
            ]
            critic_param_names = [
                'critic_input_proj', 'critic_input_norm',
                'critic_cls_token', 'critic_transformer',
                'critic_proj1', 'critic_proj2', 'critic_norm', 'critic_norm1',
                'value_scale'
            ]
            actor_params = []
            critic_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if any(apn in name for apn in actor_param_names):
                    actor_params.append(param)
                elif any(cpn in name for cpn in critic_param_names):
                    critic_params.append(param)
                else:
                    other_params.append(param)

            # ----------------- LR Tuning ---------------------------
            critic_lr = float(os.getenv('CRITIC_LR', '1e-3'))
            self.optimizer = torch.optim.Adam([
                {'params': actor_params, 'lr': float(os.getenv('ACTOR_LR', '3e-4'))},
                {'params': critic_params, 'lr': critic_lr},
                {'params': other_params, 'lr': float(os.getenv('OTHER_LR', '3e-4'))}
            ])

            # Log model size and GPU memory
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(
                LogCategory.GPU_MEMORY,
                "Model size: %d parameters", total_params
            )

            # Log GPU memory after model initialization
            log_gpu_memory()

            logger.info(LogCategory.GPU_MEMORY,
                        "Model initialized successfully")

        # One-time initial load
        if not self._did_initial_load:
            # Load from base checkpoint if it exists
            try:
                if self.model_path:
                    logger.info(LogCategory.GPU_MEMORY,
                                f"Model path configured: {self.model_path}")
                    if os.path.exists(self.model_path):
                        logger.info(LogCategory.GPU_MEMORY,
                                    "Found existing model, loading...")
                        self.loadModel(self.model_path)
            except Exception:
                pass

            # Inference workers prefer the 'latest' weights if present
            try:
                if self.py_role == "inference" and self.model_latest_path:
                    self.reloadLatestModelIfNewer(self.model_latest_path)
            except Exception:
                pass

            self._did_initial_load = True

        self._ensure_mcts_reference_model()
        self._ensure_policy_ensemble_models()

    @staticmethod
    def _parse_env_paths(raw: str):
        """Parse model paths from an env var without breaking Windows drive letters."""
        raw = str(raw or '').strip()
        if not raw:
            return []
        if ';' in raw:
            parts = raw.split(';')
        elif '\n' in raw:
            parts = raw.splitlines()
        elif os.pathsep != ':' and os.pathsep in raw:
            parts = raw.split(os.pathsep)
        elif ',' in raw:
            parts = raw.split(',')
        else:
            parts = [raw]
        return [p.strip().strip('"').strip("'") for p in parts if p and p.strip()]

    @staticmethod
    def _parse_float_list(raw: str):
        raw = str(raw or '').strip()
        if not raw:
            return []
        values = []
        for part in raw.replace(';', ',').split(','):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part))
            except Exception:
                return []
        return values

    def _ensure_mcts_reference_model(self):
        """Load an optional frozen policy reference for prefix/MCTS target mixing."""
        if self.mcts_reference_model is not None:
            return
        path = self.mcts_reference_model_path
        if not path:
            return
        if not os.path.exists(path):
            logger.warning(LogCategory.MODEL_INIT,
                           "MCTS reference model path does not exist: %s", path)
            print(f"[REF_POLICY] reference model missing: {path}", flush=True)
            return
        try:
            ref = MTGTransformerModel(
                d_model=int(os.getenv('MODEL_D_MODEL', '128')),
                nhead=int(os.getenv('MODEL_NHEAD', '4')),
                num_layers=int(os.getenv('MODEL_NUM_LAYERS', '2')),
                dim_feedforward=int(os.getenv('MODEL_DIM_FEEDFORWARD', '512')),
                cand_feat_dim=48,
            ).to(self.device)
            ref.load(path)
            ref.eval()
            for p in ref.parameters():
                p.requires_grad = False
            self.mcts_reference_model = ref
            logger.info(LogCategory.MODEL_INIT,
                        "Loaded MCTS reference model from %s", path)
            print(
                f"[REF_POLICY] loaded reference model path={path} coef={self.reference_policy_kl_coef:.4f}",
                flush=True)
        except Exception as e:
            logger.warning(LogCategory.MODEL_INIT,
                           "Failed to load MCTS reference model from %s: %s",
                           path, str(e))
            self.mcts_reference_model = None

    def _ensure_policy_ensemble_models(self):
        """Load optional frozen companion policies for eval-time ensemble diagnostics."""
        if self._policy_ensemble_loaded:
            return
        self._policy_ensemble_loaded = True
        paths = list(self.policy_ensemble_model_paths or [])
        if not paths:
            return

        loaded = []
        for path in paths:
            if not os.path.exists(path):
                logger.warning(LogCategory.MODEL_INIT,
                               "Policy ensemble model path does not exist: %s", path)
                print(f"[POLICY_ENSEMBLE] model missing: {path}", flush=True)
                continue
            try:
                ens = MTGTransformerModel(
                    d_model=int(os.getenv('MODEL_D_MODEL', '128')),
                    nhead=int(os.getenv('MODEL_NHEAD', '4')),
                    num_layers=int(os.getenv('MODEL_NUM_LAYERS', '2')),
                    dim_feedforward=int(os.getenv('MODEL_DIM_FEEDFORWARD', '512')),
                    cand_feat_dim=48,
                ).to(self.device)
                ens.load(path)
                ens.eval()
                for p in ens.parameters():
                    p.requires_grad = False
                loaded.append((path, ens))
                logger.info(LogCategory.MODEL_INIT,
                            "Loaded policy ensemble model from %s", path)
                print(f"[POLICY_ENSEMBLE] loaded model path={path}", flush=True)
            except Exception as e:
                logger.warning(LogCategory.MODEL_INIT,
                               "Failed to load policy ensemble model from %s: %s",
                               path, str(e))

        self.policy_ensemble_models = [model for _path, model in loaded]
        weights = self._parse_float_list(self.policy_ensemble_weight_spec)
        expected_with_primary = 1 + len(self.policy_ensemble_models)
        if not self.policy_ensemble_models:
            self.policy_ensemble_weights = []
            return
        if len(weights) == expected_with_primary:
            normalized = weights
        elif len(weights) == len(self.policy_ensemble_models):
            normalized = [1.0] + weights
        else:
            normalized = [1.0] * expected_with_primary
            if self.policy_ensemble_weight_spec:
                logger.warning(LogCategory.MODEL_INIT,
                               "Ignoring POLICY_ENSEMBLE_WEIGHTS length=%d expected=%d or %d",
                               len(weights), expected_with_primary,
                               len(self.policy_ensemble_models))
        normalized = [max(0.0, float(w)) for w in normalized]
        total = sum(normalized)
        if total <= 0.0:
            normalized = [1.0] * expected_with_primary
            total = float(expected_with_primary)
        self.policy_ensemble_weights = [float(w) / total for w in normalized]
        logger.info(LogCategory.MODEL_INIT,
                    "Policy ensemble active: companions=%d weights=%s",
                    len(self.policy_ensemble_models), self.policy_ensemble_weights)
        print(
            f"[POLICY_ENSEMBLE] active companions={len(self.policy_ensemble_models)} weights={self.policy_ensemble_weights}",
            flush=True)

    def _score_reference_policy_chunk(self, seq, mask, token_ids, candidate_features,
                                      candidate_ids, candidate_mask, head_idx,
                                      max_candidates, autocast_ctx):
        """Score a chunk with the frozen reference model, matching per-head routing."""
        ref = self.mcts_reference_model
        if ref is None:
            return None
        chunk_n = seq.shape[0]
        ref_probs = torch.zeros(chunk_n, max_candidates, device=self.device)
        with torch.no_grad():
            with autocast_ctx:
                for hid_val, hid_name in enumerate(HEAD_NAMES):
                    hmask = (head_idx == hid_val)
                    if not hmask.any():
                        continue
                    p_h, _v_h = ref.score_candidates(
                        seq[hmask], mask[hmask], token_ids[hmask],
                        candidate_features[hmask], candidate_ids[hmask],
                        candidate_mask[hmask], head_id=hid_name)
                    ref_probs[hmask] = p_h.float()
        return torch.clamp(ref_probs, min=1e-8, max=1.0).detach()


    # ------------------------------------------------------------------
    # Snapshot opponent helpers
    # ------------------------------------------------------------------

    # Snapshot methods - delegate to snapshot_mgr
    # Also expose attributes for backward compatibility
    @property
    def snapshot_dir(self):
        return self.snapshot_mgr.snapshot_dir

    @property
    def snapshot_save_every_steps(self):
        return self.snapshot_mgr.snapshot_save_every_steps

    @property
    def snapshot_max_files(self):
        return self.snapshot_mgr.snapshot_max_files

    @property
    def snapshot_cache_size(self):
        return self.snapshot_mgr.snapshot_cache_size

    @property
    def snapshot_models(self):
        return self.snapshot_mgr.snapshot_models

    def _get_snapshot_model(self, snap_id: str):
        return self.snapshot_mgr.get_snapshot_model(snap_id)

    def _get_policy_model(self, policy_key: str):
        # Single-backend: guarantee only one GPU model instance (no snapshot models).
        if self.backend_mode == "single":
            return self.model
        return self.snapshot_mgr.get_policy_model(policy_key, self.model)

    def _maybe_save_snapshot(self):
        self.snapshot_mgr.maybe_save_snapshot(
            self.metrics.train_step_counter, lambda path: self.saveModel(path))

    def scoreCandidatesFlat(self,
                            sequences_bytes,
                            masks_bytes,
                            token_ids_bytes,
                            candidate_features_bytes,
                            candidate_ids_bytes,
                            candidate_mask_bytes,
                            batch_size,
                            seq_len,
                            d_model,
                            max_candidates,
                            cand_feat_dim):
        return self.scoreCandidatesPolicyFlat(
            sequences_bytes,
            masks_bytes,
            token_ids_bytes,
            candidate_features_bytes,
            candidate_ids_bytes,
            candidate_mask_bytes,
            "train",
            "action",
            0,
            0,
            0,
            batch_size,
            seq_len,
            d_model,
            max_candidates,
            cand_feat_dim
        )

    def scoreCandidatesPolicyFlat(self,
                                  sequences_bytes,
                                  masks_bytes,
                                  token_ids_bytes,
                                  candidate_features_bytes,
                                  candidate_ids_bytes,
                                  candidate_mask_bytes,
                                  policy_key,
                                  head_id,
                                  pick_index,
                                  min_targets,
                                  max_targets,
                                  batch_size,
                                  seq_len,
                                  d_model,
                                  max_candidates,
                                  cand_feat_dim):
        """Score padded candidates for each state (policy over candidates + value)."""
        t_start = time.perf_counter()

        def _score_numpy_range(start: int, end: int):
            device = self.device
            lock_held = False
            if self.backend_mode == "single":
                # Single-backend: pause inference while training is updating weights.
                self._gpu_mutex.acquire()
                lock_held = True
            try:
                seq = np.frombuffer(sequences_bytes, dtype='<f4').reshape(
                    batch_size, seq_len, d_model)[start:end]
                mask = np.frombuffer(masks_bytes, dtype='<i4').reshape(
                    batch_size, seq_len)[start:end]
                tok_ids = np.frombuffer(token_ids_bytes, dtype='<i4').reshape(
                    batch_size, seq_len)[start:end]

                cand_feat = np.frombuffer(candidate_features_bytes, dtype='<f4').reshape(
                    batch_size, max_candidates, cand_feat_dim)[start:end]
                cand_ids = np.frombuffer(candidate_ids_bytes, dtype='<i4').reshape(
                    batch_size, max_candidates)[start:end]
                cand_mask = np.frombuffer(candidate_mask_bytes, dtype='<i4').reshape(
                    batch_size, max_candidates)[start:end]

                seq_t = torch.tensor(seq, dtype=torch.float32, device=device)
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device)
                cand_feat_t = torch.tensor(
                    cand_feat, dtype=torch.float32, device=device)
                cand_ids_t = torch.tensor(
                    cand_ids, dtype=torch.long, device=device)
                cand_mask_t = torch.tensor(
                    cand_mask, dtype=torch.bool, device=device)

                probe_inputs = None
                if self.python_inference_duplicate_probe:
                    probe_inputs = {
                        "seq": np.ascontiguousarray(seq),
                        "mask": np.ascontiguousarray(mask),
                        "token_ids": np.ascontiguousarray(tok_ids),
                        "candidate_features": np.ascontiguousarray(cand_feat),
                        "candidate_ids": np.ascontiguousarray(cand_ids),
                        "candidate_mask": np.ascontiguousarray(cand_mask),
                    }

                # Release numpy slices immediately
                del seq, mask, tok_ids, cand_feat, cand_ids, cand_mask

                model = self._get_policy_model(policy_key)
                if model is None:
                    # Snapshot may fail to load; ensure base model exists as fallback.
                    self._ensure_main_model_initialized()
                    model = self._get_policy_model(policy_key)
                if model is None:
                    raise RuntimeError("No policy model available for scoring")
                model.eval()
                self._ensure_policy_ensemble_models()

                acquired_gpu_lock_here = False
                if self.backend_mode != "single":
                    # Multi-backend: inference should run while GPULock is held.
                    # If Java-side lock acquisition was missed, recover by acquiring here.
                    if torch.cuda.is_available() and str(device).startswith("cuda") and not self.gpu_lock.is_locked:
                        logger.warning(
                            LogCategory.GPU_MEMORY,
                            "Inference entered without GPULock; auto-acquiring for this score call."
                        )
                        self.gpu_lock.acquire(timeout=None, process_name=self.process_name)
                        acquired_gpu_lock_here = True

                try:
                    with torch.inference_mode():
                        model_training_before = bool(getattr(model, "training", False))
                        if self.python_inference_duplicate_probe:
                            probs, value, logits, candidate_q = model.score_candidates(
                                seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t,
                                head_id, int(pick_index), int(min_targets), int(max_targets),
                                return_logits=True, return_candidate_q=True)
                            duplicate_probs, duplicate_value, duplicate_logits, _duplicate_q = model.score_candidates(
                                seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t,
                                head_id, int(pick_index), int(min_targets), int(max_targets),
                                return_logits=True, return_candidate_q=True)
                            probe_probs, probe_value = probs, value
                        else:
                            probs, value, candidate_q = model.score_candidates(
                                seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t,
                                head_id, int(pick_index), int(min_targets), int(max_targets),
                                return_candidate_q=True)
                            logits = duplicate_probs = duplicate_value = duplicate_logits = None
                            probe_probs = probe_value = None
                        if self.policy_ensemble_models:
                            weights = self.policy_ensemble_weights
                            primary_weight = float(weights[0]) if weights else 1.0
                            probs_acc = probs.float() * primary_weight
                            value_acc = value.float() * primary_weight
                            active_weight = primary_weight
                            for ens_idx, ens_model in enumerate(self.policy_ensemble_models):
                                weight_idx = ens_idx + 1
                                ens_weight = float(weights[weight_idx]) if weight_idx < len(weights) else 1.0
                                if ens_weight <= 0.0:
                                    continue
                                try:
                                    ens_probs, ens_value = ens_model.score_candidates(
                                        seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t,
                                        head_id, int(pick_index), int(min_targets), int(max_targets))
                                    if ens_probs.shape != probs.shape or ens_value.shape != value.shape:
                                        raise RuntimeError(
                                            f"shape mismatch probs={tuple(ens_probs.shape)} value={tuple(ens_value.shape)}")
                                    probs_acc = probs_acc + ens_probs.float() * ens_weight
                                    value_acc = value_acc + ens_value.float() * ens_weight
                                    active_weight += ens_weight
                                except Exception as e:
                                    if ens_idx not in self._policy_ensemble_runtime_failures:
                                        self._policy_ensemble_runtime_failures.add(ens_idx)
                                        logger.warning(
                                            LogCategory.MODEL_INIT,
                                            "Policy ensemble companion %d failed during scoring; skipping it: %s",
                                            ens_idx, str(e))
                            if active_weight > 0.0:
                                probs = probs_acc / float(active_weight)
                                valid = cand_mask_t.bool()
                                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                                probs = probs * valid.float()
                                row_sum = probs.sum(dim=-1, keepdim=True)
                                fallback = valid.float() / (valid.float().sum(dim=-1, keepdim=True) + 1e-8)
                                probs = torch.where(row_sum > 0, probs / (row_sum + 1e-8), fallback)
                                value = value_acc / float(active_weight)
                        if self.python_inference_duplicate_probe:
                            self._append_python_inference_duplicate_probe(
                                int(self.score_call_counter) + 1,
                                int(start),
                                policy_key,
                                head_id,
                                int(batch_size),
                                int(seq_len),
                                int(d_model),
                                int(max_candidates),
                                int(cand_feat_dim),
                                model,
                                model_training_before,
                                bool(getattr(model, "training", False)),
                                probe_inputs or {},
                                probe_probs,
                                probe_value,
                                logits,
                                duplicate_probs,
                                duplicate_value,
                                duplicate_logits,
                                cand_mask_t)
                finally:
                    if acquired_gpu_lock_here:
                        self.gpu_lock.release(process_name=self.process_name)

                probs_np = probs.detach().cpu().numpy()
                value_np = value.detach().cpu().numpy()
                candidate_q_np = candidate_q.detach().cpu().numpy()

                # Release GPU tensors immediately (in this scope where they're defined)
                del seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t, probs, value, candidate_q

                return probs_np, value_np, candidate_q_np
            finally:
                if lock_held:
                    try:
                        self._gpu_mutex.release()
                    except Exception:
                        pass

        def _score_with_oom_splitting(start: int, end: int):
            n = int(end - start)
            if n <= 0:
                return (
                    np.zeros((0, max_candidates), dtype=np.float32),
                    np.zeros((0, 1), dtype=np.float32),
                    np.zeros((0, max_candidates), dtype=np.float32),
                )
            # Proactive cap if configured or learned.
            cap = self._infer_safe_max if (
                self.auto_batch_enable and self._infer_safe_max) else None
            if cap is not None and n > cap:
                try:
                    self._autobatch_counts["infer_splits_cap"] += 1
                except Exception:
                    pass
                probs_parts = []
                value_parts = []
                q_parts = []
                i = start
                while i < end:
                    j = min(end, i + int(cap))
                    p, v, q = _score_with_oom_splitting(i, j)
                    probs_parts.append(p)
                    value_parts.append(v)
                    q_parts.append(q)
                    i = j
                result_probs = np.concatenate(probs_parts, axis=0)
                result_values = np.concatenate(value_parts, axis=0)
                result_q = np.concatenate(q_parts, axis=0)
                # Clean up intermediate parts
                del probs_parts, value_parts, q_parts
                return result_probs, result_values, result_q

            # Proactive paging avoidance: if we'd exceed headroom, split before we start paging/thrashing.
            if self.auto_batch_enable and self.auto_avoid_paging and torch.cuda.is_available():
                est = 0.0
                if self._infer_mb_per_sample is not None:
                    est = float(self._infer_mb_per_sample) * float(n)
                if self._should_split_for_paging(est) and n > 1:
                    try:
                        self._autobatch_counts["infer_splits_paging"] += 1
                    except Exception:
                        pass
                    mid = start + (n // 2)
                    p0, v0, q0 = _score_with_oom_splitting(start, mid)
                    p1, v1, q1 = _score_with_oom_splitting(mid, end)
                    return (
                        np.concatenate((p0, p1), axis=0),
                        np.concatenate((v0, v1), axis=0),
                        np.concatenate((q0, q1), axis=0),
                    )

            try:
                (p, v, q), extra_mb = self._measure_peak_extra_mb(
                    lambda: _score_numpy_range(start, end))
                # Update per-sample estimate from measured peak delta.
                self._update_mem_ema("infer", extra_mb, n)
                return p, v, q
            except RuntimeError as e:
                if self.auto_batch_enable and self._is_cuda_oom(e):
                    self._cuda_cleanup_after_oom()
                    if n <= 1:
                        raise
                    try:
                        self._autobatch_counts["infer_splits_oom"] += 1
                    except Exception:
                        pass
                    # Learn a smaller cap for future calls.
                    new_cap = max(1, n // 2)
                    if self._infer_safe_max is None or new_cap < int(self._infer_safe_max):
                        self._infer_safe_max = int(new_cap)
                        logger.warning(
                            LogCategory.GPU_BATCH, "AutoBatch(infer): OOM -> shrinking infer cap to %d", int(self._infer_safe_max))
                    mid = start + (n // 2)
                    p0, v0, q0 = _score_with_oom_splitting(start, mid)
                    p1, v1, q1 = _score_with_oom_splitting(mid, end)
                    result_probs = np.concatenate((p0, p1), axis=0)
                    result_values = np.concatenate((v0, v1), axis=0)
                    result_q = np.concatenate((q0, q1), axis=0)
                    # Clean up splits
                    del p0, v0, q0, p1, v1, q1
                    return result_probs, result_values, result_q
                raise

        try:
            # Fixed microbatching on the batch dimension (states), to reduce peak activations.
            n_total = int(batch_size)
            infer_chunk = int(self.infer_chunk) if hasattr(
                self, "infer_chunk") else 0
            if infer_chunk and infer_chunk > 0 and n_total > infer_chunk:
                probs_parts = []
                value_parts = []
                q_parts = []
                i = 0
                while i < n_total:
                    j = min(n_total, i + int(infer_chunk))
                    p, v, q = _score_with_oom_splitting(i, j)
                    probs_parts.append(p)
                    value_parts.append(v)
                    q_parts.append(q)
                    i = j
                probs_np = np.concatenate(probs_parts, axis=0)
                value_np = np.concatenate(value_parts, axis=0)
                candidate_q_np = np.concatenate(q_parts, axis=0)
                del probs_parts, value_parts, q_parts
            else:
                probs_np, value_np, candidate_q_np = _score_with_oom_splitting(0, n_total)

            # -------------------------------------------------------
            # Debug: confirm inputs vary + value isn't flatlined
            # -------------------------------------------------------
            try:
                self.score_call_counter += 1
                diag_every = int(os.getenv("SCORE_DIAG_EVERY", "0"))
                if diag_every > 0 and (self.score_call_counter % diag_every == 0):
                    # mask: 1=pad, 0=valid (per StateSequenceBuilder). Keep this cheap: only inspect sample 0.
                    mask_view = np.frombuffer(
                        masks_bytes, dtype='<i4').reshape(batch_size, seq_len)
                    tok_view = np.frombuffer(
                        token_ids_bytes, dtype='<i4').reshape(batch_size, seq_len)
                    seq0 = np.frombuffer(sequences_bytes, dtype='<f4').reshape(
                        batch_size, seq_len, d_model)[0]
                    valid0 = int((mask_view[0] == 0).sum()
                                 ) if batch_size > 0 else -1
                    pad0 = int((mask_view[0] != 0).sum()
                               ) if batch_size > 0 else -1
                    seq_mean = float(seq0.mean()) if batch_size > 0 else 0.0
                    seq_std = float(seq0.std()) if batch_size > 0 else 0.0
                    tok_unique0 = int(
                        np.unique(tok_view[0]).shape[0]) if batch_size > 0 else 0
                    logger.info(
                        LogCategory.MODEL_TRAIN,
                        "ScoreDiag call=%d policy=%s batch=%d | valid_tokens[0]=%d pad_tokens[0]=%d | seq0(mean=%.4f std=%.4f) tok_unique0=%d | value(mean=%.4f min=%.4f max=%.4f)",
                        int(self.score_call_counter),
                        str(policy_key),
                        int(batch_size),
                        valid0,
                        pad0,
                        seq_mean,
                        seq_std,
                        tok_unique0,
                        float(value_np.mean()) if value_np.size > 0 else 0.0,
                        float(value_np.min()) if value_np.size > 0 else 0.0,
                        float(value_np.max()) if value_np.size > 0 else 0.0,
                    )
            except Exception:
                # Don't fail inference for diagnostics
                pass

            # Increment inference counter
            self.metrics.infer_counter += 1

            # Clear GPU cache immediately to prevent memory buildup from parallel workers
            # Force GC every 50 inferences to aggressively prevent tensor reference leaks
            # if torch.cuda.is_available():
            #     if self.metrics.infer_counter % 50 == 0:
            #         import gc
            #         gc.collect()
            #     torch.cuda.empty_cache()

            # Track timing
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.metrics.update_timing_metric("infer", elapsed_ms)

            out = np.concatenate((probs_np, value_np, candidate_q_np), axis=1)
            result_bytes = out.astype('<f4').tobytes()

            # Clean up numpy arrays to prevent accumulation
            del probs_np, value_np, candidate_q_np, out
            self._log_cuda_mem("scoreCandidatesPolicyFlat:end")

            return result_bytes

        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in scoreCandidatesPolicyFlat: %s", str(e))
            raise

    # Metrics methods and properties - delegate to metrics collector
    # Counters
    @property
    def train_step_counter(self):
        return self.metrics.train_step_counter

    @train_step_counter.setter
    def train_step_counter(self, value):
        self.metrics.train_step_counter = value

    @property
    def score_call_counter(self):
        return self.metrics.score_call_counter

    @score_call_counter.setter
    def score_call_counter(self, value):
        self.metrics.score_call_counter = value

    @property
    def main_train_sample_counter(self):
        return self.metrics.main_train_sample_counter

    @main_train_sample_counter.setter
    def main_train_sample_counter(self, value):
        self.metrics.main_train_sample_counter = value

    # GAE properties
    @property
    def use_gae(self):
        return self.metrics.use_gae

    @use_gae.setter
    def use_gae(self, value):
        self.metrics.use_gae = value

    @property
    def current_gae_lambda(self):
        return self.metrics.current_gae_lambda

    @current_gae_lambda.setter
    def current_gae_lambda(self, value):
        self.metrics.current_gae_lambda = value

    @property
    def gae_enabled_step(self):
        return self.metrics.gae_enabled_step

    @gae_enabled_step.setter
    def gae_enabled_step(self, value):
        self.metrics.gae_enabled_step = value

    # Delegated methods
    def get_entropy_coefficient(self):
        return self.metrics.get_entropy_coefficient()

    def record_value_prediction(self, value_pred, won):
        self.metrics.record_value_prediction(value_pred, won)

    def get_value_metrics(self):
        return self.metrics.get_value_metrics()

    def update_gae_lambda_schedule(self):
        self.metrics.update_gae_lambda_schedule()

    def compute_gae(self, rewards, values, gamma=0.99, gae_lambda=None, dones=None):
        return self.metrics.compute_gae(rewards, values, gamma, gae_lambda, dones)

    def _joint_logp_from_probs(self, probs_safe, cand_mask_t, chosen_indices_t, chosen_count_t):
        """
        Compute joint log-prob for sequential sampling without replacement, using a fixed base probs vector.
        Matches the Java behavior policy:
        - Start with remaining = cand_mask (valid candidates)
        - At each pick, renormalize over remaining and add log(p_cond)
        - Remove chosen index from remaining
        """
        bsz, max_c = probs_safe.shape
        remaining = cand_mask_t.bool().clone()
        chosen_count_t = torch.clamp(chosen_count_t.long(), min=0, max=max_c)
        logp = torch.zeros((bsz,), dtype=torch.float32,
                           device=probs_safe.device)

        # Only iterate up to max chosen count in batch (bounded by max_c <= 64).
        max_k = int(chosen_count_t.max().item()
                    ) if chosen_count_t.numel() > 0 else 0
        for t in range(max_k):
            active = chosen_count_t > t
            if not bool(active.any().item()):
                break
            denom = (probs_safe * remaining.float()).sum(dim=1).clamp_min(1e-8)
            idx = chosen_indices_t[:, t].long()
            idx_clamped = torch.clamp(idx, min=0, max=max_c - 1)
            p = probs_safe.gather(1, idx_clamped.unsqueeze(1)).squeeze(1)
            p_cond = (p / denom).clamp_min(1e-8)
            logp = logp + active.float() * torch.log(p_cond)

            # Remove chosen from remaining for active rows
            rows = torch.nonzero(active, as_tuple=False).squeeze(1)
            if rows.numel() > 0:
                remaining[rows, idx_clamped[rows]] = False
        return logp

    def trainCandidatesFlat(self,
                            sequences_bytes,
                            masks_bytes,
                            token_ids_bytes,
                            candidate_features_bytes,
                            candidate_ids_bytes,
                            candidate_mask_bytes,
                            chosen_indices_bytes,
                            chosen_count_bytes,
                            old_logp_total_bytes,
                            old_value_bytes,
                            rewards_bytes,  # Changed from discounted_returns_bytes
                            batch_size,
                            seq_len,
                            d_model,
                            max_candidates,
                            cand_feat_dim):
        """Train on padded candidate decision steps using GAE for advantage estimation."""
        lock_held = False
        if self.backend_mode == "single":
            self._gpu_mutex.acquire()
            lock_held = True
        try:
            if self.model is None or self.optimizer is None:
                raise RuntimeError("Model not initialized")

            device = self.device

            seq = np.frombuffer(sequences_bytes, dtype='<f4').reshape(
                batch_size, seq_len, d_model)
            mask = np.frombuffer(masks_bytes, dtype='<i4').reshape(
                batch_size, seq_len)
            tok_ids = np.frombuffer(token_ids_bytes, dtype='<i4').reshape(
                batch_size, seq_len)

            cand_feat = np.frombuffer(candidate_features_bytes, dtype='<f4').reshape(
                batch_size, max_candidates, cand_feat_dim)
            cand_ids = np.frombuffer(candidate_ids_bytes, dtype='<i4').reshape(
                batch_size, max_candidates)
            cand_mask = np.frombuffer(candidate_mask_bytes, dtype='<i4').reshape(
                batch_size, max_candidates)

            chosen_indices = np.frombuffer(chosen_indices_bytes, dtype='<i4').reshape(
                batch_size, max_candidates)
            chosen_count = np.frombuffer(chosen_count_bytes, dtype='<i4').reshape(
                batch_size)
            rewards = np.frombuffer(rewards_bytes, dtype='<f4').reshape(
                batch_size)

            seq_t = torch.tensor(seq, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
            tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device)
            cand_feat_t = torch.tensor(
                cand_feat, dtype=torch.float32, device=device)
            cand_ids_t = torch.tensor(
                cand_ids, dtype=torch.long, device=device)
            cand_mask_t = torch.tensor(
                cand_mask, dtype=torch.bool, device=device)
            chosen_indices_t = torch.tensor(
                chosen_indices, dtype=torch.long, device=device)
            chosen_count_t = torch.tensor(
                chosen_count, dtype=torch.long, device=device)
            rewards_t = torch.tensor(
                rewards, dtype=torch.float32, device=device)
            old_logp_total = np.frombuffer(
                old_logp_total_bytes, dtype='<f4').reshape(batch_size)
            old_logp_t = torch.tensor(
                old_logp_total, dtype=torch.float32, device=device)
            old_value_np = np.frombuffer(
                old_value_bytes, dtype='<f4').reshape(batch_size)
            old_value_t = torch.tensor(
                old_value_np, dtype=torch.float32, device=device)

            # Release numpy arrays immediately
            del seq, mask, tok_ids, cand_feat, cand_ids, cand_mask, chosen_indices, chosen_count, rewards, old_logp_total, old_value_np

            # Check for NaN/Inf in input data
            if torch.isnan(seq_t).any() or torch.isinf(seq_t).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in input sequence - skipping batch")
                self._log_cuda_mem("trainCandidatesFlat:skip_seq_nan")
                return
            if torch.isnan(rewards_t).any() or torch.isinf(rewards_t).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in rewards - skipping batch (rewards=%s)", rewards_t.tolist())
                self._log_cuda_mem("trainCandidatesFlat:skip_rewards_nan")
                return
            if torch.isnan(cand_feat_t).any() or torch.isinf(cand_feat_t).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in candidate features - skipping batch")
                self._log_cuda_mem("trainCandidatesFlat:skip_candfeat_nan")
                return

            self.model.train()
            next_step = self.train_step_counter + 1

            # -------------------------------------------------------
            # Encoder freeze during critic warmup (optional)
            # -------------------------------------------------------
            if self.freeze_encoder_in_warmup:
                in_warmup = self.loss_schedule_enable and next_step <= self.critic_warmup_steps
                if in_warmup and not self._encoder_frozen:
                    self._set_encoder_requires_grad(False)
                    self._encoder_frozen = True
                    logger.info(LogCategory.MODEL_TRAIN,
                                "Warmup: froze encoder params for %d steps", int(self.critic_warmup_steps))
                elif (not in_warmup) and self._encoder_frozen:
                    self._set_encoder_requires_grad(True)
                    self._encoder_frozen = False
                    logger.info(LogCategory.MODEL_TRAIN,
                                "Warmup ended: unfroze encoder params")

            use_amp = bool(self.amp_enable) and torch.cuda.is_available() and str(
                device).startswith("cuda")
            autocast_ctx = torch.autocast(
                device_type="cuda", dtype=self.amp_dtype) if use_amp else nullcontext()
            scaler = self.amp_scaler if (use_amp and bool(
                self.amp_use_scaler) and self.amp_scaler is not None) else None

            self.optimizer.zero_grad(set_to_none=True)

            # Get new policy probabilities (with gradients)
            with autocast_ctx:
                probs, value = self.model.score_candidates(
                    seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t)

            # Check for NaN in model outputs before training
            if torch.isnan(probs).any() or torch.isnan(value).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "Model produced NaN outputs - skipping batch (probs_nan=%s, value_nan=%s)",
                               torch.isnan(probs).any().item(), torch.isnan(value).any().item())
                self._log_cuda_mem("trainCandidatesFlat:skip_model_nan")
                return

            # Compute advantages using GAE or simple baseline
            value_squeezed = value.squeeze(1)

            # IMPORTANT: Detach values for GAE computation to prevent gradients
            # flowing through value targets and advantages. Only the current
            # value prediction (value_squeezed) should have gradients for loss_value.
            value_detached = value_squeezed.detach()

            if self.use_gae:
                # Use GAE for advantage estimation (reduces variance, better credit assignment)
                self.update_gae_lambda_schedule()
                # Flat batches are not guaranteed to be a single ordered trajectory; prevent leakage.
                dones_t = torch.ones_like(rewards_t)
                advantages, value_targets = self.compute_gae(
                    rewards_t, value_detached, gamma=self.gamma, gae_lambda=self.current_gae_lambda, dones=dones_t)
                advantages = advantages.detach()
                value_targets = value_targets.detach()
            else:
                # Flat batches: treat each sample as terminal to avoid cross-sample leakage.
                value_targets = rewards_t
                advantages = rewards_t - value_detached

            # -------------------------------------------------------
            # Debug: value target / prediction distribution
            # -------------------------------------------------------
            diag_every = int(os.getenv("VALUE_TARGET_DIAG_EVERY", "50"))
            post_every = int(os.getenv("VALUE_POSTUPDATE_DIAG_EVERY", "0"))
            do_post = post_every > 0 and (next_step % post_every == 0)
            # Optional eval-mode baseline before update (removes dropout noise)
            v_before_eval = None
            if do_post:
                with torch.no_grad():
                    self.model.eval()
                    _p0, _v0 = self.model.score_candidates(
                        seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t
                    )
                    v_before_eval = _v0.squeeze(1).detach().float().view(-1)
                    self.model.train()
            if diag_every > 0 and (next_step % diag_every == 0):
                with torch.no_grad():
                    def _stats(name, t):
                        t = t.detach().float().view(-1)
                        if t.numel() == 0:
                            return f"{name}: empty"
                        pos = (t > 0).float().mean().item() * 100.0
                        neg = (t < 0).float().mean().item() * 100.0
                        zer = 100.0 - pos - neg
                        return (f"{name}: mean={t.mean().item():.3f} std={t.std().item():.3f} "
                                f"min={t.min().item():.3f} max={t.max().item():.3f} "
                                f"pos={pos:.1f}% neg={neg:.1f}% zero={zer:.1f}%")

                    logger.info(LogCategory.MODEL_TRAIN,
                                "ValueDiag step=%d use_gae=%s lam=%.3f batch=%d | %s | %s | %s | %s",
                                next_step, self.use_gae,
                                (self.current_gae_lambda if self.use_gae else 1.0),
                                int(batch_size),
                                _stats("value_pred", value_squeezed),
                                _stats("value_target", value_targets),
                                _stats("adv", advantages),
                                _stats("reward", rewards_t))

            # Check for NaN in advantages BEFORE normalization
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in GAE advantages (batch_size=%d, rewards=%s, values_range=[%.4f,%.4f]) - skipping batch",
                               # first 5 rewards
                               batch_size, rewards_t.tolist()[:5],
                               value_detached.min().item(), value_detached.max().item())
                self._log_cuda_mem("trainCandidatesFlat:skip_adv_nan")
                return

            # Normalize advantages for policy gradient stability
            # Use RUNNING statistics instead of per-batch to avoid destroying signal
            # when all samples in batch have similar outcomes (common in RL)
            advantages_normalized = advantages
            if self._adv_use_running and batch_size >= 1:
                # Update running statistics with current batch
                batch_mean = float(advantages.mean().item())
                batch_var = float(advantages.var().item()
                                  ) if batch_size >= 2 else 1.0
                alpha = self._adv_ema_alpha
                # Include between-batch variance (mean shift) for proper streaming variance
                mean_shift_sq = (batch_mean - self._adv_running_mean) ** 2
                combined_var = batch_var + mean_shift_sq
                self._adv_running_mean = (
                    1 - alpha) * self._adv_running_mean + alpha * batch_mean
                self._adv_running_var = (
                    1 - alpha) * self._adv_running_var + alpha * combined_var
                # Normalize using running stats (not per-batch)
                running_std = max(self._adv_running_var ** 0.5, 1e-8)
                advantages_normalized = (
                    advantages - self._adv_running_mean) / running_std
            elif batch_size >= 2:
                # Fallback to per-batch if running stats disabled
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages_normalized = (
                        advantages - advantages.mean()) / adv_std

            # Clip normalized advantages to prevent gradient explosions
            adv_clip_max = float(os.getenv('ADV_CLIP_MAX', '5.0'))
            advantages_normalized = advantages_normalized.clamp(-adv_clip_max, adv_clip_max)

            probs_safe = torch.clamp(probs, min=1e-8, max=1.0)
            new_logp = self._joint_logp_from_probs(
                probs_safe, cand_mask_t, chosen_indices_t, chosen_count_t)

            if self.use_ppo:
                # PPO: Clipped surrogate objective with numerical stability
                # Use unclamped PPO ratio in objective; clamp only log-ratio for numeric safety.
                log_ratio = (new_logp - old_logp_t).clamp(-20.0, 20.0)
                ratio_raw = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio_raw, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
                # Use normalized advantages for policy to prevent huge gradients
                loss_policy = - \
                    torch.min(ratio_raw * advantages_normalized,
                              clipped_ratio * advantages_normalized).mean()

                # PPO stats logging (mtg_ai.log) - mean/std of adv/ret/ratio
                if self._ppo_stats_every > 0 and (next_step % self._ppo_stats_every == 0):
                    with torch.no_grad():
                        adv_t = advantages.detach().float().view(-1)
                        ret_t = value_targets.detach().float().view(-1)
                        rr_t = ratio_raw.detach().float().view(-1)
                        adv_mean = float(adv_t.mean().item()
                                         ) if adv_t.numel() > 0 else 0.0
                        adv_std = float(adv_t.std().item()
                                        ) if adv_t.numel() > 1 else 0.0
                        ret_mean = float(ret_t.mean().item()
                                         ) if ret_t.numel() > 0 else 0.0
                        ret_std = float(ret_t.std().item()
                                        ) if ret_t.numel() > 1 else 0.0
                        ratio_mean = float(
                            rr_t.mean().item()) if rr_t.numel() > 0 else 1.0
                        ratio_std = float(
                            rr_t.std().item()) if rr_t.numel() > 1 else 0.0
                    logger.info(
                        LogCategory.MODEL_TRAIN,
                        "PPOStats step=%d adv(mean=%.4f std=%.4f) ret(mean=%.4f std=%.4f) ratio(mean=%.4f std=%.4f)",
                        int(next_step),
                        adv_mean, adv_std,
                        ret_mean, ret_std,
                        ratio_mean, ratio_std
                    )
            else:
                # REINFORCE: Simple policy gradient with normalized advantages
                loss_policy = -(new_logp * advantages_normalized).mean()

            # -------------------------------------------------------
            # Loss coefficients (scheduled)
            # -------------------------------------------------------
            if self.loss_schedule_enable and next_step <= self.critic_warmup_steps:
                policy_loss_coef = float(self.policy_loss_coef_warmup)
                value_loss_coef = float(self.value_loss_coef_warmup)
                entropy_loss_mult = float(self.entropy_loss_mult_warmup)
            else:
                policy_loss_coef = float(self.policy_loss_coef_main)
                value_loss_coef = float(self.value_loss_coef_main)
                entropy_loss_mult = float(self.entropy_loss_mult_main)

            # Value loss: MSE between predicted value and GAE value targets
            # Coefficient increased to 5.0 to force value head out of local minimum
            # At 1.0 coeff, value head gets stuck at -0.5 because MSE loss is "acceptable"
            # Higher coeff forces it to actually learn correct predictions
            vf_clip = float(os.getenv("PPO_VF_CLIP", "0.2"))
            if self.use_ppo and vf_clip > 0.0:
                v_pred = value_squeezed
                v_old = old_value_t.view_as(v_pred).detach()
                v_clipped = v_old + (v_pred - v_old).clamp(-vf_clip, vf_clip)
                vf_loss1 = (v_pred - value_targets).pow(2)
                vf_loss2 = (v_clipped - value_targets).pow(2)
                loss_value = value_loss_coef * \
                    (0.5 * torch.max(vf_loss1, vf_loss2).mean())
            else:
                loss_value = value_loss_coef * \
                    F.mse_loss(value_squeezed, value_targets)

            # Entropy bonus (encourage exploration with decay schedule)
            # Clamp probabilities to avoid log(0)
            probs_safe = torch.clamp(probs, min=1e-8, max=1.0)
            log_probs = torch.log(probs_safe)
            entropy = -(probs_safe * log_probs).sum(dim=-1).mean()
            entropy_coef = float(
                self.get_entropy_coefficient()) * float(entropy_loss_mult)
            loss_entropy = -entropy_coef * entropy

            loss = (policy_loss_coef * loss_policy) + loss_value + loss_entropy

            # NaN guard: skip update if loss is NaN to prevent model corruption
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(LogCategory.MODEL_TRAIN,
                               "Skipping update due to NaN/Inf loss (policy=%.4f, value=%.4f, ent=%.4f)",
                               loss_policy.item() if not torch.isnan(loss_policy) else float('nan'),
                               loss_value.item() if not torch.isnan(loss_value) else float('nan'),
                               entropy.item() if not torch.isnan(entropy) else float('nan'))
                self.optimizer.zero_grad()
                self._log_cuda_mem("trainCandidatesFlat:skip_loss_nan")
                return

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # -------------------------------------------------------
            # Debug: ensure critic gradients are nonzero
            # -------------------------------------------------------
            grad_diag_every = int(os.getenv("VALUE_GRAD_DIAG_EVERY", "0"))
            if grad_diag_every > 0 and (next_step % grad_diag_every == 0):
                with torch.no_grad():
                    def _grad_norm(needle: str):
                        s = 0.0
                        c = 0
                        for n, p in self.model.named_parameters():
                            if needle in n and p.grad is not None:
                                g = p.grad.detach()
                                s += float(g.norm(2).item())
                                c += 1
                        return s, c

                    critic_gn, critic_c = _grad_norm("critic_")
                    actor_gn, actor_c = _grad_norm("actor_")
                    enc_gn, enc_c = _grad_norm("transformer_layers")
                    vs_gn = 0.0
                    vs_has = 0
                    for n, p in self.model.named_parameters():
                        if n.endswith("value_scale") and p.grad is not None:
                            vs_gn = float(p.grad.detach().norm(2).item())
                            vs_has = 1
                            break

                    logger.info(
                        LogCategory.MODEL_TRAIN,
                        "GradDiag step=%d | critic=%.6f(n=%d) actor=%.6f(n=%d) enc=%.6f(n=%d) value_scale=%.6f(has=%d)",
                        int(next_step),
                        critic_gn, int(critic_c),
                        actor_gn, int(actor_c),
                        enc_gn, int(enc_c),
                        vs_gn, int(vs_has)
                    )

            # Check for NaN gradients before applying
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                logger.warning(LogCategory.MODEL_TRAIN,
                               "Skipping update due to NaN/Inf gradients")
                self.optimizer.zero_grad()
                self._log_cuda_mem("trainCandidatesFlat:skip_grad_nan")
                return

            if scaler is not None:
                try:
                    scaler.unscale_(self.optimizer)
                except Exception:
                    pass
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.model.max_grad_norm)
            if scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            # Log with PPO info if enabled
            if self.use_ppo:
                with torch.no_grad():
                    # Log the same stable ratio as the loss, plus approx KL.
                    approx_kl = (old_logp_t - new_logp).mean()
                    clip_frac = ((ratio_raw < 1.0 - self.ppo_epsilon) |
                                 (ratio_raw > 1.0 + self.ppo_epsilon)).float().mean()
                logger.info(LogCategory.MODEL_TRAIN,
                            "trainCandidatesFlat — loss=%.4f policy=%.4f value=%.4f ent=%.4f (coeff: %.4f) [PPO clip: %.2f%% kl=%.6f]",
                            loss.item(), loss_policy.item(), loss_value.item(), entropy.item(),
                            entropy_coef, clip_frac.item() * 100, approx_kl.item())
                # Record loss components for metrics export
                self.metrics.record_train_losses(
                    total_loss=loss.item(),
                    policy_loss=loss_policy.item(),
                    value_loss=loss_value.item(),
                    entropy=entropy.item(),
                    entropy_coef=entropy_coef,
                    clip_frac=clip_frac.item(),
                    approx_kl=approx_kl.item(),
                    batch_size=batch_size,
                    advantage_mean=float(advantages_normalized.mean().item())
                )
                # Write to CSV for post-hoc analysis
                self._write_training_losses_csv(episodes_in_batch=1)
            else:
                logger.info(LogCategory.MODEL_TRAIN,
                            "trainCandidatesFlat — loss=%.4f policy=%.4f value=%.4f ent=%.4f (coeff: %.4f)",
                            loss.item(), loss_policy.item(), loss_value.item(), entropy.item(), float(self.get_entropy_coefficient()) * float(entropy_loss_mult))
                # Record loss components for metrics export
                self.metrics.record_train_losses(
                    total_loss=loss.item(),
                    policy_loss=loss_policy.item(),
                    value_loss=loss_value.item(),
                    entropy=entropy.item(),
                    entropy_coef=entropy_coef,
                    clip_frac=0.0,
                    approx_kl=0.0,
                    batch_size=batch_size,
                    advantage_mean=float(advantages_normalized.mean().item())
                )
                # Write to CSV for post-hoc analysis
                self._write_training_losses_csv(episodes_in_batch=1)

            # -------------------------------------------------------
            # Debug: did this update move value toward target?
            # -------------------------------------------------------
            if do_post:
                with torch.no_grad():
                    self.model.eval()
                    _p1, _v1 = self.model.score_candidates(
                        seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t
                    )
                    v_after_eval = _v1.squeeze(1).detach().float().view(-1)
                    self.model.train()

                    v_before = v_before_eval if v_before_eval is not None else value_squeezed.detach().float().view(-1)
                    v_after = v_after_eval
                    v_tgt = value_targets.detach().float().view(-1)

                    mse_before = float(F.mse_loss(
                        v_before, v_tgt).item()) if v_before.numel() > 0 else 0.0
                    mse_after = float(F.mse_loss(
                        v_after, v_tgt).item()) if v_after.numel() > 0 else 0.0

                    def _s(t):
                        if t.numel() == 0:
                            return "empty"
                        pos = (t > 0).float().mean().item() * 100.0
                        return f"mean={t.mean().item():.3f} min={t.min().item():.3f} max={t.max().item():.3f} pos={pos:.1f}%"

                    logger.info(
                        LogCategory.MODEL_TRAIN,
                        "ValuePost step=%d | pred_before(%s) -> pred_after(%s) | target(%s) | mse_before=%.4f mse_after=%.4f | coefs(policy=%.2f value=%.2f entMult=%.2f)",
                        int(next_step),
                        _s(v_before),
                        _s(v_after),
                        _s(v_tgt),
                        mse_before,
                        mse_after,
                        float(policy_loss_coef),
                        float(value_loss_coef),
                        float(entropy_loss_mult),
                    )

            # -------------------------------------------------------
            #  Increment counters & checkpoint every 100 updates
            # -------------------------------------------------------
            self.train_step_counter += 1
            self.main_train_sample_counter += batch_size
            if self.train_step_counter % 100 == 0:
                # Save to persistent location, not temp directory
                if self.model_path:
                    # Overwrite main model every 100 steps for continuous checkpointing
                    ckpt_path = self.model_path
                else:
                    # Fallback to profile-aware models directory
                    ckpt_path = f'{profile_models_dir()}/model.pt'
                try:
                    self.saveModel(ckpt_path)
                    logger.info(LogCategory.MODEL_SAVE,
                                "Checkpoint saved at step %d -> %s",
                                self.train_step_counter, ckpt_path)
                except Exception as e:
                    logger.error(LogCategory.MODEL_SAVE,
                                 "Failed to save checkpoint at step %d: %s",
                                 self.train_step_counter, str(e))

            # Explicit cleanup before saving to reduce VRAM pressure
            del seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t
            del chosen_indices_t, chosen_count_t, rewards_t, old_logp_t, old_value_t, value_targets, advantages
            del loss, loss_policy, loss_value, entropy
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            if torch.cuda.is_available() and bool(int(os.getenv("CUDA_EMPTY_CACHE_AFTER_TRAIN", "0"))):
                torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Snapshot checkpoints (league opponents)
            self._maybe_save_snapshot()

            # Torch profiler step (env-gated)
            self._prof_step("train")
            self._log_cuda_mem("trainCandidatesFlat:end")
            return True

        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in trainCandidatesFlat: %s", str(e))
            raise
        finally:
            if lock_held:
                try:
                    self._gpu_mutex.release()
                except Exception:
                    pass

    def trainCandidatesMultiFlat(self,
                                 sequences_bytes,
                                 masks_bytes,
                                 token_ids_bytes,
                                 candidate_features_bytes,
                                 candidate_ids_bytes,
                                 candidate_mask_bytes,
                                 rewards_bytes,
                                 chosen_indices_bytes,
                                 chosen_count_bytes,
                                 old_logp_total_bytes,
                                 old_value_bytes,
                                 sample_weights_bytes,
                                 dones_bytes,
                                 head_ids_bytes,
                                 batch_size,
                                 seq_len,
                                 d_model,
                                 max_candidates,
                                 cand_feat_dim,
                                 archetype_labels_bytes=None,
                                 num_archetypes=0,
                                 mcts_visits_bytes=None,
                                 card_belief_labels_bytes=None,
                                 card_belief_dim=0):
        """
        Train on a batch that concatenates multiple episodes.
        dones marks episode ends (1=end-of-episode), so GAE/returns do not leak across boundaries.
        head_ids_bytes: per-step head index
            (0=action, 1=target, 2=card_select, 3=attack, 4=block, 5=mulligan).
        """
        t_start = time.perf_counter()
        lock_held = False
        if self.backend_mode == "single":
            self._gpu_mutex.acquire()
            lock_held = True

        def _train_slice(start: int, end: int):
            if self.model is None or self.optimizer is None:
                raise RuntimeError("Model not initialized")
            device = self.device

            seq = np.frombuffer(sequences_bytes, dtype='<f4').reshape(
                batch_size, seq_len, d_model)[start:end]
            mask = np.frombuffer(masks_bytes, dtype='<i4').reshape(
                batch_size, seq_len)[start:end]
            tok_ids = np.frombuffer(token_ids_bytes, dtype='<i4').reshape(
                batch_size, seq_len)[start:end]

            cand_feat = np.frombuffer(candidate_features_bytes, dtype='<f4').reshape(
                batch_size, max_candidates, cand_feat_dim)[start:end]
            cand_ids = np.frombuffer(candidate_ids_bytes, dtype='<i4').reshape(
                batch_size, max_candidates)[start:end]
            cand_mask = np.frombuffer(candidate_mask_bytes, dtype='<i4').reshape(
                batch_size, max_candidates)[start:end]

            chosen_indices = np.frombuffer(chosen_indices_bytes, dtype='<i4').reshape(
                batch_size, max_candidates)[start:end]
            chosen_count = np.frombuffer(chosen_count_bytes, dtype='<i4').reshape(
                batch_size)[start:end]
            rewards = np.frombuffer(rewards_bytes, dtype='<f4').reshape(
                batch_size)[start:end]
            old_logp_total = np.frombuffer(old_logp_total_bytes, dtype='<f4').reshape(
                batch_size)[start:end]
            old_value = np.frombuffer(old_value_bytes, dtype='<f4').reshape(
                batch_size)[start:end]
            sample_weights = np.frombuffer(sample_weights_bytes, dtype='<f4').reshape(
                batch_size)[start:end]
            dones = np.frombuffer(dones_bytes, dtype='<i4').reshape(
                batch_size)[start:end]
            head_ids = np.frombuffer(head_ids_bytes, dtype='<i4').reshape(
                batch_size)[start:end]

            # Phase 1 archetype labels (optional - may be absent from older senders).
            archetype_labels_np = None
            _num_archetypes = int(num_archetypes or 0)
            if _num_archetypes > 0 and archetype_labels_bytes:
                try:
                    archetype_labels_np = np.frombuffer(
                        archetype_labels_bytes, dtype='<i4').reshape(
                        batch_size)[start:end]
                except Exception:
                    archetype_labels_np = None

            # AlphaZero MCTS visit distribution target (optional; zero rows = no target).
            mcts_visits_np = None
            if mcts_visits_bytes:
                try:
                    mcts_visits_np = np.frombuffer(
                        mcts_visits_bytes, dtype='<f4').reshape(
                        batch_size, max_candidates)[start:end]
                except Exception:
                    mcts_visits_np = None

            # Generic card-level hidden-state belief labels. Rows filled with
            # -1 are absent and skipped by the loss.
            card_belief_labels_np = None
            _card_belief_dim = int(card_belief_dim or 0)
            if _card_belief_dim > 0 and card_belief_labels_bytes:
                try:
                    card_belief_labels_np = np.frombuffer(
                        card_belief_labels_bytes, dtype='<f4').reshape(
                        batch_size, _card_belief_dim)[start:end]
                except Exception:
                    card_belief_labels_np = None

            _nb = str(device).startswith("cuda")
            seq_t = torch.tensor(seq, dtype=torch.float32).to(device, non_blocking=_nb)
            mask_t = torch.tensor(mask, dtype=torch.bool).to(device, non_blocking=_nb)
            tok_t = torch.tensor(tok_ids, dtype=torch.long).to(device, non_blocking=_nb)
            cand_feat_t = torch.tensor(cand_feat, dtype=torch.float32).to(device, non_blocking=_nb)
            cand_ids_t = torch.tensor(cand_ids, dtype=torch.long).to(device, non_blocking=_nb)
            cand_mask_t = torch.tensor(cand_mask, dtype=torch.bool).to(device, non_blocking=_nb)
            chosen_indices_t = torch.tensor(chosen_indices, dtype=torch.long).to(device, non_blocking=_nb)
            chosen_count_t = torch.tensor(chosen_count, dtype=torch.long).to(device, non_blocking=_nb)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device, non_blocking=_nb)
            old_logp_t = torch.tensor(old_logp_total, dtype=torch.float32).to(device, non_blocking=_nb)
            old_value_t = torch.tensor(old_value, dtype=torch.float32).to(device, non_blocking=_nb)
            sample_w_t = torch.tensor(sample_weights, dtype=torch.float32).to(device, non_blocking=_nb)
            dones_t = torch.tensor(dones, dtype=torch.float32).to(device, non_blocking=_nb)
            head_idx_t = torch.tensor(head_ids, dtype=torch.long).to(device, non_blocking=_nb)

            archetype_labels_t = None
            if archetype_labels_np is not None:
                archetype_labels_t = torch.tensor(
                    archetype_labels_np, dtype=torch.long).to(device, non_blocking=_nb)

            mcts_visits_t = None
            if mcts_visits_np is not None:
                mcts_visits_t = torch.tensor(
                    mcts_visits_np, dtype=torch.float32).to(device, non_blocking=_nb)

            card_belief_labels_t = None
            if card_belief_labels_np is not None:
                card_belief_labels_t = torch.tensor(
                    card_belief_labels_np, dtype=torch.float32).to(device, non_blocking=_nb)

            if _nb:
                torch.cuda.synchronize(device)

            # Release numpy slices immediately
            del seq, mask, tok_ids, cand_feat, cand_ids, cand_mask, chosen_indices, chosen_count, rewards, old_logp_total, old_value, sample_weights, dones, head_ids
            archetype_labels_np = None
            card_belief_labels_np = None

            local_batch_size = int(end - start)
            mcts_signed_targets = bool(int(os.getenv(
                "CANDIDATE_Q_MCTS_SIGNED_TARGETS",
                os.getenv("CANDIDATE_Q_BRANCH_RETURN_TARGETS", "0"))))

            _bad = torch.stack([
                ~torch.isfinite(seq_t).all(),
                ~torch.isfinite(rewards_t).all(),
                ~torch.isfinite(cand_feat_t).all(),
            ])
            if _bad.any().item():
                which = []
                if _bad[0]:
                    which.append("sequences")
                if _bad[1]:
                    which.append("rewards")
                if _bad[2]:
                    which.append("cand_feat")
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in %s - skipping batch", ",".join(which))
                self._log_cuda_mem("trainCandidatesMultiFlat:skip_nan")
                return
            sample_w_t = torch.nan_to_num(
                sample_w_t, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(0.0)
            w_sum = sample_w_t.sum()
            if (not torch.isfinite(w_sum)) or float(w_sum.item()) <= 0.0:
                sample_w_t = torch.ones_like(sample_w_t)
                w_sum = sample_w_t.sum()
            norm_w_t = sample_w_t / w_sum.clamp_min(1e-8)
            norm_w_t = norm_w_t * float(max(1, local_batch_size))

            self.model.train()
            # Preserve the historical semantics of train_step_counter as "episodes processed".
            # trainCandidatesFlat was called once per episode; here we batch multiple episodes.
            ep_count = int(torch.clamp(dones_t.detach().sum(), min=1.0).item())
            next_step = self.train_step_counter + ep_count

            if self.freeze_encoder_in_warmup:
                in_warmup = self.loss_schedule_enable and next_step <= self.critic_warmup_steps
                if in_warmup and not self._encoder_frozen:
                    self._set_encoder_requires_grad(False)
                    self._encoder_frozen = True
                    logger.info(LogCategory.MODEL_TRAIN,
                                "Warmup: froze encoder params for %d steps", int(self.critic_warmup_steps))
                elif (not in_warmup) and self._encoder_frozen:
                    self._set_encoder_requires_grad(True)
                    self._encoder_frozen = False
                    logger.info(LogCategory.MODEL_TRAIN,
                                "Warmup ended: unfroze encoder params")

            # GPU lock is acquired by Java learner loop, not here
            # (Learner holds lock for entire training burst)
            try:
                use_amp = bool(self.amp_enable) and torch.cuda.is_available() and str(
                    device).startswith("cuda")
                autocast_ctx = torch.autocast(
                    device_type="cuda", dtype=self.amp_dtype) if use_amp else nullcontext()
                scaler = self.amp_scaler if (use_amp and bool(
                    self.amp_use_scaler) and self.amp_scaler is not None) else None

                # Chunking: cap peak VRAM by splitting forward/backward into
                # step-wise chunks. Pass 1 (below) does a no_grad forward for GAE;
                # Pass 2 (after advantage normalization) does grad forward + loss + backward
                # per chunk so activations from earlier chunks are freed before the next.
                _HEAD_NAMES = ["action", "target", "card_select", "attack", "block", "mulligan"]
                train_chunk_size = int(os.getenv("TRAIN_CHUNK_SIZE", "256"))
                if train_chunk_size <= 0 or train_chunk_size >= local_batch_size:
                    chunk_starts = [0]
                    effective_chunk = local_batch_size
                else:
                    chunk_starts = list(range(0, local_batch_size, train_chunk_size))
                    effective_chunk = train_chunk_size

                if bool(int(os.getenv("BC_DIRECT_LOSS", "0"))):
                    if mcts_visits_t is not None and not mcts_signed_targets:
                        targets_full = mcts_visits_t.float().clone()
                    else:
                        targets_full = torch.zeros(local_batch_size, max_candidates, device=device)
                    targets_full = torch.nan_to_num(targets_full, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                    row_sum = targets_full.sum(dim=-1)

                    # Fallback for older data: if no MCTS/BC distribution exists,
                    # convert chosenIndices into a uniform multi-label target.
                    empty_rows = row_sum <= 1e-6
                    if empty_rows.any():
                        fallback = torch.zeros_like(targets_full)
                        max_choices = min(int(chosen_indices_t.shape[1]), max_candidates)
                        for _pick in range(max_choices):
                            _idx = chosen_indices_t[:, _pick]
                            _valid = (
                                empty_rows
                                & (_pick < chosen_count_t)
                                & (_idx >= 0)
                                & (_idx < max_candidates)
                                & cand_mask_t.gather(1, _idx.clamp(0, max_candidates - 1).unsqueeze(1)).squeeze(1)
                            )
                            if _valid.any():
                                fallback[_valid, _idx[_valid].long()] = 1.0
                        fallback_sum = fallback.sum(dim=-1)
                        use_fallback = empty_rows & (fallback_sum > 1e-6)
                        targets_full[use_fallback] = fallback[use_fallback]
                        row_sum = targets_full.sum(dim=-1)

                    valid_rows = row_sum > 1e-6
                    if not valid_rows.any():
                        logger.warning(LogCategory.MODEL_TRAIN,
                                       "BC_DIRECT_LOSS requested but batch has no target rows")
                        return
                    if bool(int(os.getenv("BC_HARDEN_BINARY_TARGETS", "0"))):
                        valid_count_full = cand_mask_t.bool().sum(dim=-1)
                        binary_rows = valid_rows & (valid_count_full == 2)
                        if binary_rows.any():
                            masked_targets = targets_full.masked_fill(~cand_mask_t.bool(), -1.0)
                            hard_indices = masked_targets.argmax(dim=-1).long()
                            force_binary_idx = os.getenv("BC_FORCE_BINARY_TARGET_IDX", "").strip()
                            if force_binary_idx:
                                try:
                                    forced_idx = int(force_binary_idx)
                                    if 0 <= forced_idx < max_candidates:
                                        hard_indices = hard_indices.clone()
                                        hard_indices[binary_rows] = forced_idx
                                except Exception:
                                    pass
                            hard_targets = torch.zeros_like(targets_full)
                            hard_targets.scatter_(1, hard_indices.unsqueeze(1), 1.0)
                            if not getattr(self, "_bc_harden_binary_diag_emitted", False):
                                hard_indices_cpu = hard_indices[binary_rows].detach().to(torch.long).cpu()
                                head_idx_cpu = head_idx_t[binary_rows].detach().to(torch.long).cpu()
                                head_counts = torch.bincount(head_idx_cpu, minlength=len(_HEAD_NAMES)).tolist()
                                target_counts = torch.bincount(
                                    hard_indices_cpu, minlength=max_candidates).tolist()
                                logger.info(
                                    LogCategory.MODEL_TRAIN,
                                    "BC_HARDEN_BINARY_TARGETS active binaryRows=%d headCounts=%s targetArgmaxCountsFirst8=%s",
                                    int(binary_rows.detach().sum().item()),
                                    str({name: int(head_counts[i]) for i, name in enumerate(_HEAD_NAMES)}),
                                    str([int(v) for v in target_counts[:8]])
                                )
                                print(
                                    "BC_HARDEN_BINARY_TARGETS active "
                                    f"binaryRows={int(binary_rows.detach().sum().item())} "
                                    f"headCounts={{{', '.join(f'{name}: {int(head_counts[i])}' for i, name in enumerate(_HEAD_NAMES))}}} "
                                    f"targetArgmaxCountsFirst8={[int(v) for v in target_counts[:8]]}",
                                    flush=True
                                )
                                diag_file = os.getenv("BC_HARDEN_BINARY_DIAG_FILE", "").strip()
                                if diag_file:
                                    try:
                                        with open(diag_file, "a", encoding="utf-8") as _diag:
                                            _diag.write(
                                                "BC_HARDEN_BINARY_TARGETS active "
                                                f"binaryRows={int(binary_rows.detach().sum().item())} "
                                                f"headCounts={{{', '.join(f'{name}: {int(head_counts[i])}' for i, name in enumerate(_HEAD_NAMES))}}} "
                                                f"targetArgmaxCountsFirst8={[int(v) for v in target_counts[:8]]}\n"
                                            )
                                    except Exception:
                                        pass
                                self._bc_harden_binary_diag_emitted = True
                            targets_full[binary_rows] = hard_targets[binary_rows]
                            row_sum = targets_full.sum(dim=-1)
                    targets_full = targets_full / row_sum.unsqueeze(1).clamp_min(1e-6)
                    targets_full = targets_full * cand_mask_t.float()
                    targets_full = targets_full / targets_full.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                    bc_coef = float(os.getenv("BC_DIRECT_LOSS_COEF", str(self.mcts_kl_loss_coef)))
                    effective_sample_w_t = sample_w_t
                    if bool(int(os.getenv("BC_BALANCE_BINARY_TARGETS", "0"))):
                        valid_count_full = cand_mask_t.bool().sum(dim=-1)
                        balance_rows = valid_rows & (valid_count_full == 2)
                        if balance_rows.any():
                            target_idx_for_weight = targets_full.argmax(dim=-1).long()
                            idx_cpu = target_idx_for_weight[balance_rows].detach().to(torch.long).cpu()
                            counts = torch.bincount(idx_cpu, minlength=max_candidates).float()
                            active = counts > 0
                            class_count = float(active.sum().item())
                            if class_count > 1.0:
                                total_binary = float(balance_rows.detach().sum().item())
                                class_weights = torch.ones(max_candidates, device=device)
                                for _i in range(max_candidates):
                                    if counts[_i].item() > 0:
                                        class_weights[_i] = total_binary / (class_count * float(counts[_i].item()))
                                effective_sample_w_t = sample_w_t.clone()
                                effective_sample_w_t[balance_rows] = (
                                    effective_sample_w_t[balance_rows]
                                    * class_weights[target_idx_for_weight[balance_rows]]
                                )
                                diag_file = os.getenv("BC_HARDEN_BINARY_DIAG_FILE", "").strip()
                                if diag_file and not getattr(self, "_bc_balance_binary_diag_emitted", False):
                                    try:
                                        with open(diag_file, "a", encoding="utf-8") as _diag:
                                            _diag.write(
                                                "BC_BALANCE_BINARY_TARGETS active "
                                                f"binaryRows={int(balance_rows.detach().sum().item())} "
                                                f"countsFirst8={[int(v) for v in counts[:8].tolist()]} "
                                                f"weightsFirst8={[float(v) for v in class_weights[:8].detach().cpu().tolist()]}\n"
                                            )
                                    except Exception:
                                        pass
                                    self._bc_balance_binary_diag_emitted = True
                    total_bc_weight = effective_sample_w_t[valid_rows].sum().clamp_min(1e-8).detach()
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    total_rows = 0
                    for _c_start in chunk_starts:
                        _c_end = min(_c_start + effective_chunk, local_batch_size)
                        _head_idx_c = head_idx_t[_c_start:_c_end]
                        _seq_c = seq_t[_c_start:_c_end]
                        _mask_c = mask_t[_c_start:_c_end]
                        _tok_c = tok_t[_c_start:_c_end]
                        _cf_c = cand_feat_t[_c_start:_c_end]
                        _ci_c = cand_ids_t[_c_start:_c_end]
                        _cm_c = cand_mask_t[_c_start:_c_end]
                        _chunk_N = _c_end - _c_start
                        _logits_c = torch.zeros(_chunk_N, max_candidates, device=device)
                        with autocast_ctx:
                            for _hid_val, _hid_name in enumerate(_HEAD_NAMES):
                                _hmask = (_head_idx_c == _hid_val)
                                if not _hmask.any():
                                    continue
                                _, _, _logits_h = self.model.score_candidates(
                                    _seq_c[_hmask], _mask_c[_hmask], _tok_c[_hmask],
                                    _cf_c[_hmask], _ci_c[_hmask], _cm_c[_hmask],
                                    head_id=_hid_name, return_logits=True)
                                _logits_c[_hmask] = _logits_h.float()
                            _targets_c = targets_full[_c_start:_c_end]
                            _valid_c = valid_rows[_c_start:_c_end]
                            if not _valid_c.any():
                                continue
                            _log_probs_c = torch.log_softmax(_logits_c.float(), dim=-1)
                            _loss_per = -(_targets_c * _log_probs_c).sum(dim=-1)
                            _weights_c = effective_sample_w_t[_c_start:_c_end]
                            _chunk_loss = bc_coef * ((_loss_per[_valid_c] * _weights_c[_valid_c]).sum() / total_bc_weight)
                        if scaler is not None:
                            scaler.scale(_chunk_loss).backward()
                        else:
                            _chunk_loss.backward()
                        total_loss += float(_chunk_loss.detach().item())
                        total_rows += int(_valid_c.detach().sum().item())

                    if scaler is not None:
                        try:
                            scaler.unscale_(self.optimizer)
                        except Exception:
                            pass
                    _probe_name = None
                    _probe_param = None
                    for _name, _param in self.model.named_parameters():
                        if _name.startswith("policy_scorer_mulligan.") and _param.requires_grad:
                            _probe_name = _name
                            _probe_param = _param
                            break
                    _probe_before = (
                        float(_probe_param.detach().float().flatten()[0].item())
                        if _probe_param is not None and _probe_param.numel() > 0 else 0.0
                    )
                    _mull_grad_sq = 0.0
                    _mull_grad_count = 0
                    for _name, _param in self.model.named_parameters():
                        if _name.startswith("policy_scorer_mulligan.") and _param.grad is not None:
                            _gn = float(_param.grad.detach().float().norm().item())
                            _mull_grad_sq += _gn * _gn
                            _mull_grad_count += 1
                    _grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.model.max_grad_norm)
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    if not getattr(self, "_bc_grad_diag_emitted", False):
                        _probe_after = (
                            float(_probe_param.detach().float().flatten()[0].item())
                            if _probe_param is not None and _probe_param.numel() > 0 else 0.0
                        )
                        diag_file = os.getenv("BC_HARDEN_BINARY_DIAG_FILE", "").strip()
                        if diag_file:
                            try:
                                with open(diag_file, "a", encoding="utf-8") as _diag:
                                    _diag.write(
                                        "BC_DIRECT_LOSS gradDiag "
                                        f"gradNorm={float(_grad_norm):.8f} "
                                        f"mullGradNorm={(_mull_grad_sq ** 0.5):.8f} "
                                        f"mullGradParamCount={_mull_grad_count} "
                                        f"probe={_probe_name} "
                                        f"probeBefore={_probe_before:.10f} "
                                        f"probeAfter={_probe_after:.10f} "
                                        f"lrs={[float(g.get('lr', 0.0)) for g in self.optimizer.param_groups]}\n"
                                    )
                            except Exception:
                                pass
                        self._bc_grad_diag_emitted = True

                    self.train_step_counter = next_step
                    self.main_train_sample_counter += int(local_batch_size)
                    logger.info(LogCategory.MODEL_TRAIN,
                                "trainCandidatesMultiFlat BC_DIRECT_LOSS loss=%.4f rows=%d coef=%.4f batch=%d",
                                total_loss, total_rows, bc_coef, local_batch_size)
                    self.metrics.record_train_losses(
                        total_loss=total_loss,
                        policy_loss=0.0,
                        value_loss=0.0,
                        entropy=0.0,
                        entropy_coef=0.0,
                        clip_frac=0.0,
                        approx_kl=0.0,
                        batch_size=local_batch_size,
                        advantage_mean=0.0
                    )
                    self._write_training_losses_csv(episodes_in_batch=ep_count)
                    return

                # Pass 1 is only needed when we plan to backward per chunk (multi-chunk):
                # we need all values BEFORE any Pass 2 chunk starts loss computation
                # because GAE requires the full value trajectory.
                # When single-chunk (batch fits in one forward), skip Pass 1 and keep
                # the grad graph alive from a single forward that serves BOTH GAE (via
                # detach) AND the loss+backward. Saves one full forward pass.
                _multi_chunk = len(chunk_starts) > 1
                candidate_q_loss_coef = float(os.getenv("CANDIDATE_Q_LOSS_COEF", "0.0"))
                candidate_q_critical_only = bool(int(os.getenv("CANDIDATE_Q_CRITICAL_ONLY", "0")))
                candidate_q_huber_beta = float(os.getenv("CANDIDATE_Q_HUBER_BETA", "0.25"))
                candidate_q_from_mcts_targets = bool(int(os.getenv("CANDIDATE_Q_FROM_MCTS_TARGETS", "0")))
                candidate_q_mcts_signed_targets = mcts_signed_targets
                _single_pass_probs = None
                _single_pass_value = None
                _single_pass_cls = None
                _single_pass_candidate_q = None
                probs = torch.zeros(local_batch_size, max_candidates, device=device)
                value = torch.zeros(local_batch_size, 1, device=device)
                if _multi_chunk:
                    # ----------------------------------------------------------
                    # Pass 1 (no_grad, eval-mode): probs + value for GAE.
                    # eval() drops dropout so values are deterministic -- gives
                    # cleaner GAE targets than using dropout-corrupted values.
                    # ----------------------------------------------------------
                    self.model.eval()
                    try:
                        with torch.no_grad(), autocast_ctx:
                            for _c_start in chunk_starts:
                                _c_end = min(_c_start + effective_chunk, local_batch_size)
                                _head_idx_c = head_idx_t[_c_start:_c_end]
                                _seq_c = seq_t[_c_start:_c_end]
                                _mask_c = mask_t[_c_start:_c_end]
                                _tok_c = tok_t[_c_start:_c_end]
                                _cf_c = cand_feat_t[_c_start:_c_end]
                                _ci_c = cand_ids_t[_c_start:_c_end]
                                _cm_c = cand_mask_t[_c_start:_c_end]
                                for _hid_val, _hid_name in enumerate(_HEAD_NAMES):
                                    _hmask = (_head_idx_c == _hid_val)
                                    if not _hmask.any():
                                        continue
                                    _p_h, _v_h = self.model.score_candidates(
                                        _seq_c[_hmask], _mask_c[_hmask], _tok_c[_hmask],
                                        _cf_c[_hmask], _ci_c[_hmask], _cm_c[_hmask],
                                        head_id=_hid_name)
                                    _abs_idx = torch.nonzero(_hmask, as_tuple=False).view(-1) + _c_start
                                    probs[_abs_idx] = _p_h.float()
                                    value[_abs_idx] = _v_h.float()
                    finally:
                        self.model.train()

                    if torch.isnan(probs).any() or torch.isnan(value).any():
                        logger.warning(LogCategory.MODEL_TRAIN,
                                       "Model produced NaN outputs - skipping batch (probs_nan=%s, value_nan=%s)",
                                       torch.isnan(probs).any().item(), torch.isnan(value).any().item())
                        self._log_cuda_mem(
                            "trainCandidatesMultiFlat:skip_model_nan")
                        return
                else:
                    # ----------------------------------------------------------
                    # Single-pass forward (grad, train-mode): one forward serves
                    # both GAE (via .detach()) and the loss + backward below.
                    # The autograd graph stays alive for the single backward.
                    # ----------------------------------------------------------
                    with autocast_ctx:
                        _sp_probs = torch.zeros(local_batch_size, max_candidates, device=device)
                        _sp_value = torch.zeros(local_batch_size, 1, device=device)
                        _sp_cls = torch.zeros(local_batch_size, self.model.d_model, device=device)
                        _sp_candidate_q = torch.zeros(local_batch_size, max_candidates, device=device) if candidate_q_loss_coef > 0.0 else None
                        for _hid_val, _hid_name in enumerate(_HEAD_NAMES):
                            _hmask = (head_idx_t == _hid_val)
                            if not _hmask.any():
                                continue
                            if candidate_q_loss_coef > 0.0:
                                _p_h, _v_h, _cls_h, _q_h = self.model.score_candidates(
                                    seq_t[_hmask], mask_t[_hmask], tok_t[_hmask],
                                    cand_feat_t[_hmask], cand_ids_t[_hmask], cand_mask_t[_hmask],
                                    head_id=_hid_name, return_cls=True, return_candidate_q=True)
                            else:
                                _p_h, _v_h, _cls_h = self.model.score_candidates(
                                    seq_t[_hmask], mask_t[_hmask], tok_t[_hmask],
                                    cand_feat_t[_hmask], cand_ids_t[_hmask], cand_mask_t[_hmask],
                                    head_id=_hid_name, return_cls=True)
                            _sp_probs[_hmask] = _p_h.float()
                            _sp_value[_hmask] = _v_h.float()
                            _sp_cls[_hmask] = _cls_h.float()
                            if _sp_candidate_q is not None:
                                _sp_candidate_q[_hmask] = _q_h.float()
                    if torch.isnan(_sp_probs).any() or torch.isnan(_sp_value).any():
                        logger.warning(LogCategory.MODEL_TRAIN,
                                       "Model produced NaN outputs - skipping batch (probs_nan=%s, value_nan=%s)",
                                       torch.isnan(_sp_probs).any().item(), torch.isnan(_sp_value).any().item())
                        self._log_cuda_mem(
                            "trainCandidatesMultiFlat:skip_model_nan")
                        return
                    _single_pass_probs = _sp_probs
                    _single_pass_value = _sp_value
                    _single_pass_cls = _sp_cls
                    _single_pass_candidate_q = _sp_candidate_q
                    # Use detached copy for diagnostics / GAE; the live-grad copies
                    # live in _single_pass_probs / _single_pass_value for the loss.
                    probs = _sp_probs.detach()
                    value = _sp_value.detach()

                value_squeezed = value.squeeze(1)
                value_detached = value_squeezed.detach()

                if self.use_gae:
                    self.update_gae_lambda_schedule()
                    advantages, value_targets = self.compute_gae(
                        rewards_t, value_detached, gamma=self.gamma, gae_lambda=self.current_gae_lambda, dones=dones_t)
                    advantages = advantages.detach()
                    value_targets = value_targets.detach()
                else:
                    gamma = self.gamma
                    monte_carlo_returns = torch.zeros_like(rewards_t)
                    running_return = 0.0
                    for t in reversed(range(local_batch_size)):
                        if float(dones_t[t].item()) != 0.0:
                            running_return = 0.0
                        running_return = rewards_t[t] + gamma * running_return
                        monte_carlo_returns[t] = running_return
                    value_targets = monte_carlo_returns
                    advantages = monte_carlo_returns - value_detached

                diag_every = int(os.getenv("VALUE_TARGET_DIAG_EVERY", "50"))
                post_every = int(os.getenv("VALUE_POSTUPDATE_DIAG_EVERY", "0"))
                do_post = post_every > 0 and (next_step % post_every == 0)
                v_before_eval = None
                if do_post:
                    with torch.no_grad():
                        self.model.eval()
                        _v0_parts = []
                        for _hid_val, _hid_name in enumerate(_HEAD_NAMES):
                            _hmask_d = (head_idx_t == _hid_val)
                            if not _hmask_d.any():
                                continue
                            _, _v0_h = self.model.score_candidates(
                                seq_t[_hmask_d], mask_t[_hmask_d], tok_t[_hmask_d],
                                cand_feat_t[_hmask_d], cand_ids_t[_hmask_d], cand_mask_t[_hmask_d],
                                head_id=_hid_name)
                            _v0_parts.append((_hmask_d, _v0_h))
                        _v0_full = torch.zeros(local_batch_size, 1, device=device)
                        for _hmask_d, _v0_h in _v0_parts:
                            _v0_full[_hmask_d] = _v0_h.float()
                        v_before_eval = _v0_full.squeeze(1).detach().float().view(-1)
                        self.model.train()
                if diag_every > 0 and (next_step % diag_every == 0):
                    with torch.no_grad():
                        def _stats(name, t):
                            t = t.detach().float().view(-1)
                            if t.numel() == 0:
                                return f"{name}: empty"
                            pos = (t > 0).float().mean().item() * 100.0
                            neg = (t < 0).float().mean().item() * 100.0
                            zer = 100.0 - pos - neg
                            return (f"{name}: mean={t.mean().item():.3f} std={t.std().item():.3f} "
                                    f"min={t.min().item():.3f} max={t.max().item():.3f} "
                                    f"pos={pos:.1f}% neg={neg:.1f}% zero={zer:.1f}%")

                        logger.info(LogCategory.MODEL_TRAIN,
                                    "ValueDiag step=%d use_gae=%s lam=%.3f batch=%d | %s | %s | %s | %s",
                                    next_step, self.use_gae,
                                    (self.current_gae_lambda if self.use_gae else 1.0),
                                    int(batch_size),
                                    _stats("value_pred", value_squeezed),
                                    _stats("value_target", value_targets),
                                    _stats("adv", advantages),
                                    _stats("reward", rewards_t))

                if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                    logger.warning(LogCategory.MODEL_TRAIN,
                                   "NaN/Inf in advantages - skipping batch (batch_size=%d)", batch_size)
                    self._log_cuda_mem("trainCandidatesMultiFlat:skip_adv_nan")
                    return

                # Normalize advantages using RUNNING statistics (cross-batch)
                # Per-batch normalization destroys signal when outcomes are uniform
                advantages_normalized = advantages
                if self._adv_use_running and local_batch_size >= 1:
                    # Update running statistics with current batch
                    batch_mean = float(advantages.mean().item())
                    batch_var = float(advantages.var().item()
                                      ) if local_batch_size >= 2 else 1.0
                    alpha = self._adv_ema_alpha
                    # Include between-batch variance (mean shift) for proper streaming variance
                    mean_shift_sq = (batch_mean - self._adv_running_mean) ** 2
                    combined_var = batch_var + mean_shift_sq
                    self._adv_running_mean = (
                        1 - alpha) * self._adv_running_mean + alpha * batch_mean
                    self._adv_running_var = (
                        1 - alpha) * self._adv_running_var + alpha * combined_var
                    # Normalize using running stats (not per-batch)
                    running_std = max(self._adv_running_var ** 0.5, 1e-8)
                    advantages_normalized = (
                        advantages - self._adv_running_mean) / running_std
                elif local_batch_size >= 2:
                    # Fallback to per-batch if running stats disabled
                    adv_std = advantages.std()
                    if adv_std > 1e-8:
                        advantages_normalized = (
                            advantages - advantages.mean()) / adv_std

                # Clip normalized advantages to prevent gradient explosions
                adv_clip_max = float(os.getenv('ADV_CLIP_MAX', '5.0'))
                advantages_normalized = advantages_normalized.clamp(-adv_clip_max, adv_clip_max)

                # Compute loss coefficients once (shared by all chunks).
                if self.loss_schedule_enable and next_step <= self.critic_warmup_steps:
                    policy_loss_coef = float(self.policy_loss_coef_warmup)
                    value_loss_coef = float(self.value_loss_coef_warmup)
                    entropy_loss_mult = float(self.entropy_loss_mult_warmup)
                else:
                    policy_loss_coef = float(self.policy_loss_coef_main)
                    value_loss_coef = float(self.value_loss_coef_main)
                    entropy_loss_mult = float(self.entropy_loss_mult_main)
                entropy_coef = float(
                    self.get_entropy_coefficient()) * float(entropy_loss_mult)
                vf_clip = float(os.getenv("PPO_VF_CLIP", "0.2"))
                total_norm_w_sum = norm_w_t.sum().clamp_min(1e-8).detach()

                # ------------------------------------------------------------------
                # Pass 2 (grad, chunked): forward + loss + backward per chunk.
                # Per-chunk loss terms are normalized by BATCH-WIDE totals so that
                # summing chunk losses equals the original full-batch loss. Backward
                # per chunk frees activations before the next chunk's forward.
                # ------------------------------------------------------------------
                self.optimizer.zero_grad(set_to_none=True)

                total_loss = 0.0
                total_loss_policy = 0.0
                total_loss_value = 0.0
                total_loss_belief = 0.0
                total_loss_card_belief = 0.0
                total_loss_candidate_q = 0.0
                total_loss_branch_return_policy = 0.0
                total_loss_reference_policy = 0.0
                total_loss_return_contrastive = 0.0
                total_loss_value_pair_rank = 0.0
                total_loss_policy_pair_rank = 0.0
                total_loss_trajectory_pair_rank = 0.0
                total_loss_action_pair_rank = 0.0
                total_entropy_sum = 0.0
                new_logp_chunks = []
                ratio_raw_chunks = []
                mcts_debug = bool(int(os.getenv("MCTS_KL_DEBUG", "0")))
                value_pair_rank_coef = float(os.getenv("VALUE_PAIR_RANK_LOSS_COEF", "0.0"))
                value_pair_rank_margin = max(0.0, float(os.getenv("VALUE_PAIR_RANK_MARGIN", "0.20")))
                policy_pair_rank_coef = float(os.getenv("POLICY_PAIR_RANK_LOSS_COEF", "0.0"))
                policy_pair_rank_margin = max(0.0, float(os.getenv("POLICY_PAIR_RANK_MARGIN", "0.20")))
                trajectory_pair_rank_coef = float(os.getenv("TRAJECTORY_PAIR_RANK_LOSS_COEF", "0.0"))
                trajectory_pair_rank_margin = max(0.0, float(os.getenv("TRAJECTORY_PAIR_RANK_MARGIN", "0.20")))
                action_pair_rank_coef = float(os.getenv("ACTION_PAIR_RANK_LOSS_COEF", "0.0"))
                action_pair_rank_margin = max(0.0, float(os.getenv("ACTION_PAIR_RANK_MARGIN", "0.20")))
                action_pair_rank_min_gap = max(0.0, float(os.getenv("ACTION_PAIR_RANK_MIN_GAP", "0.25")))
                branch_return_policy_coef = float(os.getenv("BRANCH_RETURN_POLICY_LOSS_COEF", "0.0"))
                branch_return_policy_temp = max(
                    1e-3, float(os.getenv("BRANCH_RETURN_POLICY_TEMPERATURE", "0.50")))
                branch_return_policy_min_gap = max(
                    0.0, float(os.getenv("BRANCH_RETURN_POLICY_MIN_GAP", "0.25")))
                branch_return_policy_target_mix = max(
                    0.0, min(1.0, float(os.getenv("BRANCH_RETURN_POLICY_TARGET_MIX", "1.0"))))
                return_contrastive_coef = float(os.getenv("RETURN_CONTRASTIVE_POLICY_LOSS_COEF", "0.0"))
                return_contrastive_eps = float(os.getenv("RETURN_CONTRASTIVE_RETURN_EPS", "0.05"))
                return_contrastive_neg_prob_floor = float(os.getenv("RETURN_CONTRASTIVE_NEG_PROB_FLOOR", "0.20"))
                return_contrastive_critical_only = bool(int(os.getenv("RETURN_CONTRASTIVE_CRITICAL_ONLY", "1")))

                for _c_start in chunk_starts:
                    _c_end = min(_c_start + effective_chunk, local_batch_size)
                    _head_idx_c = head_idx_t[_c_start:_c_end]
                    _seq_c = seq_t[_c_start:_c_end]
                    _mask_c = mask_t[_c_start:_c_end]
                    _tok_c = tok_t[_c_start:_c_end]
                    _cf_c = cand_feat_t[_c_start:_c_end]
                    _ci_c = cand_ids_t[_c_start:_c_end]
                    _cm_c = cand_mask_t[_c_start:_c_end]
                    _chosen_idx_c = chosen_indices_t[_c_start:_c_end]
                    _chosen_cnt_c = chosen_count_t[_c_start:_c_end]
                    _norm_w_c = norm_w_t[_c_start:_c_end]
                    _adv_c = advantages_normalized[_c_start:_c_end]
                    _old_logp_c = old_logp_t[_c_start:_c_end]
                    _old_value_c = old_value_t[_c_start:_c_end]
                    _rewards_c = rewards_t[_c_start:_c_end]
                    _value_targets_c = value_targets[_c_start:_c_end]
                    _chunk_N = _c_end - _c_start

                    if _single_pass_probs is not None:
                        # Reuse the forward already done above (single-chunk).
                        _probs_c = _single_pass_probs
                        _value_c = _single_pass_value
                        _cls_c = _single_pass_cls
                        _candidate_q_c = _single_pass_candidate_q
                    else:
                        with autocast_ctx:
                            _probs_c = torch.zeros(_chunk_N, max_candidates, device=device)
                            _value_c = torch.zeros(_chunk_N, 1, device=device)
                            _cls_c = torch.zeros(_chunk_N, self.model.d_model, device=device)
                            _candidate_q_c = torch.zeros(_chunk_N, max_candidates, device=device) if candidate_q_loss_coef > 0.0 else None
                            for _hid_val, _hid_name in enumerate(_HEAD_NAMES):
                                _hmask = (_head_idx_c == _hid_val)
                                if not _hmask.any():
                                    continue
                                if candidate_q_loss_coef > 0.0:
                                    _p_h, _v_h, _cls_h, _q_h = self.model.score_candidates(
                                        _seq_c[_hmask], _mask_c[_hmask], _tok_c[_hmask],
                                        _cf_c[_hmask], _ci_c[_hmask], _cm_c[_hmask],
                                        head_id=_hid_name, return_cls=True, return_candidate_q=True)
                                else:
                                    _p_h, _v_h, _cls_h = self.model.score_candidates(
                                        _seq_c[_hmask], _mask_c[_hmask], _tok_c[_hmask],
                                        _cf_c[_hmask], _ci_c[_hmask], _cm_c[_hmask],
                                        head_id=_hid_name, return_cls=True)
                                _probs_c[_hmask] = _p_h.float()
                                _value_c[_hmask] = _v_h.float()
                                _cls_c[_hmask] = _cls_h.float()
                                if _candidate_q_c is not None:
                                    _candidate_q_c[_hmask] = _q_h.float()

                    _probs_safe_c = torch.clamp(_probs_c, min=1e-8, max=1.0)
                    _new_logp_c = self._joint_logp_from_probs(
                        _probs_safe_c, _cm_c, _chosen_idx_c, _chosen_cnt_c)

                    if self.use_ppo:
                        # Tighten log-ratio clamp from [-20, 20] to [-5, 5] so
                        # ratio_raw <= e^5 ~= 148x. The old [-20, 20] bound
                        # allowed ratios up to e^20 ~= 5e8, which produced
                        # policy_loss in the millions when advantage clamped to
                        # ±5 met a pathological log-ratio (2026-04-22 value head
                        # collapse). The collapsed network outputs near-constant
                        # values regardless of state, killing training signal.
                        _log_ratio_clip = float(os.getenv("PPO_LOG_RATIO_CLIP", "5.0"))
                        _log_ratio_c = (_new_logp_c - _old_logp_c).clamp(-_log_ratio_clip, _log_ratio_clip)
                        _ratio_raw_c = torch.exp(_log_ratio_c)
                        _clipped_ratio_c = torch.clamp(
                            _ratio_raw_c, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
                        _policy_obj_c = torch.min(
                            _ratio_raw_c * _adv_c, _clipped_ratio_c * _adv_c)
                        _loss_policy_c = -(_policy_obj_c * _norm_w_c).sum() / total_norm_w_sum

                        # Extra safety: if policy loss exploded despite the clamp
                        # (e.g., many samples hit the ratio cap simultaneously),
                        # skip this update entirely. The data is too stale.
                        _policy_loss_threshold = float(os.getenv("PPO_POLICY_LOSS_SKIP_THRESHOLD", "100.0"))
                        if (policy_loss_coef > 0.0
                                and torch.isfinite(_loss_policy_c)
                                and _loss_policy_c.abs().item() > _policy_loss_threshold):
                            logger.warning(LogCategory.MODEL_TRAIN,
                                "Skipping update: policy_loss=%.1f exceeds threshold %.1f (stale rollouts or pathological ratios)",
                                float(_loss_policy_c.item()), _policy_loss_threshold)
                            self.optimizer.zero_grad(set_to_none=True)
                            return
                    else:
                        _loss_policy_c = -((_new_logp_c * _adv_c) * _norm_w_c).sum() / total_norm_w_sum

                    _value_c_sq = _value_c.squeeze(1)
                    if self.use_ppo and vf_clip > 0.0:
                        _v_old_c = _old_value_c.view_as(_value_c_sq).detach()
                        _v_clipped_c = _v_old_c + (_value_c_sq - _v_old_c).clamp(-vf_clip, vf_clip)
                        _vf1_c = (_value_c_sq - _value_targets_c).pow(2)
                        _vf2_c = (_v_clipped_c - _value_targets_c).pow(2)
                        _vf_max_c = torch.max(_vf1_c, _vf2_c)
                        _loss_value_c = value_loss_coef * (0.5 * (_vf_max_c * _norm_w_c).sum() / total_norm_w_sum)
                    else:
                        _vf_c = (_value_c_sq - _value_targets_c).pow(2)
                        _loss_value_c = value_loss_coef * ((_vf_c * _norm_w_c).sum() / total_norm_w_sum)

                    # Decision-local value ranking for paired branch states.
                    # Paired episodes come from generic terminal branch
                    # outcomes, not action text or card-specific rules.
                    _loss_value_pair_rank_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if value_pair_rank_coef > 0.0 and _chunk_N >= 2:
                        _pair_left_c = torch.arange(0, _chunk_N - 1, 2, device=device)
                        if _pair_left_c.numel() > 0:
                            _pair_right_c = _pair_left_c + 1
                            _reward_gap_c = _rewards_c[_pair_left_c] - _rewards_c[_pair_right_c]
                            _value_gap_c = _value_c_sq[_pair_left_c] - _value_c_sq[_pair_right_c]
                            _valid_pair_c = (
                                torch.isfinite(_reward_gap_c)
                                & torch.isfinite(_value_gap_c)
                                & (_reward_gap_c.abs() > 1e-6)
                            )
                            if _valid_pair_c.any():
                                _pair_sign_c = torch.sign(_reward_gap_c[_valid_pair_c]).detach()
                                _pair_margin_c = value_pair_rank_margin - (
                                    _pair_sign_c * _value_gap_c[_valid_pair_c]
                                )
                                _pair_loss_c = F.relu(_pair_margin_c)
                                _pair_w_c = _reward_gap_c[_valid_pair_c].abs().detach().clamp(min=1e-6, max=2.0)
                                _loss_value_pair_rank_c = value_pair_rank_coef * (
                                    (_pair_loss_c * _pair_w_c).sum() / _pair_w_c.sum().clamp_min(1e-8)
                                )

                    _log_probs_c = torch.log(_probs_safe_c)
                    _entropy_per_step_c = -(_probs_safe_c * _log_probs_c).sum(dim=-1)
                    _loss_entropy_c = -entropy_coef * _entropy_per_step_c.sum() / float(max(local_batch_size, 1))

                    # Decision-local policy ranking for paired branch records.
                    # Adjacent records are generic terminal branch alternatives:
                    # the higher-return record's selected action should receive
                    # higher log probability than the lower-return record's
                    # selected action. This is a policy-side analogue of the
                    # value pair-rank loss and does not inspect card/action text.
                    _loss_policy_pair_rank_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if policy_pair_rank_coef > 0.0 and _chunk_N >= 2:
                        _pair_left_c = torch.arange(0, _chunk_N - 1, 2, device=device)
                        if _pair_left_c.numel() > 0:
                            _pair_right_c = _pair_left_c + 1
                            _reward_gap_c = _rewards_c[_pair_left_c] - _rewards_c[_pair_right_c]
                            _logp_gap_c = _new_logp_c[_pair_left_c] - _new_logp_c[_pair_right_c]
                            _valid_pair_c = (
                                torch.isfinite(_reward_gap_c)
                                & torch.isfinite(_logp_gap_c)
                                & (_reward_gap_c.abs() > 1e-6)
                            )
                            if _valid_pair_c.any():
                                _pair_sign_c = torch.sign(_reward_gap_c[_valid_pair_c]).detach()
                                _rank_margin_c = policy_pair_rank_margin - (
                                    _pair_sign_c * _logp_gap_c[_valid_pair_c]
                                )
                                _pair_loss_c = F.softplus(_rank_margin_c)
                                _pair_w_c = _reward_gap_c[_valid_pair_c].abs().detach().clamp(min=1e-6, max=2.0)
                                _loss_policy_pair_rank_c = policy_pair_rank_coef * (
                                    (_pair_loss_c * _pair_w_c).sum() / _pair_w_c.sum().clamp_min(1e-8)
                                )

                    # Trajectory-level policy ranking for adjacent branch
                    # episodes. The collector emits paired branch episodes in
                    # adjacent order; this compares the sum of selected-action
                    # log-probs over each whole episode using only terminal
                    # outcome gaps. It is generic preference learning: no
                    # action text, card names, or deck-specific rules.
                    _loss_trajectory_pair_rank_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if trajectory_pair_rank_coef > 0.0 and _chunk_N >= 2:
                        _done_c = dones_t[_c_start:_c_end].view(-1)
                        _done_idx_c = torch.nonzero(_done_c > 0.5, as_tuple=False).view(-1)
                        _episode_ranges_c = []
                        _ep_start_c = 0
                        for _ep_end_t in _done_idx_c.detach().cpu().tolist():
                            _ep_end = int(_ep_end_t)
                            if _ep_end >= _ep_start_c:
                                _episode_ranges_c.append((_ep_start_c, _ep_end))
                            _ep_start_c = _ep_end + 1
                        if len(_episode_ranges_c) >= 2:
                            _traj_losses_c = []
                            _traj_weights_c = []
                            for _pair_i in range(0, len(_episode_ranges_c) - 1, 2):
                                _l0, _l1 = _episode_ranges_c[_pair_i]
                                _r0, _r1 = _episode_ranges_c[_pair_i + 1]
                                _reward_gap_t = _rewards_c[_l1] - _rewards_c[_r1]
                                if (not torch.isfinite(_reward_gap_t)) or float(_reward_gap_t.detach().abs().item()) <= 1e-6:
                                    continue
                                _logp_gap_t = _new_logp_c[_l0:_l1 + 1].sum() - _new_logp_c[_r0:_r1 + 1].sum()
                                if not torch.isfinite(_logp_gap_t):
                                    continue
                                _pair_sign_t = torch.sign(_reward_gap_t).detach()
                                _traj_losses_c.append(F.softplus(
                                    trajectory_pair_rank_margin - (_pair_sign_t * _logp_gap_t)))
                                _traj_weights_c.append(_reward_gap_t.detach().abs().clamp(min=1e-6, max=2.0))
                            if _traj_losses_c:
                                _traj_loss_t = torch.stack(_traj_losses_c)
                                _traj_weight_t = torch.stack(_traj_weights_c)
                                _loss_trajectory_pair_rank_c = trajectory_pair_rank_coef * (
                                    (_traj_loss_t * _traj_weight_t).sum()
                                    / _traj_weight_t.sum().clamp_min(1e-8)
                                )

                    # Decision-local action ranking from signed branch returns.
                    # This is weaker than matching the full branch-return target
                    # distribution: it only asks the current policy to rank the
                    # best observed sibling above the worst observed sibling.
                    # The targets are generic terminal outcomes and do not use
                    # action text, card names, or deck-specific rules.
                    _loss_action_pair_rank_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if (action_pair_rank_coef > 0.0
                            and mcts_signed_targets
                            and mcts_visits_t is not None):
                        _branch_rank_c = mcts_visits_t[_c_start:_c_end].float()
                        _obs_rank_c = (
                            torch.isfinite(_branch_rank_c)
                            & (_branch_rank_c >= -1.0)
                            & (_branch_rank_c <= 1.0)
                            & _cm_c.bool()
                        )
                        _obs_count_rank_c = _obs_rank_c.sum(dim=-1)
                        _best_rank_ret_c, _best_rank_idx_c = _branch_rank_c.masked_fill(
                            ~_obs_rank_c, -1e9).max(dim=-1)
                        _worst_rank_ret_c, _worst_rank_idx_c = _branch_rank_c.masked_fill(
                            ~_obs_rank_c, 1e9).min(dim=-1)
                        _rank_gap_c = _best_rank_ret_c - _worst_rank_ret_c
                        _valid_rank_rows_c = (
                            (_obs_count_rank_c >= 2)
                            & torch.isfinite(_rank_gap_c)
                            & (_rank_gap_c >= action_pair_rank_min_gap)
                        )
                        if _valid_rank_rows_c.any():
                            _rank_rows_c = torch.arange(_chunk_N, device=device)
                            _best_lp_c = _log_probs_c[
                                _rank_rows_c, _best_rank_idx_c.clamp(0, max_candidates - 1)]
                            _worst_lp_c = _log_probs_c[
                                _rank_rows_c, _worst_rank_idx_c.clamp(0, max_candidates - 1)]
                            _logp_gap_c = _best_lp_c - _worst_lp_c
                            _rank_per_c = F.softplus(action_pair_rank_margin - _logp_gap_c)
                            _rank_w_c = _rank_gap_c.detach().clamp(min=1e-6, max=2.0)
                            _rank_denom_c = _rank_w_c[_valid_rank_rows_c].sum().clamp_min(1e-8)
                            _loss_action_pair_rank_c = action_pair_rank_coef * (
                                (_rank_per_c[_valid_rank_rows_c] * _rank_w_c[_valid_rank_rows_c]).sum()
                                / _rank_denom_c
                            )

                    # Action-conditioned terminal value. In online RL this
                    # trains Q(s,a) only for the action actually taken, using
                    # the Monte Carlo terminal return target. For terminal
                    # counterfactual imports, CANDIDATE_Q_FROM_MCTS_TARGETS=1
                    # trains the Q scorer from branch-derived candidate targets
                    # carried in the MCTS target tensor.
                    _loss_candidate_q_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if candidate_q_loss_coef > 0.0 and _candidate_q_c is not None:
                        if candidate_q_from_mcts_targets and mcts_visits_t is not None:
                            _mcts_q_c = mcts_visits_t[_c_start:_c_end].float()
                            if candidate_q_mcts_signed_targets:
                                _target_mask_c = (
                                    torch.isfinite(_mcts_q_c)
                                    & (_mcts_q_c >= -1.0)
                                    & (_mcts_q_c <= 1.0)
                                    & cand_mask_t[_c_start:_c_end].bool()
                                )
                            else:
                                _target_mask_c = (_mcts_q_c > 1e-8) & cand_mask_t[_c_start:_c_end].bool()
                            if candidate_q_critical_only:
                                _critical_c = (_cf_c[:, :, 24] > 0.5) | (_head_idx_c.view(-1, 1) == 5)
                                _target_mask_c = _target_mask_c & _critical_c
                            if _target_mask_c.any():
                                if candidate_q_mcts_signed_targets:
                                    _q_target_c = _mcts_q_c.clamp(-1.0, 1.0).detach()
                                else:
                                    _q_target_c = (2.0 * _mcts_q_c - 1.0).clamp(-1.0, 1.0).detach()
                                _q_per_c = F.smooth_l1_loss(
                                    _candidate_q_c.float(),
                                    _q_target_c.float(),
                                    beta=candidate_q_huber_beta,
                                    reduction='none')
                                _loss_candidate_q_c = candidate_q_loss_coef * _q_per_c[_target_mask_c].mean()
                        else:
                            _first_idx_c = _chosen_idx_c[:, 0].long()
                            _has_choice_c = (_chosen_cnt_c > 0) & (_first_idx_c >= 0) & (_first_idx_c < max_candidates)
                            if _has_choice_c.any():
                                _safe_idx_c = _first_idx_c.clamp(0, max_candidates - 1)
                                _row_idx_c = torch.arange(_chunk_N, device=device)
                                _selected_feat_c = _cf_c[_row_idx_c, _safe_idx_c]
                                _selected_is_spell_c = _selected_feat_c[:, 24] > 0.5
                                _selected_is_mull_c = (_head_idx_c == 5)
                                if candidate_q_critical_only:
                                    _eligible_q_c = _has_choice_c & (_selected_is_spell_c | _selected_is_mull_c)
                                else:
                                    _eligible_q_c = _has_choice_c
                                if _eligible_q_c.any():
                                    _selected_q_c = _candidate_q_c[_row_idx_c, _safe_idx_c]
                                    _q_target_c = _value_targets_c.detach().clamp(-1.0, 1.0)
                                    _q_per_c = F.smooth_l1_loss(
                                        _selected_q_c.float(),
                                        _q_target_c.float(),
                                        beta=candidate_q_huber_beta,
                                        reduction='none')
                                    _q_w_c = _norm_w_c.detach()
                                    _q_denom_c = _q_w_c[_eligible_q_c].sum().clamp_min(1e-8)
                                    _loss_candidate_q_c = candidate_q_loss_coef * (
                                        (_q_per_c[_eligible_q_c] * _q_w_c[_eligible_q_c]).sum() / _q_denom_c
                                    )

                    # Phase 1 belief auxiliary loss: classify opponent's deck archetype
                    # from the shared encoder CLS. Only samples with a known label
                    # (>= 0) contribute; padded/off-meta steps are ignored.
                    _loss_belief_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if (archetype_labels_t is not None
                            and _num_archetypes > 0
                            and self.belief_loss_coef > 0.0):
                        _arc_c = archetype_labels_t[_c_start:_c_end]
                        _valid_c = (_arc_c >= 0)
                        if _valid_c.any():
                            with autocast_ctx:
                                _belief_logits_c = self.model.belief_logits_from_cls(_cls_c)
                            _sel_logits = _belief_logits_c[_valid_c].float()
                            _sel_labels = _arc_c[_valid_c].long()
                            _loss_belief_c = self.belief_loss_coef * torch.nn.functional.cross_entropy(
                                _sel_logits, _sel_labels, reduction='mean')

                    # Generic card-level belief auxiliary loss: regress the
                    # normalized hidden hand+library count vector from public
                    # state. This is deck-pool generic; rows with -1 sentinels
                    # are skipped.
                    _loss_card_belief_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    _model_card_dim = int(getattr(self.model, "card_belief_dim", 0) or 0)
                    if (card_belief_labels_t is not None
                            and _card_belief_dim > 0
                            and _model_card_dim == _card_belief_dim
                            and self.card_belief_loss_coef > 0.0):
                        _card_labels_c = card_belief_labels_t[_c_start:_c_end]
                        _card_valid_c = torch.isfinite(_card_labels_c).all(dim=-1) & (_card_labels_c >= 0.0).all(dim=-1)
                        if _card_valid_c.any():
                            with autocast_ctx:
                                _card_logits_c = self.model.card_belief_logits_from_cls(_cls_c)
                            _card_pred_c = torch.sigmoid(_card_logits_c[_card_valid_c].float())
                            _card_target_c = _card_labels_c[_card_valid_c].float().clamp(0.0, 1.0)
                            _loss_card_belief_c = self.card_belief_loss_coef * torch.nn.functional.mse_loss(
                                _card_pred_c, _card_target_c, reduction='mean')

                    # AlphaZero policy distillation loss: KL(policy || MCTS_visits)
                    # for steps where MCTS was run (non-zero visit distribution).
                    _loss_mcts_kl_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if (mcts_visits_t is not None
                            and self.mcts_kl_loss_coef > 0.0
                            and not mcts_signed_targets):
                        _mcts_c = mcts_visits_t[_c_start:_c_end]
                        # Mask steps that have any positive visit target.
                        _mcts_row_sum = _mcts_c.sum(dim=-1)
                        _mcts_valid = (_mcts_row_sum > 1e-6)
                        anchor_source = "none"
                        if _mcts_valid.any():
                            _probs_c_valid = _probs_safe_c[_mcts_valid]
                            _targets_valid = _mcts_c[_mcts_valid]
                            # Normalize each MCTS row (in case not exactly 1.0).
                            _targets_valid = _targets_valid / _targets_valid.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                            target_policy_mix = float(os.getenv("MCTS_TARGET_POLICY_MIX", "1.0"))
                            target_policy_mix = max(0.0, min(1.0, target_policy_mix))
                            anchor_source = "current"
                            if target_policy_mix < 1.0:
                                ref_probs_c = self._score_reference_policy_chunk(
                                    _seq_c, _mask_c, _tok_c, _cf_c, _ci_c, _cm_c,
                                    _head_idx_c, max_candidates, autocast_ctx)
                                if ref_probs_c is not None:
                                    anchor = ref_probs_c[_mcts_valid]
                                    anchor_source = "reference"
                                else:
                                    anchor = _probs_c_valid.detach()
                                anchor = anchor / anchor.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                                _targets_valid = (
                                    (target_policy_mix * _targets_valid)
                                    + ((1.0 - target_policy_mix) * anchor)
                                )
                                _targets_valid = _targets_valid / _targets_valid.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                            # KL(targets || probs) = sum(targets * (log(targets) - log(probs)))
                            # We only care about the gradient on probs: sum(-targets * log(probs)) + const
                            # Use cross-entropy form which is what matters for gradient.
                            _log_probs_valid = torch.log(_probs_c_valid.clamp(min=1e-8))
                            _ce = -(_targets_valid * _log_probs_valid).sum(dim=-1)
                            if bool(int(os.getenv("MCTS_TARGET_ROW_SUM_WEIGHT_ENABLE", "0"))):
                                _row_w = _mcts_row_sum[_mcts_valid].detach().float()
                                _row_w = torch.nan_to_num(
                                    _row_w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                                _row_w_max = float(os.getenv("MCTS_TARGET_ROW_SUM_WEIGHT_MAX", "10.0"))
                                if _row_w_max > 0:
                                    _row_w = _row_w.clamp(max=_row_w_max)
                                _row_w_sum = _row_w.sum().clamp_min(1e-8)
                                _loss_mcts_kl_c = self.mcts_kl_loss_coef * ((_ce * _row_w).sum() / _row_w_sum)
                            else:
                                _loss_mcts_kl_c = self.mcts_kl_loss_coef * _ce.mean()
                        if mcts_debug and _c_start == 0:
                            try:
                                _head_counts = torch.bincount(_head_idx_c.detach().cpu(), minlength=len(_HEAD_NAMES)).tolist()
                                _valid_count = int(_mcts_valid.detach().sum().item())
                                _mean_ce = float(_ce.detach().mean().item()) if _mcts_valid.any() else 0.0
                                _msg = (
                                    "MCTS_KL_DEBUG "
                                    f"coef={float(self.mcts_kl_loss_coef):.3f} "
                                    f"target_policy_mix={float(os.getenv('MCTS_TARGET_POLICY_MIX', '1.0')):.3f} "
                                    f"anchor={anchor_source} "
                                    f"batch={int(local_batch_size)} "
                                    f"chunk={int(_chunk_N)} "
                                    f"head_counts={_head_counts} "
                                    f"valid={_valid_count} "
                                    f"mean_ce={_mean_ce:.6f} "
                                    f"loss={float(_loss_mcts_kl_c.detach().item()):.6f}"
                                )
                                print(_msg, flush=True)
                                _debug_file = os.getenv("MCTS_KL_DEBUG_FILE", "").strip()
                                if _debug_file:
                                    _rows = min(8, int(_chunk_N))
                                    _target_rows = _mcts_c[:_rows, :min(8, max_candidates)].detach().cpu().tolist()
                                    _prob_rows = _probs_safe_c[:_rows, :min(8, max_candidates)].detach().cpu().tolist()
                                    _chosen_rows = _chosen_idx_c[:_rows, :min(8, max_candidates)].detach().cpu().tolist()
                                    os.makedirs(os.path.dirname(_debug_file), exist_ok=True)
                                    with open(_debug_file, "a", encoding="utf-8") as _fh:
                                        _fh.write(_msg + "\n")
                                        _fh.write(f"targets={_target_rows}\n")
                                        _fh.write(f"probs={_prob_rows}\n")
                                        _fh.write(f"chosen={_chosen_rows}\n")
                            except Exception:
                                pass

                    # Signed terminal branch-return policy loss. Branch-return
                    # rows use mcts_visits_t as per-candidate terminal returns
                    # in [-1, 1] with -2 as unobserved. This trains the policy
                    # on generic terminal outcome preferences among branch-
                    # evaluated legal candidates without action text or
                    # strategy-specific labels.
                    _loss_branch_return_policy_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if (branch_return_policy_coef > 0.0
                            and mcts_signed_targets
                            and mcts_visits_t is not None):
                        _branch_c = mcts_visits_t[_c_start:_c_end].float()
                        _obs_c = (
                            torch.isfinite(_branch_c)
                            & (_branch_c >= -1.0)
                            & (_branch_c <= 1.0)
                            & _cm_c.bool()
                        )
                        _obs_count_c = _obs_c.sum(dim=-1)
                        _masked_returns_c = _branch_c.masked_fill(~_obs_c, -1e9)
                        _best_ret_c = _masked_returns_c.max(dim=-1).values
                        _worst_ret_c = _branch_c.masked_fill(~_obs_c, 1e9).min(dim=-1).values
                        _gap_c = _best_ret_c - _worst_ret_c
                        _valid_branch_rows_c = (_obs_count_c >= 2) & (_gap_c >= branch_return_policy_min_gap)
                        if _valid_branch_rows_c.any():
                            _scores_c = (_branch_c / branch_return_policy_temp).masked_fill(~_obs_c, -1e9)
                            _target_c = torch.softmax(_scores_c.float(), dim=-1)
                            _obs_probs_c = (_probs_safe_c * _obs_c.float()).clamp(min=1e-8)
                            _obs_probs_c = _obs_probs_c / _obs_probs_c.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                            if branch_return_policy_target_mix < 1.0:
                                _target_c = (
                                    branch_return_policy_target_mix * _target_c
                                    + (1.0 - branch_return_policy_target_mix) * _obs_probs_c.detach()
                                )
                                _target_c = _target_c / _target_c.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                            _branch_ce_c = -(_target_c * torch.log(_obs_probs_c.clamp(min=1e-8))).sum(dim=-1)
                            _row_w_c = _gap_c.detach().clamp(min=1e-6, max=2.0)
                            _denom_c = _row_w_c[_valid_branch_rows_c].sum().clamp_min(1e-8)
                            _loss_branch_return_policy_c = branch_return_policy_coef * (
                                (_branch_ce_c[_valid_branch_rows_c] * _row_w_c[_valid_branch_rows_c]).sum()
                                / _denom_c
                            )

                    # Generic frozen-reference policy anchor. This loss uses
                    # the current legal candidate set and a frozen checkpoint's
                    # policy distribution. It is independent of MCTS targets and
                    # stays thesis-clean because it does not inspect action text,
                    # card names, or Spy-specific state.
                    _loss_reference_policy_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if self.reference_policy_kl_coef > 0.0 and self.mcts_reference_model is not None:
                        ref_probs_c = self._score_reference_policy_chunk(
                            _seq_c, _mask_c, _tok_c, _cf_c, _ci_c, _cm_c,
                            _head_idx_c, max_candidates, autocast_ctx)
                        if ref_probs_c is not None:
                            _legal_c = (_cm_c > 0.5).float()
                            _ref_target_c = (ref_probs_c.float() * _legal_c).clamp(min=0.0)
                            _ref_target_c = _ref_target_c / _ref_target_c.sum(
                                dim=-1, keepdim=True).clamp(min=1e-8)
                            _cur_probs_ref_c = (_probs_safe_c * _legal_c).clamp(min=1e-8)
                            _cur_probs_ref_c = _cur_probs_ref_c / _cur_probs_ref_c.sum(
                                dim=-1, keepdim=True).clamp(min=1e-8)
                            _ref_ce_c = -(_ref_target_c.detach() * torch.log(
                                _cur_probs_ref_c.clamp(min=1e-8))).sum(dim=-1)
                            _loss_reference_policy_c = self.reference_policy_kl_coef * _ref_ce_c.mean()

                    # Terminal-only critical-action contrastive loss. Won
                    # trajectories imitate selected critical actions; lost
                    # trajectories push down high-probability selected critical
                    # actions. By default this is restricted to spell choices
                    # and mulligan decisions so ordinary land sequencing is not
                    # punished just because a game was lost later.
                    _loss_return_contrastive_c = torch.zeros((), device=device, dtype=_loss_policy_c.dtype)
                    if return_contrastive_coef > 0.0:
                        _first_idx_c = _chosen_idx_c[:, 0].long()
                        _has_choice_c = (_chosen_cnt_c > 0) & (_first_idx_c >= 0) & (_first_idx_c < max_candidates)
                        if _has_choice_c.any():
                            _safe_idx_c = _first_idx_c.clamp(0, max_candidates - 1)
                            _row_idx_c = torch.arange(_chunk_N, device=device)
                            _selected_feat_c = _cf_c[_row_idx_c, _safe_idx_c]
                            _selected_is_spell_c = _selected_feat_c[:, 24] > 0.5
                            _selected_is_mull_c = (_head_idx_c == 5)
                            if return_contrastive_critical_only:
                                _eligible_c = _has_choice_c & (_selected_is_spell_c | _selected_is_mull_c)
                            else:
                                _eligible_c = _has_choice_c
                            if _eligible_c.any():
                                _joint_prob_c = torch.exp(_new_logp_c).clamp(1e-6, 1.0 - 1e-6)
                                _ret_c = _value_targets_c.detach()
                                _pos_c = _eligible_c & (_ret_c > return_contrastive_eps)
                                _neg_c = _eligible_c & (_ret_c < -return_contrastive_eps) & (
                                    _joint_prob_c > return_contrastive_neg_prob_floor)
                                _valid_contrast_c = _pos_c | _neg_c
                                if _valid_contrast_c.any():
                                    _per_c = torch.zeros_like(_new_logp_c)
                                    _per_c[_pos_c] = -_new_logp_c[_pos_c]
                                    _per_c[_neg_c] = -torch.log1p(-_joint_prob_c[_neg_c])
                                    _ret_w_c = _ret_c.abs().clamp(max=1.0)
                                    _contrast_w_c = (_norm_w_c * _ret_w_c).detach()
                                    _denom_c = _contrast_w_c[_valid_contrast_c].sum().clamp_min(1e-8)
                                    _loss_return_contrastive_c = return_contrastive_coef * (
                                        (_per_c[_valid_contrast_c] * _contrast_w_c[_valid_contrast_c]).sum()
                                        / _denom_c
                                    )

                    _chunk_loss = (
                        (policy_loss_coef * _loss_policy_c)
                        + _loss_value_c
                        + _loss_value_pair_rank_c
                        + _loss_candidate_q_c
                        + _loss_entropy_c
                        + _loss_belief_c
                        + _loss_card_belief_c
                        + _loss_mcts_kl_c
                        + _loss_policy_pair_rank_c
                        + _loss_trajectory_pair_rank_c
                        + _loss_action_pair_rank_c
                        + _loss_branch_return_policy_c
                        + _loss_reference_policy_c
                        + _loss_return_contrastive_c
                    )

                    if torch.isnan(_chunk_loss) or torch.isinf(_chunk_loss):
                        logger.warning(LogCategory.MODEL_TRAIN,
                                       "Skipping update due to NaN/Inf loss in chunk [%d:%d] (policy=%.4f value=%.4f)",
                                       _c_start, _c_end,
                                       _loss_policy_c.item() if not torch.isnan(_loss_policy_c) else float('nan'),
                                       _loss_value_c.item() if not torch.isnan(_loss_value_c) else float('nan'))
                        self.optimizer.zero_grad(set_to_none=True)
                        self._log_cuda_mem("trainCandidatesMultiFlat:skip_loss_nan")
                        return

                    if scaler is not None:
                        scaler.scale(_chunk_loss).backward()
                    else:
                        _chunk_loss.backward()

                    total_loss += _chunk_loss.item()
                    total_loss_policy += _loss_policy_c.item()
                    total_loss_value += _loss_value_c.item()
                    try:
                        total_loss_belief += float(_loss_belief_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_card_belief += float(_loss_card_belief_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_candidate_q += float(_loss_candidate_q_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_branch_return_policy += float(_loss_branch_return_policy_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_reference_policy += float(_loss_reference_policy_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_return_contrastive += float(_loss_return_contrastive_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_value_pair_rank += float(_loss_value_pair_rank_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_policy_pair_rank += float(_loss_policy_pair_rank_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_trajectory_pair_rank += float(_loss_trajectory_pair_rank_c.item())
                    except Exception:
                        pass
                    try:
                        total_loss_action_pair_rank += float(_loss_action_pair_rank_c.item())
                    except Exception:
                        pass
                    total_entropy_sum += _entropy_per_step_c.detach().sum().item()
                    new_logp_chunks.append(_new_logp_c.detach())
                    if self.use_ppo:
                        ratio_raw_chunks.append(_ratio_raw_c.detach())

                # Reassemble batch-wide tensors for post-backward logging.
                new_logp = torch.cat(new_logp_chunks) if new_logp_chunks else torch.zeros(local_batch_size, device=device)
                if self.use_ppo:
                    ratio_raw = torch.cat(ratio_raw_chunks) if ratio_raw_chunks else torch.ones(local_batch_size, device=device)

                    if self._ppo_stats_every > 0 and (next_step % self._ppo_stats_every == 0):
                        with torch.no_grad():
                            adv_t = advantages.detach().float().view(-1)
                            ret_t = value_targets.detach().float().view(-1)
                            rr_t = ratio_raw.detach().float().view(-1)
                            adv_mean = float(adv_t.mean().item()) if adv_t.numel() > 0 else 0.0
                            adv_std = float(adv_t.std().item()) if adv_t.numel() > 1 else 0.0
                            ret_mean = float(ret_t.mean().item()) if ret_t.numel() > 0 else 0.0
                            ret_std = float(ret_t.std().item()) if ret_t.numel() > 1 else 0.0
                            ratio_mean = float(rr_t.mean().item()) if rr_t.numel() > 0 else 1.0
                            ratio_std = float(rr_t.std().item()) if rr_t.numel() > 1 else 0.0
                        logger.info(
                            LogCategory.MODEL_TRAIN,
                            "PPOStats step=%d adv(mean=%.4f std=%.4f) ret(mean=%.4f std=%.4f) ratio(mean=%.4f std=%.4f)",
                            int(next_step),
                            adv_mean, adv_std,
                            ret_mean, ret_std,
                            ratio_mean, ratio_std
                        )

                # Wrap aggregates as 0-d tensors so downstream `.item()` /
                # torch.isnan() calls continue to work unchanged.
                loss = torch.tensor(total_loss, device=device)
                loss_policy = torch.tensor(total_loss_policy, device=device)
                loss_value = torch.tensor(total_loss_value, device=device)
                entropy = torch.tensor(total_entropy_sum / float(max(local_batch_size, 1)), device=device)

                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break
                if has_nan_grad:
                    logger.warning(LogCategory.MODEL_TRAIN,
                                   "Skipping update due to NaN/Inf gradients")
                    self.optimizer.zero_grad()
                    self._log_cuda_mem(
                        "trainCandidatesMultiFlat:skip_grad_nan")
                    return

                if scaler is not None:
                    try:
                        scaler.unscale_(self.optimizer)
                    except Exception:
                        pass
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.model.max_grad_norm)
                if scaler is not None:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
            finally:
                # GPU lock released by Java learner loop after entire burst
                pass

            if self.use_ppo:
                with torch.no_grad():
                    approx_kl = (old_logp_t - new_logp).mean()
                    clip_frac = ((ratio_raw < 1.0 - self.ppo_epsilon) |
                                 (ratio_raw > 1.0 + self.ppo_epsilon)).float().mean()
                logger.info(LogCategory.MODEL_TRAIN,
                            "trainCandidatesMultiFlat — loss=%.4f policy=%.4f value=%.4f valueRank=%.4f policyPair=%.4f trajPair=%.4f actionPair=%.4f candQ=%.4f branchRetPolicy=%.4f ent=%.4f belief=%.4f cardBelief=%.4f refPolicy=%.4f (coeff: %.4f) [PPO clip: %.2f%% kl=%.6f]",
                            loss.item(), loss_policy.item(), loss_value.item(), total_loss_value_pair_rank, total_loss_policy_pair_rank, total_loss_trajectory_pair_rank, total_loss_action_pair_rank, total_loss_candidate_q, total_loss_branch_return_policy, entropy.item(), total_loss_belief, total_loss_card_belief, total_loss_reference_policy,
                            entropy_coef, clip_frac.item() * 100, approx_kl.item())
                # Record loss components for metrics export
                self.metrics.record_train_losses(
                    total_loss=loss.item(),
                    policy_loss=loss_policy.item(),
                    value_loss=loss_value.item(),
                    entropy=entropy.item(),
                    entropy_coef=entropy_coef,
                    clip_frac=clip_frac.item(),
                    approx_kl=approx_kl.item(),
                    batch_size=local_batch_size,
                    advantage_mean=float(advantages_normalized.mean().item())
                )
                # Write to CSV for post-hoc analysis
                self._write_training_losses_csv(episodes_in_batch=ep_count)
            else:
                logger.info(LogCategory.MODEL_TRAIN,
                            "trainCandidatesMultiFlat — loss=%.4f policy=%.4f value=%.4f valueRank=%.4f policyPair=%.4f trajPair=%.4f actionPair=%.4f candQ=%.4f branchRetPolicy=%.4f ent=%.4f belief=%.4f cardBelief=%.4f refPolicy=%.4f (coeff: %.4f)",
                            loss.item(), loss_policy.item(), loss_value.item(), total_loss_value_pair_rank, total_loss_policy_pair_rank, total_loss_trajectory_pair_rank, total_loss_action_pair_rank, total_loss_candidate_q, total_loss_branch_return_policy, entropy.item(), total_loss_belief, total_loss_card_belief, total_loss_reference_policy, float(self.get_entropy_coefficient()) * float(entropy_loss_mult))
                # Record loss components for metrics export
                self.metrics.record_train_losses(
                    total_loss=loss.item(),
                    policy_loss=loss_policy.item(),
                    value_loss=loss_value.item(),
                    entropy=entropy.item(),
                    entropy_coef=entropy_coef,
                    clip_frac=0.0,
                    approx_kl=0.0,
                    batch_size=local_batch_size,
                    advantage_mean=float(advantages_normalized.mean().item())
                )
                # Write to CSV for post-hoc analysis
                self._write_training_losses_csv(episodes_in_batch=ep_count)

            if do_post:
                with torch.no_grad():
                    self.model.eval()
                    _v1_parts = []
                    for _hid_val, _hid_name in enumerate(_HEAD_NAMES):
                        _hmask_d = (head_idx_t == _hid_val)
                        if not _hmask_d.any():
                            continue
                        _, _v1_h = self.model.score_candidates(
                            seq_t[_hmask_d], mask_t[_hmask_d], tok_t[_hmask_d],
                            cand_feat_t[_hmask_d], cand_ids_t[_hmask_d], cand_mask_t[_hmask_d],
                            head_id=_hid_name)
                        _v1_parts.append((_hmask_d, _v1_h))
                    _v1_full = torch.zeros(local_batch_size, 1, device=device)
                    for _hmask_d, _v1_h in _v1_parts:
                        _v1_full[_hmask_d] = _v1_h.float()
                    v_after_eval = _v1_full.squeeze(1).detach().float().view(-1)
                    self.model.train()

                    v_before = v_before_eval if v_before_eval is not None else value_squeezed.detach().float().view(-1)
                    v_after = v_after_eval
                    v_tgt = value_targets.detach().float().view(-1)

                    mse_before = float(F.mse_loss(
                        v_before, v_tgt).item()) if v_before.numel() > 0 else 0.0
                    mse_after = float(F.mse_loss(
                        v_after, v_tgt).item()) if v_after.numel() > 0 else 0.0

                    def _s(t):
                        if t.numel() == 0:
                            return "empty"
                        pos = (t > 0).float().mean().item() * 100.0
                        return f"mean={t.mean().item():.3f} min={t.min().item():.3f} max={t.max().item():.3f} pos={pos:.1f}%"

                    logger.info(
                        LogCategory.MODEL_TRAIN,
                        "ValuePost step=%d | pred_before(%s) -> pred_after(%s) | target(%s) | mse_before=%.4f mse_after=%.4f | coefs(policy=%.2f value=%.2f entMult=%.2f)",
                        int(next_step),
                        _s(v_before),
                        _s(v_after),
                        _s(v_tgt),
                        mse_before,
                        mse_after,
                        float(policy_loss_coef),
                        float(value_loss_coef),
                        float(entropy_loss_mult),
                    )

            prev_step = int(self.train_step_counter)
            self.train_step_counter += ep_count
            self.main_train_sample_counter += local_batch_size

            # Explicit cleanup of training tensors
            del seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t
            del chosen_indices_t, chosen_count_t, value_targets, advantages, rewards_t, old_logp_t, old_value_t, sample_w_t, norm_w_t, dones_t
            del loss, loss_policy, loss_value, entropy
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            if torch.cuda.is_available() and bool(int(os.getenv("CUDA_EMPTY_CACHE_AFTER_TRAIN", "0"))):
                torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Save once per 100 episodes (avoid missing the boundary when ep_count > 1)
            if (self.train_step_counter // 100) > (prev_step // 100):
                if self.model_path:
                    ckpt_path = self.model_path
                else:
                    ckpt_path = f'{profile_models_dir()}/model.pt'
                try:
                    self.saveModel(ckpt_path)
                    logger.info(LogCategory.MODEL_SAVE,
                                "Checkpoint saved at step %d -> %s",
                                self.train_step_counter, ckpt_path)
                except Exception as e:
                    logger.error(LogCategory.MODEL_SAVE,
                                 "Failed to save checkpoint at step %d: %s",
                                 self.train_step_counter, str(e))

            self._maybe_save_snapshot()

            # Periodic CUDA cache flush to prevent memory fragmentation
            # Increased frequency to combat memory accumulation from model reloads
            if self.train_step_counter % 100 == 0 and torch.cuda.is_available():
                import gc
                gc.collect()  # Force Python GC to release tensor references
                torch.cuda.empty_cache()
                logger.debug(LogCategory.GPU_MEMORY,
                             "CUDA cache flushed + GC at step %d", self.train_step_counter)

            return True

        def _episode_boundaries():
            d = np.frombuffer(dones_bytes, dtype='<i4').reshape(batch_size)
            ends = np.where(d != 0)[0].tolist()
            if not ends or ends[-1] != (batch_size - 1):
                ends.append(batch_size - 1)
            spans = []
            s = 0
            for e in ends:
                spans.append((s, int(e) + 1))
                s = int(e) + 1
            # Clean up temporary arrays
            del d, ends
            return spans

        def _estimated_train_extra_mb(steps: int):
            per_step = self._train_mb_per_step
            if per_step is None or float(per_step) <= 0.0:
                try:
                    per_step = float(os.getenv("AUTO_TRAIN_MB_PER_STEP_INIT", "2.0"))
                except Exception:
                    per_step = 2.0
            return max(0.0, float(per_step) * float(max(1, int(steps))))

        def _train_spans(spans):
            if not spans:
                return True
            # Optional proactive cap by episodes (if set or learned).
            cap_eps = self._train_safe_max_episodes if (
                self.auto_batch_enable and self._train_safe_max_episodes) else None
            if cap_eps is not None and len(spans) > int(cap_eps):
                try:
                    self._autobatch_counts["train_splits_cap"] += 1
                except Exception:
                    pass
                out = True
                i = 0
                while i < len(spans):
                    j = min(len(spans), i + int(cap_eps))
                    out = _train_spans(spans[i:j]) and out
                    i = j
                return out

            start = spans[0][0]
            end = spans[-1][1]
            steps = int(end - start)
            estimated_extra_mb = _estimated_train_extra_mb(steps)
            # Fixed microbatching (episode-aligned) to reduce peak activations / avoid paging cliffs.
            max_steps_cap = int(os.getenv("TRAIN_MULTI_MAX_STEPS", "512"))
            if max_steps_cap > 0 and steps > int(max_steps_cap) and len(spans) > 1:
                groups = []
                cur = []
                cur_steps = 0
                for (s, e) in spans:
                    st = int(e - s)
                    if cur and (cur_steps + st) > int(max_steps_cap):
                        groups.append(cur)
                        cur = [(s, e)]
                        cur_steps = st
                    else:
                        cur.append((s, e))
                        cur_steps += st
                if cur:
                    groups.append(cur)
                out = True
                for g in groups:
                    out = _train_spans(g) and out
                return out
            # Hard VRAM guard: never start a training allocation that is expected to
            # eat our configured headroom. Multi-episode batches split; single
            # episodes wait for headroom instead of falling into WDDM shared RAM.
            if self.auto_batch_enable and torch.cuda.is_available():
                has_room = self._train_has_vram_headroom(
                    estimated_extra_mb,
                    wait=False,
                    tag=f"train:{steps}steps:{len(spans)}episodes",
                )
                if not has_room:
                    if len(spans) > 1:
                        try:
                            self._autobatch_counts["train_splits_paging"] += 1
                        except Exception:
                            pass
                        mid = len(spans) // 2
                        ok0 = _train_spans(spans[:mid])
                        ok1 = _train_spans(spans[mid:])
                        return ok0 and ok1
                    if not self._train_has_vram_headroom(
                        estimated_extra_mb,
                        wait=True,
                        tag=f"train:{steps}steps:single_episode",
                    ):
                        raise RuntimeError(
                            "VRAM_GUARD_NO_HEADROOM "
                            f"steps={steps} estimated_extra_mb={estimated_extra_mb:.1f} "
                            f"free_mb={self._autobatch_last_free_mb:.1f} "
                            f"need_mb={self.cuda_mgr._train_guard_last_need_mb:.1f}"
                        )
            # Proactive paging avoidance: estimate extra VRAM for this slice and split episodes if needed.
            if self.auto_batch_enable and self.auto_avoid_paging and torch.cuda.is_available() and len(spans) > 1:
                if self._should_split_for_paging(estimated_extra_mb):
                    try:
                        self._autobatch_counts["train_splits_paging"] += 1
                    except Exception:
                        pass
                    mid = len(spans) // 2
                    ok0 = _train_spans(spans[:mid])
                    ok1 = _train_spans(spans[mid:])
                    return ok0 and ok1
            try:
                out, extra_mb = self._measure_peak_extra_mb(
                    lambda: _train_slice(start, end))
                # Update per-step estimate from measured peak delta.
                self._update_mem_ema("train", extra_mb, max(1, steps))
                return out
            except RuntimeError as e:
                if self.auto_batch_enable and self._is_cuda_oom(e):
                    self._cuda_cleanup_after_oom()
                    if len(spans) <= 1:
                        # If even a single-episode chunk OOMs, we can't recover here.
                        raise
                    try:
                        self._autobatch_counts["train_splits_oom"] += 1
                    except Exception:
                        pass
                    new_cap = max(1, len(spans) // 2)
                    if self._train_safe_max_episodes is None or new_cap < int(self._train_safe_max_episodes):
                        self._train_safe_max_episodes = int(new_cap)
                        logger.warning(
                            LogCategory.GPU_BATCH, "AutoBatch(train): OOM -> shrinking train episode cap to %d", int(self._train_safe_max_episodes))
                    mid = len(spans) // 2
                    ok0 = _train_spans(spans[:mid])
                    ok1 = _train_spans(spans[mid:])
                    return ok0 and ok1
                raise

        try:
            spans = _episode_boundaries()
            result = _train_spans(spans)

            # Track timing
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.metrics.update_timing_metric("train", elapsed_ms)
            self._log_cuda_mem("trainCandidatesMultiFlat:end")

            return result
        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in trainCandidatesMultiFlat: %s", str(e))
            raise
        finally:
            if lock_held:
                try:
                    self._gpu_mutex.release()
                except Exception:
                    pass

    # Persistence methods - delegate to persistence but keep model logic here
    @property
    def model_path(self):
        return self.persistence.model_path

    @property
    def model_latest_path(self):
        return self.persistence.model_latest_path

    @property
    def _did_initial_load(self):
        return self.persistence._did_initial_load

    @_did_initial_load.setter
    def _did_initial_load(self, value):
        self.persistence._did_initial_load = value

    def saveModel(self, path):
        # Pack optimizer state and training counters for persistence
        extra = {
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_step_counter': self.train_step_counter,
            'main_train_sample_counter': self.main_train_sample_counter,
            'gae_enabled_step': self.metrics.gae_enabled_step,
        }
        self.persistence.save_model(self.model, path, extra_state=extra)

    def _heal_collapsed_scales(self):
        """Detect and reset scale / output-layer params that compress the value
        head output. Two failure modes handled:
          1. self_attn.scale ~0 -> uniform attention, bag-of-words encoder
          2. value_scale << 1.0 + critic_proj2 shrunk -> compressed value output,
             e.g., post-PPO-ratio-explosion where policy_loss in the millions
             destabilized the value head into producing |output| ~0.05.
        Returns list of reset param names.
        """
        if self.model is None:
            return []
        reset = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name.endswith('self_attn.scale') and param.numel() == 1:
                    if abs(float(param.item())) < 0.1:
                        param.data.fill_(1.0)
                        reset.append(name)

            # Opt-in value-head unstick: triggered by env var because the
            # heuristic for "stuck" can fire on healthy but still-learning
            # models. Only run on explicit user request.
            if os.getenv("VALUE_HEAD_UNSTICK", "0") == "1":
                for name, param in self.model.named_parameters():
                    # value_scale: scalar multiplier on value-head output.
                    # Pre-collapse sits near 1.0; shrunk value_scale mechanically
                    # suppresses separation between win/loss predictions.
                    if name == "value_scale" and param.numel() == 1:
                        current = float(param.item())
                        if current < 0.8:
                            param.data.fill_(1.0)
                            reset.append(f"{name} ({current:.3f}->1.0)")
                    # critic_proj2.weight: final linear to scalar value.
                    # If std is unusually small after collapse, amplify 3x to
                    # restore output magnitude without destroying learned
                    # direction (sign structure preserved).
                    if name == "critic_proj2.weight":
                        w_std = float(param.std().item())
                        if w_std < 0.15:
                            factor = 3.0
                            param.data.mul_(factor)
                            reset.append(f"{name} (scaled {factor}x, was std={w_std:.3f})")
                    # critic_proj2.bias: zero it so bias doesn't offset the
                    # amplified weight output.
                    if name == "critic_proj2.bias":
                        param.data.zero_()
                        reset.append(f"{name} (zeroed)")

        if reset:
            logger.warning(
                LogCategory.MODEL_LOAD,
                "Auto-healed %d params: %s",
                len(reset), reset,
            )
        return reset

    def loadModel(self, path):
        if self.model is None:
            self.initializeModel()
        extra = self.persistence.load_model(self.model, path)

        # Auto-heal collapsed scales before we rebuild optimizer state from the
        # saved dict. Adam's momentum for those params is driving them back to
        # zero, so we also discard the saved optimizer state when we heal.
        healed = self._heal_collapsed_scales()
        skipped_incompatible = extra.get('_skipped_incompatible_state_keys') if extra else None
        self._record_model_load_determinism_gate(path, extra, skipped_incompatible, healed)
        if self.fail_on_skipped_incompatible and skipped_incompatible:
            sample = ", ".join(str(x) for x in list(skipped_incompatible)[:12])
            raise RuntimeError(
                "SKIPPED_INCOMPATIBLE_CHECKPOINT_TENSORS "
                f"count={len(skipped_incompatible)} sample=[{sample}]"
            )
        if healed and extra and extra.get('optimizer_state_dict') is not None:
            extra = dict(extra)
            extra['optimizer_state_dict'] = None
            logger.warning(
                LogCategory.MODEL_LOAD,
                "Cleared saved optimizer_state_dict because scale collapse was healed; "
                "momentum from old (broken) gradients would otherwise re-collapse the scale.",
            )
        if skipped_incompatible and extra and extra.get('optimizer_state_dict') is not None:
            extra = dict(extra)
            extra['optimizer_state_dict'] = None
            logger.warning(
                LogCategory.MODEL_LOAD,
                "Cleared saved optimizer_state_dict because %d checkpoint tensors were skipped "
                "for shape incompatibility.",
                len(skipped_incompatible),
            )

        # Restore optimizer state and training counters if present. Repair and
        # distillation passes can intentionally reset Adam momentum/LR while
        # keeping model weights by setting LOAD_OPTIMIZER_STATE=0.
        load_optimizer_state = bool(int(os.getenv("LOAD_OPTIMIZER_STATE", "1")))
        if extra and self.optimizer:
            if load_optimizer_state and 'optimizer_state_dict' in extra and extra['optimizer_state_dict'] is not None:
                try:
                    self.optimizer.load_state_dict(extra['optimizer_state_dict'])
                    logger.info(LogCategory.MODEL_LOAD, "Restored optimizer state")
                except Exception as e:
                    logger.warning(LogCategory.MODEL_LOAD, "Could not restore optimizer state: %s", e)
            elif not load_optimizer_state and 'optimizer_state_dict' in extra and extra['optimizer_state_dict'] is not None:
                logger.info(LogCategory.MODEL_LOAD, "Skipped optimizer state restore because LOAD_OPTIMIZER_STATE=0")
            
            reset_training_state = bool(int(os.getenv("RESET_TRAINING_STATE_ON_LOAD", "0")))
            if reset_training_state:
                self.train_step_counter = 0
                self.main_train_sample_counter = 0
                self.metrics.gae_enabled_step = 0 if self.metrics.use_gae else None
                logger.info(
                    LogCategory.MODEL_LOAD,
                    "Reset training counters on load: train_step_counter=0 main_train_sample_counter=0 gae_enabled_step=%s",
                    str(self.metrics.gae_enabled_step),
                )
            else:
                if 'train_step_counter' in extra:
                    self.train_step_counter = int(extra['train_step_counter'])
                    logger.info(LogCategory.MODEL_LOAD, "Restored train_step_counter: %d", self.train_step_counter)

                if 'main_train_sample_counter' in extra:
                    self.main_train_sample_counter = int(extra['main_train_sample_counter'])
                    logger.info(LogCategory.MODEL_LOAD, "Restored main_train_sample_counter: %d", self.main_train_sample_counter)

                if 'gae_enabled_step' in extra:
                    self.metrics.gae_enabled_step = extra['gae_enabled_step']
                    if self.metrics.gae_enabled_step is not None:
                        logger.info(LogCategory.MODEL_LOAD, "Restored gae_enabled_step: %d", self.metrics.gae_enabled_step)
            
            # Log summary of restored training state
            entropy_coef = self.get_entropy_coefficient()
            logger.info(LogCategory.MODEL_LOAD, 
                       "Training state restored: train_steps=%d, entropy_coef=%.4f, samples=%d",
                       self.train_step_counter, entropy_coef, self.main_train_sample_counter)

    def saveLatestModelAtomic(self, path=None):
        # Include the same extra state as saveModel so the entropy-decay
        # counter and optimizer momentum survive trainer restarts. Without
        # this, every restart rewinds train_step_counter to the last full
        # save (often 0), and entropy never decays across restarts.
        extra = {
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_step_counter': self.train_step_counter,
            'main_train_sample_counter': self.main_train_sample_counter,
            'gae_enabled_step': self.metrics.gae_enabled_step,
        }
        return self.persistence.save_latest_model_atomic(self.model, path, extra_state=extra)

    def reloadLatestModelIfNewer(self, path=None):
        if self.model is None:
            self.initializeModel()
        return self.persistence.reload_latest_model_if_newer(self.model, path)

    # GPU lock control for Java learner loop
    def acquireGPULock(self):
        """Acquire GPU lock (called by Java before training burst)."""
        self.gpu_lock.acquire(timeout=None, process_name=self.process_name)

    def releaseGPULock(self):
        """Release GPU lock (called by Java after training burst)."""
        self.gpu_lock.release(process_name=self.process_name)

    def trainMulligan(self, features_bytes, decisions_bytes, outcomes_bytes, game_lengths_bytes, early_land_scores_bytes, overrides_bytes, batch_size):
        """
        Train the legacy standalone mulligan model. By default this keeps the old
        survival + land-drop shaping path; MULLIGAN_TERMINAL_ONLY=1 uses only the
        terminal game outcome target.

        Args:
            features_bytes: Raw bytes of mulligan features [batch_size * 71 * 4]
                Format: [mulligan_num(1), land_count(1), creature_count(1), avg_cmc(1), hand_ids(7), deck_ids(60)]
            decisions_bytes: Raw bytes of decisions (0=mulligan, 1=keep) [batch_size * 4]  
            outcomes_bytes: Raw bytes of game outcomes (1=win, 0=loss) [batch_size * 4]
            game_lengths_bytes: Raw bytes of game lengths in turns [batch_size * 4] for survival reward
            early_land_scores_bytes: Raw bytes of early land scores [batch_size * 4] (0-1 on-curve fraction, -1 if no data)
            overrides_bytes: Raw bytes of override flags [batch_size * 4] (1.0=overridden, 0.0=not overridden)
            batch_size: Number of decisions in this game

        Reward shaping:
        - Wins always get target=1.0
        - Losses scaled by: survival_alpha * survival + land_drop_alpha * on_curve_score
        - Short losses with missed land drops = strong negative signal
        - Long losses with good land drops = hand was acceptable, lost for other reasons
        - Overridden decisions (0-land keeps, all-land keeps) get forced target=0.0 regardless of game outcome
        """
        t_start = time.perf_counter()
        # Use lock to prevent concurrent training (avoids gradient conflicts)
        with self.mulligan_lock:
            lock_held = False
            if self.backend_mode == "single" and str(self.mulligan_device).startswith("cuda"):
                self._gpu_mutex.acquire()
                lock_held = True
            try:
                # --------------------------
                # Parse and enqueue to replay
                # --------------------------
                bs = int(batch_size)
                if bs <= 0:
                    mulligan_logger.info(
                        LogCategory.MODEL_TRAIN,
                        "Mulligan replay skipped - reason=empty_batch bs=%d", int(
                            bs)
                    )
                    return True

                features_np = np.frombuffer(
                    features_bytes, dtype='<f4').reshape(bs, 71)
                decisions_np = np.frombuffer(
                    decisions_bytes, dtype='<f4').reshape(bs)
                outcomes_np = np.frombuffer(
                    outcomes_bytes, dtype='<f4').reshape(bs)
                game_lengths_np = np.frombuffer(
                    game_lengths_bytes, dtype='<i4').reshape(bs).astype(np.float32)
                early_land_np = np.frombuffer(
                    early_land_scores_bytes, dtype='<f4').reshape(bs)
                overrides_np = np.frombuffer(
                    overrides_bytes, dtype='<f4').reshape(bs)

                stored = 0
                dropped_nonfinite = 0
                for i in range(bs):
                    f = np.array(features_np[i], dtype=np.float32, copy=True)
                    d = float(decisions_np[i])
                    o = float(outcomes_np[i])
                    gl = float(game_lengths_np[i])
                    els = float(early_land_np[i])
                    ovr = float(overrides_np[i])
                    if not np.isfinite(f).all() or not np.isfinite([d, o, gl, els, ovr]).all():
                        dropped_nonfinite += 1
                        continue
                    is_keep = bool(d > 0.5)
                    was_overridden = bool(ovr > 0.5)
                    # Store tuple: (features, action_keep(1/0), outcome, game_length, mulligan_num, early_land_score, was_overridden)
                    action_keep = 1 if is_keep else 0
                    mulligan_num = float(f[0]) if f.shape[0] > 0 else 0.0
                    sample = (f, action_keep, o, gl, mulligan_num, els, was_overridden)
                    if is_keep:
                        self._mull_replay_keep.append(sample)
                        self._mull_action_hist.append(1)
                    else:
                        self._mull_replay_mull.append(sample)
                        self._mull_action_hist.append(0)
                    stored += 1

                replay_keep = int(len(self._mull_replay_keep))
                replay_mull = int(len(self._mull_replay_mull))
                replay_total = replay_keep + replay_mull

                # Replay-based stats for degenerate gating (windowed over recent samples)
                hist_n = int(len(self._mull_action_hist))
                hist_keep_rate = float(
                    sum(self._mull_action_hist)) / float(hist_n) if hist_n > 0 else 0.0
                hist_all_same = bool(
                    hist_keep_rate <= 0.0 or hist_keep_rate >= 1.0) if hist_n > 0 else True

                # Decide whether we will actually do an optimizer step
                mb = int(max(1, self._mull_replay_batch_size))
                min_samples = int(max(1, self._mull_replay_min_samples))

                if replay_total < min_samples or replay_total < mb:
                    mulligan_logger.info(
                        LogCategory.MODEL_TRAIN,
                        "Mulligan replay skipped - reason=replay_not_ready stored=%d dropped_nonfinite=%d replay_total=%d keep=%d mull=%d min=%d mb=%d hist_keep_rate=%.3f",
                        int(stored),
                        int(dropped_nonfinite),
                        int(replay_total),
                        int(replay_keep),
                        int(replay_mull),
                        int(min_samples),
                        int(mb),
                        float(hist_keep_rate),
                    )
                    return True

                # Warmup gating moved to replay stats (not per-call batch)
                warmup_steps = int(
                    os.getenv("MULLIGAN_TRAIN_WARMUP_STEPS", "50"))
                min_keep_rate = float(
                    os.getenv("MULLIGAN_TRAIN_MIN_KEEP_RATE", "0.10"))
                max_keep_rate = float(
                    os.getenv("MULLIGAN_TRAIN_MAX_KEEP_RATE", "0.90"))
                do_gate = (self.mulligan_train_step_counter < warmup_steps) and (
                    hist_n >= min(50, self._mull_replay_stats_window))
                if do_gate and (hist_all_same or hist_keep_rate < min_keep_rate or hist_keep_rate > max_keep_rate):
                    if hist_all_same:
                        reason = "replay_warmup_gate_all_same"
                    elif hist_keep_rate < min_keep_rate:
                        reason = "replay_warmup_gate_keep_rate_low"
                    else:
                        reason = "replay_warmup_gate_keep_rate_high"
                    mulligan_logger.info(
                        LogCategory.MODEL_TRAIN,
                        "Mulligan replay skipped - reason=%s replay_total=%d keep=%d mull=%d hist_keep_rate=%.3f hist_n=%d step=%d/%d",
                        str(reason),
                        int(replay_total),
                        int(replay_keep),
                        int(replay_mull),
                        float(hist_keep_rate),
                        int(hist_n),
                        int(self.mulligan_train_step_counter),
                        int(warmup_steps),
                    )
                    return True

                # --------------------------
                # Sample minibatch (stratified)
                # --------------------------
                n_keep = 0
                n_mull = 0
                if self._mull_replay_stratified:
                    half = mb // 2
                    n_keep = min(replay_keep, half)
                    n_mull = min(replay_mull, mb - n_keep)
                    # Try to keep it balanced when possible
                    if n_keep < half and replay_mull > n_mull:
                        extra = min(replay_mull - n_mull, half - n_keep)
                        n_mull += extra
                    if n_mull < half and replay_keep > n_keep:
                        extra = min(replay_keep - n_keep, half - n_mull)
                        n_keep += extra
                    # Fill remainder from whichever has more left
                    rem = mb - (n_keep + n_mull)
                    if rem > 0:
                        if (replay_keep - n_keep) >= (replay_mull - n_mull):
                            n_keep += min(rem, replay_keep - n_keep)
                        else:
                            n_mull += min(rem, replay_mull - n_mull)
                else:
                    # Unstratified: just sample from combined pool later
                    n_keep = min(replay_keep, mb)
                    n_mull = min(replay_mull, mb - n_keep)

                keep_samples = []
                mull_samples = []
                if n_keep > 0:
                    idx = self._mull_rng.choice(
                        replay_keep, size=int(n_keep), replace=False)
                    buf = list(self._mull_replay_keep)
                    keep_samples = [buf[int(i)] for i in idx]
                if n_mull > 0:
                    idx = self._mull_rng.choice(
                        replay_mull, size=int(n_mull), replace=False)
                    buf = list(self._mull_replay_mull)
                    mull_samples = [buf[int(i)] for i in idx]
                # Oversample minority class (with replacement) to avoid "always keep" drift when mull is rare
                if self._mull_replay_stratified and self._mull_replay_oversample_minority and mb > 1:
                    target_half = mb // 2
                    if replay_mull > 0 and len(mull_samples) < target_half:
                        need = int(target_half - len(mull_samples))
                        idx = self._mull_rng.choice(
                            replay_mull, size=need, replace=True)
                        buf = list(self._mull_replay_mull)
                        mull_samples.extend([buf[int(i)] for i in idx])
                    if replay_keep > 0 and len(keep_samples) < target_half:
                        need = int(target_half - len(keep_samples))
                        idx = self._mull_rng.choice(
                            replay_keep, size=need, replace=True)
                        buf = list(self._mull_replay_keep)
                        keep_samples.extend([buf[int(i)] for i in idx])

                batch = keep_samples + mull_samples
                if not batch:
                    mulligan_logger.info(
                        LogCategory.MODEL_TRAIN,
                        "Mulligan replay skipped - reason=empty_minibatch replay_total=%d keep=%d mull=%d mb=%d",
                        int(replay_total), int(replay_keep), int(
                            replay_mull), int(mb)
                    )
                    return True
                random.shuffle(batch)

                feats_mb = np.stack([b[0] for b in batch], axis=0).astype(
                    np.float32, copy=False)
                action_keep_mb = np.array(
                    [b[1] for b in batch], dtype=np.int64)
                outcomes_mb = np.array([b[2] for b in batch], dtype=np.float32)
                game_len_mb = np.array([b[3] for b in batch], dtype=np.float32)
                mull_num_mb = np.array([b[4] for b in batch], dtype=np.float32)
                early_land_mb = np.array([b[5] for b in batch], dtype=np.float32)
                overridden_mb = np.array([b[6] for b in batch], dtype=np.float32)

                mb_size = int(feats_mb.shape[0])
                mb_keep_rate = float(
                    (action_keep_mb > 0).mean()) if mb_size > 0 else 0.0
                outcome_mean = float(
                    outcomes_mb.mean()) if mb_size > 0 else 0.0

                # Validate minibatch before touching GPU
                if feats_mb.shape[1] != 71:
                    mulligan_logger.info(
                        LogCategory.MODEL_TRAIN,
                        "Mulligan replay skipped - reason=bad_features_shape mb=%d feat_dim=%d",
                        int(mb_size), int(feats_mb.shape[1])
                    )
                    return True
                if not np.isfinite(feats_mb).all() or not np.isfinite(outcomes_mb).all() or not np.isfinite(game_len_mb).all() or not np.isfinite(mull_num_mb).all() or not np.isfinite(early_land_mb).all() or not np.isfinite(overridden_mb).all():
                    mulligan_logger.info(
                        LogCategory.MODEL_TRAIN,
                        "Mulligan replay skipped - reason=non_finite_minibatch mb=%d keep_rate=%.3f outcome_mean=%.3f",
                        int(mb_size), float(mb_keep_rate), float(outcome_mean)
                    )
                    return True

                # Now init model if needed
                if self.mulligan_model is None:
                    self.initializeMulliganModel()

                # --------------------------
                # Train step on minibatch
                # --------------------------
                self.mulligan_model.train()
                self.mulligan_optimizer.zero_grad()

                mdev = self.mulligan_device
                use_gpu_lock = (str(mdev).startswith("cuda")
                                and self.backend_mode != "single")
                # Multi-backend + mulligan on GPU: use inter-process GPULock.
                if use_gpu_lock:
                    self.gpu_lock.acquire(
                        timeout=None, process_name=self.process_name)
                try:
                    features_t = torch.tensor(
                        feats_mb, dtype=torch.float32, device=mdev)
                    action_keep_t = torch.tensor(
                        action_keep_mb, dtype=torch.long, device=mdev)
                    outcomes_t = torch.tensor(
                        outcomes_mb, dtype=torch.float32, device=mdev)

                    logits = self.mulligan_model(features_t)  # [mb, 1]
                    if logits is None or logits.ndim != 2 or logits.shape[0] != mb_size:
                        skip_reason = "bad_logit_shape"
                        train_count = 0
                        skipped_count = mb_size
                        loss = torch.tensor(0.0, device=mdev)
                    else:
                        logits = logits.squeeze(1)  # [mb]
                        # Clamp logits to prevent sigmoid saturation
                        # At +/-5, sigmoid is 0.993/0.007 -- still exploratory
                        logits = logits.clamp(-5.0, 5.0)

                        terminal_only = str(os.getenv("MULLIGAN_TERMINAL_ONLY", "0")).lower() in ("1", "true", "yes")
                        if terminal_only:
                            reward = outcomes_t
                        else:
                            # Legacy reward shaping path.
                            survival_alpha = float(
                                os.getenv("MULLIGAN_SURVIVAL_ALPHA", "0.3"))
                            survival_max = float(
                                os.getenv("MULLIGAN_SURVIVAL_MAX_TURNS", "12"))
                            land_drop_alpha = float(
                                os.getenv("MULLIGAN_LAND_DROP_ALPHA", "0.2"))
                            game_len_t = torch.tensor(
                                game_len_mb, dtype=torch.float32, device=mdev)
                            survival = (game_len_t.clamp(
                                min=1.0, max=survival_max) / survival_max)
                            early_land_t = torch.tensor(
                                early_land_mb, dtype=torch.float32, device=mdev)
                            has_land_data = (early_land_t >= 0.0).float()
                            land_bonus = land_drop_alpha * early_land_t.clamp(min=0.0) * has_land_data
                            reward = outcomes_t + \
                                (1.0 - outcomes_t) * (survival_alpha * survival + land_bonus)
                            # Override mask: overridden decisions get reward=0
                            overridden_t = torch.tensor(
                                overridden_mb, dtype=torch.float32, device=mdev)
                            override_mask = (overridden_t > 0.5).float()
                            reward = (1.0 - override_mask) * reward

                        # REINFORCE: log P(action_taken)
                        # P(keep) = sigmoid(logit), P(mull) = 1 - sigmoid(logit)
                        # log P(keep) = logit - softplus(logit) = -softplus(-logit)
                        # log P(mull) = -softplus(logit)
                        action_keep_float = action_keep_t.float()
                        log_prob = action_keep_float * logits - F.softplus(logits)

                        # Baseline: EMA of reward for variance reduction
                        if not hasattr(self, '_mull_reward_baseline'):
                            self._mull_reward_baseline = 0.5
                        advantage = reward - self._mull_reward_baseline
                        self._mull_reward_baseline = (
                            0.99 * self._mull_reward_baseline +
                            0.01 * float(reward.mean().item()))

                        # Policy gradient loss
                        pg_loss = -(advantage.detach() * log_prob).mean()

                        # Entropy bonus to prevent collapse
                        entropy_coeff = float(
                            os.getenv("MULLIGAN_ENTROPY_COEFF", "0.20"))
                        p_keep = torch.sigmoid(logits)
                        entropy = -(p_keep * F.softplus(-logits) +
                                    (1.0 - p_keep) * (-F.softplus(logits)))
                        entropy_bonus = entropy.mean()

                        loss = pg_loss - entropy_coeff * entropy_bonus

                        if not torch.isfinite(loss).item():
                            skip_reason = "non_finite_loss"
                            train_count = 0
                            skipped_count = mb_size
                            loss = torch.tensor(0.0, device=mdev)
                        else:
                            skip_reason = ""
                            train_count = mb_size
                            skipped_count = 0

                    if train_count > 0:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.mulligan_model.parameters(), 1.0)
                        self.mulligan_optimizer.step()

                    # Stats for logging (keep wire-compatible names)
                    with torch.no_grad():
                        p_keep_val = torch.sigmoid(logits).mean().item() if 'logits' in locals() and logits.numel() > 0 else 0.5
                        q_keep_mean = p_keep_val
                        q_mull_mean = 1.0 - p_keep_val
                        q_taken_mean = p_keep_val
                        target_mean = float(reward.mean().item()) if 'reward' in locals() and reward.numel() > 0 else 0.0

                finally:
                    if use_gpu_lock:
                        self.gpu_lock.release(process_name=self.process_name)

                # Switch back to eval for inference safety
                self.mulligan_optimizer.zero_grad()
                self.mulligan_model.eval()

                # Logging: train vs skip
                if train_count > 0:
                    mulligan_logger.info(
                        LogCategory.MODEL_TRAIN,
                        "Mulligan REINFORCE trained - loss=%.4f, mb=%d, keep_rate=%.3f, replay_total=%d (keep=%d mull=%d), hist_keep_rate=%.3f, reward_mean=%.3f, P_keep_avg=%.3f, P_mull_avg=%.3f, baseline=%.3f",
                        float(loss.item()),
                        int(train_count),
                        float(mb_keep_rate),
                        int(replay_total),
                        int(replay_keep),
                        int(replay_mull),
                        float(hist_keep_rate),
                        float(target_mean),
                        float(q_keep_mean),
                        float(q_mull_mean),
                        float(self._mull_reward_baseline if hasattr(self, '_mull_reward_baseline') else 0.5),
                    )
                else:
                    mulligan_logger.info(
                        LogCategory.MODEL_TRAIN,
                        "Mulligan REINFORCE skipped - reason=%s, mb=%d, keep_rate=%.3f, replay_total=%d (keep=%d mull=%d), hist_keep_rate=%.3f, outcome_mean=%.3f",
                        str(skip_reason) if 'skip_reason' in locals() else "unknown",
                        int(mb_size),
                        float(mb_keep_rate),
                        int(replay_total),
                        int(replay_keep),
                        int(replay_mull),
                        float(hist_keep_rate),
                        float(outcome_mean),
                    )

                # Track training iterations only on actual optimizer steps
                if train_count > 0:
                    self.mulligan_train_step_counter += 1
                    self.mulligan_train_sample_counter += int(train_count)

                # Track timing
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                self.metrics.update_timing_metric("mulligan", elapsed_ms)

                return True

            except Exception as e:
                logger.error(LogCategory.SYSTEM_ERROR,
                             f"Error in trainMulligan: {str(e)}")
                raise
            finally:
                if lock_held:
                    try:
                        self._gpu_mutex.release()
                    except Exception:
                        pass

    def saveMulliganModel(self):
        """Save mulligan model separately"""
        try:
            if self.mulligan_model is None:
                logger.warning(LogCategory.MODEL_SAVE,
                               "Mulligan model not initialized, nothing to save")
                return

            checkpoint = {
                'model_state_dict': self.mulligan_model.state_dict(),
                'optimizer_state_dict': self.mulligan_optimizer.state_dict(),
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(
                self.mulligan_model_path), exist_ok=True)
            torch.save(checkpoint, self.mulligan_model_path)
            logger.info(LogCategory.MODEL_SAVE,
                        f"Mulligan model saved to {self.mulligan_model_path}")

        except Exception as e:
            logger.error(LogCategory.MODEL_SAVE,
                         f"Failed to save mulligan model: {e}")

    def getMainModelTrainingStats(self):
        """Get main model training statistics (iterations and samples)"""
        # Convert to Java HashMap for Py4J compatibility
        from py4j.java_gateway import java_import
        java_import(gateway.jvm, 'java.util.HashMap')
        result = gateway.jvm.HashMap()
        result.put('train_steps', int(self.train_step_counter))
        result.put('train_samples', int(self.main_train_sample_counter))
        return result

    def getHealthStats(self):
        """Get training health statistics (OOMs, errors, etc.)"""
        from py4j.java_gateway import java_import
        java_import(gateway.jvm, 'java.util.HashMap')
        result = gateway.jvm.HashMap()
        result.put('gpu_oom_count', int(self.cuda_mgr.get_oom_count()))
        return result

    def resetHealthStats(self):
        """Reset health statistics counters."""
        self.cuda_mgr.reset_oom_count()

    def recordGameResult(self, lastValuePrediction, won):
        """
        Record game result for value head quality tracking and auto-GAE.
        Called from Java after each game ends.

        Args:
            lastValuePrediction: The final value prediction from the model
            won: True if the RL player won the game
        """
        self.record_value_prediction(lastValuePrediction, won)

    def getValueHeadMetrics(self):
        """
        Get current value head quality metrics (callable from Java).
        Returns a HashMap with accuracy, avg_win, avg_loss, samples.
        """
        from py4j.java_gateway import java_import
        java_import(gateway.jvm, 'java.util.HashMap')
        metrics = self.get_value_metrics()
        result = gateway.jvm.HashMap()
        result.put('accuracy', float(metrics['accuracy']))
        result.put('avg_win', float(metrics['avg_win']))
        result.put('avg_loss', float(metrics['avg_loss']))
        result.put('samples', int(metrics['samples']))
        result.put('use_gae', self.use_gae)
        return result

    def _write_training_losses_csv(self, episodes_in_batch):
        """Write training loss components to CSV file for post-hoc analysis."""
        try:
            import csv
            from datetime import datetime
            
            # Create directories if needed
            os.makedirs(os.path.dirname(self.training_losses_csv_path), exist_ok=True)
            
            # Check if we need to write header
            file_exists = os.path.exists(self.training_losses_csv_path)
            
            with open(self.training_losses_csv_path, 'a', newline='') as f:
                if not file_exists or not self._training_losses_header_written:
                    # Write header
                    f.write('step,timestamp,total_loss,policy_loss,value_loss,entropy,entropy_coef,clip_frac,approx_kl,batch_size,episodes_in_batch,advantage_mean\n')
                    self._training_losses_header_written = True
                
                # Write data row
                timestamp = datetime.now().isoformat()
                f.write(f'{self.train_step_counter},{timestamp},{self.metrics.latest_total_loss:.6f},{self.metrics.latest_policy_loss:.6f},{self.metrics.latest_value_loss:.6f},{self.metrics.latest_entropy:.6f},{self.metrics.latest_entropy_coef:.6f},{self.metrics.latest_clip_frac:.6f},{self.metrics.latest_approx_kl:.6f},{self.metrics.latest_batch_size},{episodes_in_batch},{self.metrics.latest_advantage_mean:.6f}\n')
        except Exception as e:
            # Non-critical - don't fail training over CSV logging
            logger.warning(LogCategory.MODEL_TRAIN, f"Failed to write training_losses.csv: {e}")

    def getAutoBatchMetrics(self):
        """
        Export auto-batching decisions + timing metrics for Prometheus/Grafana via Java.
        Returns a HashMap with counters, caps, estimates, and operation timings.
        """
        from py4j.java_gateway import java_import
        java_import(gateway.jvm, 'java.util.HashMap')
        result = gateway.jvm.HashMap()
        role = str(getattr(self, "py_role", "learner")).strip().lower()
        port = str(os.getenv("PY4J_PORT", "")).strip()
        worker = f"{role}_port{port}_pid{os.getpid()}"
        result.put("worker", worker)
        result.put("role", role)
        result.put("infer_safe_max", int(self._infer_safe_max)
                   if self._infer_safe_max is not None else 0)
        result.put("train_safe_max_episodes", int(self._train_safe_max_episodes)
                   if self._train_safe_max_episodes is not None else 0)
        result.put("infer_mb_per_sample", float(self._infer_mb_per_sample)
                   if self._infer_mb_per_sample is not None else 0.0)
        result.put("train_mb_per_step", float(self._train_mb_per_step)
                   if self._train_mb_per_step is not None else 0.0)

        # Refresh mem info best-effort so gauges are live.
        try:
            self._desired_free_mb()
        except Exception:
            pass
        result.put("free_mb", float(self._autobatch_last_free_mb))
        result.put("total_mb", float(self._autobatch_last_total_mb))
        result.put("desired_free_mb", float(
            self._autobatch_last_desired_free_mb))

        # Add timing metrics
        timing = self.metrics.get_timing_metrics()
        result.put("train_time_ms", float(timing['train_time_ms']))
        result.put("infer_time_ms", float(timing['infer_time_ms']))
        result.put("mulligan_time_ms", float(timing['mulligan_time_ms']))

        for k, v in self._autobatch_counts.items():
            try:
                result.put(k, int(v))
            except Exception:
                pass
        return result

    def getTrainingLossMetrics(self):
        """
        Export latest training loss components for Prometheus/Grafana via Java.
        Returns a HashMap with total_loss, policy_loss, value_loss, entropy, etc.
        """
        from py4j.java_gateway import java_import
        java_import(gateway.jvm, 'java.util.HashMap')
        m = self.metrics.get_training_loss_metrics()
        result = gateway.jvm.HashMap()
        for k, v in m.items():
            result.put(k, float(v))
        return result

    def shutdown(self):
        """Allows Java to cleanly shut down the Python gateway."""
        logger.info(LogCategory.SYSTEM_CLEANUP,
                    "Received shutdown request from Java.")
        cleanup_temp_files()

        # We must use a separate thread to shut down the gateway,
        # otherwise the call will deadlock as it's waiting for a response.
        def _shutdown():
            time.sleep(1)  # Give Java a moment to receive the response
            # Note that gateway.shutdown() is not enough since the script would
            # still be running, just without an active gateway.
            os._exit(0)

        threading.Thread(target=_shutdown).start()
        # Immediately return to Java so it doesn't block
        return "OK"

    def getOptimalBatchSize(self):
        """Calculate optimal batch size based on available GPU memory (callable from Java)"""
        try:
            return calculate_optimal_batch_size()
        except Exception as e:
            logger.error(LogCategory.GPU_MEMORY,
                         "Error in getOptimalBatchSize: %s", str(e))
            return 10000  # Safe fallback

    def getDeviceInfo(self):
        """Return a short diagnostic string about CUDA/device placement (callable from Java)."""
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            dev = str(self.device) if hasattr(self, "device") else "unknown"
            name = ""
            if cuda_ok:
                try:
                    name = torch.cuda.get_device_name(0)
                except Exception:
                    name = ""
            opt = None
            try:
                opt = calculate_optimal_batch_size()
            except Exception:
                opt = None
            return f"device={dev} cuda_available={cuda_ok} gpu={name} optimal_batch={opt}"
        except Exception as e:
            return f"device_info_error={e}"

    class Java:
        implements = ["mage.player.ai.rl.PythonBatchService"]


if __name__ == "__main__":
    exit_code = 0
    try:
        # CLI args (used by Java launcher); env vars still supported.
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--port", type=int, default=None)
        parser.add_argument("--role", type=str, default=None)
        args, _ = parser.parse_known_args()

        py4j_port = int(args.port) if args.port is not None else int(
            os.getenv("PY4J_PORT", "25334"))
        py_role = (args.role or os.getenv(
            "PY_ROLE", "learner")).strip().lower()
        os.environ["PY4J_PORT"] = str(py4j_port)
        os.environ["PY_ROLE"] = py_role

        # Start Py4J gateway with retry logic
        max_retries = 5
        retry_delay = 2  # seconds
        gateway = None

        for attempt in range(max_retries):
            try:
                logger.info(LogCategory.SYSTEM_INIT,
                            f"Attempting to start Py4J gateway (attempt {attempt + 1}/{max_retries})")
                gateway = ClientServer(
                    java_parameters=JavaParameters(),
                    python_parameters=PythonParameters(port=py4j_port),
                    python_server_entry_point=PythonEntryPoint()
                )
                logger.info(LogCategory.SYSTEM_INIT,
                            f"Python ML service started on port {py4j_port} (role={py_role})")
                break
            except Exception as e:
                logger.error(LogCategory.SYSTEM_ERROR,
                             "Failed to start Py4J gateway (attempt %d/%d): %s", attempt + 1, max_retries, str(e))
                if attempt < max_retries - 1:
                    logger.info(LogCategory.SYSTEM_INIT,
                                "Retrying in %d seconds...", retry_delay)
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        "Failed to start Py4J gateway after %d attempts" % max_retries)

        # Keep the server alive with connection monitoring
        last_connection_check = time.time()
        connection_check_interval = 5  # Check connection every 5 seconds

        while True:
            try:
                # Check connection periodically
                current_time = time.time()
                if current_time - last_connection_check >= connection_check_interval:
                    if gateway is None:
                        logger.error(LogCategory.SYSTEM_ERROR,
                                     "Py4J gateway connection lost (gateway is None), attempting to reconnect...")
                        try:
                            gateway = ClientServer(
                                java_parameters=JavaParameters(),
                                python_parameters=PythonParameters(
                                    port=py4j_port),
                                python_server_entry_point=PythonEntryPoint()
                            )
                            logger.info(LogCategory.SYSTEM_INIT,
                                        f"Successfully reconnected to Py4J gateway on port {py4j_port}")
                        except Exception as e:
                            logger.error(LogCategory.SYSTEM_ERROR,
                                         "Failed to reconnect to Py4J gateway: %s", str(e))
                    last_connection_check = current_time

                # Sleep to prevent busy waiting
                time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(LogCategory.SYSTEM_ERROR,
                             "Error in main loop: %s", str(e))
                time.sleep(1)  # Sleep before retrying
    except KeyboardInterrupt:
        logger.info(LogCategory.SYSTEM_CLEANUP,
                    "Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(LogCategory.SYSTEM_ERROR,
                     "Fatal error in gateway thread: %s", str(e))
        exit_code = 1
    finally:
        cleanup_temp_files()
        if gateway is not None:
            try:
                gateway.shutdown()
            except Exception as e:
                logger.error(LogCategory.SYSTEM_ERROR,
                             "Error during gateway shutdown: %s", str(e))
        logger.info(LogCategory.SYSTEM_CLEANUP, "Python ML service stopped")
        sys.exit(exit_code)
