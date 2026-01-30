from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
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

# Import new modules
from logging_utils import logger, LogCategory, TEMP_DIR, script_dir, log_file
from cuda_manager import CUDAManager
from snapshot_manager import SnapshotManager
from metrics_collector import MetricsCollector
from model_persistence import ModelPersistence
from mulligan_model import MulliganNet
from gpu_lock import GPULock

# Now we can safely log initialization
logger.info(LogCategory.SYSTEM_INIT, f"Logging to file: {log_file}")
logger.info(LogCategory.SYSTEM_INIT,
            f"Created temporary directory for shared memory: {TEMP_DIR}")

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


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.py_role = os.getenv("PY_ROLE", "learner").strip().lower()

        # GPU coordination lock (inter-process)
        self.gpu_lock = GPULock()
        self.process_name = f"{self.py_role}_{os.getpid()}"

        # Initialize helper modules
        self.cuda_mgr = CUDAManager(self.py_role)
        self.snapshot_mgr = SnapshotManager(self.device)
        self.metrics = MetricsCollector()
        self.persistence = ModelPersistence()

        # Mulligan model
        self.mulligan_model = None
        self.mulligan_optimizer = None
        self.mulligan_model_path = os.getenv('MULLIGAN_MODEL_PATH',
                                             'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/mulligan_model.pt')
        self.mulligan_lock = threading.Lock()
        self._mull_target1_window = int(
            os.getenv("MULLIGAN_TARGET1_WINDOW", "500"))
        self._mull_target1_log_every = int(
            os.getenv("MULLIGAN_TARGET1_LOG_EVERY", "20"))
        self._mull_target1_hist = deque(
            maxlen=max(1, self._mull_target1_window))

        # PPO configuration
        self.ppo_epsilon = float(os.getenv('PPO_EPSILON', '0.2'))
        self.use_ppo = bool(int(os.getenv('USE_PPO', '1')))
        self._ppo_stats_every = int(os.getenv("PPO_STATS_EVERY", "50"))

        # Loss scheduling
        self.loss_schedule_enable = bool(
            int(os.getenv('LOSS_SCHEDULE_ENABLE', '1')))
        self.critic_warmup_steps = int(os.getenv('CRITIC_WARMUP_STEPS', '200'))
        self.freeze_encoder_in_warmup = bool(
            int(os.getenv('FREEZE_ENCODER_IN_WARMUP', '1')))
        self._encoder_frozen = False

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

        logger.info(LogCategory.GPU_MEMORY, "Using device: %s", self.device)

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

    def _update_mem_ema(self, kind: str, extra_mb: float, n: int):
        self.cuda_mgr.update_mem_ema(kind, extra_mb, n)

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

    def initializeModel(self):
        """Initialize the model and optimizer"""
        logger.info(LogCategory.GPU_MEMORY, "Initializing model")
        if self.model is None:
            self.model = MTGTransformerModel().to(self.device)

            # Separate higher LR for actor head to help logits move early
            actor_param_names = [
                'actor_proj1', 'actor_proj2', 'actor_norm', 'actor_norm1'
            ]
            actor_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if any(apn in name for apn in actor_param_names):
                    actor_params.append(param)
                else:
                    other_params.append(param)

            # ----------------- LR Tuning ---------------------------
            # Slower, more stable learning rates to prevent policy collapse.
            self.optimizer = torch.optim.Adam([
                {'params': actor_params, 'lr': 1e-4},
                # Increased from 5e-5 to match actor LR
                {'params': other_params, 'lr': 1e-4}
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

        # One-time initial load + mulligan init
        if not self._did_initial_load:
            try:
                if self.mulligan_model is None:
                    self.initializeMulliganModel()
            except Exception:
                pass

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

    def initializeMulliganModel(self):
        """Initialize the mulligan model (separate from main model)"""
        logger.info(LogCategory.MODEL_INIT,
                    "Initializing card-level mulligan model")

        # vocab_size=65536 (same as main model), embed_dim=32, max_hand=7, max_deck=60
        self.mulligan_model = MulliganNet(
            vocab_size=65536, embed_dim=32, max_hand=7, max_deck=60).to(self.device)
        self.mulligan_optimizer = torch.optim.Adam(
            self.mulligan_model.parameters(), lr=1e-3)

        # Try to load existing mulligan model
        if os.path.exists(self.mulligan_model_path):
            try:
                checkpoint = torch.load(
                    self.mulligan_model_path, map_location=self.device)
                self.mulligan_model.load_state_dict(
                    checkpoint['model_state_dict'])
                self.mulligan_optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                logger.info(LogCategory.MODEL_LOAD,
                            f"Loaded existing mulligan model from {self.mulligan_model_path}")
            except Exception as e:
                logger.warning(LogCategory.MODEL_LOAD,
                               f"Failed to load mulligan model, starting fresh: {e}")
        else:
            logger.info(LogCategory.MODEL_INIT,
                        "No existing mulligan model found, starting with random initialization")

        logger.info(LogCategory.MODEL_INIT, "Mulligan model initialized")

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
                                  batch_size,
                                  seq_len,
                                  d_model,
                                  max_candidates,
                                  cand_feat_dim):
        """Score padded candidates for each state (policy over candidates + value)."""
        t_start = time.perf_counter()

        def _score_numpy_range(start: int, end: int):
            if self.model is None:
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

            seq_t = torch.tensor(seq, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
            tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device)
            cand_feat_t = torch.tensor(
                cand_feat, dtype=torch.float32, device=device)
            cand_ids_t = torch.tensor(
                cand_ids, dtype=torch.long, device=device)
            cand_mask_t = torch.tensor(
                cand_mask, dtype=torch.bool, device=device)

            # Release numpy slices immediately
            del seq, mask, tok_ids, cand_feat, cand_ids, cand_mask

            model = self._get_policy_model(policy_key)
            model.eval()

            # Hard requirement: inference must only run while GPULock is held
            if torch.cuda.is_available() and str(device).startswith("cuda") and not self.gpu_lock.is_locked:
                raise RuntimeError(
                    "GPULock is required for inference. Call acquireGPULock() before scoring.")

            with torch.no_grad():
                probs, value = model.score_candidates(
                    seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t)

            probs_np = probs.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()

            # Release GPU tensors immediately (in this scope where they're defined)
            del seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t, probs, value
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()

            return probs_np, value_np

        def _score_with_oom_splitting(start: int, end: int):
            n = int(end - start)
            if n <= 0:
                return np.zeros((0, max_candidates), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
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
                i = start
                while i < end:
                    j = min(end, i + int(cap))
                    p, v = _score_with_oom_splitting(i, j)
                    probs_parts.append(p)
                    value_parts.append(v)
                    i = j
                result_probs = np.concatenate(probs_parts, axis=0)
                result_values = np.concatenate(value_parts, axis=0)
                # Clean up intermediate parts
                del probs_parts, value_parts
                return result_probs, result_values

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
                    p0, v0 = _score_with_oom_splitting(start, mid)
                    p1, v1 = _score_with_oom_splitting(mid, end)
                    return np.concatenate((p0, p1), axis=0), np.concatenate((v0, v1), axis=0)

            try:
                (p, v), extra_mb = self._measure_peak_extra_mb(
                    lambda: _score_numpy_range(start, end))
                # Update per-sample estimate from measured peak delta.
                self._update_mem_ema("infer", extra_mb, n)
                return p, v
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
                    p0, v0 = _score_with_oom_splitting(start, mid)
                    p1, v1 = _score_with_oom_splitting(mid, end)
                    result_probs = np.concatenate((p0, p1), axis=0)
                    result_values = np.concatenate((v0, v1), axis=0)
                    # Clean up splits
                    del p0, v0, p1, v1
                    return result_probs, result_values
                raise

        try:
            probs_np, value_np = _score_with_oom_splitting(0, int(batch_size))

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

            out = np.concatenate((probs_np, value_np), axis=1)
            result_bytes = out.astype('<f4').tobytes()

            # Clean up numpy arrays to prevent accumulation
            del probs_np, value_np, out

            return result_bytes

        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in scoreCandidatesPolicyFlat: %s", str(e))
            raise

    def predictMulligan(self, features):
        """
        Predict mulligan decision using Q-learning neural network.

        Args:
            features: List/array of mulligan features (68-dim vector)
                Format: [mulligan_num(1), hand_card_ids(7), deck_card_ids(60)]

        Returns:
            float: Deterministic keep indicator (0.0 = mulligan, 1.0 = keep)
                   KEEP if Q_keep >= Q_mull else MULL
        """
        try:
            if self.mulligan_model is None:
                self.initializeMulliganModel()

            # Convert to tensor - should be 68 dims: 1 + 7 + 60
            features_tensor = torch.tensor(
                features, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Forward pass (no gradient needed for inference)
            self.mulligan_model.eval()
            with torch.no_grad():
                q_values = self.mulligan_model(features_tensor)  # [1, 2]
                q_keep = q_values[0, 0].item()
                q_mull = q_values[0, 1].item()

            return 1.0 if q_keep >= q_mull else 0.0

        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in predictMulligan: %s", str(e))
            # On error, default to 50/50
            return 0.5

    def predictMulliganScores(self, features):
        """
        Return raw two-headed mulligan scores as little-endian float32 bytes: [Q_keep, Q_mull].
        """
        try:
            if self.mulligan_model is None:
                self.initializeMulliganModel()

            features_tensor = torch.tensor(
                features, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.mulligan_model.eval()
            with torch.no_grad():
                q_values = self.mulligan_model(features_tensor)  # [1, 2]
                q_keep = float(q_values[0, 0].item())
                q_mull = float(q_values[0, 1].item())

            out = np.array([q_keep, q_mull], dtype='<f4')
            return out.tobytes()
        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in predictMulliganScores: %s", str(e))
            return np.array([0.0, 0.0], dtype='<f4').tobytes()

    # Metrics methods and properties - delegate to metrics collector
    # Counters
    @property
    def train_step_counter(self):
        return self.metrics.train_step_counter

    @train_step_counter.setter
    def train_step_counter(self, value):
        self.metrics.train_step_counter = value

    @property
    def mulligan_train_step_counter(self):
        return self.metrics.mulligan_train_step_counter

    @mulligan_train_step_counter.setter
    def mulligan_train_step_counter(self, value):
        self.metrics.mulligan_train_step_counter = value

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

    @property
    def mulligan_train_sample_counter(self):
        return self.metrics.mulligan_train_sample_counter

    @mulligan_train_sample_counter.setter
    def mulligan_train_sample_counter(self, value):
        self.metrics.mulligan_train_sample_counter = value

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
                return
            if torch.isnan(rewards_t).any() or torch.isinf(rewards_t).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in rewards - skipping batch (rewards=%s)", rewards_t.tolist())
                return
            if torch.isnan(cand_feat_t).any() or torch.isinf(cand_feat_t).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in candidate features - skipping batch")
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

            self.optimizer.zero_grad()

            # Get new policy probabilities (with gradients)
            probs, value = self.model.score_candidates(
                seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t)

            # Check for NaN in model outputs before training
            if torch.isnan(probs).any() or torch.isnan(value).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "Model produced NaN outputs - skipping batch (probs_nan=%s, value_nan=%s)",
                               torch.isnan(probs).any().item(), torch.isnan(value).any().item())
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
                    rewards_t, value_detached, gamma=0.99, gae_lambda=self.current_gae_lambda, dones=dones_t)
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
                self._adv_running_mean = (
                    1 - alpha) * self._adv_running_mean + alpha * batch_mean
                self._adv_running_var = (
                    1 - alpha) * self._adv_running_var + alpha * batch_var
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
                return

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
                return

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.model.max_grad_norm)
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
            else:
                logger.info(LogCategory.MODEL_TRAIN,
                            "trainCandidatesFlat — loss=%.4f policy=%.4f value=%.4f ent=%.4f (coeff: %.4f)",
                            loss.item(), loss_policy.item(), loss_value.item(), entropy.item(), float(self.get_entropy_coefficient()) * float(entropy_loss_mult))

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
                    # Fallback to models directory
                    ckpt_path = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt"
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
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Snapshot checkpoints (league opponents)
            self._maybe_save_snapshot()

            # Torch profiler step (env-gated)
            self._prof_step("train")
            return True

        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in trainCandidatesFlat: %s", str(e))
            raise

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
                                 dones_bytes,
                                 batch_size,
                                 seq_len,
                                 d_model,
                                 max_candidates,
                                 cand_feat_dim):
        """
        Train on a batch that concatenates multiple episodes.
        dones marks episode ends (1=end-of-episode), so GAE/returns do not leak across boundaries.
        """
        t_start = time.perf_counter()

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
            dones = np.frombuffer(dones_bytes, dtype='<i4').reshape(
                batch_size)[start:end]

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
            old_logp_t = torch.tensor(
                old_logp_total, dtype=torch.float32, device=device)
            old_value_t = torch.tensor(
                old_value, dtype=torch.float32, device=device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

            # Release numpy slices immediately
            del seq, mask, tok_ids, cand_feat, cand_ids, cand_mask, chosen_indices, chosen_count, rewards, old_logp_total, old_value, dones

            local_batch_size = int(end - start)

            if torch.isnan(seq_t).any() or torch.isinf(seq_t).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in input sequence - skipping batch")
                return
            if torch.isnan(rewards_t).any() or torch.isinf(rewards_t).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in rewards - skipping batch (rewards=%s)", rewards_t.tolist())
                return
            if torch.isnan(cand_feat_t).any() or torch.isinf(cand_feat_t).any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "NaN/Inf in candidate features - skipping batch")
                return

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
                self.optimizer.zero_grad()

                probs, value = self.model.score_candidates(
                    seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t)

                if torch.isnan(probs).any() or torch.isnan(value).any():
                    logger.warning(LogCategory.MODEL_TRAIN,
                                   "Model produced NaN outputs - skipping batch (probs_nan=%s, value_nan=%s)",
                                   torch.isnan(probs).any().item(), torch.isnan(value).any().item())
                    return

                value_squeezed = value.squeeze(1)
                value_detached = value_squeezed.detach()

                if self.use_gae:
                    self.update_gae_lambda_schedule()
                    advantages, value_targets = self.compute_gae(
                        rewards_t, value_detached, gamma=0.99, gae_lambda=self.current_gae_lambda, dones=dones_t)
                    advantages = advantages.detach()
                    value_targets = value_targets.detach()
                else:
                    gamma = 0.99
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
                        _p0, _v0 = self.model.score_candidates(
                            seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t
                        )
                        v_before_eval = _v0.squeeze(
                            1).detach().float().view(-1)
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
                    self._adv_running_mean = (
                        1 - alpha) * self._adv_running_mean + alpha * batch_mean
                    self._adv_running_var = (
                        1 - alpha) * self._adv_running_var + alpha * batch_var
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

                probs_safe = torch.clamp(probs, min=1e-8, max=1.0)
                new_logp = self._joint_logp_from_probs(
                    probs_safe, cand_mask_t, chosen_indices_t, chosen_count_t)

                if self.use_ppo:
                    log_ratio = (new_logp - old_logp_t).clamp(-20.0, 20.0)
                    ratio_raw = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(
                        ratio_raw, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
                    loss_policy = - \
                        torch.min(ratio_raw * advantages_normalized,
                                  clipped_ratio * advantages_normalized).mean()

                    # PPO stats logging (mtg_ai.log) - mean/std of adv/ret/ratio
                    if self._ppo_stats_every > 0 and (next_step % self._ppo_stats_every == 0):
                        with torch.no_grad():
                            adv_t = advantages.detach().float().view(-1)
                            ret_t = value_targets.detach().float().view(-1)
                            rr_t = ratio_raw.detach().float().view(-1)
                            adv_mean = float(
                                adv_t.mean().item()) if adv_t.numel() > 0 else 0.0
                            adv_std = float(adv_t.std().item()
                                            ) if adv_t.numel() > 1 else 0.0
                            ret_mean = float(
                                ret_t.mean().item()) if ret_t.numel() > 0 else 0.0
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
                    loss_policy = -(new_logp * advantages_normalized).mean()

                if self.loss_schedule_enable and next_step <= self.critic_warmup_steps:
                    policy_loss_coef = float(self.policy_loss_coef_warmup)
                    value_loss_coef = float(self.value_loss_coef_warmup)
                    entropy_loss_mult = float(self.entropy_loss_mult_warmup)
                else:
                    policy_loss_coef = float(self.policy_loss_coef_main)
                    value_loss_coef = float(self.value_loss_coef_main)
                    entropy_loss_mult = float(self.entropy_loss_mult_main)

                vf_clip = float(os.getenv("PPO_VF_CLIP", "0.2"))
                if self.use_ppo and vf_clip > 0.0:
                    v_pred = value_squeezed
                    v_old = old_value_t.view_as(v_pred).detach()
                    v_clipped = v_old + \
                        (v_pred - v_old).clamp(-vf_clip, vf_clip)
                    vf_loss1 = (v_pred - value_targets).pow(2)
                    vf_loss2 = (v_clipped - value_targets).pow(2)
                    loss_value = value_loss_coef * \
                        (0.5 * torch.max(vf_loss1, vf_loss2).mean())
                else:
                    loss_value = value_loss_coef * \
                        F.mse_loss(value_squeezed, value_targets)

                probs_safe = torch.clamp(probs, min=1e-8, max=1.0)
                log_probs = torch.log(probs_safe)
                entropy = -(probs_safe * log_probs).sum(dim=-1).mean()
                entropy_coef = float(
                    self.get_entropy_coefficient()) * float(entropy_loss_mult)
                loss_entropy = -entropy_coef * entropy

                loss = (policy_loss_coef * loss_policy) + \
                    loss_value + loss_entropy

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(LogCategory.MODEL_TRAIN,
                                   "Skipping update due to NaN/Inf loss (policy=%.4f, value=%.4f, ent=%.4f)",
                                   loss_policy.item() if not torch.isnan(loss_policy) else float('nan'),
                                   loss_value.item() if not torch.isnan(loss_value) else float('nan'),
                                   entropy.item() if not torch.isnan(entropy) else float('nan'))
                    self.optimizer.zero_grad()
                    return

                loss.backward()

                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break
                if has_nan_grad:
                    logger.warning(LogCategory.MODEL_TRAIN,
                                   "Skipping update due to NaN/Inf gradients")
                    self.optimizer.zero_grad()
                    return

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.model.max_grad_norm)
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
                            "trainCandidatesMultiFlat — loss=%.4f policy=%.4f value=%.4f ent=%.4f (coeff: %.4f) [PPO clip: %.2f%% kl=%.6f]",
                            loss.item(), loss_policy.item(), loss_value.item(), entropy.item(),
                            entropy_coef, clip_frac.item() * 100, approx_kl.item())
            else:
                logger.info(LogCategory.MODEL_TRAIN,
                            "trainCandidatesMultiFlat — loss=%.4f policy=%.4f value=%.4f ent=%.4f (coeff: %.4f)",
                            loss.item(), loss_policy.item(), loss_value.item(), entropy.item(), float(self.get_entropy_coefficient()) * float(entropy_loss_mult))

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

            prev_step = int(self.train_step_counter)
            self.train_step_counter += ep_count
            self.main_train_sample_counter += local_batch_size

            # Explicit cleanup of training tensors
            del seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t
            del chosen_indices_t, chosen_count_t, value_targets, advantages, rewards_t, old_logp_t, old_value_t, dones_t
            del loss, loss_policy, loss_value, entropy
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Save once per 100 episodes (avoid missing the boundary when ep_count > 1)
            if (self.train_step_counter // 100) > (prev_step // 100):
                if self.model_path:
                    ckpt_path = self.model_path
                else:
                    ckpt_path = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt"
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
            # Proactive paging avoidance: estimate extra VRAM for this slice and split episodes if needed.
            if self.auto_batch_enable and self.auto_avoid_paging and torch.cuda.is_available() and len(spans) > 1:
                est = 0.0
                if self._train_mb_per_step is not None:
                    est = float(self._train_mb_per_step) * float(steps)
                if self._should_split_for_paging(est):
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

            return result
        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in trainCandidatesMultiFlat: %s", str(e))
            raise

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
        self.persistence.save_model(self.model, path)

    def loadModel(self, path):
        if self.model is None:
            self.initializeModel()
        self.persistence.load_model(self.model, path)

    def saveLatestModelAtomic(self, path=None):
        return self.persistence.save_latest_model_atomic(self.model, path)

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

    def trainMulligan(self, features_bytes, decisions_bytes, outcomes_bytes, land_counts_bytes, batch_size):
        """
        Train mulligan model using Q-learning from game outcomes.

        Args:
            features_bytes: Raw bytes of mulligan features [batch_size * 68 * 4]
                Format: [mulligan_num(1), hand_ids(7), deck_ids(60)] per sample
            decisions_bytes: Raw bytes of decisions (0=mulligan, 1=keep) [batch_size * 4]  
            outcomes_bytes: Raw bytes of game outcomes (1=win, 0=loss) [batch_size * 4]
            land_counts_bytes: Raw bytes of land counts [batch_size * 4] (unused in Q-learning)
            batch_size: Number of decisions in this game

        Q-Learning approach:
        - For each decision, train only the Q-value of the action taken
        - Target = game outcome (1.0 for win, 0.0 for loss)
        - Per-decision weighting: each decision weighted by 1/batch_size
        """
        t_start = time.perf_counter()
        # Use lock to prevent concurrent training (avoids gradient conflicts)
        with self.mulligan_lock:
            try:
                if self.mulligan_model is None:
                    self.initializeMulliganModel()

                # Parse inputs: 1 + 7 + 60 = 68 features per sample
                features = np.frombuffer(
                    features_bytes, dtype='<f4').reshape(batch_size, 68)
                decisions = np.frombuffer(
                    decisions_bytes, dtype='<f4').reshape(batch_size)
                outcomes = np.frombuffer(
                    outcomes_bytes, dtype='<f4').reshape(batch_size)
                land_counts_np = np.frombuffer(
                    land_counts_bytes, dtype='<i4').reshape(batch_size).astype(np.float32)

                # Convert to tensors
                features_t = torch.tensor(
                    features, dtype=torch.float32, device=self.device)
                decisions_t = torch.tensor(
                    decisions, dtype=torch.float32, device=self.device)
                outcomes_t = torch.tensor(
                    outcomes, dtype=torch.float32, device=self.device)
                land_counts_t = torch.tensor(
                    land_counts_np, dtype=torch.float32, device=self.device)

                # Release numpy arrays immediately (but keep land_counts_np for heuristic below)
                del features, decisions, outcomes

                # Training mode
                self.mulligan_model.train()
                self.mulligan_optimizer.zero_grad()

                # Acquire GPU lock before mulligan training
                self.gpu_lock.acquire(
                    timeout=None, process_name=self.process_name)
                try:
                    # Q-Learning: Train the Q-value of the action taken toward game outcome
                    # All decisions get the same game outcome (win=1.0, loss=0.0)

                    # Forward pass to get Q-values [batch, 2] where [:, 0]=Q_keep, [:, 1]=Q_mull
                    q_values = self.mulligan_model(
                        features_t)  # [batch_size, 2]

                    # Extract action indices
                    # decisions_t: 1.0=keep, 0.0=mulligan
                    # We want: index 0 for keep (Q_keep), index 1 for mulligan (Q_mull)
                    # [batch_size] - 0 for keep, 1 for mulligan
                    action_indices = (decisions_t <= 0.5).long()

                    # Get Q-value for the action taken using gather
                    # q_values: [batch_size, 2], action_indices: [batch_size] -> [batch_size, 1]
                    q_taken = q_values.gather(
                        # [batch_size]
                        1, action_indices.unsqueeze(1)).squeeze(1)

                    targets = outcomes_t  # [batch_size]

                    # Early label gating to avoid collapse (e.g., always-mull early)
                    warmup_steps = int(
                        os.getenv("MULLIGAN_TRAIN_WARMUP_STEPS", "50"))
                    min_keep_rate = float(
                        os.getenv("MULLIGAN_TRAIN_MIN_KEEP_RATE", "0.10"))
                    max_keep_rate = float(
                        os.getenv("MULLIGAN_TRAIN_MAX_KEEP_RATE", "0.90"))
                    keep_rate = float((decisions_t > 0.5).float().mean(
                    ).item()) if decisions_t.numel() > 0 else 0.0
                    all_same = bool((decisions_t > 0.5).all().item() or (
                        decisions_t <= 0.5).all().item()) if decisions_t.numel() > 0 else True
                    if self.mulligan_train_step_counter < warmup_steps and (all_same or keep_rate < min_keep_rate or keep_rate > max_keep_rate):
                        train_count = 0
                        skipped_count = batch_size
                        loss = torch.tensor(0.0, device=self.device)
                    else:
                        # Mulligan cost shaping (applies to MULL actions)
                        mull_base = float(
                            os.getenv("MULLIGAN_COST_BASE", "0.02"))
                        mull_max = float(
                            os.getenv("MULLIGAN_COST_MAX", "0.15"))
                        mull_num = features_t[:, 0].clamp(min=0.0)
                        mull_cost = (mull_base * (mull_num + 1.0)
                                     ).clamp(max=mull_max)
                        is_mull = (action_indices == 1).float()
                        targets = targets - is_mull * mull_cost

                        # Land-count shaping for KEEP (annealed toward 0)
                        land_init = float(
                            os.getenv("MULLIGAN_LAND_SHAPING_INIT", "0.05"))
                        land_decay_steps = float(
                            os.getenv("MULLIGAN_LAND_SHAPING_DECAY_STEPS", "5000"))
                        t = float(self.mulligan_train_step_counter)
                        shaping_scale = max(
                            0.0, 1.0 - (t / max(1.0, land_decay_steps)))
                        shaping_scale = land_init * shaping_scale

                        if shaping_scale > 0.0:
                            lc = land_counts_t.clamp(min=0.0, max=7.0)
                            is_keep = (action_indices == 0).float()
                            good = ((lc >= 2.0) & (lc <= 3.0)).float()
                            low = (lc <= 1.0).float()
                            high = (lc >= 4.0).float()
                            land_shaping = shaping_scale * \
                                (good - 1.0 * low - 0.5 * high)
                            targets = targets + is_keep * land_shaping

                        # Rolling target_1_rate over last N mulligan decisions (for mtg_ai.log)
                        try:
                            bits = (targets.detach() >= 0.5).to(
                                torch.int32).cpu().tolist()
                            for b in bits:
                                self._mull_target1_hist.append(int(b))
                        except Exception:
                            pass
                        if self._mull_target1_log_every > 0 and (self.mulligan_train_step_counter % self._mull_target1_log_every == 0):
                            try:
                                n = len(self._mull_target1_hist)
                                if n > 0:
                                    rate = float(
                                        sum(self._mull_target1_hist)) / float(n)
                                else:
                                    rate = 0.0
                                logger.info(
                                    LogCategory.MODEL_TRAIN,
                                    "MulliganTarget1 window=%d rate=%.3f (n=%d)",
                                    int(self._mull_target1_hist.maxlen or 0),
                                    float(rate),
                                    int(n)
                                )
                            except Exception:
                                pass

                        # Per-decision weighting: 1/batch_size for each decision
                        weight_per_decision = 1.0 / batch_size

                        # Huber loss with per-decision weighting
                        per_sample = F.smooth_l1_loss(
                            q_taken, targets, reduction='none')
                        loss = (per_sample * weight_per_decision).sum()

                        train_count = batch_size
                        skipped_count = 0

                    if train_count > 0:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.mulligan_model.parameters(), 1.0)
                        self.mulligan_optimizer.step()
                finally:
                    # Release GPU lock after mulligan training
                    self.gpu_lock.release(process_name=self.process_name)

                # Clear gradients and set to eval mode
                self.mulligan_optimizer.zero_grad()
                self.mulligan_model.eval()

                # Calculate Q-value statistics for logging
                with torch.no_grad():
                    q_keep_mean = q_values[:, 0].mean().item()
                    q_mull_mean = q_values[:, 1].mean().item()
                    q_taken_mean = q_taken.mean().item()

                logger.info(LogCategory.MODEL_TRAIN,
                            f"Mulligan Q-learning trained - loss={loss.item():.4f}, " +
                            f"decisions={train_count}, skipped={skipped_count}, keep_rate={keep_rate:.3f}, " +
                            f"Q_keep_avg={q_keep_mean:.3f}, Q_mull_avg={q_mull_mean:.3f}, " +
                            f"Q_taken_avg={q_taken_mean:.3f}, target={outcomes_t[0].item():.1f}")

                # Track training iterations
                self.mulligan_train_step_counter += 1
                self.mulligan_train_sample_counter += train_count

                # Explicit cleanup of mulligan tensors
                del features_t, decisions_t, outcomes_t, land_counts_t
                del q_values, action_indices, q_taken, targets, loss
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Clear CUDA cache to free stale references
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Track timing
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                self.metrics.update_timing_metric("mulligan", elapsed_ms)

                return True

            except Exception as e:
                logger.error(LogCategory.SYSTEM_ERROR,
                             f"Error in trainMulligan: {str(e)}")
                raise

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

    def getMulliganModelTrainingStats(self):
        """Get mulligan model training statistics (iterations and samples)"""
        # Convert to Java HashMap for Py4J compatibility
        from py4j.java_gateway import java_import
        java_import(gateway.jvm, 'java.util.HashMap')
        result = gateway.jvm.HashMap()
        result.put('train_steps', int(self.mulligan_train_step_counter))
        result.put('train_samples', int(self.mulligan_train_sample_counter))
        return result

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
