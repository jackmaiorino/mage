from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
from mtg_transformer import MTGTransformerModel
import numpy as np
import torch
import torch.nn.functional as F
import logging
import sys
import os
import signal
import time
import tempfile
import threading
import queue
from collections import deque
import struct

# Logging categories


class LogCategory:
    # Set to True to enable logging for each category
    MODEL_INIT = True  # Enable model initialization logging
    MODEL_PREDICT = True  # Enable prediction logging
    MODEL_TRAIN = True
    MODEL_SAVE = True  # Enable model save/load logging
    MODEL_LOAD = True

    GPU_MEMORY = False  # Enable GPU memory logging
    GPU_BATCH = True  # Enable batch processing logging
    GPU_CLEANUP = False

    SYSTEM_INIT = True  # Enable system initialization logging
    SYSTEM_CLEANUP = True
    SYSTEM_ERROR = True

    PERFORMANCE_BATCH = True  # Enable performance logging
    PERFORMANCE_MEMORY = True
    PERFORMANCE_TIMING = True

    # Default category is always enabled
    DEFAULT = True


# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, 'mtg_ai.log')

# Create a temporary directory for shared memory files
TEMP_DIR = tempfile.mkdtemp(prefix='mtg_ai_')

# Custom formatter that handles the category field


class CategoryFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'category'):
            record.category = LogCategory.DEFAULT
        return super().format(record)

# Adapter using stdlib LoggerAdapter to attach category per-call while keeping
# the original logger.info(category, msg, ...) call pattern for minimal diff.


class CategoryAdapter(logging.LoggerAdapter):
    """Thin wrapper around LoggerAdapter that accepts (category, msg) style."""

    def _log_with_category(self, level, category, msg, *args, **kwargs):
        """Centralised category-aware logging.

        • If *category* is a boolean flag (as used in ``LogCategory``) and is
          ``False``, the call is a no-op—effectively silencing that log line.
        • Otherwise, we attach the category to *extra* so the formatter can
          display it.
        """

        # Fast-skip if the caller explicitly disabled this category.
        if isinstance(category, bool) and category is False:
            return

        extra = kwargs.pop('extra', {})
        extra['category'] = category
        self.logger.log(level, msg, *args, extra=extra, **kwargs)

    def info(self, category, msg, *args, **kwargs):
        self._log_with_category(logging.INFO, category, msg, *args, **kwargs)

    def warning(self, category, msg, *args, **kwargs):
        self._log_with_category(
            logging.WARNING, category, msg, *args, **kwargs)

    def error(self, category, msg, *args, **kwargs):
        self._log_with_category(logging.ERROR, category, msg, *args, **kwargs)

    def debug(self, category, msg, *args, **kwargs):
        self._log_with_category(logging.DEBUG, category, msg, *args, **kwargs)

    # Preserve access to underlying logger attributes
    def __getattr__(self, item):
        return getattr(self.logger, item)


# Configure base logging first
base_logger = logging.getLogger('mtg_ai')

# Default to WARNING to reduce verbosity; override via MTG_AI_LOG_LEVEL env var.
log_level = os.getenv("MTG_AI_LOG_LEVEL", "WARNING").upper()
base_logger.setLevel(getattr(logging, log_level, logging.WARNING))

# Create formatters
formatter = CategoryFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(category)s] - %(message)s')

# Create handlers
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()

# Add formatter to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
base_logger.addHandler(file_handler)
base_logger.addHandler(console_handler)

# Use CategoryAdapter so existing call sites remain unchanged
logger = CategoryAdapter(base_logger, {})

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


# ---------------------------------------------------------------------------
#  SimpleBatcher – synchronous, single-threaded helper for prediction/training
# ---------------------------------------------------------------------------

class SimpleBatcher:
    """A minimal replacement for the previous ModelBatcher.

    • Runs prediction/training in the caller thread – no extra queues, locks.
    • Still keeps the public .predict() / .train() signatures so Java code
      does not need to change.
    """

    def __init__(self, model: MTGTransformerModel, device: torch.device, optimizer: torch.optim.Optimizer):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    # -------------------------- byte helpers ---------------------------
    @staticmethod
    def _bytes_to_tensor(buf: bytes, shape, np_dtype: str, torch_dtype, device=None):
        """Converts a byte buffer to a torch.Tensor, specifying the numpy and torch data types."""
        arr = np.frombuffer(buf, dtype=np_dtype).reshape(*shape)
        return torch.tensor(arr, dtype=torch_dtype, device=device)

    # ------------------------------ API -------------------------------
    def predict(self, sequences_bytes, masks_bytes, action_masks_bytes, batch_size, seq_len, d_model, max_actions):
        seq = self._bytes_to_tensor(
            sequences_bytes, (batch_size, seq_len, d_model), '<f4', torch.float32, device=self.device)
        mask = self._bytes_to_tensor(
            masks_bytes, (batch_size, seq_len), '<i4', torch.bool, device=self.device)

        # Convert action masks
        act_mask = self._bytes_to_tensor(
            action_masks_bytes, (batch_size, max_actions), '<i4', torch.bool, device=self.device)

        self.model.eval()
        with torch.no_grad():
            _, policy_probs, value_scores = self.model(seq, mask, act_mask)

        # Interleave policy and value scores for Java-side unpacking
        policy_np = policy_probs.cpu().numpy()
        value_np = value_scores.cpu().numpy()

        # Combine them into a new array of shape [batch_size, num_actions + 1]
        interleaved_results = np.concatenate((policy_np, value_np), axis=1)

        # Flatten to a single byte array in the format Java expects
        return interleaved_results.tobytes()

    def train(self, sequences_bytes, masks_bytes, policy_scores_bytes, discounted_returns_bytes,
              action_types_bytes, action_combos_bytes, batch_size, seq_len, d_model, max_actions):
        """Trains the model using discounted returns as value targets."""
        seq = self._bytes_to_tensor(
            sequences_bytes, (batch_size, seq_len, d_model), '<f4', torch.float32, device=self.device)
        mask = self._bytes_to_tensor(
            masks_bytes, (batch_size, seq_len), '<i4', torch.bool, device=self.device)

        # Target action indices (long)
        tgt_idx = self._bytes_to_tensor(
            policy_scores_bytes, (batch_size,), '<i4', torch.long, device=self.device)

        # Target for the value function is the discounted return calculated in Java
        reward_tensor = self._bytes_to_tensor(
            discounted_returns_bytes, (batch_size, 1), '<f4', torch.float32, device=self.device)

        # -------------- MICRO-BATCH GRADIENT ACCUMULATION ----------------
        MICRO_BATCHES = 4  # tune if needed; must divide batch_size
        if batch_size % MICRO_BATCHES != 0:
            MICRO_BATCHES = 1  # fallback – no accumulation

        micro_bs = batch_size // MICRO_BATCHES

        self.model.train()
        self.optimizer.zero_grad()

        for mb in range(MICRO_BATCHES):
            mb_slice = slice(mb * micro_bs, (mb + 1) * micro_bs)

            mb_seq = seq[mb_slice]
            mb_mask = mask[mb_slice]
            mb_tgt_idx = tgt_idx[mb_slice]
            mb_reward = reward_tensor[mb_slice]

            logits, pred_policy, pred_val = self.model(mb_seq, mb_mask)

            # --- losses (same as before) ---
            value_loss_coeff = 0.5
            loss_value = value_loss_coeff * F.mse_loss(pred_val, mb_reward)

            # pad/trim logits
            if logits.shape[1] != max_actions:
                if logits.shape[1] < max_actions:
                    pad = torch.zeros(micro_bs, max_actions -
                                      logits.shape[1], device=self.device)
                    logits_padded = torch.cat([logits, pad], dim=1)
                else:
                    logits_padded = logits[:, :max_actions]
            else:
                logits_padded = logits

            logits_padded = torch.nan_to_num(
                logits_padded, nan=0.0, posinf=20.0, neginf=-20.0)
            logits_padded = torch.clamp(logits_padded, -20.0, 20.0)

            log_probs = F.log_softmax(logits_padded, dim=-1)
            selected_log_probs = log_probs.gather(
                1, mb_tgt_idx.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                advantage = (mb_reward - pred_val).squeeze(1)
                advantage = torch.clamp(advantage, -1.0, 1.0)

            loss_policy = -(selected_log_probs * advantage).mean()

            entropy = -(pred_policy * torch.log(pred_policy + 1e-9)
                        ).sum(dim=-1).mean()
            entropy_coeff = -0.05
            loss_entropy = entropy_coeff * entropy

            loss = loss_policy + loss_value + loss_entropy

            # Scale loss by 1/M to keep gradient magnitude constant across accumulation
            loss = loss / MICRO_BATCHES

            loss.backward()

        # After accumulation – clip and step
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.model.max_grad_norm)
        self.optimizer.step()

        # Log training metrics
        logger.info(
            LogCategory.MODEL_TRAIN,
            "Train step — loss_value: %.4f, loss_policy: %.4f, loss_entropy: %.4f",
            loss_value.item(), loss_policy.item(), loss_entropy.item()
        )

        # -----------------------------------------------------
        #  High-precision logging of policy logits (first item)
        # -----------------------------------------------------
        try:
            sample_logits = logits[0].detach().cpu().numpy()
            logger.info(LogCategory.MODEL_TRAIN,
                        "Policy logits sample (precise): %s",
                        np.array2string(sample_logits, precision=8, suppress_small=False))
        except Exception as e:
            logger.warning(LogCategory.MODEL_TRAIN,
                           "Failed to log policy logits: %s", str(e))

        # Free unused GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(LogCategory.GPU_CLEANUP,
                        "Cleared CUDA cache after training step")

        return True


class PythonEntryPoint:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.batcher = None
        # Get model path from environment variable
        self.model_path = os.getenv('MTG_MODEL_PATH')
        # ---------------------------------------------------------------
        # Training step counter for periodic checkpointing
        # ---------------------------------------------------------------
        self.train_step_counter = 0  # counts successful trainFlat calls
        logger.info(
            LogCategory.GPU_MEMORY,
            "Using device: %s", self.device
        )
        if self.model_path:
            logger.info(LogCategory.GPU_MEMORY,
                        f"Model path configured: {self.model_path}")
            if os.path.exists(self.model_path):
                logger.info(LogCategory.GPU_MEMORY,
                            "Found existing model, loading...")
                self.loadModel(self.model_path)
            else:
                logger.info(LogCategory.GPU_MEMORY,
                            "No existing model found, will initialize new model")

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
                {'params': other_params, 'lr': 5e-5}
            ])
            self.batcher = SimpleBatcher(
                self.model, self.device, self.optimizer)

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

    def predictBatchFlat(self, sequences_bytes, masks_bytes, action_masks_bytes, batch_size, seq_len, d_model, max_actions):
        try:
            start_time = time.time()
            if self.model is None:
                raise RuntimeError("Model not initialized")

            # Log GPU memory before prediction
            log_gpu_memory()

            # Use the batcher for prediction
            result = self.batcher.predict(
                sequences_bytes, masks_bytes, action_masks_bytes, batch_size, seq_len, d_model, max_actions)

            # Log GPU memory after prediction
            log_gpu_memory()

            # Proactively clear unused GPU cache to avoid fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(LogCategory.GPU_CLEANUP,
                            "Cleared CUDA cache after prediction")

            total_time = time.time() - start_time
            logger.info(
                LogCategory.GPU_MEMORY, f"Total predictBatch operation took {total_time:.3f} seconds")

            # Ensure result is a single byte array
            if isinstance(result, list):
                result = b''.join(result)
            return result

        except Exception as e:
            logger.error(LogCategory.GPU_MEMORY,
                         f"Error in predictBatchFlat: {str(e)}")
            logger.error(
                LogCategory.GPU_MEMORY, f"Input shapes at error - batch_size: {batch_size}, seq_len: {seq_len}, d_model: {d_model}")
            raise

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
        """Score padded candidates for each state (policy over candidates + value)."""
        try:
            if self.model is None:
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

            seq_t = torch.tensor(seq, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
            tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device)
            cand_feat_t = torch.tensor(
                cand_feat, dtype=torch.float32, device=device)
            cand_ids_t = torch.tensor(cand_ids, dtype=torch.long, device=device)
            cand_mask_t = torch.tensor(cand_mask, dtype=torch.bool, device=device)

            self.model.eval()
            with torch.no_grad():
                probs, value = self.model.score_candidates(
                    seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t)

            probs_np = probs.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()

            out = np.concatenate((probs_np, value_np), axis=1)
            return out.astype('<f4').tobytes()

        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in scoreCandidatesFlat: %s", str(e))
            raise

    def trainCandidatesFlat(self,
                            sequences_bytes,
                            masks_bytes,
                            token_ids_bytes,
                            candidate_features_bytes,
                            candidate_ids_bytes,
                            candidate_mask_bytes,
                            chosen_index_bytes,
                            discounted_returns_bytes,
                            batch_size,
                            seq_len,
                            d_model,
                            max_candidates,
                            cand_feat_dim):
        """Train on padded candidate decision steps."""
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

            chosen = np.frombuffer(chosen_index_bytes, dtype='<i4').reshape(
                batch_size)
            returns = np.frombuffer(discounted_returns_bytes, dtype='<f4').reshape(
                batch_size, 1)

            seq_t = torch.tensor(seq, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
            tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device)
            cand_feat_t = torch.tensor(
                cand_feat, dtype=torch.float32, device=device)
            cand_ids_t = torch.tensor(cand_ids, dtype=torch.long, device=device)
            cand_mask_t = torch.tensor(cand_mask, dtype=torch.bool, device=device)
            chosen_t = torch.tensor(chosen, dtype=torch.long, device=device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            self.model.train()
            self.optimizer.zero_grad()

            probs, value = self.model.score_candidates(
                seq_t, mask_t, tok_t, cand_feat_t, cand_ids_t, cand_mask_t)

            # Policy loss: REINFORCE with baseline (advantage)
            log_probs = torch.log(probs + 1e-9)
            selected = log_probs.gather(
                1, chosen_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                advantage = (returns_t - value).squeeze(1)
                advantage = torch.clamp(advantage, -1.0, 1.0)

            loss_policy = -(selected * advantage).mean()

            # Value loss
            loss_value = 0.5 * F.mse_loss(value, returns_t)

            # Entropy bonus (encourage exploration)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss_entropy = -0.01 * entropy

            loss = loss_policy + loss_value + loss_entropy
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.model.max_grad_norm)
            self.optimizer.step()

            logger.info(LogCategory.MODEL_TRAIN,
                        "trainCandidatesFlat — loss=%.4f policy=%.4f value=%.4f ent=%.4f",
                        loss.item(), loss_policy.item(), loss_value.item(), entropy.item())
            return True

        except Exception as e:
            logger.error(LogCategory.SYSTEM_ERROR,
                         "Error in trainCandidatesFlat: %s", str(e))
            raise

    def trainFlat(self, sequences_bytes, masks_bytes, policy_scores_bytes, discounted_returns_bytes,
                  action_types_bytes, action_combos_bytes, batch_size, seq_len, d_model, max_actions):
        """Batch training using direct byte array conversion"""
        # Trace entry so Java side can verify training invocations
        logger.info(LogCategory.MODEL_TRAIN,
                    "trainFlat called — batch_size=%d, max_actions=%d",
                    batch_size, max_actions)

        # Skip update if no actions provided (e.g., bookkeeping rows)
        if max_actions <= 0:
            logger.warning(LogCategory.MODEL_TRAIN,
                           "max_actions == 0, skipping gradient step")
            return True

        try:
            start_time = time.time()
            if self.model is None:
                raise RuntimeError("Model not initialized")

            # Log GPU memory before training
            log_gpu_memory()

            # Use the batcher for training - note the signature change
            result = self.batcher.train(sequences_bytes, masks_bytes, policy_scores_bytes, discounted_returns_bytes,
                                        action_types_bytes, action_combos_bytes, batch_size, seq_len, d_model, max_actions)

            # Log GPU memory after training
            log_gpu_memory()

            # Free unused GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(LogCategory.GPU_CLEANUP,
                            "Cleared CUDA cache after training step")

            total_time = time.time() - start_time
            logger.info(
                LogCategory.GPU_MEMORY, f"Total trainFlat operation took {total_time:.3f} seconds")

            # -------------------------------------------------------
            #  Increment counter & checkpoint every 100 updates
            # -------------------------------------------------------
            self.train_step_counter += 1
            if self.train_step_counter % 100 == 0:
                ckpt_path = self.model_path or os.path.join(
                    TEMP_DIR, f"checkpoint_step_{self.train_step_counter}.pt")
                try:
                    self.saveModel(ckpt_path)
                    logger.info(LogCategory.MODEL_SAVE,
                                "Checkpoint saved at step %d -> %s",
                                self.train_step_counter, ckpt_path)
                except Exception as e:
                    logger.error(LogCategory.MODEL_SAVE,
                                 "Failed to save checkpoint at step %d: %s",
                                 self.train_step_counter, str(e))

            return result

        except Exception as e:
            logger.error(LogCategory.GPU_MEMORY,
                         f"Error in trainFlat: {str(e)}")
            logger.error(
                LogCategory.GPU_MEMORY, f"Input shapes at error - batch_size: {batch_size}, seq_len: {seq_len}, d_model: {d_model}")
            raise

    def saveModel(self, path):
        """Save model state"""
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")
            self.model.save(path)
            logger.info(
                LogCategory.GPU_MEMORY,
                "Model saved to %s", path
            )
        except Exception as e:
            logger.error(LogCategory.GPU_MEMORY,
                         f"Error saving model: {str(e)}")
            raise

    def loadModel(self, path):
        """Load model state"""
        try:
            if self.model is None:
                self.initializeModel()
            self.model.load(path)
            logger.info(
                LogCategory.GPU_MEMORY,
                "Model loaded from %s", path
            )
        except Exception as e:
            logger.error(LogCategory.GPU_MEMORY,
                         f"Error loading model: {str(e)}")
            raise

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
    try:
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
                    python_parameters=PythonParameters(),
                    python_server_entry_point=PythonEntryPoint()
                )
                logger.info(LogCategory.SYSTEM_INIT,
                            f"Python ML service started and ready for connections")
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
                                python_parameters=PythonParameters(),
                                python_server_entry_point=PythonEntryPoint()
                            )
                            logger.info(LogCategory.SYSTEM_INIT,
                                        "Successfully reconnected to Py4J gateway")
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
    finally:
        cleanup_temp_files()
        if gateway is not None:
            try:
                gateway.shutdown()
            except Exception as e:
                logger.error(LogCategory.SYSTEM_ERROR,
                             "Error during gateway shutdown: %s", str(e))
        logger.info(LogCategory.SYSTEM_CLEANUP, "Python ML service stopped")
