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
    GPU_CLEANUP = True

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
    def _bytes_to_tensor(buf: bytes, shape, dtype=torch.float32, device=None, big_endian=False):
        np_dtype = ">f4" if big_endian else "<f4"
        arr = np.frombuffer(buf, dtype=np_dtype)

        # Cast according to requested dtype
        if dtype == torch.bool:
            arr = arr.astype(np.bool_)
        elif dtype == torch.long:
            arr = arr.astype(np.int64)
        else:
            arr = arr.astype(np.float32)

        arr = arr.reshape(*shape)
        return torch.tensor(arr, dtype=dtype, device=device)

    # ------------------------------ API -------------------------------
    def predict(self, sequences_bytes, masks_bytes, batch_size, seq_len, d_model):
        seq = self._bytes_to_tensor(
            sequences_bytes, (batch_size, seq_len, d_model), device=self.device)
        mask = self._bytes_to_tensor(
            masks_bytes, (batch_size, seq_len), dtype=torch.bool, device=self.device)

        self.model.eval()
        with torch.no_grad():
            _, policy_probs, value_scores = self.model(seq, mask)

        # Flatten to byte array: policy_probs then value_scores
        return policy_probs.cpu().numpy().tobytes() + value_scores.cpu().numpy().tobytes()

    def train(self, sequences_bytes, masks_bytes, policy_scores_bytes, value_scores_bytes,
              action_types_bytes, action_combos_bytes, batch_size, seq_len, d_model, max_actions, reward):
        """Very simple supervised-like update: minimise MSE on value head.
        Real PPO logic was in old ModelBatcher; keeping a placeholder so the
        Java side doesn't break. Returns True synchronously.
        """
        seq = self._bytes_to_tensor(
            sequences_bytes, (batch_size, seq_len, d_model), device=self.device)
        mask = self._bytes_to_tensor(
            masks_bytes, (batch_size, seq_len), dtype=torch.bool, device=self.device)

        # Target value tensor (float32)
        tgt_val = self._bytes_to_tensor(
            value_scores_bytes, (batch_size, 1), device=self.device)

        # Target action indices (long)
        tgt_idx = self._bytes_to_tensor(
            policy_scores_bytes, (batch_size,), dtype=torch.long, device=self.device)

        self.model.train()
        self.optimizer.zero_grad()
        logits, pred_policy, pred_val = self.model(seq, mask)

        # Value regression loss
        loss_value = F.mse_loss(pred_val, tgt_val)

        # Policy supervised loss (cross entropy)
        # logits shape: [B, num_actions] ; tgt_idx shape: [B]
        if logits.shape[1] != max_actions:
            # Pad/trim logits to max_actions
            if logits.shape[1] < max_actions:
                pad = torch.zeros(batch_size, max_actions -
                                  logits.shape[1], device=self.device)
                logits_padded = torch.cat([logits, pad], dim=1)
            else:
                logits_padded = logits[:, :max_actions]
        else:
            logits_padded = logits

        # Sanity-check target indices and clamp to valid range
        with torch.no_grad():
            bad_mask = (tgt_idx < 0) | (tgt_idx >= max_actions)
            if bad_mask.any():
                logger.warning(LogCategory.MODEL_TRAIN,
                               "Found %d out-of-range target indices; clamping.",
                               bad_mask.sum().item())
                tgt_idx = torch.clamp(tgt_idx, 0, max_actions - 1)

        # Clamp logits to a safe numeric interval and remove NaN/Inf again
        logits_padded = torch.nan_to_num(
            logits_padded, nan=0.0, posinf=20.0, neginf=-20.0)
        logits_padded = torch.clamp(logits_padded, -20.0, 20.0)

        loss_policy = F.cross_entropy(logits_padded, tgt_idx)

        loss = loss_value + loss_policy

        # Compute policy entropy for monitoring (with small epsilon for log)
        with torch.no_grad():
            entropy = -(pred_policy * torch.log(pred_policy + 1e-9)
                        ).sum(dim=-1).mean()

        loss.backward()
        # Gradient clipping to mitigate exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.model.max_grad_norm)
        self.optimizer.step()

        # Log training metrics
        logger.info(
            LogCategory.MODEL_TRAIN,
            "Train step — loss_value: %.4f, loss_policy: %.4f, entropy: %.4f",
            loss_value.item(), loss_policy.item(), entropy.item()
        )

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
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
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

    def predictBatchFlat(self, sequences_bytes, masks_bytes, batch_size, seq_len, d_model):
        try:
            start_time = time.time()
            if self.model is None:
                raise RuntimeError("Model not initialized")

            # Log GPU memory before prediction
            log_gpu_memory()

            # Use the batcher for prediction
            result = self.batcher.predict(
                sequences_bytes, masks_bytes, batch_size, seq_len, d_model)

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

    def trainFlat(self, sequences_bytes, masks_bytes, policy_scores_bytes, value_scores_bytes,
                  action_types_bytes, action_combos_bytes, batch_size, seq_len, d_model, max_actions, reward):
        """Batch training using direct byte array conversion"""
        # Trace entry so Java side can verify training invocations
        logger.info(LogCategory.MODEL_TRAIN,
                    "trainFlat called — batch_size=%d, max_actions=%d, reward=%.3f",
                    batch_size, max_actions, reward)

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

            # Use the batcher for training
            result = self.batcher.train(sequences_bytes, masks_bytes, policy_scores_bytes, value_scores_bytes,
                                        action_types_bytes, action_combos_bytes, batch_size, seq_len, d_model, max_actions, reward)

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
