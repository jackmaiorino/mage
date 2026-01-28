import os
import time
import torch
from logging_utils import logger, LogCategory


class ModelPersistence:
    """Handles model save/load operations with atomic write support."""

    def __init__(self):
        self.model_path = os.getenv('MTG_MODEL_PATH')
        self.model_latest_path = os.getenv('MODEL_LATEST_PATH', '').strip()
        self._latest_loaded_mtime = 0.0
        self._did_initial_load = False

    def save_model(self, model, path):
        """Save model state."""
        try:
            if model is None:
                raise RuntimeError("Model not initialized")
            model.save(path)
            logger.info(LogCategory.GPU_MEMORY, "Model saved to %s", path)
        except Exception as e:
            logger.error(LogCategory.GPU_MEMORY, "Error saving model: %s", str(e))
            raise

    def load_model(self, model, path):
        """Load model state."""
        try:
            model.load(path)
            logger.info(LogCategory.GPU_MEMORY, "Model loaded from %s", path)
        except Exception as e:
            logger.error(LogCategory.GPU_MEMORY, "Error loading model: %s", str(e))
            raise

    def save_latest_model_atomic(self, model, path=None):
        """
        Save a 'latest weights' file atomically (tmp -> replace) for inference workers to reload.
        """
        p = (path or self.model_latest_path or "").strip()
        if not p:
            return False
        tmp = p + ".tmp"
        self.save_model(model, tmp)
        # Windows can fail replace() if another process is reading the target.
        for i in range(20):
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                time.sleep(0.05 * (i + 1))
        else:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise
        try:
            self._latest_loaded_mtime = float(os.path.getmtime(p))
        except Exception:
            pass
        return True

    def reload_latest_model_if_newer(self, model, path=None):
        """
        If the latest weights file exists and is newer than what we have loaded, reload it.
        Returns True if reloaded.
        """
        p = (path or self.model_latest_path or "").strip()
        if not p:
            return False
        try:
            mtime = float(os.path.getmtime(p))
        except Exception:
            return False
        if mtime <= float(self._latest_loaded_mtime):
            return False
        self.load_model(model, p)
        self._latest_loaded_mtime = mtime
        
        # Aggressively clear CUDA cache after model reload to prevent memory accumulation
        # Model reloads create temporary allocations that fragment CUDA memory over time
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return True
