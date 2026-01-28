import torch
import os
from logging_utils import logger, LogCategory


class CUDAManager:
    """Manages CUDA memory, OOM handling, and auto-batching."""

    def __init__(self, py_role="learner"):
        self.py_role = py_role
        
        # Auto-batching config
        self.auto_batch_enable = bool(int(os.getenv('AUTO_BATCH_ENABLE', '1')))
        self.auto_avoid_paging = bool(int(os.getenv('AUTO_AVOID_PAGING', '1')))
        self.auto_target_used_frac = float(os.getenv('AUTO_TARGET_USED_FRAC', '0.85'))
        self.auto_min_free_mb = float(os.getenv('AUTO_MIN_FREE_MB', '1024'))
        self.auto_mem_ema_alpha = float(os.getenv('AUTO_MEM_EMA_ALPHA', '0.2'))
        
        # Memory tracking
        self._infer_mb_per_sample = None
        self._train_mb_per_step = None
        self._autobatch_last_free_mb = 0.0
        self._autobatch_last_total_mb = 0.0
        self._autobatch_last_desired_free_mb = 0.0

    def is_cuda_oom(self, e: Exception) -> bool:
        try:
            msg = str(e).lower()
            return ("cuda out of memory" in msg) or ("cublas_status_alloc_failed" in msg)
        except Exception:
            return False

    def cuda_cleanup_after_oom(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def cuda_mem_info_mb(self):
        """Return (free_mb, total_mb) using torch.cuda.mem_get_info if available."""
        if not torch.cuda.is_available():
            return None
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_mb = float(free_b) / (1024.0 * 1024.0)
            total_mb = float(total_b) / (1024.0 * 1024.0)
            self._autobatch_last_free_mb = float(free_mb)
            self._autobatch_last_total_mb = float(total_mb)
            return free_mb, total_mb
        except Exception:
            return None

    def desired_free_mb(self):
        info = self.cuda_mem_info_mb()
        if info is None:
            return None
        free_mb, total_mb = info
        frac_headroom = max(0.0, (1.0 - float(self.auto_target_used_frac))) * float(total_mb)
        desired = max(float(self.auto_min_free_mb), float(frac_headroom))
        self._autobatch_last_desired_free_mb = float(desired)
        return desired

    def should_split_for_paging(self, estimated_extra_mb: float):
        if not self.auto_batch_enable or not self.auto_avoid_paging:
            return False
        info = self.cuda_mem_info_mb()
        if info is None:
            return False
        free_mb, total_mb = info
        desired_free = self.desired_free_mb()
        if desired_free is None:
            return False
        need = max(0.0, float(estimated_extra_mb))
        return float(free_mb) < (float(desired_free) + need)

    def update_mem_ema(self, kind: str, extra_mb: float, n: int):
        if not torch.cuda.is_available():
            return
        if n <= 0:
            return
        kind = (kind or "").strip().lower()
        per = max(0.0, float(extra_mb)) / float(n)
        a = float(self.auto_mem_ema_alpha)
        a = 0.2 if not (0.0 < a <= 1.0) else a
        if kind == "infer":
            cur = self._infer_mb_per_sample
            self._infer_mb_per_sample = per if cur is None else (a * per + (1.0 - a) * float(cur))
        elif kind == "train":
            cur = self._train_mb_per_step
            self._train_mb_per_step = per if cur is None else (a * per + (1.0 - a) * float(cur))

    def measure_peak_extra_mb(self, fn):
        """
        Run fn() and return (result, peak_extra_mb). Uses reserved-memory peaks to be robust
        to caching; 'extra' is computed as peak_reserved - base_reserved.
        """
        if not torch.cuda.is_available():
            return fn(), 0.0
        try:
            base_reserved = float(torch.cuda.memory_reserved())
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            out = fn()
            try:
                peak_reserved = float(torch.cuda.max_memory_reserved())
            except Exception:
                peak_reserved = float(torch.cuda.memory_reserved())
            extra_b = max(0.0, peak_reserved - base_reserved)
            return out, float(extra_b) / (1024.0 * 1024.0)
        except Exception:
            return fn(), 0.0

    def get_auto_batch_metrics(self):
        return {
            'infer_mb_per_sample': self._infer_mb_per_sample,
            'train_mb_per_step': self._train_mb_per_step,
            'last_free_mb': self._autobatch_last_free_mb,
            'last_total_mb': self._autobatch_last_total_mb,
            'last_desired_free_mb': self._autobatch_last_desired_free_mb,
        }
