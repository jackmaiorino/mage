import torch
import os
import time
import gc
from logging_utils import logger, LogCategory


class CUDAManager:
    """Manages CUDA memory, OOM handling, and auto-batching."""

    def __init__(self, py_role="learner"):
        self.py_role = py_role

        # OOM tracking
        self._oom_count = 0

        # Auto-batching config
        self.auto_batch_enable = bool(int(os.getenv('AUTO_BATCH_ENABLE', '1')))
        self.auto_avoid_paging = bool(int(os.getenv('AUTO_AVOID_PAGING', '1')))
        self.auto_target_used_frac = float(
            os.getenv('AUTO_TARGET_USED_FRAC', '0.85'))
        self.auto_min_free_mb = float(os.getenv('AUTO_MIN_FREE_MB', '1024'))
        self.auto_mem_ema_alpha = float(os.getenv('AUTO_MEM_EMA_ALPHA', '0.2'))

        # Learner VRAM guard. This is intentionally preventive: on Windows/WDDM
        # overcommitted CUDA allocations can spill into shared RAM and make a
        # batch look "hung" instead of failing quickly. The guard splits or
        # waits before large training allocations cross that line.
        self.train_vram_guard_enable = bool(int(os.getenv('TRAIN_VRAM_GUARD_ENABLE', '1')))
        self.train_min_free_mb = float(os.getenv(
            'TRAIN_MIN_FREE_VRAM_MB',
            os.getenv('AUTO_MIN_FREE_MB', '1536'),
        ))
        self.train_target_used_frac = float(os.getenv(
            'TRAIN_MAX_USED_VRAM_FRAC',
            os.getenv('AUTO_TARGET_USED_FRAC', '0.85'),
        ))
        self.train_vram_guard_retries = max(0, int(os.getenv('TRAIN_VRAM_GUARD_RETRIES', '2')))
        self.train_vram_guard_retry_ms = max(0, int(os.getenv('TRAIN_VRAM_GUARD_RETRY_MS', '250')))
        self.train_vram_guard_max_wait_ms = max(0, int(os.getenv('TRAIN_VRAM_GUARD_MAX_WAIT_MS', '120000')))
        train_step_init = os.getenv('AUTO_TRAIN_MB_PER_STEP_INIT', '').strip()

        # Memory tracking
        self._infer_mb_per_sample = None
        self._train_mb_per_step = None
        if train_step_init:
            try:
                init_value = float(train_step_init)
                if init_value > 0.0:
                    self._train_mb_per_step = init_value
            except Exception:
                self._train_mb_per_step = None
        self._autobatch_last_free_mb = 0.0
        self._autobatch_last_total_mb = 0.0
        self._autobatch_last_desired_free_mb = 0.0
        self._train_guard_waits = 0
        self._train_guard_wait_ms = 0.0
        self._train_guard_flushes = 0
        self._train_guard_blocks = 0
        self._train_guard_last_need_mb = 0.0
        self._train_guard_last_estimated_extra_mb = 0.0

    def is_cuda_oom(self, e: Exception) -> bool:
        try:
            msg = str(e).lower()
            is_oom = ("cuda out of memory" in msg) or (
                "cublas_status_alloc_failed" in msg)
            if is_oom:
                self._oom_count += 1
                logger.warning(
                    LogCategory.GPU_MEMORY, "CUDA OOM detected (total: %d): %s", self._oom_count, str(e)[:200])
            return is_oom
        except Exception:
            return False

    def get_oom_count(self) -> int:
        """Get total number of CUDA OOM errors encountered."""
        return self._oom_count

    def reset_oom_count(self):
        """Reset OOM counter."""
        self._oom_count = 0

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
        frac_headroom = max(
            0.0, (1.0 - float(self.auto_target_used_frac))) * float(total_mb)
        desired = max(float(self.auto_min_free_mb), float(frac_headroom))
        self._autobatch_last_desired_free_mb = float(desired)
        return desired

    def train_desired_free_mb(self):
        info = self.cuda_mem_info_mb()
        if info is None:
            return None
        _free_mb, total_mb = info
        target_used_frac = max(0.05, min(0.98, float(self.train_target_used_frac)))
        frac_headroom = max(0.0, (1.0 - target_used_frac)) * float(total_mb)
        desired = max(float(self.train_min_free_mb), float(frac_headroom))
        self._autobatch_last_desired_free_mb = float(desired)
        return desired

    def train_has_headroom(self, estimated_extra_mb: float, wait: bool = False, tag: str = "train") -> bool:
        if not self.train_vram_guard_enable:
            return True
        if not torch.cuda.is_available():
            return True

        need_extra = max(0.0, float(estimated_extra_mb or 0.0))
        self._train_guard_last_estimated_extra_mb = need_extra
        started = time.monotonic()
        attempts = max(1, int(self.train_vram_guard_retries) + 1)

        while True:
            for attempt in range(attempts):
                info = self.cuda_mem_info_mb()
                desired = self.train_desired_free_mb()
                if info is None or desired is None:
                    return True
                free_mb, _total_mb = info
                need_mb = float(desired) + need_extra
                self._train_guard_last_need_mb = need_mb
                if float(free_mb) >= need_mb:
                    return True

                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._train_guard_flushes += 1
                except Exception:
                    pass

                info = self.cuda_mem_info_mb()
                if info is not None:
                    free_mb, _total_mb = info
                    if float(free_mb) >= need_mb:
                        return True

                if attempt < (attempts - 1):
                    sleep_s = float(self.train_vram_guard_retry_ms) / 1000.0
                    if sleep_s > 0.0:
                        self._train_guard_waits += 1
                        time.sleep(sleep_s)
                        self._train_guard_wait_ms += sleep_s * 1000.0

            if not wait:
                self._train_guard_blocks += 1
                return False

            elapsed_ms = (time.monotonic() - started) * 1000.0
            if self.train_vram_guard_max_wait_ms > 0 and elapsed_ms >= self.train_vram_guard_max_wait_ms:
                self._train_guard_blocks += 1
                logger.warning(
                    LogCategory.GPU_MEMORY,
                    "Train VRAM guard timed out tag=%s free=%.0fMB need=%.0fMB extra=%.0fMB waited=%.0fms",
                    str(tag),
                    float(self._autobatch_last_free_mb),
                    float(self._train_guard_last_need_mb),
                    float(need_extra),
                    float(elapsed_ms),
                )
                return False

            sleep_s = max(0.05, float(self.train_vram_guard_retry_ms) / 1000.0)
            self._train_guard_waits += 1
            time.sleep(sleep_s)
            self._train_guard_wait_ms += sleep_s * 1000.0

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
            self._infer_mb_per_sample = per if cur is None else (
                a * per + (1.0 - a) * float(cur))
        elif kind == "train":
            cur = self._train_mb_per_step
            self._train_mb_per_step = per if cur is None else (
                a * per + (1.0 - a) * float(cur))

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
            'train_guard_waits': self._train_guard_waits,
            'train_guard_wait_ms': self._train_guard_wait_ms,
            'train_guard_flushes': self._train_guard_flushes,
            'train_guard_blocks': self._train_guard_blocks,
            'train_guard_last_need_mb': self._train_guard_last_need_mb,
            'train_guard_last_estimated_extra_mb': self._train_guard_last_estimated_extra_mb,
        }
