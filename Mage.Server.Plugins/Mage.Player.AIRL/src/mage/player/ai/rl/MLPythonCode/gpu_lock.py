import os
import sys
import time
import tempfile
import threading
from logging_utils import logger, LogCategory

# Windows-compatible file locking
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl


class GPULock:
    """
    File-based inter-process lock for GPU access.
    Only one Python process (learner or inference worker) can hold GPU at a time.
    Cross-platform compatible (Windows + Unix).
    """
    
    def __init__(self, lock_file=None):
        if lock_file is None:
            lock_dir = os.path.join(tempfile.gettempdir(), 'mage_gpu_locks')
            os.makedirs(lock_dir, exist_ok=True)
            py4j_port = os.environ.get('PY4J_PORT', '')
            base_port = os.environ.get('PY4J_BASE_PORT', '')
            lock_id = base_port or py4j_port or 'global'
            lock_file = os.path.join(lock_dir, f'gpu_{lock_id}.lock')
        
        self.lock_file = lock_file
        self.file_handle = None
        self.is_locked = False
        self.is_windows = sys.platform == 'win32'
        self._local_mutex = threading.Lock()
        self._local_ref_count = 0
    
    def acquire(self, timeout=None, process_name="unknown"):
        """
        Acquire GPU lock. Blocks until lock is available or timeout expires.
        
        Args:
            timeout: Max seconds to wait (None = wait forever)
            process_name: Name for logging (e.g., "learner", "infer_worker_0")
        
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()

        while True:
            fh = None
            try:
                with self._local_mutex:
                    # Re-entrant within this process: if already held, just bump ref count.
                    if self.is_locked and self.file_handle is not None:
                        self._local_ref_count += 1
                        logger.debug(
                            LogCategory.GPU_MEMORY,
                            "GPU lock re-entered by %s (pid=%d, ref=%d)",
                            process_name, os.getpid(), self._local_ref_count
                        )
                        return True

                    # Open file for writing (creates if doesn't exist)
                    fh = open(self.lock_file, 'w')

                    if self.is_windows:
                        # Windows: Lock first byte using msvcrt
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        # Unix: Use fcntl
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Write process info for debugging
                    fh.write(f"{process_name}:{os.getpid()}:{time.time()}\n")
                    fh.flush()

                    self.file_handle = fh
                    self.is_locked = True
                    self._local_ref_count = 1

                logger.debug(LogCategory.GPU_MEMORY, 
                            f"GPU lock acquired by {process_name} (pid={os.getpid()})")
                return True
                
            except (IOError, OSError, ValueError) as e:
                # Lock is held by another process
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.warning(LogCategory.GPU_MEMORY,
                                      f"GPU lock timeout for {process_name} after {elapsed:.1f}s")
                        return False
                
                # Wait a bit before retry
                time.sleep(0.01)  # 10ms
    
    def release(self, process_name="unknown"):
        """Release GPU lock."""
        with self._local_mutex:
            if not self.is_locked or self.file_handle is None:
                return
            if self._local_ref_count > 1:
                self._local_ref_count -= 1
                logger.debug(
                    LogCategory.GPU_MEMORY,
                    "GPU lock release (decrement) by %s (pid=%d, ref=%d)",
                    process_name, os.getpid(), self._local_ref_count
                )
                return

            fh = self.file_handle
            self._local_ref_count = 0
            try:
                if self.is_windows:
                    # Windows: Unlock first byte
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    # Unix: Use fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

                fh.close()
                logger.debug(LogCategory.GPU_MEMORY,
                            f"GPU lock released by {process_name} (pid={os.getpid()})")
            except Exception as e:
                logger.warning(LogCategory.GPU_MEMORY,
                              f"Error releasing GPU lock: {e}")
            finally:
                self.file_handle = None
                self.is_locked = False
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
