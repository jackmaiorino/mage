import os
import sys
import time
import tempfile
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
            lock_file = os.path.join(lock_dir, 'gpu.lock')
        
        self.lock_file = lock_file
        self.file_handle = None
        self.is_locked = False
        self.is_windows = sys.platform == 'win32'
    
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
            try:
                # Open file for writing (creates if doesn't exist)
                self.file_handle = open(self.lock_file, 'w')
                
                if self.is_windows:
                    # Windows: Lock first byte using msvcrt
                    msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Unix: Use fcntl
                    fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Write process info for debugging
                self.file_handle.write(f"{process_name}:{os.getpid()}:{time.time()}\n")
                self.file_handle.flush()
                
                self.is_locked = True
                logger.debug(LogCategory.GPU_MEMORY, 
                            f"GPU lock acquired by {process_name} (pid={os.getpid()})")
                return True
                
            except (IOError, OSError) as e:
                # Lock is held by another process
                if self.file_handle:
                    self.file_handle.close()
                    self.file_handle = None
                
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
        if not self.is_locked or self.file_handle is None:
            return
        
        try:
            if self.is_windows:
                # Windows: Unlock first byte
                msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                # Unix: Use fcntl
                fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
            
            self.file_handle.close()
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
