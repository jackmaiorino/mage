import logging
import os
import tempfile

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


class CategoryFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'category'):
            record.category = LogCategory.DEFAULT
        return super().format(record)


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


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger():
    """Configure and return the global logger."""
    base_logger = logging.getLogger('mtg_ai')

    # Default to WARNING to reduce verbosity; override via MTG_AI_LOG_LEVEL env var.
    log_level = os.getenv("MTG_AI_LOG_LEVEL", "WARNING").upper()
    base_logger.setLevel(getattr(logging, log_level, logging.WARNING))

    # Create formatters
    formatter = CategoryFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(category)s] - %(message)s')

    # Create handlers with immediate flushing
    file_handler = FlushFileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Add formatter to handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    base_logger.addHandler(file_handler)
    base_logger.addHandler(console_handler)

    # Use CategoryAdapter so existing call sites remain unchanged
    return CategoryAdapter(base_logger, {})


# Global logger instance
logger = setup_logger()
