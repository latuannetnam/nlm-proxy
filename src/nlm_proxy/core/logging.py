"""Centralized logging configuration.

All modules should use get_logger() to obtain a logger instance.
Call setup_logging() once at application startup.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from nlm_proxy.core.config import get_logging_settings

_initialized = False


def setup_logging(debug: bool = False) -> logging.Logger:
    """Initialize logging from .env file settings.

    This should be called once at application startup. Subsequent calls
    are safe but will be no-ops unless reset_logging() is called first.

    Args:
        debug: If True, override log level to DEBUG regardless of config

    Returns:
        The root nlm_proxy logger
    """
    global _initialized

    if _initialized:
        return logging.getLogger("nlm_proxy")

    settings = get_logging_settings()

    # Create root logger for the package
    root_logger = logging.getLogger("nlm_proxy")

    # Determine log level
    if debug:
        level = logging.DEBUG
    else:
        level = getattr(logging, settings.level.upper(), logging.INFO)

    root_logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(settings.format)

    # Console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler (if path configured and not empty)
    if settings.file and settings.file.strip():
        try:
            log_path = Path(os.path.expanduser(settings.file))
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=settings.max_size,
                backupCount=settings.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # Log to console if file handler fails
            root_logger.warning(f"Could not create log file handler: {e}")

    # Prevent propagation to root logger to avoid duplicate logs
    root_logger.propagate = False

    _initialized = True

    root_logger.debug(
        f"Logging initialized: level={settings.level}, "
        f"file={settings.file or 'disabled'}"
    )

    return root_logger


def reset_logging() -> None:
    """Reset logging state to allow re-initialization.

    Primarily useful for testing.
    """
    global _initialized
    _initialized = False

    # Clear handlers from the root logger
    root_logger = logging.getLogger("nlm_proxy")
    root_logger.handlers.clear()


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the nlm_proxy namespace.

    Args:
        name: Logger name. Handles various input formats:
            - Full path like "nlm_proxy.mcp.server" -> used as-is
            - Module __name__ like "nlm_proxy.core.client" -> used as-is
            - Short name like "api" -> becomes "nlm_proxy.api"

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)  # Recommended
        logger = get_logger("api")     # Short form
    """
    if name.startswith("nlm_proxy."):
        return logging.getLogger(name)
    elif "nlm_proxy" in name:
        # Handle edge cases where nlm_proxy is in the name but not at start
        return logging.getLogger(name)
    else:
        return logging.getLogger(f"nlm_proxy.{name}")
