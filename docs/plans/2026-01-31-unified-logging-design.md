# Unified Logging System Design

**Date:** 2026-01-31
**Status:** Pending approval

## Overview

Refactor the logging system so all code uses the same logging logic, with configuration via `.env` file using `pydantic-settings`.

## Current State

| File | Logger Name | Problem |
|------|-------------|---------|
| `mcp/server.py` | `nlm_proxy.mcp` | Custom config in `main()` |
| `openai/server.py` | `__name__` | Uses `logging.basicConfig()` |
| `openai/session.py` | `__name__` | No configuration |
| `core/client.py` | `notebooklm_mcp.api` | Inconsistent name, hardcoded WARNING |

**Issues:**
- Inconsistent logger names
- Duplicate configuration in each server
- No centralized control
- No environment file support

## Design

### Logger Hierarchy

```
nlm_proxy              (root - all logs flow here)
├── nlm_proxy.mcp      (MCP server)
├── nlm_proxy.openai   (OpenAI proxy)
├── nlm_proxy.api      (API client)
└── nlm_proxy.session  (Session store)
```

### Configuration Settings

Using `pydantic-settings` for type-safe configuration:

```python
# src/nlm_proxy/core/config.py
from pydantic_settings import BaseSettings

class LoggingSettings(BaseSettings):
    level: str = "INFO"
    file: str = "~/.nlm-proxy/logs/nlm-proxy.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_size: int = 10485760  # 10 MB
    backup_count: int = 5

    model_config = {
        "env_prefix": "NLM_PROXY_LOG_",
        "env_file": ["~/.nlm-proxy/.env", ".env"],
        "extra": "ignore",
    }

class Settings(BaseSettings):
    """Main settings container. Extensible for future config."""

    model_config = {
        "env_file": ["~/.nlm-proxy/.env", ".env"],
        "extra": "ignore",
    }

# Singleton instance
_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

### Logging Module

```python
# src/nlm_proxy/core/logging.py
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from nlm_proxy.core.config import LoggingSettings

_initialized = False

def setup_logging(debug: bool = False) -> logging.Logger:
    """Initialize logging from .env file. Call once at startup.

    Args:
        debug: If True, override log level to DEBUG

    Returns:
        The root nlm_proxy logger
    """
    global _initialized

    if _initialized:
        return logging.getLogger("nlm_proxy")

    settings = LoggingSettings()

    # Create root logger
    root_logger = logging.getLogger("nlm_proxy")
    level = logging.DEBUG if debug else getattr(logging, settings.level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(settings.format)

    # Console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if path configured)
    if settings.file:
        log_path = Path(os.path.expanduser(settings.file))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=settings.max_size,
            backupCount=settings.backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    _initialized = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under nlm_proxy namespace.

    Args:
        name: Logger name. If not starting with 'nlm_proxy.',
              extracts module name and prefixes it.

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)  # "nlm_proxy.mcp.server" -> "nlm_proxy.mcp.server"
        logger = get_logger("api")     # -> "nlm_proxy.api"
    """
    if name.startswith("nlm_proxy."):
        return logging.getLogger(name)
    elif "nlm_proxy" in name:
        # Handle __name__ like "nlm_proxy.core.client"
        return logging.getLogger(name)
    else:
        return logging.getLogger(f"nlm_proxy.{name}")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NLM_PROXY_LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `NLM_PROXY_LOG_FILE` | `~/.nlm-proxy/logs/nlm-proxy.log` | Log file path (empty to disable) |
| `NLM_PROXY_LOG_FORMAT` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Python logging format |
| `NLM_PROXY_LOG_MAX_SIZE` | `10485760` | Max file size before rotation (10 MB) |
| `NLM_PROXY_LOG_BACKUP_COUNT` | `5` | Number of rotated files to keep |

### Example `.env` File

```bash
# ~/.nlm-proxy/.env

# ============================================
# NLM Proxy Configuration
# ============================================

# --- Logging ---
NLM_PROXY_LOG_LEVEL=INFO
NLM_PROXY_LOG_FILE=~/.nlm-proxy/logs/nlm-proxy.log
NLM_PROXY_LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
NLM_PROXY_LOG_MAX_SIZE=10485760
NLM_PROXY_LOG_BACKUP_COUNT=5
```

## Implementation Plan

### Step 1: Create config module
- Create `src/nlm_proxy/core/config.py` with `LoggingSettings` and `Settings` classes

### Step 2: Create logging module
- Create `src/nlm_proxy/core/logging.py` with `setup_logging()` and `get_logger()`

### Step 3: Update core exports
- Modify `src/nlm_proxy/core/__init__.py` to export new functions

### Step 4: Update CLI
- Modify `src/nlm_proxy/cli.py` to call `setup_logging(debug=args.debug)` at startup

### Step 5: Migrate MCP server
- Modify `src/nlm_proxy/mcp/server.py`:
  - Remove custom logging configuration from `main()`
  - Replace `mcp_logger = logging.getLogger("nlm_proxy.mcp")` with `get_logger(__name__)`
  - Remove the duplicate handler setup

### Step 6: Migrate OpenAI server
- Modify `src/nlm_proxy/openai/server.py`:
  - Remove `logging.basicConfig()` from `main()`
  - Replace `logger = logging.getLogger(__name__)` with `get_logger(__name__)`

### Step 7: Migrate session store
- Modify `src/nlm_proxy/openai/session.py`:
  - Replace `logger = logging.getLogger(__name__)` with `get_logger(__name__)`

### Step 8: Migrate API client
- Modify `src/nlm_proxy/core/client.py`:
  - Replace `logger = logging.getLogger("notebooklm_mcp.api")` with `get_logger("nlm_proxy.api")`
  - Remove hardcoded `logger.setLevel(logging.WARNING)`

### Step 9: Verify dependencies
- Ensure `pydantic-settings` is in `pyproject.toml` dependencies

### Step 10: Test
- Run `nlm-proxy serve mcp` and verify console logging
- Run `nlm-proxy serve mcp --debug` and verify DEBUG level
- Create `.env` file and verify settings are loaded
- Verify log file is created and rotates correctly

## CLI Behavior

| Command | Behavior |
|---------|----------|
| `nlm-proxy serve mcp` | Uses `.env` settings (default: INFO) |
| `nlm-proxy serve mcp --debug` | Overrides to DEBUG level |
| `nlm-proxy serve openai` | Uses same `.env` settings |

## Benefits

1. **Single source of truth** - All logging configured in one place
2. **Type-safe configuration** - Pydantic validates settings
3. **Environment file support** - Easy to configure without code changes
4. **Extensible** - `Settings` class ready for future config (auth, server, etc.)
5. **Dual output** - Console for immediate feedback, file for persistence
6. **Log rotation** - Prevents disk space issues
