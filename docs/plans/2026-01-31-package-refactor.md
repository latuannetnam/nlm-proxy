# Package Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the project into a single package with clear separation: `nlm_proxy.core` (standalone library), `nlm_proxy.mcp` (MCP server), and `nlm_proxy.openai` (OpenAI proxy), with optional extras for each server.

**Architecture:** The `core` module contains `NotebookLMClient`, authentication, and exceptions with zero framework dependencies. The `mcp` and `openai` modules are thin wrappers that translate their protocols to core client calls. A unified CLI (`nlm-proxy`) provides subcommands for all functionality.

**Tech Stack:** Python 3.11+, httpx, FastMCP (optional), FastAPI/uvicorn (optional), argparse

---

## Task 1: Create Core Module Structure

**Files:**
- Create: `src/nlm_proxy/__init__.py`
- Create: `src/nlm_proxy/core/__init__.py`
- Create: `src/nlm_proxy/core/exceptions.py`

**Step 1: Create package root**

```python
# src/nlm_proxy/__init__.py
"""NLM Proxy - NotebookLM client library with MCP and OpenAI interfaces."""

__version__ = "0.2.0"
```

**Step 2: Create exceptions module**

```python
# src/nlm_proxy/core/exceptions.py
"""Custom exceptions for NLM Proxy."""


class NLMProxyError(Exception):
    """Base exception for all NLM Proxy errors."""
    pass


class AuthenticationError(NLMProxyError):
    """Token expired, invalid, or missing."""
    pass


class RateLimitError(NLMProxyError):
    """API rate limit exceeded."""
    pass


class NotFoundError(NLMProxyError):
    """Notebook, source, or resource not found."""
    pass


class APIError(NLMProxyError):
    """Generic API error from NotebookLM."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
```

**Step 3: Create core module exports**

```python
# src/nlm_proxy/core/__init__.py
"""Core NotebookLM client library."""

from .exceptions import (
    NLMProxyError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    APIError,
)

__all__ = [
    "NLMProxyError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "APIError",
]
```

**Step 4: Commit**

```bash
git add src/nlm_proxy/
git commit -m "feat: create nlm_proxy package structure with exceptions"
```

---

## Task 2: Move Constants to Core

**Files:**
- Move: `src/notebooklm_mcp/constants.py` → `src/nlm_proxy/core/constants.py`
- Modify: `src/nlm_proxy/core/__init__.py`

**Step 1: Copy constants file**

```bash
cp src/notebooklm_mcp/constants.py src/nlm_proxy/core/constants.py
```

**Step 2: Update core exports**

Add to `src/nlm_proxy/core/__init__.py`:

```python
from .constants import CodeMapper

__all__ = [
    # ... existing exports ...
    "CodeMapper",
]
```

**Step 3: Commit**

```bash
git add src/nlm_proxy/core/constants.py src/nlm_proxy/core/__init__.py
git commit -m "feat: move constants to core module"
```

---

## Task 3: Move Auth to Core

**Files:**
- Move: `src/notebooklm_mcp/auth.py` → `src/nlm_proxy/core/auth.py`
- Modify: `src/nlm_proxy/core/__init__.py`

**Step 1: Copy auth file**

```bash
cp src/notebooklm_mcp/auth.py src/nlm_proxy/core/auth.py
```

**Step 2: Update imports in auth.py**

Replace any internal imports to use new paths. Change:
```python
from .constants import ...
```
to:
```python
from nlm_proxy.core.constants import ...
```

**Step 3: Update core exports**

Add to `src/nlm_proxy/core/__init__.py`:

```python
from .auth import TokenManager, load_tokens, save_tokens

__all__ = [
    # ... existing exports ...
    "TokenManager",
    "load_tokens",
    "save_tokens",
]
```

Note: Actual function names may differ - check `auth.py` and export what exists.

**Step 4: Commit**

```bash
git add src/nlm_proxy/core/auth.py src/nlm_proxy/core/__init__.py
git commit -m "feat: move auth to core module"
```

---

## Task 4: Move Client to Core

**Files:**
- Move: `src/notebooklm_mcp/api_client.py` → `src/nlm_proxy/core/client.py`
- Modify: `src/nlm_proxy/core/__init__.py`

**Step 1: Copy client file**

```bash
cp src/notebooklm_mcp/api_client.py src/nlm_proxy/core/client.py
```

**Step 2: Update imports in client.py**

Replace internal imports:
```python
from .constants import ...
from .auth import ...
```
to:
```python
from nlm_proxy.core.constants import ...
from nlm_proxy.core.auth import ...
```

**Step 3: Add exception usage**

Where the client raises generic exceptions, update to use custom exceptions:
```python
from nlm_proxy.core.exceptions import AuthenticationError, RateLimitError, NotFoundError, APIError
```

**Step 4: Update core exports**

Add to `src/nlm_proxy/core/__init__.py`:

```python
from .client import NotebookLMClient

__all__ = [
    "NotebookLMClient",
    # ... existing exports ...
]
```

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/client.py src/nlm_proxy/core/__init__.py
git commit -m "feat: move client to core module"
```

---

## Task 5: Create MCP Module

**Files:**
- Create: `src/nlm_proxy/mcp/__init__.py`
- Move: `src/notebooklm_mcp/server.py` → `src/nlm_proxy/mcp/server.py`

**Step 1: Copy server file**

```bash
mkdir -p src/nlm_proxy/mcp
cp src/notebooklm_mcp/server.py src/nlm_proxy/mcp/server.py
```

**Step 2: Update imports in server.py**

Replace:
```python
from .api_client import NotebookLMClient
from .auth import ...
from .constants import ...
```
to:
```python
from nlm_proxy.core import NotebookLMClient, load_tokens
from nlm_proxy.core.constants import CodeMapper
```

**Step 3: Create MCP module init with lazy import**

```python
# src/nlm_proxy/mcp/__init__.py
"""MCP server for NotebookLM."""


def create_server():
    """Create and return the MCP server instance."""
    try:
        from .server import mcp
    except ImportError as e:
        raise ImportError(
            "MCP dependencies not installed. Run: pip install nlm-proxy[mcp]"
        ) from e
    return mcp


def run_server(debug: bool = False, transport: str = "stdio", port: int = 8000):
    """Run the MCP server."""
    try:
        from .server import main as server_main
    except ImportError as e:
        raise ImportError(
            "MCP dependencies not installed. Run: pip install nlm-proxy[mcp]"
        ) from e
    # Pass arguments to server main function
    server_main(debug=debug, transport=transport, port=port)


__all__ = ["create_server", "run_server"]
```

**Step 4: Commit**

```bash
git add src/nlm_proxy/mcp/
git commit -m "feat: create mcp module with server"
```

---

## Task 6: Create OpenAI Module

**Files:**
- Create: `src/nlm_proxy/openai/__init__.py`
- Move: `src/notebooklm_mcp/openai_proxy.py` → `src/nlm_proxy/openai/server.py`
- Move: `src/notebooklm_mcp/openai_types.py` → `src/nlm_proxy/openai/types.py`
- Move: `src/notebooklm_mcp/session_store.py` → `src/nlm_proxy/openai/session.py`

**Step 1: Copy files**

```bash
mkdir -p src/nlm_proxy/openai
cp src/notebooklm_mcp/openai_proxy.py src/nlm_proxy/openai/server.py
cp src/notebooklm_mcp/openai_types.py src/nlm_proxy/openai/types.py
cp src/notebooklm_mcp/session_store.py src/nlm_proxy/openai/session.py
```

**Step 2: Update imports in server.py**

Replace:
```python
from .api_client import NotebookLMClient
from .auth import ...
from .openai_types import ...
from .session_store import ...
```
to:
```python
from nlm_proxy.core import NotebookLMClient, load_tokens
from nlm_proxy.openai.types import ...
from nlm_proxy.openai.session import ...
```

**Step 3: Update imports in types.py and session.py**

Check for any internal imports and update to new paths.

**Step 4: Create OpenAI module init with lazy import**

```python
# src/nlm_proxy/openai/__init__.py
"""OpenAI-compatible proxy for NotebookLM."""


def create_app():
    """Create and return the FastAPI app instance."""
    try:
        from .server import app
    except ImportError as e:
        raise ImportError(
            "OpenAI proxy dependencies not installed. Run: pip install nlm-proxy[openai]"
        ) from e
    return app


def run_server(host: str = "0.0.0.0", port: int = 8080, session_ttl: int = 86400):
    """Run the OpenAI proxy server."""
    try:
        from .server import main as server_main
    except ImportError as e:
        raise ImportError(
            "OpenAI proxy dependencies not installed. Run: pip install nlm-proxy[openai]"
        ) from e
    server_main(host=host, port=port, session_ttl=session_ttl)


__all__ = ["create_app", "run_server"]
```

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/
git commit -m "feat: create openai module with proxy server"
```

---

## Task 7: Create Unified CLI

**Files:**
- Create: `src/nlm_proxy/cli.py`

**Step 1: Create CLI with subcommands**

```python
# src/nlm_proxy/cli.py
"""Unified CLI for NLM Proxy."""

import argparse
import sys


def cmd_serve_mcp(args):
    """Run the MCP server."""
    from nlm_proxy.mcp import run_server
    run_server(debug=args.debug, transport=args.transport, port=args.port)


def cmd_serve_openai(args):
    """Run the OpenAI proxy server."""
    from nlm_proxy.openai import run_server
    run_server(host=args.host, port=args.port, session_ttl=args.session_ttl)


def cmd_auth_extract(args):
    """Extract authentication tokens."""
    # Import and run the existing auth extraction logic
    from nlm_proxy.core.auth import extract_tokens_interactive
    extract_tokens_interactive()


def cmd_auth_test(args):
    """Test if current tokens are valid."""
    from nlm_proxy.core import NotebookLMClient, load_tokens, AuthenticationError

    try:
        tokens = load_tokens()
        client = NotebookLMClient(tokens)
        notebooks = client.list_notebooks()
        print(f"Authentication successful! Found {len(notebooks)} notebooks.")
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="nlm-proxy",
        description="NotebookLM client library with MCP and OpenAI interfaces",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run a server")
    serve_subparsers = serve_parser.add_subparsers(dest="server", help="Server type")

    # serve mcp
    mcp_parser = serve_subparsers.add_parser("mcp", help="Run MCP server")
    mcp_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )
    mcp_parser.set_defaults(func=cmd_serve_mcp)

    # serve openai
    openai_parser = serve_subparsers.add_parser("openai", help="Run OpenAI proxy")
    openai_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    openai_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    openai_parser.add_argument(
        "--session-ttl",
        type=int,
        default=86400,
        help="Session TTL in seconds (default: 86400 = 24h)",
    )
    openai_parser.set_defaults(func=cmd_serve_openai)

    # auth command
    auth_parser = subparsers.add_parser("auth", help="Authentication management")
    auth_subparsers = auth_parser.add_subparsers(dest="action", help="Auth action")

    # auth extract
    extract_parser = auth_subparsers.add_parser("extract", help="Extract tokens from browser")
    extract_parser.set_defaults(func=cmd_auth_extract)

    # auth test
    test_parser = auth_subparsers.add_parser("test", help="Test current tokens")
    test_parser.set_defaults(func=cmd_auth_test)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve" and not args.server:
        serve_parser.print_help()
        sys.exit(1)

    if args.command == "auth" and not args.action:
        auth_parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/nlm_proxy/cli.py
git commit -m "feat: add unified CLI with serve and auth subcommands"
```

---

## Task 8: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Read current pyproject.toml**

Review the current file to understand existing dependencies.

**Step 2: Update pyproject.toml**

Replace/update the project configuration:

```toml
[project]
name = "nlm-proxy"
version = "0.2.0"
description = "NotebookLM client library with MCP and OpenAI interfaces"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    # Core dependencies only - check existing deps and include what's needed
    "httpx>=0.27",
]

[project.optional-dependencies]
mcp = [
    "mcp>=1.0",
]
openai = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
]
all = [
    "nlm-proxy[mcp,openai]",
]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]

[project.scripts]
nlm-proxy = "nlm_proxy.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/nlm_proxy"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "mcp: requires mcp extras",
    "openai: requires openai extras",
]
```

Note: Check existing `pyproject.toml` for additional dependencies (tenacity, etc.) and include them in core dependencies.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: update pyproject.toml with new package structure and extras"
```

---

## Task 9: Update Test Imports

**Files:**
- Modify: All files in `tests/`

**Step 1: Find all test files**

```bash
ls tests/
```

**Step 2: Update imports in each test file**

Replace:
```python
from notebooklm_mcp.api_client import NotebookLMClient
from notebooklm_mcp.auth import ...
from notebooklm_mcp.server import ...
```
to:
```python
from nlm_proxy.core import NotebookLMClient, load_tokens
from nlm_proxy.mcp import create_server
from nlm_proxy.openai import create_app
```

**Step 3: Add pytest markers for server tests**

For MCP tests:
```python
import pytest

@pytest.mark.mcp
def test_mcp_functionality():
    ...
```

For OpenAI tests:
```python
import pytest

@pytest.mark.openai
def test_openai_functionality():
    ...
```

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: update test imports for new package structure"
```

---

## Task 10: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md with new commands and structure**

Key sections to update:

1. **Project Overview** - Update package name and description
2. **Development Commands**:
   ```bash
   # Install dependencies
   uv tool install .

   # Reinstall after code changes
   uv cache clean && uv tool install --force .

   # Run the MCP server
   nlm-proxy serve mcp

   # Run with Debug logging
   nlm-proxy serve mcp --debug

   # Run as HTTP server
   nlm-proxy serve mcp --transport http --port 8000

   # Run OpenAI proxy
   nlm-proxy serve openai --port 8080

   # Test authentication
   nlm-proxy auth test
   ```

3. **Architecture** - Update structure diagram:
   ```
   src/nlm_proxy/
   ├── __init__.py      # Package version
   ├── cli.py           # Unified CLI entry point
   ├── core/
   │   ├── __init__.py  # Public exports
   │   ├── client.py    # NotebookLMClient
   │   ├── auth.py      # Token management
   │   ├── constants.py # Code mappings
   │   └── exceptions.py # Custom exceptions
   ├── mcp/
   │   ├── __init__.py  # Lazy imports
   │   └── server.py    # FastMCP tools
   └── openai/
       ├── __init__.py  # Lazy imports
       ├── server.py    # FastAPI routes
       ├── session.py   # Session management
       └── types.py     # Pydantic models
   ```

4. **OpenAI-Compatible Proxy** - Update command examples

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new package structure"
```

---

## Task 11: Remove Old Package

**Files:**
- Delete: `src/notebooklm_mcp/` (entire directory)

**Step 1: Verify new package works**

```bash
uv cache clean && uv tool install --force .
nlm-proxy --help
nlm-proxy serve mcp --help
nlm-proxy serve openai --help
nlm-proxy auth test
```

**Step 2: Remove old package directory**

```bash
rm -rf src/notebooklm_mcp/
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old notebooklm_mcp package"
```

---

## Task 12: Final Verification

**Step 1: Clean install and test**

```bash
uv cache clean
uv tool install --force .
```

**Step 2: Test CLI commands**

```bash
nlm-proxy --help
nlm-proxy serve --help
nlm-proxy serve mcp --help
nlm-proxy serve openai --help
nlm-proxy auth --help
```

**Step 3: Run tests**

```bash
uv run pytest
uv run pytest -m "not mcp and not openai"  # Core only
uv run pytest -m mcp                        # MCP tests
uv run pytest -m openai                     # OpenAI tests
```

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final cleanup after refactor"
```

---

## Summary

| Task | Description | Estimated Changes |
|------|-------------|-------------------|
| 1 | Create core module structure | 3 new files |
| 2 | Move constants to core | 1 file move |
| 3 | Move auth to core | 1 file move, import updates |
| 4 | Move client to core | 1 file move, import updates |
| 5 | Create MCP module | 2 files (init + move) |
| 6 | Create OpenAI module | 4 files (init + 3 moves) |
| 7 | Create unified CLI | 1 new file |
| 8 | Update pyproject.toml | 1 file modify |
| 9 | Update test imports | Multiple test files |
| 10 | Update documentation | 1 file modify |
| 11 | Remove old package | 1 directory delete |
| 12 | Final verification | Testing only |
