# Configuration System

## Overview

Unified configuration using pydantic-settings with precedence:
**CLI args > Environment variables > .env files > Defaults**

## Settings Classes

| Class | Prefix | Purpose |
|-------|--------|---------|
| `SharedSettings` | `NLM_PROXY_` | Global settings (debug, auth_dir) |
| `MCPSettings` | `NLM_PROXY_MCP_` | MCP server settings |
| `OpenAISettings` | `NLM_PROXY_OPENAI_` | OpenAI proxy settings |
| `AuthSettings` | `NLM_PROXY_AUTH_` | Authentication settings |
| `LoggingSettings` | `NLM_PROXY_LOG_` | Logging settings |

## Environment Variables

```bash
# Shared
NLM_PROXY_DEBUG=false
NLM_PROXY_AUTH_DIR=~/.nlm-proxy

# MCP Server
NLM_PROXY_MCP_PORT=8000
NLM_PROXY_MCP_TRANSPORT=stdio

# OpenAI Proxy
NLM_PROXY_OPENAI_HOST=0.0.0.0
NLM_PROXY_OPENAI_PORT=8080
NLM_PROXY_OPENAI_SESSION_TTL=86400

# Authentication
NLM_PROXY_AUTH_CHROME_PORT=9222
NLM_PROXY_AUTH_AUTO_LAUNCH=true

# Logging
NLM_PROXY_LOG_LEVEL=INFO
NLM_PROXY_LOG_FILE=~/.nlm-proxy/logs/nlm-proxy.log
```

## .env File Locations

Priority (first found wins):
1. `.env` in current directory
2. `~/.nlm-proxy/.env`

## Usage in Code

```python
from nlm_proxy.core.config import (
    get_shared_settings,
    get_mcp_settings,
    get_openai_settings,
    get_auth_settings,
    get_logging_settings,
)

# Singleton instances
shared = get_shared_settings()
print(shared.debug)  # False by default

mcp = get_mcp_settings()
print(mcp.port)  # 8000 by default
```

## CLI Override Example

```bash
# Environment sets port to 9000
export NLM_PROXY_OPENAI_PORT=9000

# CLI overrides to 8080
nlm-proxy serve openai --port 8080  # Uses 8080
```

## Reference

See `.env.example` in project root for all options.
