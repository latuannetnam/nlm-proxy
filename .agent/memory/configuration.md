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
| `SmartRoutingSettings` | `NLM_PROXY_ROUTING_` | Smart routing and source fetching |
| `TracingSettings` | `NLM_PROXY_OTEL_` | OpenTelemetry tracing settings |

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

# Smart Routing
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=your-api-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
NLM_PROXY_ROUTING_ROUTER_MODEL_NAME=knowledge-finder
NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS=  # comma-separated notebook IDs
NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL=3600
NLM_PROXY_ROUTING_SOURCE_FETCH_CONCURRENCY=10  # max parallel source fetches
NLM_PROXY_ROUTING_MAX_SOURCE_TITLES=15  # max source titles in selection prompt

# OpenTelemetry Tracing
NLM_PROXY_OTEL_ENABLED=false
NLM_PROXY_OTEL_ENDPOINT=localhost:4317
NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy
NLM_PROXY_OTEL_PROTOCOL=grpc  # grpc or http
NLM_PROXY_OTEL_API_KEY=  # Bearer token for authentication
NLM_PROXY_OTEL_INSECURE=true  # true=plain text, false=TLS
NLM_PROXY_OTEL_VERIFY_CERT=true  # Skip cert validation (HTTP only)
NLM_PROXY_OTEL_CA_CERT_PATH=  # Path to custom CA certificate
NLM_PROXY_OTEL_REQUEST_MAX_LENGTH=500  # max chars of user query (0=disable)
NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH=1000  # max chars of response (0=disable)
```

## Tracing Authentication & TLS

| Variable | Default | Description |
|----------|---------|-------------|
| `NLM_PROXY_OTEL_PROTOCOL` | `grpc` | Exporter protocol: `grpc` or `http` |
| `NLM_PROXY_OTEL_API_KEY` | (none) | Bearer token for collector authentication |
| `NLM_PROXY_OTEL_INSECURE` | `true` | `true`=plain text, `false`=TLS enabled |
| `NLM_PROXY_OTEL_VERIFY_CERT` | `true` | Skip cert validation (HTTP only) |
| `NLM_PROXY_OTEL_CA_CERT_PATH` | (none) | Path to private CA certificate |

### Common Configurations

**Local development (plain text):**
```bash
NLM_PROXY_OTEL_PROTOCOL=grpc
NLM_PROXY_OTEL_INSECURE=true
```

**Production with private CA:**
```bash
NLM_PROXY_OTEL_PROTOCOL=http
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_CA_CERT_PATH=/etc/ssl/otel-ca.pem
NLM_PROXY_OTEL_API_KEY=your-bearer-token
```

**Development with self-signed cert (skip verify):**
```bash
NLM_PROXY_OTEL_PROTOCOL=http
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_VERIFY_CERT=false
```

**Note:** gRPC protocol does not support `verify_cert=false`. Use HTTP protocol for skip-verify scenarios.

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
