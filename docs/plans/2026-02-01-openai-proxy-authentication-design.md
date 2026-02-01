# OpenAI Proxy Authentication Design

## Overview

Add API key authentication to the OpenAI proxy to secure remote deployments and ensure compatibility with OpenAI SDK clients.

## Goals

1. **Security for remote deployment** - Prevent unauthorized access when proxy is exposed on network
2. **OpenAI client compatibility** - Support standard `Authorization: Bearer <key>` header

## Design

### Configuration

**New Setting in `OpenAISettings`:**
```python
api_key: str  # Required, no default
```

**Environment Variable:** `NLM_PROXY_OPENAI_API_KEY`

**Startup Behavior:**
- If not set → raise `ValueError` with message: `"NLM_PROXY_OPENAI_API_KEY is required. Set it to a secure random string."`
- Proxy will not start without this configured
- No opt-out mechanism - authentication is always enforced

### Request Authentication

**Header Format:**
```
Authorization: Bearer <api_key>
```

**Middleware Implementation:**
```python
def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header format")

    token = authorization.removeprefix("Bearer ")
    if not secrets.compare_digest(token, settings.api_key):
        raise HTTPException(401, "Invalid API key")
```

**Protected Endpoints:**
- `POST /v1/chat/completions`
- `GET /v1/models`
- `POST /v1/embeddings`
- `GET /v1/sessions*` (admin routes)

### Error Response Format

OpenAI-compatible JSON:
```json
{
  "error": {
    "message": "Invalid API key",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

### Security Considerations

- Uses `secrets.compare_digest()` for timing-safe comparison
- Returns generic error messages to avoid leaking information
- No localhost exception or disable flag - always enforced

## Client Usage

```python
# OpenAI Python SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://your-server:8080/v1",
    api_key="your-nlm-proxy-key"
)
```

```bash
# curl
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer your-nlm-proxy-key"
```

```yaml
# Open WebUI
OPENAI_API_KEY=your-nlm-proxy-key
OPENAI_API_BASE_URL=http://your-server:8080/v1
```

## Files to Modify

1. `src/nlm_proxy/core/config.py` - Add `api_key` to `OpenAISettings`
2. `src/nlm_proxy/openai/server.py` - Add auth dependency to routes
3. `.env.example` - Add `NLM_PROXY_OPENAI_API_KEY` example
4. `.claude/memory/openai-proxy.md` - Document authentication setup
