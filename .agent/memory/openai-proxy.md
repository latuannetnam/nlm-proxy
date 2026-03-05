# OpenAI-Compatible Proxy

Connect any OpenAI client to NotebookLM.

## Authentication

The OpenAI proxy requires API key authentication on all `/v1/*` endpoints.

### Setup

Set the required environment variable:

```bash
export NLM_PROXY_OPENAI_API_KEY="your-secret-key-here"
nlm-proxy serve openai
```

Or in `.env`:

```
NLM_PROXY_OPENAI_API_KEY=your-secret-key-here
```

### Error Responses

Missing or invalid API key returns 401:
```json
{
  "error": {
    "message": "Invalid API key",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

## Usage

```bash
nlm-proxy serve openai --port 8080
nlm-proxy serve openai --port 8080 --session-ttl 3600  # 1 hour
nlm-proxy serve openai --host 127.0.0.1 --port 8000
```

**Options:**
- `--host`: Bind address (default: 0.0.0.0)
- `--port`: Port (default: 8080)
- `--session-ttl`: Session TTL in seconds (default: 86400)

## Server Architecture

The OpenAI proxy uses an `AgentCore` singleton (shared with MCP server) for routing, caching, and NLM query delegation.

### Startup Initialization (`main()`)

1. Creates `NotebookLMClient` with authentication
2. Creates `NotebookCache` (proactive, blocking initial fetch)
3. Creates `ResponseCache` (three-layer)
4. Creates LangChain `ChatModel` via `create_chat_model()` factory
5. Creates `LangChainLLMClient` (for L3 cache verification)
6. Creates `SessionStore` (chat_id → conversation_id mapping)
7. Creates `AgentCore` singleton → wires all components
8. Stores everything on `app.state`

### Request Routing

```
POST /v1/chat/completions
  │
  ├─ model == "knowledge-finder"
  │     └─→ handle_smart_routing() — four-phase pipeline (see below)
  │
  └─ model == notebook_id
        └─→ chat_completions() — direct NotebookLM query
              ├─→ AgentCore.handle_direct_query() for cache check
              └─→ NLM query (streaming or non-streaming)
```

### Four-Phase Smart Routing Pipeline (`handle_smart_routing()`)

```
Phase 0: Pre-routing cache check
  └─ AgentCore.route() checks global L1 cache → instant hit skips routing

Phase 1: LangGraph routing
  └─ AgentCore.route() → LangGraph StateGraph
     ├─ classify_node() → NOTEBOOKLM | LLM_TASK
     └─ select_notebook_node() → notebook_id (with ACL filtering)

Phase 2: Post-routing cache check
  └─ ResponseCache.lookup_async() → three-layer lookup

Phase 3: Execute
  ├─ 3a streaming: stream_smart_response() — SSE with reasoning_content
  └─ 3b non-streaming: _handle_non_streaming() — JSON response
```

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Chat (streaming + non-streaming) |
| `GET /v1/models` | List notebooks as models |
| `POST /v1/embeddings` | Returns 501 |
| `GET /health` | Health check |
| `GET /v1/sessions` | List active sessions |
| `DELETE /v1/sessions/{chat_id}` | Delete session |
| `GET /v1/sessions/stats` | Session statistics |
| `GET /v1/cache/stats` | Cache hit/miss metrics |
| `DELETE /v1/cache` | Clear all cache entries |
| `DELETE /v1/cache/{notebook_id}` | Clear notebook cache |

## Session Persistence

- First query → Creates NotebookLM conversation
- Follow-up queries → Reuses `conversation_id` (via `AgentCore.get_conversation_id()`)
- Different chats → Separate conversations
- Sessions expire after TTL
- `SessionStore` lives in `core/session.py` (shared component)

**Open WebUI requirement:**
```bash
ENABLE_FORWARD_USER_INFO_HEADERS=true
```

## Python SDK Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="your-nlm-proxy-key")

# List notebooks
for model in client.models.list():
    print(f"{model.id}: {model.name}")

# Chat
response = client.chat.completions.create(
    model="<notebook-uuid>",
    messages=[{"role": "user", "content": "Summarize key points"}],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Custom Parameters

Pass via `extra_body`:
```python
extra_body={
    "conversation_id": "prev-id",
    "include_thinking": True,
    "bypass_cache": True,  # Skip response cache, fetch fresh answer
}
```
