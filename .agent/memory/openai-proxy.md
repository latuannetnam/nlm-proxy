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
- Follow-up queries → Reuses `conversation_id`
- Different chats → Separate conversations
- Sessions expire after TTL

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
