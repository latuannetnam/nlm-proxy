# Smart Request Routing Architecture

This document describes the smart routing feature that automatically classifies incoming requests and routes them to either NotebookLM or an external LLM.

## Overview

Smart routing uses an external LLM (e.g., GPT-4o-mini) to analyze each request and determine:
1. Whether it's a knowledge query (route to NotebookLM) or a general LLM task (route to external LLM)
2. Which notebook is most relevant for knowledge queries

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OpenAI Proxy Server                             │
│                                                                         │
│  ┌─────────────┐    ┌──────────────────────────────────────────────┐   │
│  │   Request   │───>│  Model = "smart-router"?                     │   │
│  │  /v1/chat   │    │                                              │   │
│  │ completions │    │  YES ──────────────────────────────────────┐ │   │
│  └─────────────┘    │                                            │ │   │
│                     │  NO ───> Direct NotebookLM query           │ │   │
│                     │          (model = notebook_id)             │ │   │
│                     └──────────────────────────────────────────┬─┘ │   │
│                                                                │   │   │
│  ┌─────────────────────────────────────────────────────────────▼───┐   │
│  │                      SmartRouter                                │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                     │   │
│  │  │ classify_request│───>│  RequestType    │                     │   │
│  │  │  (LLM call)     │    │  NOTEBOOKLM or  │                     │   │
│  │  └─────────────────┘    │  LLM_TASK       │                     │   │
│  │                         └────────┬────────┘                     │   │
│  │                                  │                              │   │
│  │         ┌────────────────────────┴────────────────────┐         │   │
│  │         │                                             │         │   │
│  │         ▼                                             ▼         │   │
│  │  ┌──────────────┐                          ┌──────────────┐     │   │
│  │  │ NOTEBOOKLM   │                          │ LLM_TASK     │     │   │
│  │  │              │                          │              │     │   │
│  │  │ select_      │                          │ Passthrough  │     │   │
│  │  │ notebook()   │                          │ to External  │     │   │
│  │  │ (LLM call)   │                          │ LLM          │     │   │
│  │  └──────┬───────┘                          └──────┬───────┘     │   │
│  │         │                                         │             │   │
│  └─────────┼─────────────────────────────────────────┼─────────────┘   │
│            │                                         │                 │
│            ▼                                         ▼                 │
│  ┌──────────────────┐                     ┌──────────────────┐         │
│  │   NotebookLM     │                     │   External LLM   │         │
│  │   (via client)   │                     │   (OpenAI SDK)   │         │
│  └──────────────────┘                     └──────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. ExternalLLMClient (`core/llm_client.py`)

A reusable client for calling OpenAI-compatible LLM endpoints using the official OpenAI SDK.

**Features:**
- Lazy initialization of AsyncOpenAI client
- Non-streaming `complete()` for classification tasks
- Streaming `stream()` for LLM task passthrough
- Configurable base URL, API key, and model

**Usage:**
```python
from nlm_proxy.core.llm_client import ExternalLLMClient

client = ExternalLLMClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o-mini"
)

# Simple completion
result = await client.complete("Classify this request...", max_tokens=50)

# Streaming
async for chunk in await client.stream(messages):
    print(chunk.choices[0].delta.content)

await client.close()
```

### 2. NotebookCache (`openai/notebook_cache.py`)

Thread-safe TTL cache for notebook summaries to avoid repeated API calls.

**Features:**
- Configurable TTL (default: 1 hour)
- Thread-safe operations with locking
- Automatic expiration on access
- Stores: id, title, summary, topics, cached_at

**Data Structure:**
```python
@dataclass
class NotebookInfo:
    id: str
    title: str
    summary: str
    topics: list[str]
    cached_at: float
```

### 3. SmartRouter (`openai/router.py`)

The main routing logic that classifies requests and selects notebooks.

**Key Classes:**

```python
class RequestType(Enum):
    NOTEBOOKLM = "notebooklm"  # Knowledge queries
    LLM_TASK = "llm_task"      # General LLM tasks

@dataclass
class RoutingDecision:
    request_type: RequestType
    notebook_id: str | None = None
    reasoning: str = ""
```

**Routing Flow:**

1. `route(query)` - Main entry point
2. `classify_request(query)` - Determines NOTEBOOKLM vs LLM_TASK
3. `select_notebook(query)` - If NOTEBOOKLM, finds best notebook
4. Returns `RoutingDecision` with type, notebook_id, and reasoning

### 4. Prompt Templates (`openai/prompts/`)

External text files for LLM prompts, loaded via `load_prompt()`.

**classify_request.txt:**
```
You are a request classifier. Analyze the user's request and determine its type.

Request types:
1. "notebooklm" - Questions seeking information, facts, or knowledge
2. "llm_task" - Meta-tasks that don't require document knowledge

User request:
{query}

Respond with ONLY one word: "notebooklm" or "llm_task"
```

**select_notebook.txt:**
```
You are a notebook selector. Given the user's query and available notebooks
with their summaries, select the most relevant notebook.

Available notebooks:
{notebooks_json}

User query:
{query}

Respond with ONLY the notebook_id (UUID) of the most relevant notebook.
```

### 5. Server Integration (`openai/server.py`)

**Key Functions:**

- `handle_smart_routing()` - Entry point when model="smart-router"
- `stream_smart_response()` - Streaming with reasoning_content for routing decision

**Request Flow:**
```
POST /v1/chat/completions
  │
  ├─ model == "smart-router"
  │     └─> handle_smart_routing()
  │           ├─> Extract chat_id from headers/metadata
  │           ├─> Lookup conversation_id from SessionStore
  │           ├─> router.route(query)
  │           ├─> If LLM_TASK: stream from external LLM
  │           └─> If NOTEBOOKLM: stream from selected notebook
  │                 └─> Save conversation_id to SessionStore
  │
  └─ model == notebook_id
        └─> Direct NotebookLM query (existing flow)
```

### 6. Session Mapping (Conversation Continuity)

The smart router supports mapping Open Web UI `chat_id` to NotebookLM `conversation_id` for multi-turn conversations.

**How it works:**

1. **Extract `chat_id`**: From `X-OpenWebUI-Chat-Id` header or `metadata.chat_id` in request body
2. **Lookup**: Check `SessionStore` for existing `conversation_id` mapped to this `chat_id`
3. **Reuse**: If found, pass `conversation_id` to NotebookLM query for conversation continuity
4. **Save**: After query, save the new/returned `conversation_id` back to `SessionStore`

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Open Web UI    │────>│   Smart Router    │────>│   NotebookLM    │
│  (chat_id)      │     │   (SessionStore)  │     │ (conversation_id)│
└─────────────────┘     └───────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌───────────────────┐
                        │  chat_id ──────>  │
                        │  conversation_id  │
                        │  (TTL: 24 hours)  │
                        └───────────────────┘
```

**Behavior:**
- First message in a chat: Creates new NotebookLM conversation, stores mapping
- Subsequent messages: Reuses existing conversation for context continuity
- Different chats: Each gets its own NotebookLM conversation
- Session expiry: Configurable TTL (default 24 hours)

## Configuration

All settings use the `NLM_PROXY_ROUTING_` prefix.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NLM_PROXY_ROUTING_LLM_BASE_URL` | `https://api.openai.com/v1` | Base URL for external LLM |
| `NLM_PROXY_ROUTING_LLM_API_KEY` | (required) | API key for external LLM |
| `NLM_PROXY_ROUTING_LLM_MODEL` | `gpt-4o-mini` | Model for classification |
| `NLM_PROXY_ROUTING_ROUTER_MODEL_NAME` | `smart-router` | Model name that triggers routing |
| `NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS` | (empty = all) | Comma-separated notebook IDs |
| `NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL` | `3600` | Cache TTL in seconds |

**Example `.env`:**
```bash
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-api-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
NLM_PROXY_ROUTING_ROUTER_MODEL_NAME=smart-router
NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL=3600
```

## Logging Tags

The smart routing feature uses specific logging tags for debugging:

| Tag | Component | Description |
|-----|-----------|-------------|
| `[LLM]` | ExternalLLMClient | External LLM API calls |
| `[ROUTER]` | SmartRouter | Classification and notebook selection |
| `[SMART-ROUTER]` | Server | High-level routing decisions |

**Example log output:**
```
INFO  [ROUTER] Starting routing for query: What are the key findings...
DEBUG [ROUTER] Classifying request: What are the key findings...
DEBUG [LLM] Calling complete: model=gpt-4o-mini, max_tokens=50
DEBUG [LLM] Response: notebooklm
INFO  [ROUTER] Classified as NOTEBOOKLM query
DEBUG [ROUTER] Selecting notebook for query: What are the key findings...
DEBUG [ROUTER] Using 3 cached notebooks
INFO  [ROUTER] Selected notebook: Research Notes (ID: abc-123)
INFO  [ROUTER] Routing to NotebookLM: abc-123
INFO  [SMART-ROUTER] Decision: notebooklm, notebook=abc-123
```

## API Usage

### Using the Smart Router

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-nlm-proxy-api-key"
)

# Use "smart-router" as the model
response = client.chat.completions.create(
    model="smart-router",
    messages=[{"role": "user", "content": "What does the research say about X?"}],
    stream=True
)

for chunk in response:
    # Routing reasoning appears in reasoning_content
    if hasattr(chunk.choices[0].delta, 'reasoning_content'):
        reasoning = chunk.choices[0].delta.reasoning_content
        if reasoning:
            print(f"[Routing] {reasoning}", end="")

    # Actual response in content
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Response Format

The smart router returns routing decisions in `reasoning_content`:

```json
{
  "id": "chatcmpl-abc123",
  "model": "smart-router",
  "choices": [{
    "delta": {
      "reasoning_content": "Selected notebook: Research Notes (ID: abc-123)\n\n",
      "content": "Based on the research..."
    }
  }]
}
```

## Error Handling

### No Notebooks Available
If no notebooks exist or all are filtered out:
- Returns `RoutingDecision` with `notebook_id=None`
- Falls back to external LLM

### Classification Failure
If external LLM call fails:
- Exception propagates to caller
- HTTP 500 returned to client

### Notebook Summary Fetch Failure
If fetching a notebook summary fails:
- Warning logged
- Notebook cached with empty summary
- Selection continues with available info

## Performance Considerations

### Caching Strategy
- Notebook summaries cached for 1 hour (configurable)
- First request after cache expiry fetches all summaries
- Subsequent requests use cache

### LLM Call Optimization
- Classification uses `max_tokens=50` (single word response)
- Notebook selection uses `max_tokens=100` (UUID response)
- `temperature=0.0` for deterministic classification

### Parallel Operations
- Notebook list and summary fetches are sequential (API limitation)
- Consider pre-warming cache on server startup for large notebook collections

## Manual Verification

### Step 1: Configure Routing

Add to `~/.nlm-proxy/.env`:

```bash
# Smart Routing Configuration
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-openai-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
```

Alternative providers:

```bash
# OpenRouter
NLM_PROXY_ROUTING_LLM_BASE_URL=https://openrouter.ai/api/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-or-xxx
NLM_PROXY_ROUTING_LLM_MODEL=anthropic/claude-3-haiku

# Ollama (local)
NLM_PROXY_ROUTING_LLM_BASE_URL=http://localhost:11434/v1
NLM_PROXY_ROUTING_LLM_API_KEY=ollama
NLM_PROXY_ROUTING_LLM_MODEL=llama3.2
```

### Step 2: Start the Server

```bash
nlm-proxy serve openai --port 8080
```

With debug logging:

```bash
NLM_PROXY_DEBUG=true nlm-proxy serve openai --port 8080
```

### Step 3: Verify smart-router in Models List

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" | jq '.data[].id'
```

**Expected:** `"smart-router"` appears first, followed by notebook IDs.

### Step 4: Test NotebookLM Query Routing

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart-router",
    "messages": [{"role": "user", "content": "What information is in my notebooks?"}],
    "stream": true
  }'
```

Or for powershell
```bash
$body = @'
{
  "model": "smart-router",
  "messages": [
    { "role": "user", "content": "What information is in my notebooks?" }
  ],
  "stream": true
}
'@

curl -N http://localhost:9999/v1/chat/completions `
  -H "Authorization: Bearer $env:NLM_PROXY_OPENAI_API_KEY" `
  -H "Content-Type: application/json" `
  -d $body
```

**Expected:**
- First chunk has `reasoning_content` with "Selected notebook: ..."
- Subsequent chunks have `content` with NotebookLM answer

### Step 5: Test LLM Task Routing

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart-router",
    "messages": [{"role": "user", "content": "Summarize what we discussed"}],
    "stream": false
  }'
```

**Expected:**
- `reasoning_content`: "Classified as LLM task (not a notebook query)"
- `content`: Response from external LLM

### Step 6: Test with Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-api-key"  # NLM_PROXY_OPENAI_API_KEY
)

# Streaming with smart routing
response = client.chat.completions.create(
    model="smart-router",
    messages=[{"role": "user", "content": "What's in my research notes?"}],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
        print(f"[Routing] {delta.reasoning_content}", end="")
    if delta.content:
        print(delta.content, end="")
```

## File Structure

```
src/nlm_proxy/
├── core/
│   ├── config.py          # SmartRoutingSettings
│   └── llm_client.py      # ExternalLLMClient
└── openai/
    ├── notebook_cache.py  # NotebookCache, NotebookInfo
    ├── router.py          # SmartRouter, RequestType, RoutingDecision
    ├── server.py          # handle_smart_routing, stream_smart_response
    └── prompts/
        ├── __init__.py    # load_prompt()
        ├── classify_request.txt
        └── select_notebook.txt
```
