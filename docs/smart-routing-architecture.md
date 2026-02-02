# Smart Request Routing Architecture

This document describes the smart routing feature that automatically classifies incoming requests and routes them to either NotebookLM or an external LLM.

## Overview

Smart routing uses an external LLM (e.g., GPT-4o-mini) to analyze each request and determine:
1. Whether it's a knowledge query (route to NotebookLM) or a general LLM task (route to external LLM)
2. Which notebook is most relevant for knowledge queries, using **source-level information** for precise matching

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OpenAI Proxy Server                             │
│                                                                         │
│  ┌─────────────┐    ┌──────────────────────────────────────────────┐   │
│  │   Request   │───>│  Model = "knowledge-finder"?                 │   │
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
│  │  │ (LLM call    │                          │ LLM          │     │   │
│  │  │ with sources)│                          │              │     │   │
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

## Source-Level Routing

The smart router uses **source-level information** to make more accurate notebook selections:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Notebook Selection with Sources                      │
│                                                                         │
│  User Query: "What does the Attention Is All You Need paper say?"       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    NotebookCache (Warm)                          │   │
│  │                                                                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │ Notebook: "ML Research"                                  │    │   │
│  │  │ Summary: "Notes about machine learning..."               │    │   │
│  │  │ Sources:                                                 │    │   │
│  │  │   ├─ [PDF] "Attention Is All You Need.pdf"  ◄── MATCH!  │    │   │
│  │  │   ├─ [URL] "pytorch.org/docs"                            │    │   │
│  │  │   └─ [PDF] "BERT Paper.pdf"                              │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │                                                                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │ Notebook: "Project Notes"                                │    │   │
│  │  │ Summary: "General project documentation..."              │    │   │
│  │  │ Sources:                                                 │    │   │
│  │  │   ├─ [text] "Meeting Notes"                              │    │   │
│  │  │   └─ [URL] "github.com/project"                          │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Selection LLM receives:                                                │
│  - Notebook summaries and topics                                        │
│  - Source counts by type (e.g., {"pdf": 2, "url": 1})                  │
│  - Source titles (up to 15)                                            │
│                                                                         │
│  Result: Routes to "ML Research" (matches source title)                 │
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

Proactive thread-safe cache for notebook summaries **and source information** with background refresh.

**Features:**
- **Proactive initialization**: Blocking fetch at server startup to warm the cache
- **Source-level fetching**: Fetches source summaries and keywords for each notebook
- **Parallel source fetching**: Uses `asyncio.Semaphore` for controlled concurrency
- **Background refresh**: Automatic refresh at 80% of TTL (prevents expiration)
- **Thread-safe operations**: All cache operations protected with locks
- **Graceful shutdown**: Background thread stops cleanly on server shutdown
- **Graceful degradation**: If source fetch fails, keeps source with basic info
- **Shared instance**: Single cache instance shared across all router instances via `app.state`

**Architecture:**
```python
# Server startup (in openai/server.py)
@app.on_event("startup")
async def startup():
    # Create shared cache - blocks until initial fetch completes
    app.state.notebook_cache = NotebookCache(
        nlm_client=nlm_client,
        ttl_seconds=config.summary_cache_ttl,
        source_fetch_concurrency=config.source_fetch_concurrency
    )
    # Background refresh thread started automatically

# Router uses shared cache (always warm with source info)
router = SmartRouter(
    nlm_client=nlm_client,
    notebook_cache=app.state.notebook_cache,  # Shared cache
    ...
)

# Background refresh loop (internal to NotebookCache)
while not shutdown:
    sleep(ttl * 0.8)  # Refresh at 80% of TTL
    fetch_all_summaries()  # Refresh notebooks AND sources
```

**Data Structures:**
```python
@dataclass
class SourceInfo:
    """Cached source information for a notebook."""
    id: str
    title: str
    source_type: str      # "pdf", "url", "text", "gdoc", etc.
    summary: str          # AI-generated summary (stored, not sent to LLM)
    keywords: list[str]   # AI-extracted keywords

@dataclass
class NotebookInfo:
    """Cached notebook information."""
    id: str
    title: str
    summary: str
    topics: list[str]
    cached_at: float
    sources: list[SourceInfo]  # All sources in the notebook

    # Computed properties for routing
    @property
    def source_count(self) -> int:
        """Total number of sources."""
        return len(self.sources)

    @property
    def source_types(self) -> dict[str, int]:
        """Count by type, e.g., {"pdf": 2, "url": 3}"""
        ...

    @property
    def source_titles(self) -> list[str]:
        """List of source titles (truncated to 100 chars)."""
        ...
```

**Cache Lifecycle:**
```
Server Start                    Refresh 1 (48 min)         Refresh 2 (96 min)
     |                                |                           |
     v                                v                           v
[Initial Fetch]-------------[Background Refresh]--------[Background Refresh]---...
     |                                |                           |
     ├─ List notebooks                ├─ List notebooks           |
     ├─ For each notebook:            ├─ For each notebook:       |
     │   ├─ Fetch summary             │   ├─ Fetch summary        |
     │   ├─ Fetch sources list        │   ├─ Fetch sources list   |
     │   └─ For each source:          │   └─ For each source:     |
     │       └─ Fetch guide           │       └─ Fetch guide      |
     │          (parallel, max 10)    │          (parallel)       |
     |                                |                           |
  (blocks startup)              (async in bg)              (async in bg)
```

**Parallel Source Fetching:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Parallel Source Fetching Flow                         │
│                                                                         │
│  Semaphore: max 10 concurrent requests (configurable)                   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Notebook "ML Research" has 5 sources                             │  │
│  │                                                                   │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │  │
│  │  │ Source1 │ │ Source2 │ │ Source3 │ │ Source4 │ │ Source5 │    │  │
│  │  │ (slot 1)│ │ (slot 2)│ │ (slot 3)│ │ (slot 4)│ │ (slot 5)│    │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │  │
│  │       │           │           │           │           │          │  │
│  │       └───────────┴───────────┼───────────┴───────────┘          │  │
│  │                               │                                   │  │
│  │                     asyncio.gather()                              │  │
│  │                               │                                   │  │
│  │                               ▼                                   │  │
│  │                    [SourceInfo, SourceInfo, ...]                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  If any source fetch fails:                                            │
│  - Log warning                                                          │
│  - Keep source with title/type only (graceful degradation)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. SmartRouter (`openai/router.py`)

The main routing logic that classifies requests and selects notebooks using source-level information.

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
3. `select_notebook(query)` - If NOTEBOOKLM, finds best notebook using source info
4. Returns `RoutingDecision` with type, notebook_id, and reasoning

**Notebook Selection Data:**
```python
# Data sent to selection LLM
notebooks_info = [
    {
        "id": "abc-123",
        "title": "ML Research",
        "summary": "Notes about machine learning...",
        "topics": ["transformers", "attention", "NLP"],
        "source_count": 5,
        "source_types": {"pdf": 3, "url": 2},
        "source_titles": [
            "Attention Is All You Need.pdf",
            "BERT Paper.pdf",
            "GPT-3 Paper.pdf",
            "pytorch.org/docs",
            "huggingface.co/docs"
        ]
    },
    ...
]
```

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
with their summaries and sources, select the most relevant notebook.

Available notebooks:
{notebooks_json}

User query:
{query}

Selection criteria (in order of importance):
1. **Source titles** - If the query mentions a specific document, paper, URL,
   or file name, prioritize notebooks containing sources with matching titles
2. **Source types** - Match query intent to source types (e.g., "PDF paper"
   queries should prefer notebooks with PDF sources)
3. **Notebook summary** - Consider how well the notebook's overall topic
   matches the query
4. **Topics** - Use suggested topics as additional context for relevance

Respond with ONLY the notebook_id (UUID) of the most relevant notebook.
```

### 5. Server Integration (`openai/server.py`)

**Key Functions:**

- `handle_smart_routing()` - Entry point when model="knowledge-finder"
- `stream_smart_response()` - Streaming with reasoning_content for routing decision

**Request Flow:**
```
POST /v1/chat/completions
  │
  ├─ model == "knowledge-finder"
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
| `NLM_PROXY_ROUTING_ROUTER_MODEL_NAME` | `knowledge-finder` | Model name that triggers routing |
| `NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS` | (empty = all) | Comma-separated notebook IDs |
| `NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `NLM_PROXY_ROUTING_SOURCE_FETCH_CONCURRENCY` | `10` | Max parallel source summary fetches |
| `NLM_PROXY_ROUTING_MAX_SOURCE_TITLES` | `15` | Max source titles in selection prompt |

**Example `.env`:**
```bash
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-api-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
NLM_PROXY_ROUTING_ROUTER_MODEL_NAME=knowledge-finder
NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL=3600
NLM_PROXY_ROUTING_SOURCE_FETCH_CONCURRENCY=10
NLM_PROXY_ROUTING_MAX_SOURCE_TITLES=15
```

## Logging Tags

The smart routing feature uses specific logging tags for debugging:

| Tag | Component | Description |
|-----|-----------|-------------|
| `[LLM]` | ExternalLLMClient | External LLM API calls |
| `[ROUTER]` | SmartRouter | Classification and notebook selection |
| `[SMART-ROUTER]` | Server | High-level routing decisions |
| `[CACHE]` | NotebookCache | Cache operations and source fetching |

**Example log output:**
```
INFO  [CACHE] Performing initial notebook fetch...
DEBUG [CACHE] Found 3 notebooks in NotebookLM
DEBUG [CACHE] Fetching summary for: ML Research (abc-123)
DEBUG [CACHE] Cached ML Research: 5 sources
DEBUG [CACHE] Fetching summary for: Project Notes (def-456)
DEBUG [CACHE] Cached Project Notes: 2 sources
INFO  [CACHE] Initial fetch complete: 3 notebooks cached
DEBUG [CACHE] Background refresh thread started

INFO  [ROUTER] Starting routing for query: What does the Attention paper say...
DEBUG [ROUTER] Classifying request: What does the Attention paper say...
DEBUG [LLM] Calling complete: model=gpt-4o-mini, max_tokens=50
DEBUG [LLM] Response: notebooklm
INFO  [ROUTER] Classified as NOTEBOOKLM query
DEBUG [ROUTER] Selecting notebook for query: What does the Attention paper say...
DEBUG [ROUTER] Using 3 cached notebooks
DEBUG [ROUTER] Asking LLM to select from 3 notebooks
INFO  [ROUTER] Selected notebook: ML Research (ID: abc-123)
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

# Use "knowledge-finder" as the model
response = client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": "What does the Attention Is All You Need paper say about transformers?"}],
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
  "model": "knowledge-finder",
  "choices": [{
    "delta": {
      "reasoning_content": "Selected notebook: ML Research (ID: abc-123)\n\n",
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

### Source Summary Fetch Failure
If fetching a source summary fails:
- Warning logged
- Source cached with title and type only (graceful degradation)
- Routing continues with partial source info

## Performance Considerations

### Proactive Caching Strategy
- **Initial fetch**: All notebook summaries and sources fetched at server startup (blocking)
- **Parallel source fetching**: Sources fetched concurrently with semaphore (default: 10)
- **Background refresh**: Cache refreshed automatically at 80% of TTL
- **Always warm**: Cache never expires during normal operation
- **No request delays**: Routing requests never wait for cache population

**Cache Timeline:**
```
Server Start          Refresh 1 (48 min)    Refresh 2 (96 min)
     |                      |                      |
     v                      v                      v
[Initial Fetch]------[Background Refresh]---[Background Refresh]---...
     |                      |                      |
  (blocks)            (async in bg)          (async in bg)
     |                      |                      |
  Fetches:             Fetches:              Fetches:
  - 3 notebooks        - 3 notebooks         - 3 notebooks
  - 12 sources         - 12 sources          - 12 sources
  - (parallel x10)     - (parallel x10)      - (parallel x10)
     |                      |                      |
Cache TTL: 60 min    Cache still fresh      Cache still fresh
```

### LLM Call Optimization
- Classification uses `max_tokens=50` (single word response)
- Notebook selection uses `max_tokens=100` (UUID response)
- `temperature=0.0` for deterministic classification

### Parallel Operations
- **Source summaries fetched in parallel** with configurable concurrency (default: 10)
- **Notebook summary + sources list** fetched in parallel per notebook
- Background refresh runs independently from request handling
- No blocking on request path after initial startup

### Token Efficiency
- Source titles truncated to 100 characters
- Max 15 source titles per notebook (configurable)
- Notebook summaries truncated to 500 characters
- Max 5 topics per notebook

## Manual Verification

### Step 1: Configure Routing

Add to `~/.nlm-proxy/.env`:

```bash
# Smart Routing Configuration
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-openai-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
NLM_PROXY_ROUTING_SOURCE_FETCH_CONCURRENCY=10
NLM_PROXY_ROUTING_MAX_SOURCE_TITLES=15
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

**Expected startup logs:**
```
INFO  [CACHE] Performing initial notebook fetch...
DEBUG [CACHE] Found 3 notebooks in NotebookLM
DEBUG [CACHE] Fetching summary for: ML Research (abc-123)
DEBUG [CACHE] Cached ML Research: 5 sources
...
INFO  [CACHE] Initial fetch complete: 3 notebooks cached
```

### Step 3: Verify knowledge-finder in Models List

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" | jq '.data[].id'
```

**Expected:** `"knowledge-finder"` appears first, followed by notebook IDs.

### Step 4: Test Source-Specific Query Routing

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "knowledge-finder",
    "messages": [{"role": "user", "content": "What does the Attention Is All You Need paper say about transformers?"}],
    "stream": true
  }'
```

**Expected:**
- Routes to notebook containing "Attention Is All You Need.pdf" source
- First chunk has `reasoning_content` with "Selected notebook: ..."
- Subsequent chunks have `content` with NotebookLM answer

### Step 5: Test LLM Task Routing

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "knowledge-finder",
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

# Streaming with smart routing - source-specific query
response = client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": "What's in the BERT paper?"}],
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
│   ├── config.py          # SmartRoutingSettings (with source_fetch_concurrency)
│   └── llm_client.py      # ExternalLLMClient
└── openai/
    ├── notebook_cache.py  # NotebookCache, NotebookInfo, SourceInfo
    ├── router.py          # SmartRouter, RequestType, RoutingDecision
    ├── server.py          # handle_smart_routing, stream_smart_response
    └── prompts/
        ├── __init__.py    # load_prompt()
        ├── classify_request.txt
        └── select_notebook.txt  # Updated with source-aware selection
```
