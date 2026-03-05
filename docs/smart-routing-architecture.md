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
│  │                  AgentCore (shared singleton)                   │   │
│  │                                                                 │   │
│  │  Phase 0: Pre-routing cache (global L1)                         │   │
│  │                                                                 │   │
│  │  Phase 1: LangGraph Routing StateGraph                          │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                     │   │
│  │  │ classify_node   │───>│  request_type   │                     │   │
│  │  │  (LLM call)     │    │  notebooklm or  │                     │   │
│  │  └─────────────────┘    │  llm_task       │                     │   │
│  │                         └────────┬────────┘                     │   │
│  │                                  │                              │   │
│  │         ┌────────────────────────┴────────────────────┐         │   │
│  │         │                                             │         │   │
│  │         ▼                                             ▼         │   │
│  │  ┌──────────────┐                          ┌──────────────┐     │   │
│  │  │ notebooklm   │                          │ llm_task     │     │   │
│  │  │              │                          │              │     │   │
│  │  │ select_      │                          │ Passthrough  │     │   │
│  │  │ notebook_    │                          │ to External  │     │   │
│  │  │ node()       │                          │ LLM          │     │   │
│  │  │ (LLM call    │                          │              │     │   │
│  │  │ with sources)│                          │              │     │   │
│  │  └──────┬───────┘                          └──────┬───────┘     │   │
│  │         │                                         │             │   │
│  └─────────┼─────────────────────────────────────────┼─────────────┘   │
│            │                                         │                 │
│            ▼                                         ▼                 │
│  ┌──────────────────┐                     ┌──────────────────┐         │
│  │   NotebookLM     │                     │   External LLM   │         │
│  │   (via client)   │                     │   (LangChain     │         │
│  │                  │                     │    ChatModel)    │         │
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
│  - Source descriptions with keywords and summaries (first 10 sources)  │
│  - Source titles only (remaining sources)                              │
│                                                                         │
│  Result: Routes to "ML Research" (matches source keywords/title)        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. LangChainLLMClient (`core/llm_client.py`)

A LangChain-based client for calling external LLM providers.

**Features:**
- `create_chat_model()` factory supporting openai, anthropic, ollama, azure via LangChain
- `LangChainLLMClient` wrapper with simplified interface
- Non-streaming `ainvoke()` for classification/selection
- Streaming `astream()` for LLM task passthrough

**Usage:**
```python
from nlm_proxy.core.llm_client import create_chat_model, LangChainLLMClient

chat_model = create_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)

client = LangChainLLMClient(chat_model)

# Classification/selection
result = await chat_model.ainvoke(messages)

# Streaming (LLM task passthrough)
async for chunk in client.astream(messages):
    print(chunk.content, end="")
```

### 2. NotebookCache (`core/notebook_cache.py`)

Proactive thread-safe cache for notebook summaries **and source information** with background refresh.

**Features:**
- **Proactive initialization**: Blocking fetch at server startup to warm the cache
- **Source-level fetching**: Fetches source summaries and keywords for each notebook
- **Parallel source fetching**: Uses `asyncio.Semaphore` for controlled concurrency
- **Background refresh**: Automatic refresh at 80% of TTL (prevents expiration)
- **Thread-safe operations**: All cache operations protected with locks
- **Graceful shutdown**: Background thread stops cleanly on server shutdown
- **Graceful degradation**: If source fetch fails, keeps source with basic info
- **Shared instance**: Single cache instance shared via `AgentCore` (used by both OpenAI + MCP)

**Architecture:**
```python
# Server startup (in openai/server.py:main())
notebook_cache = NotebookCache(
    nlm_client=nlm_client,
    ttl_seconds=config.summary_cache_ttl,
    source_fetch_concurrency=config.source_fetch_concurrency
)

# AgentCore uses notebook_cache for routing
agent_core = AgentCore(
    nlm_client, notebook_cache, response_cache, chat_model
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

    def get_source_descriptions(
        self,
        max_sources: int = 10,
        max_keywords: int = 5,
        summary_max_chars: int = 80
    ) -> list[dict]:
        """Get source info with keywords and truncated summaries.
        Returns: [{"title": "...", "keywords": [...], "summary": "..."}, ...]
        Sources beyond max_sources get title only.
        """
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

### 3. LangGraph Routing Graph (`core/routing_graph.py`)

A LangGraph `StateGraph` that replaces the linear `SmartRouter` class with a graph of composable nodes.

**Key Classes:**

```python
class RouterState(TypedDict):
    query: str
    messages: list                         # Reserved for LangGraph memory
    request_type: str | None               # "notebooklm" | "llm_task"
    notebook_id: str | None
    reasoning: str
    available_notebooks: list[dict]
    allowed_notebooks: list[str] | None    # Per-request ACL

@dataclass
class RoutingDecision:                     # Returned by AgentCore.route()
    request_type: str
    notebook_id: str | None = None
    reasoning: str = ""
    cache_result: object | None = None     # CachedResponse on cache hit
    cache_hit_type: str | None = None
    conversation_id: str | None = None
```

**Graph Structure:**

```
START → classify_node → route_after_classify
                           ├─ "notebooklm" → select_notebook_node → END
                           └─ "llm_task"   → END
```

**Routing Flow:**

1. `AgentCore.route(query, options)` — Main entry point
2. Phase 0: Pre-routing global L1 cache check
3. Phase 1: `routing_graph.ainvoke({query, allowed_notebooks})`
   - `classify_node()` — LLM call to determine notebooklm vs llm_task
   - `select_notebook_node()` — LLM call to find best notebook (with ACL filtering)
4. Returns `RoutingDecision` with type, notebook_id, and reasoning
5. On error: fallback to first available notebook (if `agent_fallback_on_error=true`)

**Notebook Selection Data:**
```python
# Data sent to selection LLM (with source_descriptions_enabled=True)
notebooks_info = [
    {
        "id": "abc-123",
        "title": "ML Research",
        "summary": "Notes about machine learning...",
        "topics": ["transformers", "attention", "NLP"],
        "source_count": 5,
        "source_types": {"pdf": 3, "url": 2},
        "sources": [
            # First 10 sources get full descriptions
            {
                "title": "Attention Is All You Need.pdf",
                "keywords": ["transformer", "attention", "encoder-decoder"],
                "summary": "Introduces the Transformer architecture."
            },
            {
                "title": "BERT Paper.pdf",
                "keywords": ["BERT", "pre-training", "NLP"],
                "summary": "Presents bidirectional encoder representations."
            },
            # Remaining sources get title only
            {"title": "GPT-3 Paper.pdf"},
            {"title": "pytorch.org/docs"},
            {"title": "huggingface.co/docs"}
        ]
    },
    ...
]

# With source_descriptions_enabled=False (legacy mode)
notebooks_info = [
    {
        "id": "abc-123",
        ...
        "source_titles": [
            "Attention Is All You Need.pdf",
            "BERT Paper.pdf",
            ...
        ]
    }
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
1. **Source keywords** - Match query terms to source keywords
   (e.g., "neural networks" matches keywords ["neural", "deep learning"])
2. **Source summaries** - Match query intent to source descriptions
   (e.g., "how transformers work" matches summary about attention mechanisms)
3. **Source titles** - If the query mentions a specific document, paper, URL,
   or file name, prioritize notebooks containing sources with matching titles
4. **Source types** - Match query intent to source types (e.g., "PDF paper"
   queries should prefer notebooks with PDF sources)
5. **Notebook summary** - Consider how well the notebook's overall topic
   matches the query
6. **Topics** - Use suggested topics as additional context for relevance

Respond with ONLY the notebook_id (UUID) of the most relevant notebook.
```

### 5. Server Integration (`openai/server.py`)

**Key Functions:**

- `handle_smart_routing()` - Entry point when model="knowledge-finder" (four-phase pipeline)
- `stream_smart_response()` - Phase 3a: Streaming with reasoning_content for routing decision
- `_handle_non_streaming()` - Phase 3b: Non-streaming JSON response

**Four-Phase Pipeline:**
```
POST /v1/chat/completions (model = "knowledge-finder")
  │
  └─> handle_smart_routing()
        │
        ├─ Phase 0: Pre-routing cache check
        │     └─ AgentCore.route() checks global L1 → instant hit skips routing
        │
        ├─ Phase 1: LangGraph routing
        │     └─ AgentCore.route() → classify_node + select_notebook_node
        │
        ├─ Phase 2: Post-routing cache check
        │     └─ ResponseCache.lookup_async() → two-layer lookup (L1 exact + L2 embedding)
        │     └─ HIT → return cached response + X-Cache-Status: HIT
        │
        └─ Phase 3: Execute
              ├─ 3a: stream_smart_response() — SSE with reasoning_content
              │     ├─ If LLM_TASK: stream from LangChain ChatModel
              │     └─ If NOTEBOOKLM: stream from selected notebook
              │           ├─ Save conversation_id via AgentCore session helpers
              │           └─ Store response in ResponseCache
              └─ 3b: _handle_non_streaming() — JSON response
```

**Direct Notebook Queries** (model == notebook_id):
```
chat_completions()
  ├─> AgentCore.handle_direct_query() for cache check
  │     └─ HIT → return cached response + X-Cache-Status: HIT
  └─> NLM query (streaming or non-streaming)
        ├─ Save conversation_id to SessionStore
        └─ Store response in ResponseCache
```

### 6. Response Cache (`core/response_cache.py`)

Two-layer cache that eliminates 40-50s latency for repeated/similar queries:

> **Design Note:** L3 (LLM verification) was removed because it consistently misjudged HIT/MISS decisions in production, decreasing cache precision while adding 1-2s latency per lookup. L2 embedding similarity with a tuned threshold (0.93) proved more reliable and faster.

```
┌────────────────────────────────────────────────────────────────────┐
│                      ResponseCache Lookup                          │
│                                                                    │
│  Query: "What is the attention mechanism?"                        │
│  Notebook: "ML Research"                                          │
│                                                                    │
│  Layer 1: Exact Match (hash-based)                                │
│  ┌─────────────────────────────┐                                  │
│  │ SHA-256(notebook+query)     │── HIT ──> Return immediately     │
│  │ LRU eviction, TTL check    │                                  │
│  └─────────────┬───────────────┘                                  │
│                │ MISS                                              │
│                ▼                                                   │
│  Layer 2: Embedding Similarity (HuggingFace + NumPy)              │
│  ┌─────────────────────────────┐                                  │
│  │ Cosine similarity ≥ 0.93   │── HIT ──> Return cached response  │
│  │ Best match above threshold │── MISS ─> No match found          │
│  └─────────────────────────────┘                                  │
│                                                                    │
│  On L2 HIT: Create alias for instant L1 hits on future repeats   │
└────────────────────────────────────────────────────────────────────┘
```

**Auto-Invalidation:**
- `NotebookCache.on_sources_changed` → `ResponseCache.invalidate_notebook`
- When notebook sources change, all cached responses for that notebook are cleared

**Configuration:** Uses `NLM_PROXY_CACHE_` prefix. See `.env.example` for details.

### 7. Session Mapping (Conversation Continuity)

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
| `NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_ENABLED` | `true` | Include source keywords and summaries |
| `NLM_PROXY_ROUTING_SOURCE_MAX_KEYWORDS` | `5` | Max keywords per source |
| `NLM_PROXY_ROUTING_SOURCE_SUMMARY_MAX_CHARS` | `80` | Max chars of source summary |
| `NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_MAX_SOURCES` | `10` | Sources with full descriptions (rest title only) |

**Example `.env`:**
```bash
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-api-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
NLM_PROXY_ROUTING_ROUTER_MODEL_NAME=knowledge-finder
NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL=3600
NLM_PROXY_ROUTING_SOURCE_FETCH_CONCURRENCY=10
NLM_PROXY_ROUTING_MAX_SOURCE_TITLES=15

# Source descriptions (enhanced routing)
NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_ENABLED=true
NLM_PROXY_ROUTING_SOURCE_MAX_KEYWORDS=5
NLM_PROXY_ROUTING_SOURCE_SUMMARY_MAX_CHARS=80
NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_MAX_SOURCES=10
```

## Per-Request ACL Filtering

The smart router supports **per-request access control lists (ACL)** to restrict which notebooks a user can access based on their permissions. This enables multi-tenant scenarios where different users have access to different subsets of notebooks.

### Two-Layer Filtering Architecture

```
Layer 1 (Server-Wide Cache): NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS env var
    → Controls which notebooks get cached at all (server-wide)
    → Applied at cache initialization

Layer 2 (Per-Request ACL): metadata.allowed_notebooks in request body
    → Filters cached notebooks for each individual request
    → Applied during notebook selection
```

These layers compose naturally: the cache holds only server-allowed notebooks, and per-request ACL further restricts access per user.

### Usage

Send `allowed_notebooks` in the request metadata to restrict notebook access:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": "What's in my research notes?"}],
    extra_body={
        "metadata": {
            "allowed_notebooks": ["nb-abc-123", "nb-def-456"]
        }
    }
)
```

### ACL Behavior

| Scenario | `metadata.allowed_notebooks` | Behavior |
|----------|------------------------------|----------|
| No metadata | (missing) | All cached notebooks accessible |
| Null value | `null` | All cached notebooks accessible |
| Wildcard | `["*"]` | All cached notebooks accessible (normalized to null) |
| Specific IDs | `["nb-1", "nb-2"]` | Only `nb-1` and `nb-2` considered for selection |
| Empty list | `[]` | Returns error: "No accessible notebooks for this user" |
| Non-matching IDs | `["nb-999"]` | Returns error if no IDs match cached notebooks |

**Important Notes:**
- ACL filtering **only applies to NOTEBOOKLM requests**. LLM_TASK classifications bypass ACL filtering entirely.
- The router returns an error message instead of falling back to external LLM when ACL blocks all notebooks. This prevents unauthorized access.
- ACL is enforced during notebook selection—the LLM classification step is not affected.

### Integration Example (Azure AD Groups)

```python
# Example: knowledge-finder-bot backend
# Maps Azure AD group membership to allowed notebooks

# User's AD groups from authentication
user_groups = ["engineering", "research"]

# Map groups to notebook IDs (server-side configuration)
GROUP_TO_NOTEBOOKS = {
    "engineering": ["nb-eng-docs", "nb-api-specs"],
    "research": ["nb-ml-papers", "nb-experiments"],
    "leadership": ["*"]  # Wildcard: access all
}

# Resolve allowed notebooks for this user
allowed_notebooks = []
for group in user_groups:
    notebooks = GROUP_TO_NOTEBOOKS.get(group, [])
    if "*" in notebooks:
        allowed_notebooks = ["*"]  # Wildcard takes precedence
        break
    allowed_notebooks.extend(notebooks)

# Send to nlm-proxy
response = client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": user_query}],
    extra_body={
        "metadata": {
            "allowed_notebooks": allowed_notebooks
        }
    }
)
```

### OpenTelemetry Attributes

ACL filtering adds the following span attributes to `smart_router.select_notebook`:

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `acl_filter_applied` | `bool` | Whether ACL filtering was active | `true` |
| `acl_allowed_count` | `int` | Number of notebook IDs in ACL | `3` |
| `acl_matched_count` | `int` | Number of notebooks that passed ACL filter | `2` |
| `candidates_count` | `int` | Final number of notebooks after ACL filtering | `2` |

**Example trace query (ClickHouse):**

```sql
-- Count ACL rejections (no accessible notebooks)
SELECT
    count() as rejection_count,
    SpanAttributes['acl_allowed_count'] as allowed_count
FROM nlm_traces.routing_traces
WHERE SpanName = 'smart_router.select_notebook'
  AND SpanAttributes['acl_matched_count'] = '0'
GROUP BY allowed_count;

-- Average ACL filtering effectiveness
SELECT
    avg(toInt32(SpanAttributes['acl_matched_count'])) as avg_matched,
    avg(toInt32(SpanAttributes['acl_allowed_count'])) as avg_allowed
FROM nlm_traces.routing_traces
WHERE SpanName = 'smart_router.select_notebook'
  AND SpanAttributes['acl_filter_applied'] = 'true';
```

### Logging

ACL filtering produces debug logs for troubleshooting:

```
DEBUG [SMART-ROUTER] ACL filter: 2 allowed notebooks
DEBUG [ROUTER] ACL filter applied: 5 → 2 notebooks (allowed: ['nb-abc-123', 'nb-def-456'])
```

When ACL blocks all notebooks:

```
WARN [ROUTER] No accessible notebooks for this user (ACL: ['nb-999'])
```

## Logging Tags

The smart routing feature uses specific logging tags for debugging:

| Tag | Component | Description |
|-----|-----------|-------------|
| `[LLM]` | LangChainLLMClient | External LLM API calls |
| `[ROUTING]` | LangGraph routing graph | Classification and notebook selection |
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

INFO  [ROUTING] Starting routing for query: What does the Attention paper say...
DEBUG [ROUTING] Classifying request: What does the Attention paper say...
DEBUG [LLM] complete: prompt=You are a request classifier...
DEBUG [LLM] complete result: notebooklm
INFO  [ROUTING] Classified as notebooklm query
DEBUG [ROUTING] Selecting notebook for query: What does the Attention paper say...
DEBUG [ROUTING] Using 3 cached notebooks
DEBUG [ROUTING] Asking LLM to select from 3 notebooks
INFO  [ROUTING] Selected notebook: ML Research (ID: abc-123)
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
│   ├── config.py          # SmartRoutingSettings, TracingSettings
│   ├── llm_client.py      # ExternalLLMClient
│   └── tracing.py         # OpenTelemetry initialization and utilities
└── openai/
    ├── notebook_cache.py  # NotebookCache, NotebookInfo, SourceInfo
    ├── router.py          # SmartRouter, RequestType, RoutingDecision
    ├── server.py          # handle_smart_routing, stream_smart_response
    └── prompts/
        ├── __init__.py    # load_prompt()
        ├── classify_request.txt
        └── select_notebook.txt  # Updated with source-aware selection
```

## OpenTelemetry Tracing

The smart router is fully instrumented with OpenTelemetry for observability and debugging.

### Span Hierarchy

Each routing request creates a trace with nested spans:

```
smart_router.handle_request (parent span - full request lifecycle)
├── user_query: "What does the Attention paper say?"
├── response_content: "The Transformer architecture..." (truncated)
├── response_source: "notebooklm" or "llm"
│
└── AgentCore.route() → LangGraph StateGraph
    ├── request_type: "notebooklm"
    ├── notebook_id: "abc-123"
    │
    ├── classify_node (LangGraph node)
    │   ├── classification_result: "notebooklm"
    │   └── llm_model: "gpt-4o-mini"
    │
    └── select_notebook_node (LangGraph node)
        ├── acl_filter_applied, acl_matched_count
        ├── candidates_count: 3
        ├── selected_notebook_id: "abc-123"
        └── selected_notebook_title: "ML Research"
```

### Enabling Tracing

```bash
# Environment variables
export NLM_PROXY_OTEL_ENABLED=true
export NLM_PROXY_OTEL_ENDPOINT=http://localhost:4317
export NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy

# Start tracing infrastructure
docker compose -f docker-compose.otel.yml up -d

# Start proxy with tracing
nlm-proxy serve openai --port 8080
```

### Tracing Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NLM_PROXY_OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `NLM_PROXY_OTEL_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint (gRPC) |
| `NLM_PROXY_OTEL_SERVICE_NAME` | `nlm-proxy` | Service name in traces |
| `NLM_PROXY_OTEL_REQUEST_MAX_LENGTH` | `500` | Max chars of user query to store (0 to disable) |
| `NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH` | `1000` | Max chars of response to store (0 to disable) |

### Querying Traces

With ClickHouse storage, you can analyze routing patterns:

```sql
-- Average routing time by request type
SELECT
    SpanAttributes['request_type'] as request_type,
    count() as count,
    avg(Duration)/1000000 as avg_duration_ms
FROM nlm_traces.routing_traces
WHERE SpanName = 'smart_router.route'
GROUP BY request_type;

-- Most selected notebooks
SELECT
    SpanAttributes['selected_notebook_title'] as notebook,
    count() as selections
FROM nlm_traces.routing_traces
WHERE SpanName = 'smart_router.select_notebook'
GROUP BY notebook
ORDER BY selections DESC;
```

See [Tracing Guide](TRACING.md) for complete setup and query examples.
