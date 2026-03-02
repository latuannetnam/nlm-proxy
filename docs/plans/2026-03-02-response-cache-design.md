# Response Cache + Source ID Pre-caching Design

## Problem

NotebookLM takes 40-50 seconds to generate a response. Every query — including repeated or semantically identical questions — incurs this full latency. Additionally, every query makes an unnecessary `get_notebook()` RPC call to fetch source IDs that rarely change.

## Goals

1. **Response Cache**: Return instant responses for repeated/similar queries (0s instead of 40-50s)
2. **Source ID Provider**: Read source IDs from `NotebookCache` instead of calling `get_notebook()` per query (save ~1-3s)

## Scale

- **300+ users**, frequent chatbot usage
- ~1,500-3,000 queries/day, ~1,000-2,000 unique
- ~5-20 notebooks
- Primary language: Vietnamese + English mixed

## Key Decisions

- **Global cache**: Shared across ALL users and conversations. If User A's query is cached, User B gets the cached response instantly.
- **Three-layer lookup**: Exact hash match (0ms) → Embedding pre-filter via fastembed (~10-30ms) → LLM verification (~1-2s)
- **Semantic match scope**: Only available in smart routing mode (where an external LLM is already configured). Direct notebook queries use hash-only matching.
- **First-turn only**: Cache check happens at the **server level** before session store lookup. A query is first-turn when no `conversation_id` exists in the request or session store.
- **Embedding tech**: `fastembed` (ONNX-based) with **multilingual model** (`intfloat/multilingual-e5-small`), CPU by default, optional GPU. NumPy for vectorized similarity (free — already a fastembed dependency).
- **Per-request bypass**: `bypass_cache` field on `ChatCompletionRequest` to force fresh NLM query.

---

## Response Cache

### Cache Entry

```python
@dataclass
class CachedResponse:
    query: str                     # Original query text (for LLM comparison)
    query_hash: str                # Normalized hash key for exact match
    notebook_id: str               # Which notebook answered this
    answer: str                    # The full answer text
    thinking: str | None           # Thinking/reasoning text (collected during streaming)
    conversation_id: str           # The conversation_id from NLM (for follow-ups)
    embedding: list[float] | None  # Pre-normalized query embedding vector
    cached_at: float               # time.time() when cached
    hit_count: int = 0             # Number of cache hits
```

### Per-Request Cache Bypass

Clients can bypass the cache on any request to force a fresh answer from NotebookLM.

**nlm-proxy API** — `bypass_cache` is an explicit field on `ChatCompletionRequest`:
```python
# Added to ChatCompletionRequest (types.py):
bypass_cache: bool = False

# Client usage:
response = client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": "What are the key points?"}],
    extra_body={"bypass_cache": True}
)
```

**Behavior when `bypass_cache=True`**:
- Skip all 3 cache lookup layers (no hash, no embedding, no LLM check)
- Query NotebookLM directly
- **Still store the fresh response in cache** afterward (updates/refreshes the cache entry)

#### Chatbot Integration (knowledge-finder-bot)

Two ways for end users to trigger cache bypass:

**Option A: `/fresh` command prefix**
```
User: /fresh What are the key points?
Bot: [queries NLM directly, bypassing cache]
```
The chatbot detects the `/fresh` prefix, strips it, and passes `extra_body={"bypass_cache": True}` to nlm-proxy.

**Option B: "Get fresh answer" button on cached responses**
When a response is served from cache, the chatbot includes a Teams Adaptive Card button:
```
┌─────────────────────────────────────┐
│ [Answer from cache]                 │
│                                     │
│ ⚡ Answered from cache              │
│ [🔄 Get fresh answer]              │
└─────────────────────────────────────┘
```
Clicking the button re-sends the same query with `bypass_cache=True`. The fresh response replaces the cached one.

---

### Cache Eligibility: First-Turn Detection

Cache check happens at the **server level** (`server.py`), **before** session store lookup. This is critical because `client.query()` always receives a `conversation_id` (either from session store or freshly generated).

```
Request arrives at server
  │
  ├─ Step 1: Extract conversation_id from request
  ├─ Step 2: Check session store for stored conversation_id
  │
  ├─ is_first_turn = (request.conversation_id is None
  │                    AND session store has no stored conv_id for this chat_id)
  │
  ├─ is_first_turn?
  │   ├─ YES → Check response cache (L1 → L2 → L3)
  │   │         If HIT → return cached response (skip NLM entirely)
  │   │         If MISS → continue to NLM query
  │   └─ NO  → Skip cache (follow-up query, context-dependent)
  │
  ├─ Step 3: Session store lookup (load existing conversation_id)
  └─ Step 4: Call client.query() / client.query_stream()
```

**Why at server level**: By the time the request reaches `client.query()`, the conversation_id has already been set (either from session store or generated). Only the server knows if this is truly a first-turn query.

---

### Three-Layer Cache Lookup

```
New query arrives (notebook_id known, is_first_turn=True)
  │
  ├─ bypass_cache=True? → Skip cache, go directly to NLM
  │                        (still stores result in cache afterward)
  │
  ├─ Layer 1: EXACT MATCH (hash, ~0ms)
  │   key = hash(notebook_id + query.strip().lower())
  │   "What are the key points?" == "What are the key points?" → HIT ✅
  │
  ├─ Layer 2: EMBEDDING PRE-FILTER (~10-30ms) — smart routing only
  │   Compute embedding of new query via fastembed
  │   NumPy vectorized cosine similarity against notebook partition
  │   Select top-K most similar candidates (K=10)
  │   Early termination if similarity >= 0.95
  │
  ├─ Layer 3: LLM VERIFICATION (~1-2s) — smart routing only
  │   Send new query + top-K candidates to ExternalLLMClient
  │   LLM confirms: semantically equivalent? → HIT ✅ or MISS
  │
  └─ MISS → Query NotebookLM (40-50s), compute embedding, store in cache
```

**Layer 1** runs for ALL queries (both direct and smart-routed).
**Layers 2+3** run ONLY for smart-routed queries, on Layer 1 miss.

---

### ⚠️ Design Decision: Raw Response on Cache Hit (No LLM Adaptation)

**On any cache hit (exact or semantic), the cached NLM response is returned AS-IS to the client. We do NOT use LLM to rewrite, adapt, or reformat the cached response.**

**Considered and rejected**: Using LLM to post-process the cached response to better match the new query's phrasing (e.g., "key points" → "main takeaways"). This was rejected for the following reasons:

| Factor | Raw Response | LLM-Adapted |
|---|---|---|
| **Speed** | **~0ms** ✅ | ~1-2s (defeats cache purpose) ❌ |
| **Accuracy** | **100% NLM original** ✅ | Risk of distortion/hallucination ❌ |
| **Grounding** | Source-grounded (NLM) ✅ | LLM may invent or lose detail ❌ |
| **Cost** | **Free** ✅ | LLM tokens per hit ❌ |

**Core principle**: If Layer 3 confirms two queries are semantically equivalent — meaning they "would produce the same answer from the same knowledge base" — then the cached answer **IS** the correct answer. No adaptation is needed.

**If the queries would produce different answers** (different scope, format, or language), Layer 3 should classify them as **NO MATCH**, and NLM generates the proper response from scratch. We prefer strict matching over fuzzy adaptation:

```
"What are the key points?"  ≈  "Summarize the main takeaways"  → MATCH (same answer)
"What are the main risks?"  ≠  "List the top 3 risks"          → NO MATCH (different format)
"Explain the budget"        ≠  "How much is the budget?"       → NO MATCH (different depth)
```

**This means Layer 3's prompt must be strict**: only match when the answers would be truly interchangeable. Near-misses should go to NLM for a fresh, properly formatted response. A 40-50s authentic answer is better than a 1-2s hallucination-prone adaptation.

---

### Layer 2: Embedding Pre-filter (fastembed + NumPy)

#### Embedding Model: fastembed (Multilingual)

- **Default model**: `intfloat/multilingual-e5-small` (384 dims, ~100MB)
- **Why multilingual**: Chatbot serves Vietnamese + English queries. English-only models (`bge-small-en`) would fail to match "Tóm tắt điểm chính" ≈ "Nêu các ý chính".
- **Local inference**: ~10-30ms on CPU, ~2-5ms on GPU. No API calls, no cost.
- **ONNX Runtime backend**: Quantized models, auto-detects GPU/CPU at runtime.

**GPU / CPU Support**:

| | CPU (`fastembed`) | GPU (`fastembed-gpu`) |
|---|---|---|
| **Install** | `pip install fastembed` | `pip install fastembed-gpu` |
| **Backend** | `onnxruntime` (~20MB) | `onnxruntime-gpu` (~500MB) |
| **Requirements** | None | CUDA drivers + cuDNN |
| **Latency** | ~10-30ms per query | ~2-5ms per query |
| **Default** | ✅ Yes | Manual opt-in |

**pyproject.toml extras**:
```toml
[project.optional-dependencies]
cache = ["fastembed>=0.4"]
cache-gpu = ["fastembed-gpu>=0.4"]
all = ["fastembed>=0.4", ...]
```

**Graceful degradation** — if `fastembed` is not installed, semantic matching is disabled:
```python
class ResponseCache:
    def __init__(self, ..., semantic_enabled: bool = True):
        self._embedding_model = None
        if semantic_enabled:
            try:
                from fastembed import TextEmbedding
                self._embedding_model = TextEmbedding(self._embedding_model_name)
                logger.info("[CACHE] Embedding model loaded (fastembed)")
            except ImportError:
                logger.info("[CACHE] fastembed not installed, semantic matching disabled")
```

#### Vector Similarity: NumPy Vectorized (Zero-Cost Dependency)

`fastembed` already depends on `numpy` (via `onnxruntime`), so NumPy is available at no extra install cost. This gives us **100-400x faster** similarity search vs pure Python.

**Storage structure — partitioned by notebook**:
```python
self._cache_by_notebook: dict[str, list[CachedResponse]]  # notebook_id → entries
self._cache_by_hash: dict[str, CachedResponse]             # query_hash → entry (L1)
self._notebook_matrices: dict[str, np.ndarray]              # notebook_id → (n, 384) matrix
self._matrix_dirty: dict[str, bool]                         # lazy rebuild flag
```

**Optimizations (stacked)**:

1. **Pre-normalize embeddings** at store time → cosine similarity = dot product
2. **Partition by notebook_id** → search only within relevant notebook
3. **NumPy matrix-vector multiply** → all similarities in one operation
4. **Early termination** → if similarity ≥ 0.95, skip LLM verification entirely
5. **Lazy matrix rebuild** → only rebuild when cache changes

**Similarity search implementation**:
```python
import numpy as np

def _find_similar(self, query_emb: np.ndarray, notebook_id: str, top_k: int = 10):
    entries = self._cache_by_notebook.get(notebook_id, [])
    if not entries:
        return []
    
    # Rebuild matrix if dirty (new entries added)
    if self._matrix_dirty.get(notebook_id, True):
        self._rebuild_matrix(notebook_id)
    
    matrix = self._notebook_matrices[notebook_id]  # (n, 384)
    
    # Single matrix-vector multiply — all dot products at once
    similarities = matrix @ query_emb  # (n,) — pre-normalized = cosine sim
    
    # Early termination: near-perfect match → skip LLM verification
    max_sim = similarities.max()
    if max_sim >= 0.95:
        best_idx = int(similarities.argmax())
        return [(max_sim, entries[best_idx])]
    
    # Filter by threshold and get top-K
    mask = similarities >= self._similarity_threshold
    if not mask.any():
        return []
    
    filtered_idx = np.where(mask)[0]
    sorted_idx = filtered_idx[np.argsort(similarities[filtered_idx])[::-1]][:top_k]
    return [(float(similarities[i]), entries[i]) for i in sorted_idx]
```

**Performance at scale**:

| Cache size | Per notebook (10 nb) | NumPy similarity | Pure Python (for comparison) |
|---|---|---|---|
| 500 entries | ~50/nb | **~0.02ms** | ~2-5ms |
| 1000 entries | ~100/nb | **~0.05ms** | ~5-10ms |
| 2000 entries | ~200/nb | **~0.1ms** | ~15-40ms |
| 5000 entries | ~500/nb | **~0.2ms** | ~40-100ms |

---

### Layer 3: LLM Verification

**Purpose**: Confirm that embedding-similar candidates are truly semantically equivalent.

**Prompt** (multilingual with Vietnamese examples):
```
You are a cache lookup assistant. Determine if the new question is asking
essentially the same thing as any previously cached question. Two questions
match if they would produce the same answer from the same knowledge base.

Rules:
- Match: same intent, just different wording
  "What are the key points?" ≈ "Summarize the main takeaways"
  "Tóm tắt điểm chính" ≈ "Nêu các ý chính"
- No match: related topic but different scope or different info requested
  "What happened in Q1?" ≠ "What happened in Q2?"
  "List the team members" ≠ "Who is the project lead?"

New question: "{new_query}"

Cached questions:
1. "{cached_query_1}"
2. "{cached_query_2}"
...

Reply with ONLY the number of the matching question, or -1 if no match.
```

**Response parsing**:
```python
def _parse_semantic_match(self, response: str, num_candidates: int) -> int | None:
    """Parse LLM response to matched index (0-based) or None."""
    text = response.strip()
    if text == "-1" or text.lower() in ("none", "no match"):
        return None
    try:
        index = int(text) - 1  # 1-based → 0-based
        if 0 <= index < num_candidates:
            return index
    except ValueError:
        pass
    match = re.search(r'\b(\d+)\b', text)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < num_candidates:
            return index
    return None
```

**Error handling**:
| Scenario | Behavior |
|---|---|
| Embedding model not loaded | Skip Layers 2-3, proceed to NLM |
| LLM timeout (>5s) | Skip Layer 3, proceed to NLM |
| LLM returns garbage | Parse returns None → miss |
| No candidates above threshold | Skip Layer 3, proceed to NLM |
| 0 cached entries for notebook | Skip Layers 2-3 entirely |
| Similarity ≥ 0.95 | Skip Layer 3, return directly (near-exact match) |

---

### Storage: In-Memory LRU + TTL

- **LRU capacity**: Default 1000 (covers full work day for 300+ users, ~15MB)
- **TTL**: Default 4 hours (balances hit rate vs freshness)
- **Thread-safe**: `threading.Lock`
- **Memory**: ~15KB per entry → 1000 entries ≈ 15MB
- **Volatile**: Cache is lost on process restart (known limitation, documented below)

### Where to Implement

**New module: `core/response_cache.py`** — contains `ResponseCache` class with all three layers.

Cache lookup happens at the **server level** (`server.py` / `handle_smart_routing`) before calling `client.query()` / `client.query_stream()`. This is necessary for correct first-turn detection and `bypass_cache` handling.

### How `ExternalLLMClient` Reaches `ResponseCache`

The `ExternalLLMClient` for Layer 3 is created **separately in `server.py` startup** (not pulled from `SmartRouter`):

```python
# In server.py lifespan:
llm_client = ExternalLLMClient(
    base_url=routing_settings.llm_base_url,
    api_key=routing_settings.llm_api_key,
    model=routing_settings.llm_model,
)

response_cache = ResponseCache(
    llm_client=llm_client,      # For Layer 3 semantic verification
    semantic_enabled=cache_settings.semantic_match_enabled,
    embedding_model=cache_settings.embedding_model,
    ...
)
app.state.response_cache = response_cache
```

### Streaming Cache Behavior

- **Cache HIT on streaming request**: Yield cached answer as a single chunk (instant delivery). Thinking text included as `reasoning_content` if available.
- **Cache MISS on streaming request**: Stream normally from NLM. Collect both answer text and thinking text during streaming. Store complete response (answer + thinking) in cache after stream finishes.
- **Cache HIT on non-streaming request**: Return cached response dict immediately.

**Thinking text collection** (during streaming cache miss):
```python
# In the streaming path (server.py):
thinking_text = ""
answer_text = ""
async for chunk in client.query_stream(...):
    if chunk["type"] == "thinking":
        thinking_text = chunk["text"]  # cumulative
    elif chunk["type"] == "answer":
        answer_text = chunk["text"]    # cumulative
    yield chunk  # forward to client

# After stream completes, cache the full response:
response_cache.store(
    notebook_id=notebook_id,
    query=query_text,
    answer=answer_text,
    thinking=thinking_text,
    conversation_id=conversation_id,
)
```

### Cache Invalidation

1. **TTL-based**: Entries expire after configurable seconds
2. **Source change auto-invalidation**: When `NotebookCache` detects a notebook's sources have changed during periodic refresh, all `ResponseCache` entries for that notebook are invalidated (see NotebookCache Integration below)
3. **Manual purge**: `clear_response_cache(notebook_id=None)` — clear all or per-notebook

### Cache Hit Signaling to Clients

**Problem**: `ChatCompletionResponse` and `ChatCompletionChunk` have no `metadata` field. We need a way for the chatbot to detect cache hits.

**Solution — two complementary mechanisms**:

1. **HTTP response header**: `X-Cache-Status: HIT_EXACT`, `HIT_SEMANTIC`, `MISS`, or `BYPASS`
   - Works for both streaming and non-streaming
   - Chatbot reads via `response.headers`

2. **`system_fingerprint` encoding**: Include cache status in the existing field:
   - Cache hit: `"cache_exact_conv_{conv_id}"` or `"cache_semantic_conv_{conv_id}"`
   - Cache miss: `"conv_{conv_id}"` (unchanged from current behavior)

```python
# In server.py:
if cache_result.hit:
    response.headers["X-Cache-Status"] = f"HIT_{cache_result.hit_type.upper()}"
    fingerprint = f"cache_{cache_result.hit_type}_conv_{conv_id}"
else:
    response.headers["X-Cache-Status"] = "MISS"
    fingerprint = f"conv_{conv_id}"
```

- Logging: `[CACHE] HIT (exact|semantic) / MISS / BYPASS` at info level

---

## NotebookCache Integration

The existing `NotebookCache` (periodic background refresh) serves two roles for the response cache:

### 1. Source ID Provider

`NotebookCache` already proactively fetches and caches source information (`SourceInfo.id`) for all notebooks. Instead of a separate source ID cache, `query()`/`query_stream()` reads source IDs directly from `NotebookCache`:

```python
# In query()/query_stream() — before querying NLM:
if source_ids is None and notebook_cache:
    info = notebook_cache.get(notebook_id)
    if info and info.sources:
        source_ids = [s.id for s in info.sources]

# Fallback only when NotebookCache is unavailable (e.g., MCP mode)
if source_ids is None:
    notebook_data = await self.get_notebook(notebook_id)
    source_ids = self._extract_source_ids_from_notebook(notebook_data)
```

**Benefits**:
- Eliminates `get_notebook()` RPC call on every query (saves ~1-3s)
- No separate cache to maintain — `NotebookCache` handles refresh
- Source IDs are always fresh (proactively refreshed at 80% of TTL)

### 2. Source Change Detection → Auto-Invalidation

When `NotebookCache` refreshes and detects that a notebook's sources have changed (different source IDs vs previous refresh), it **automatically invalidates** all `ResponseCache` entries for that notebook.

**Implementation via callback**:
```python
class NotebookCache:
    def __init__(self, ..., on_sources_changed: Callable[[str], None] | None = None):
        self._on_sources_changed = on_sources_changed

    def set(self, notebook_id, title, summary, topics, sources=None):
        with self._lock:
            old_info = self._cache.get(notebook_id)
            old_source_ids = {s.id for s in old_info.sources} if old_info else set()
            new_source_ids = {s.id for s in (sources or [])}

            # Store updated info
            self._cache[notebook_id] = NotebookInfo(...)

            # Detect source changes → notify ResponseCache
            if old_source_ids and old_source_ids != new_source_ids:
                logger.info(
                    f"[CACHE] Sources changed for {notebook_id}: "
                    f"{len(old_source_ids)} → {len(new_source_ids)}"
                )
                if self._on_sources_changed:
                    self._on_sources_changed(notebook_id)
```

**Wiring in server startup** (`openai/server.py`):
```python
# Create ResponseCache first
response_cache = ResponseCache(...)

# Create NotebookCache with invalidation callback
notebook_cache = NotebookCache(
    nlm_client=client,
    on_sources_changed=response_cache.invalidate_notebook
)
```

**Invalidation flow**:
```
NotebookCache background refresh (every ~48 min)
  │
  ├─ Fetches notebook sources
  ├─ Compares with previous source IDs
  ├─ Sources changed?
  │   ├─ YES → calls on_sources_changed(notebook_id)
  │   │         → ResponseCache.invalidate_notebook(notebook_id)
  │   │         → All cached responses for that notebook cleared
  │   │         → NumPy matrix for that notebook rebuilt
  │   └─ NO  → No action
```

---

## Configuration

New `CacheSettings` class with `NLM_PROXY_CACHE_` prefix:

| Variable | Default | Description |
|---|---|---|
| `NLM_PROXY_CACHE_RESPONSE_CACHE_ENABLED` | `true` | Enable response caching |
| `NLM_PROXY_CACHE_RESPONSE_CACHE_TTL` | `14400` (4h) | Response cache TTL (seconds) |
| `NLM_PROXY_CACHE_RESPONSE_CACHE_MAX_ENTRIES` | `1000` | Max cached responses (LRU) |
| `NLM_PROXY_CACHE_SEMANTIC_MATCH_ENABLED` | `true` | Enable semantic matching (smart routing only) |
| `NLM_PROXY_CACHE_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | fastembed model for embeddings |
| `NLM_PROXY_CACHE_SIMILARITY_THRESHOLD` | `0.7` | Minimum cosine similarity for candidates |
| `NLM_PROXY_CACHE_SIMILARITY_EXACT_THRESHOLD` | `0.95` | Threshold for skipping LLM verification |
| `NLM_PROXY_CACHE_SEMANTIC_MATCH_TOP_K` | `10` | Max candidates sent to LLM |

---

## Architecture Diagram

```
Client Request (bypass_cache field on ChatCompletionRequest)
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  OpenAI Server (server.py)                               │
│  1. First-turn detection (before session store lookup)   │
│  2. Response cache check (L1 → L2 → L3)                 │
│  3. Session store lookup (load conversation_id)          │
│  4. Cache HIT → return immediately (headers: X-Cache)    │
│     Cache MISS → call client, store result after         │
└──────────────┬───────────────────────────────────────────┘
               │ MISS
               ▼
┌──────────────────────────────────────────────────────────┐
│  NotebookLMClient                                        │
│  source IDs from NotebookCache                           │
│  query_stream() / query() → NotebookLM API              │
└──────────────────────────────────────────────────────────┘
               ▲
               │ on_sources_changed(notebook_id)
┌──────────────┴───────────────────────────────────────────┐
│  NotebookCache (openai/notebook_cache.py)                 │
│  Background refresh every ~48 min                         │
│  Detects source changes → invalidates ResponseCache       │
│  Provides source IDs (no separate cache needed)           │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  ResponseCache (core/response_cache.py)                   │
│  L1: Exact hash match (0ms)                               │
│  L2: fastembed multilingual + NumPy cosine (~0.1ms)       │
│  L3: ExternalLLMClient semantic verification (~1-2s)      │
│  LRU 1000 + TTL 4h, partitioned by notebook               │
│  Global across all users, volatile (lost on restart)      │
└──────────────────────────────────────────────────────────┘
```

---

## What Changes, What Doesn't

| Component | Changes? | Details |
|---|---|---|
| `core/response_cache.py` | **NEW** | `ResponseCache` class with 3-layer lookup, NumPy |
| `core/config.py` | **MODIFY** | Add `CacheSettings` class |
| `core/client.py` | **MODIFY** | Read source IDs from NotebookCache |
| `openai/notebook_cache.py` | **MODIFY** | Add `on_sources_changed` callback, source change detection |
| `openai/server.py` | **MODIFY** | Cache check at server level, wire caches, `ExternalLLMClient` init, `X-Cache-Status` headers |
| `openai/types.py` | **MODIFY** | Add `bypass_cache: bool = False` to `ChatCompletionRequest` |
| `pyproject.toml` | **MODIFY** | Add `cache` and `cache-gpu` extras |
| `mcp/server.py` | No change | Benefits from hash-only cache via `ResponseCache` if wired |
| Chatbot `nlm/client.py` | **MODIFY** | Pass `bypass_cache` in `extra_body`, read `X-Cache-Status` header |
| Chatbot `bot/bot.py` | **MODIFY** | Add `/fresh` command + "Get fresh answer" button on cache hits |

---

## Edge Cases

1. **Same question, different users/conversations**: Cache HIT — key ignores user/conversation IDs
2. **Follow-up questions**: Not cached — detected at server level before session store lookup
3. **Embedding similar but different intent**: LLM verification rejects false positives
4. **Similarity ≥ 0.95**: Skip LLM verification — near-exact match returned directly
5. **LLM or embedding unavailable**: Layers 2-3 skipped, falls through to NLM
6. **fastembed not installed**: Semantic matching disabled, hash-only cache still works
7. **Empty/error responses**: Not cached
8. **Cache disabled**: `response_cache_enabled=false` bypasses entirely
9. **Concurrent identical queries**: First populates cache, second may still miss (acceptable in v1)
10. **1000+ cached entries**: NumPy vectorized similarity ~0.1ms (100-400x faster than pure Python)
11. **GPU available**: ONNX Runtime auto-detects, no config needed
12. **`bypass_cache=True`**: Skips all cache layers, queries NLM directly, still stores result
13. **`/fresh` prefix with empty query**: Chatbot rejects with helpful error message
14. **Sources changed on refresh**: `NotebookCache` auto-invalidates all response cache entries for that notebook
15. **NotebookCache unavailable** (MCP mode): Falls back to `get_notebook()` for source IDs, no semantic matching
16. **ACL-filtered notebooks**: Cache is keyed on `notebook_id`. If routing selects a different notebook for different users (due to ACL), they get different cache entries — correct behavior
17. **Server restart**: All cached responses are lost. Cache rebuilds organically as queries come in. No warm-up mechanism in v1.
18. **Vietnamese + English mixed queries**: Multilingual embedding model handles cross-language similarity
