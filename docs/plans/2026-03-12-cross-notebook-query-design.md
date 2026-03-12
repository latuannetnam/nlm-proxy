# Cross-Notebook Query Support — Design Document v2

> **Supersedes**: `docs/plans/2026-02-02-cross-notebook-query-design.md` (pre-LangGraph refactor)
>
> **Date**: 2026-03-12
>
> **Status**: Design Review

## 1. Problem Statement

NLM Proxy's smart router currently selects **one notebook** per query. For complex queries that span multiple knowledge domains (e.g., "How does our networking architecture affect billing cycle calculations?"), a single notebook often cannot provide a complete answer. Users need the system to query multiple notebooks and combine insights.

### Constraints

| Constraint | Detail |
|-----------|--------|
| **NotebookLM API is 1:1** | Each `query()`/`query_stream()` targets exactly one `notebook_id` |
| **Query latency** | 3–40s per notebook query |
| **Daily quota** | 500 chat requests/day per Pro account |
| **Grounding precision** | NotebookLM answers are source-grounded; LLM post-processing risks losing this |

### Design Goals

1. **Preserve TTFT** — Primary notebook streams immediately (~800ms TTFT, same as today)
2. **Transparency** — Synthesized answers include quoted originals from each notebook
3. **Quota efficiency** — LLM decides when cross-notebook is needed (not always-on)
4. **Graceful degradation** — Any failure degrades to single-notebook behavior
5. **Security** — No ACL leaks through cached cross-notebook responses

---

## 2. Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | Approach 1: Extended Select Node | Zero routing latency increase, low regression risk, preserves tracing granularity |
| **Response format** | LLM synthesis + quoted originals | User gets convenient combined answer AND can verify against raw notebook responses |
| **Streaming strategy** | Stream primary first, append synthesis | Preserves existing TTFT; synthesis is bonus content appended after primary |
| **Triggering** | LLM-detected (router decides) | Conserves quota — only triggers when query genuinely spans multiple notebooks |
| **Cache strategy** | Store under primary notebook_id, skip global L1 for cross-notebook | Prevents ACL leaks while preserving notebook-scoped L1 and L2 |
| **Error handling** | Graceful degradation at every layer | Worst case = today's single-notebook behavior |

---

## 3. Architecture

### 3.1 Routing Graph (Unchanged Structure)

The LangGraph maintains its current two-node structure. Only `select_notebook_node` is enhanced:

```
START → classify_node → route_after_classify
                           ├── "notebooklm" → select_notebook_node (enhanced) → END
                           └── "llm_task"   → END
```

### 3.2 Enhanced RouterState

```python
class RouterState(TypedDict):
    query: str
    messages: list
    request_type: str | None          # "notebooklm" | "llm_task"
    notebook_id: str | None            # Primary notebook UUID
    secondary_notebook_ids: list[str]  # NEW — secondary notebook UUIDs
    cross_notebook: bool               # NEW — whether cross-notebook was triggered
    reasoning: str
    available_notebooks: list[dict]
    allowed_notebooks: list[str] | None
```

### 3.3 Enhanced RoutingDecision

```python
@dataclass
class RoutingDecision:
    request_type: str
    notebook_id: str | None = None
    secondary_notebook_ids: list[str] = field(default_factory=list)  # NEW
    cross_notebook: bool = False                                      # NEW
    reasoning: str = ""
    cache_result: object | None = None
    cache_hit_type: str | None = None
    conversation_id: str | None = None
```

### 3.4 End-to-End Flow (Cross-Notebook)

```
Time 0ms:     POST /v1/chat/completions (model="knowledge-finder")
              │
              ├─ Phase 0: Pre-routing global L1 cache check
              │   └─ (No cross-notebook responses in global L1 by design)
              │
              ├─ Phase 1: LangGraph routing
              │   ├─ classify_node: "notebooklm" (~600ms)
              │   └─ select_notebook_node (enhanced): (~700ms)
              │       → primary: notebook-A
              │       → secondary: [notebook-B]
              │       → cross_notebook: true
              │
              ├─ Phase 2: Post-routing cache check — SKIPPED for cross-notebook
              │   └─ (Single-notebook queries still use Phase 2)
              │   └─ Reason: a cached single-notebook response would bypass
              │      the cross-notebook flow, returning stale single-source answer
              │
              └─ Phase 3: Execute (cross-notebook flow)
                  │
Time ~1.3s:       ├─ Launch secondary queries as background tasks
                  │   └─ asyncio.create_task(_query_secondary(notebook-B, query))
                  │       (constrained by global semaphore)
                  │
Time ~1.5s:       ├─ ⚡ Stream primary (notebook-A) to client
                  │   └─ SSE chunks flowing...
                  │
Time ~5-30s:      ├─ Primary stream completes
                  │   └─ Stream progress: "📚 Cross-referencing additional sources..."
                  │
Time ~5-35s:      ├─ Await secondary results (likely already done)
                  │
Time ~6-36s:      ├─ LLM synthesis call (if secondaries returned content)
                  │   └─ Synthesize + format with quoted originals
                  │
Time ~7-37s:      └─ Stream synthesis + quotes to client
                      └─ Stream [DONE]
```

### 3.5 Response Format

```markdown
[Primary notebook answer streams here as normal...]

---

📚 **Cross-referenced from additional sources:**

[LLM-synthesized combined insight — 2-4 sentences highlighting what
the secondary notebooks add beyond the primary answer]

> 📗 **From "Billing System Docs":**
> [Original NotebookLM answer from secondary notebook, verbatim]

> 📘 **From "Network Architecture":**
> [Original NotebookLM answer from secondary notebook, verbatim]
```

---

## 4. Component Changes

### 4.1 Configuration (`core/config.py`)

Add to `SmartRoutingSettings`:

```python
# Cross-notebook query settings
cross_notebook_enabled: bool = True
cross_notebook_max_secondary: int = 2           # Max secondary notebooks per query
cross_notebook_concurrency: int = 5             # Global semaphore limit for background queries
cross_notebook_synthesis_enabled: bool = True    # Whether to run LLM synthesis
cross_notebook_timeout: int = 30                # Max seconds to wait for secondaries after primary
cross_notebook_secondary_max_chars: int = 2000  # Max chars of secondary answer in synthesis prompt
# NOTE: section_marker is a hardcoded constant in server.py (not configurable via env var
# because multi-line markdown formatting is unwieldy as an environment variable)
```

New env vars: `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_*`

### 4.2 Select Notebook Prompt Enhancement (`openai/prompts/select_notebook.txt`)

The prompt is enhanced to also assess cross-notebook need. **The cross-notebook assessment section is only appended when `cross_notebook_enabled=true`** — when disabled, the prompt remains unchanged and the LLM response format stays as-is (single UUID).

Key additions (conditional):

```
CROSS-NOTEBOOK ASSESSMENT:
After selecting the primary notebook, assess whether the query would benefit
from supplementary information from other notebooks.

Trigger cross-notebook ONLY when:
- The query explicitly spans multiple topics covered by DIFFERENT notebooks
- Example: "How does X (from notebook A) relate to Y (from notebook B)?"
- Example: "Compare the approach in docs A with the approach in docs B"

Do NOT trigger cross-notebook when:
- The primary notebook likely contains the full answer
- The query is simple/single-topic
- Example: "What is X?" — single notebook is sufficient

Respond with JSON:
{
  "notebook_id": "PRIMARY-UUID",
  "secondary_notebook_ids": ["UUID-2"],  // empty array if no cross-notebook needed
  "cross_notebook": true,                // or false
  "reasoning": "..."
}
```

### 4.3 Routing Graph (`core/routing_graph.py`)

**Changes to `select_notebook_node`:**
- **Conditionally** append cross-notebook assessment to prompt (only when `cross_notebook_enabled=true`)
- Parse JSON response (currently parses UUID from free text)
- Extract `secondary_notebook_ids` and `cross_notebook` flag
- Apply ACL filtering to secondary notebooks too
- Fallback: if JSON parsing fails, fall back to current UUID extraction (backward compatible — feature silently disabled)

```python
# Conditional prompt construction:
cross_notebook_section = ""
if routing_settings and routing_settings.cross_notebook_enabled:
    cross_notebook_section = load_prompt("select_notebook_cross_section")

prompt = prompt_template.format(
    notebooks_json=json.dumps(notebooks_info, indent=2),
    query=query,
    cross_notebook_instructions=cross_notebook_section,
)
```

**New span attributes:**
- `cross_notebook_detected: bool`
- `secondary_notebook_ids: list[str]`
- `secondary_notebook_count: int`

### 4.4 Agent Core (`core/agent.py`)

**Changes to `RoutingDecision`:**
- Add `secondary_notebook_ids: list[str] = field(default_factory=list)`
- Add `cross_notebook: bool = False`

**Changes to `route()`:**
- Extract new fields from `RouterState` using `.get()` with safe defaults (LangGraph state may not contain these keys if `select_notebook_node` fell back to UUID extraction)
- Record new span attributes

```python
# Safe extraction from LangGraph state (new fields may be absent):
decision = RoutingDecision(
    request_type=state["request_type"],
    notebook_id=state.get("notebook_id"),
    secondary_notebook_ids=state.get("secondary_notebook_ids", []),  # safe default
    cross_notebook=state.get("cross_notebook", False),               # safe default
    reasoning=state.get("reasoning", ""),
    conversation_id=options.conversation_id,
)
```

### 4.5 Server Streaming (`openai/server.py`)

**New functions:**

```python
# Module-level semaphore
_cross_notebook_semaphore: asyncio.Semaphore | None = None

# Hardcoded formatting constant (not env-configurable, see §4.1 note)
CROSS_NOTEBOOK_SECTION_MARKER = "\n\n---\n\n📚 **Cross-referenced from additional sources:**\n\n"

async def _query_secondary(
    agent_core: AgentCore,
    notebook_id: str,
    notebook_title: str,  # Needed for formatting quoted originals
    query: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> dict | None:
    """Background query to a secondary notebook with semaphore + timeout."""

async def _synthesize_cross_notebook(
    chat_model,
    query: str,
    primary_answer: str,
    secondary_results: list[dict],  # [{notebook_title, answer}]
) -> str | None:
    """LLM synthesis of cross-notebook results. Returns None if no synthesis needed."""

async def stream_cross_notebook_response(
    agent_core: AgentCore,
    decision: RoutingDecision,
    query: str,
    request: ChatCompletionRequest,
    chat_id: str | None,
    tracing_settings,
    routing_settings,
):
    """Phase 3 streaming handler for cross-notebook queries."""
```

**Changes to `handle_smart_routing()`:**

- **Phase 2 bypass**: Skip post-routing cache check when `decision.cross_notebook=true` (a cached single-notebook response would prevent the cross-notebook flow from running)
- After routing, check `decision.cross_notebook`
- If true → `stream_cross_notebook_response()` (streaming only)
- If false → existing `stream_smart_response()` (unchanged)

```python
# Phase 2: Post-routing cache — skip for cross-notebook to avoid stale single-notebook hits
if (
    decision.request_type == "notebooklm"
    and not decision.cross_notebook              # ← NEW: skip for cross-notebook
    and not options.bypass_cache
    and agent_core.response_cache
):
    cache_result, hit_type = await agent_core.response_cache.lookup_async(...)
    # ... existing cache hit logic ...

# Phase 3: Execute
if decision.cross_notebook and decision.secondary_notebook_ids:
    if request.stream:
        return StreamingResponse(stream_cross_notebook_response(...))
    else:
        # V1: Non-streaming cross-notebook falls back to primary-only
        # (synthesis requires streaming to work naturally)
        logger.info("[CROSS-NOTEBOOK] Non-streaming request — returning primary-only")
        return await _handle_non_streaming(agent_core, decision, query, request, chat_id, tracing_settings)
else:
    # Existing single-notebook path (unchanged)
    if request.stream:
        return StreamingResponse(stream_smart_response(...))
    else:
        return await _handle_non_streaming(...)
```

> **Design decision**: Non-streaming cross-notebook is deferred to V2. In V1, non-streaming requests with `cross_notebook=true` fall back to primary-only via the existing `_handle_non_streaming()` path. This is acceptable because the vast majority of OpenAI-compatible clients (Open WebUI, Cursor) use streaming.

### 4.6 Response Cache (`core/response_cache.py`)

**Changes to `CachedResponse`:**
```python
@dataclass
class CachedResponse:
    # ... existing fields ...
    secondary_notebook_ids: list[str] = field(default_factory=list)  # NEW
```

**Changes to `store()`:**
- Accept optional `secondary_notebook_ids` parameter
- If `secondary_notebook_ids` is non-empty, **skip** global hash index storage (ACL safety)

**Changes to `invalidate_notebook()`:**
- After clearing entries where `entry.notebook_id == notebook_id`, also scan for entries where `notebook_id in entry.secondary_notebook_ids` and remove those too

**Caller update** — `stream_cross_notebook_response()` in `server.py` must pass `secondary_notebook_ids` when storing the composite response:

```python
# After primary stream + synthesis completes, store the composite response:
agent_core.response_cache.store(
    notebook_id=decision.notebook_id,
    query=query,
    answer=full_composite_response,          # primary + synthesis + quoted originals
    thinking=previous_thinking or None,
    conversation_id=conversation_id,
    embedding=embedding,
    secondary_notebook_ids=decision.secondary_notebook_ids,  # ← NEW
)
```

### 4.7 New Prompt: Synthesis (`openai/prompts/synthesize_cross_notebook.txt`)

```
You are a research assistant synthesizing information from multiple knowledge bases.

USER QUERY:
{query}

PRIMARY ANSWER (already shown to user — do not repeat):
{primary_answer_truncated}

SECONDARY SOURCE FINDINGS:
{secondary_results}

INSTRUCTIONS:
1. Determine if secondary findings add SIGNIFICANT NEW information beyond the primary answer.
2. If YES:
   - Write a concise synthesis (2-4 sentences max).
   - Focus ONLY on new information not covered in the primary answer.
   - Do NOT repeat what was in the primary answer.
   - Do NOT include citation numbers like [1], [2].
   - Keep under 200 words.
3. If NO (secondary findings are redundant, irrelevant, or empty):
   - Return exactly: NO_SYNTHESIS_NEEDED

OUTPUT:
```

---

## 5. Risk Mitigations — Detailed Design

### 5.1 🔴 ACL Leak via Cached Cross-Notebook Responses

**Risk**: User A (ACL: notebooks `[1, 2, 3]`) asks a cross-notebook query → primary notebook 1 + secondary notebook 2. The composite response is cached in the **global L1 hash index** (query-only hash, no notebook_id). User B (ACL: `[1, 3]`) asks the same query → global L1 returns the cached response containing notebook 2's content — **ACL violation**.

**Design**:

```
                    ┌─────────────────────────────────────────────┐
                    │            store() Decision Tree              │
                    │                                               │
                    │  secondary_notebook_ids is empty?             │
                    │       YES ──→ Store in global L1 index ✅     │
                    │       NO  ──→ SKIP global L1 index ⛔        │
                    │               (only store in notebook L1)     │
                    └─────────────────────────────────────────────┘
```

**Changes to `response_cache.py`:**

```python
def store(
    self,
    notebook_id: str,
    query: str,
    answer: str,
    thinking: str | None,
    conversation_id: str,
    embedding: list[float] | None = None,
    secondary_notebook_ids: list[str] | None = None,  # NEW
) -> None:
    # ... existing entry creation ...

    entry = CachedResponse(
        # ... existing fields ...
        secondary_notebook_ids=secondary_notebook_ids or [],  # NEW
    )

    # Store in notebook-scoped L1 (always)
    self._cache_by_hash[query_hash] = entry

    # Store in global L1 index ONLY if single-notebook response
    if not secondary_notebook_ids:
        global_hash = self._compute_global_hash(query)
        self._global_hash_index[global_hash] = entry
    else:
        logger.debug(
            "[CACHE] Skipping global L1 for cross-notebook response "
            "(primary=%s, secondaries=%s)",
            notebook_id[:12], [s[:12] for s in secondary_notebook_ids],
        )

    # ... rest of store logic ...
```

**Verification**: Unit test confirming cross-notebook responses are NOT retrievable via `lookup_global()`.

---

### 5.2 🟠 Follow-Up Questions Lose Secondary Context

**Risk**: User asks cross-notebook query → gets synthesis mentioning "billing cycle X from Billing Docs." Follow-up: "Tell me more about billing cycle X" → routed to primary (Networking) notebook which doesn't know about billing cycles.

**Design**:

```
                     Query 1: Cross-notebook
                     ┌─────────────────────────────────────┐
                     │ Primary: Networking (conv_id=abc)    │
                     │ Secondary: Billing (conv_id=xyz)     │
                     │ SessionStore: chat_id → conv_id=abc  │ ← only primary tracked
                     └─────────────────────────────────────┘

                     Query 2: Follow-up "billing cycle X"
                     ┌─────────────────────────────────────────────┐
                     │ classify: notebooklm                         │
                     │ select: routes to Billing (query mentions    │
                     │         billing → different notebook!)       │
                     │ conv_id: None (new conversation)             │
                     │                                               │
                     │ Result: Gets answer from Billing notebook ✅  │
                     │ (Router re-evaluates independently each time) │
                     └─────────────────────────────────────────────┘
```

**V1 behavior (accept limitation)**:
- Only track primary notebook's `conversation_id` in `SessionStore`
- The router's independent per-query classification handles most cases naturally — if the follow-up mentions billing, the router will route to the Billing notebook
- Edge case: follow-up doesn't clearly indicate which notebook → may go to wrong one

**V2 enhancement (future — not in this implementation)**:

```python
# SessionStore could be extended:
class CrossNotebookSession:
    primary_notebook_id: str
    primary_conversation_id: str
    secondary_notebook_ids: list[str]  # Hints for re-routing
    last_query_was_cross_notebook: bool

# select_notebook_node could use this hint:
if session.last_query_was_cross_notebook:
    # Boost relevance scores for notebooks in session.secondary_notebook_ids
    pass
```

**Documentation**: Add a note to `docs/smart-routing-architecture.md` explaining this limitation.

---

### 5.3 🟠 Secondary Queries Create Orphan Conversations

**Risk**: Each secondary notebook query creates a new NotebookLM conversation. These orphan conversations:
- Consume the 500/day chat quota
- Accumulate in NotebookLM's conversation list
- Are never continued (one-shot)

**Design**:

```
                     Cross-notebook query flow:
                     ┌──────────────────────────┐
                     │ Primary query             │
                     │ notebook-A, conv_id=abc   │ ← tracked, reusable
                     └──────────────────────────┘

                     ┌──────────────────────────┐
                     │ Secondary query (one-shot)│
                     │ notebook-B, conv_id=xyz   │ ← orphan, not tracked
                     └──────────────────────────┘
                     │ Why not track?            │
                     │ - Would need per-notebook  │
                     │   conversation mapping     │
                     │ - Secondary notebooks may  │
                     │   change per query          │
                     │ - Complexity not worth it   │
                     │   for supplementary queries │
                     └──────────────────────────┘
```

**Quota impact analysis**:
- Conservative estimate: 20% of queries trigger cross-notebook (LLM-detected)
- Average 1.5 secondaries per cross-notebook query
- 100 daily queries × 20% × 1.5 = **30 extra conversations/day** (~6% of 500 quota)
- Acceptable overhead given the value provided

**Mitigation**: No code change needed. LLM-detected triggering (§5.7) naturally limits frequency. The `cross_notebook_max_secondary=2` cap bounds the worst case.

---

### 5.4 🟡 Streaming Pause Between Primary and Synthesis

**Risk**: After primary answer finishes streaming (~5-30s), there's silence while:
1. Awaiting secondary results (may already be done, or up to 30s more)
2. Running synthesis LLM call (~500-1500ms)

Some clients interpret silence as "stream complete" and may close the connection. Users see a frozen UI.

**Design**:

```
Primary stream             Gap            Synthesis stream
━━━━━━━━━━━━━━━━━━━━━━━━┃━━━━━━━━━━━━━━┃━━━━━━━━━━━━━━━━━━━
 chunks flowing...       │  ← Fill gap  │  synthesis + quotes
                         │  with signal │
                         ▼              ▼
```

**Implementation in `stream_cross_notebook_response()`:**

```python
async def stream_cross_notebook_response(...):
    # [1] Stream routing reasoning
    yield sse_chunk(reasoning_content=decision.reasoning + "\n\n")

    # [2] Stream primary notebook response
    async for chunk in agent_core.query_stream(decision.notebook_id, query, ...):
        yield sse_chunk(...)  # Normal primary streaming
        accumulated_primary += delta_text

    # [3] Progress indicator (fills the gap)
    yield sse_chunk(
        content="\n\n⏳ *Cross-referencing additional sources...*\n"
    )

    # [4] Await secondary results (with timeout)
    try:
        secondary_results = await asyncio.wait_for(
            asyncio.gather(*secondary_tasks, return_exceptions=True),
            timeout=routing_settings.cross_notebook_timeout,
        )
    except asyncio.TimeoutError:
        secondary_results = []  # Timed out → skip synthesis
        yield sse_chunk(
            content="\n*⚠️ Secondary sources timed out — showing primary answer only.*\n"
        )

    # [5] Synthesis + quoted originals (if we have results)
    if valid_secondary_results:
        synthesis = await _synthesize_cross_notebook(...)
        yield sse_chunk(content=section_marker)
        if synthesis:
            yield sse_chunk(content=synthesis + "\n\n")
        for result in valid_secondary_results:
            yield sse_chunk(content=format_quoted_original(result))

    # [6] Final [DONE]
    yield sse_chunk(finish_reason="stop")
    yield "data: [DONE]\n\n"
```

**Key detail**: The progress indicator is streamed as `content` (not `reasoning_content`) so clients display it inline with the response text.

---

### 5.5 🟡 Error Cascade

**Risk**: Five independent components can fail: primary query, each secondary query (N), and the synthesis LLM call. Need a clear contract for each combination.

**Design — Degradation State Machine:**

```
                    ┌──────────────┐
                    │ Start Stream │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Primary Query │
                    └──────┬───────┘
                           │
                    ┌──────┴──────┐
                    │             │
                FailPrimary    SuccessPrimary
                    │             │
                    ▼             ▼
              ┌──────────┐  ┌──────────────┐
              │ HTTP 500  │  │ Check         │
              │ (standard │  │ Secondaries   │
              │  error)   │  └──────┬────────┘
              └──────────┘         │
                           ┌──────┴──────────────┐
                           │                      │
                    AllSecondariesFail    SomeSecondariesOK
                           │                      │
                           ▼                      ▼
                    ┌──────────────┐     ┌──────────────────┐
                    │ Return only   │     │ Attempt Synthesis │
                    │ primary answer│     └──────┬───────────┘
                    │ (no synthesis)│            │
                    └──────────────┘     ┌──────┴──────┐
                                         │             │
                                   SynthesisFails  SynthesisOK
                                         │             │
                                         ▼             ▼
                                  ┌────────────┐ ┌──────────────────┐
                                  │ Primary +   │ │ Full cross-       │
                                  │ raw quoted  │ │ notebook response │
                                  │ secondaries │ │ (synthesis +      │
                                  │ (no synth)  │ │  quoted originals)│
                                  └────────────┘ └──────────────────┘
```

**Implementation pattern:**

```python
async def stream_cross_notebook_response(...):
    # Primary always streams (errors propagate normally)
    # ... stream primary ...

    # Collect secondary results with per-task error handling
    secondary_results = []
    for task_result in gathered_results:
        if isinstance(task_result, Exception):
            logger.warning("[CROSS-NOTEBOOK] Secondary query failed: %s", task_result)
            # Record in tracing
            secondary_failures += 1
        elif task_result is None:
            # Semaphore skip
            secondary_skipped += 1
        else:
            secondary_results.append(task_result)

    if not secondary_results:
        # All secondaries failed/skipped → close stream (primary already sent)
        add_span_attributes(cross_notebook_degraded=True)
        yield final_chunk()
        return

    # Attempt synthesis
    try:
        synthesis = await _synthesize_cross_notebook(...)
    except Exception as e:
        logger.warning("[CROSS-NOTEBOOK] Synthesis failed: %s", e)
        synthesis = None  # Fall through to raw quotes

    # Stream whatever we have
    yield sse_chunk(content=section_marker)
    if synthesis and synthesis != "NO_SYNTHESIS_NEEDED":
        yield sse_chunk(content=synthesis + "\n\n")
    for result in secondary_results:
        yield sse_chunk(content=format_quoted_original(result))
```

**Tracing**: Each degradation path records `cross_notebook_degraded=True` and `degradation_reason` as span attributes for monitoring.

---

### 5.6 🟡 Background Task Cleanup on Disconnect

**Risk**: Secondary queries launched via `asyncio.create_task()` continue running even after client disconnects (closes the HTTP connection mid-primary-stream). These orphan tasks waste NLM API calls.

**Design (V1 — Accept the waste):**

```
Client connects → Primary stream starts → Secondary tasks launched
                                                ↓
Client disconnects  ─→  Primary generator stops (StopAsyncIteration)
                        BUT secondary tasks continue ─→ Results discarded
                        ↓
                   Semaphore released when tasks complete (~5-30s)
```

**Why this is acceptable for V1:**
- Secondary tasks are bounded by global semaphore (default: 5 concurrent)
- Each task runs for at most `cross_notebook_timeout` seconds (default: 30)
- Worst case: 5 × 30s = 150 seconds of wasted NLM queries
- In practice, most secondaries complete before the primary stream finishes
- Client disconnects are rare for streamed responses

**V2 Enhancement Design (not implemented now, documented for future):**

```python
# V2: Cancellation via asyncio.Event
cancel_event = asyncio.Event()

async def _query_secondary_cancellable(
    agent_core, notebook_id, query, semaphore, cancel_event
):
    async with semaphore:
        if cancel_event.is_set():
            return None  # Don't start if already cancelled
        try:
            result = await asyncio.wait_for(
                agent_core.query(notebook_id, query),
                timeout=30,
            )
            return result
        except asyncio.CancelledError:
            return None

# In the generator:
async def stream_cross_notebook_response(...):
    try:
        # ... stream primary + synthesis ...
        pass
    except GeneratorExit:
        # Client disconnected
        cancel_event.set()
        for task in secondary_tasks:
            task.cancel()
```

---

### 5.7 🟢 Select Prompt Over/Under-Triggering

**Risk**: The enhanced `select_notebook_node` prompt must decide when cross-notebook is beneficial. Two failure modes:
- **Over-trigger**: LLM suggests secondaries for "What is X?" → wasted quota
- **Under-trigger**: LLM never suggests secondaries → feature is dead

**Design — Multi-layer defense:**

**Layer 1: Prompt engineering with few-shot examples**

```
CROSS-NOTEBOOK ASSESSMENT:
After selecting the primary notebook, decide if secondary notebooks would add
SIGNIFICANT supplementary information.

TRIGGER cross-notebook when:
✅ Query spans topics from DIFFERENT notebooks:
   - "How does our network setup affect billing?" (Network + Billing)
   - "Compare the API approach in project A with project B"
✅ Query asks to combine or contrast information:
   - "What are all the security policies across our docs?"
   - "Summarize everything we know about customer X"

DO NOT trigger when:
❌ Query is answerable by a single notebook:
   - "What is the transformer architecture?" → ML Research notebook only
   - "List the API endpoints" → API Docs only
❌ Query is simple/factual:
   - "When was the project started?"
   - "Who is the team lead?"
❌ Primary notebook clearly covers the full answer:
   - Primary has sources directly matching the query topic
```

**Layer 2: Configuration kill-switch + limits**

```python
class SmartRoutingSettings:
    cross_notebook_enabled: bool = True           # Master switch
    cross_notebook_max_secondary: int = 2          # Hard cap
```

**Layer 3: Monitoring — Grafana dashboard query**

```sql
-- Alert if cross-notebook trigger rate exceeds 40% (likely over-triggering)
SELECT
    countIf(Attributes['cross_notebook_detected'] = 'true') * 100.0
    / count() as trigger_rate_pct
FROM traces
WHERE SpanName = 'smart_router.select_notebook'
  AND Timestamp > now() - INTERVAL 1 HOUR
-- Alert if trigger_rate_pct > 40 or < 5 (under-triggering check)
```

**Layer 4: Fallback parsing**

If the LLM returns malformed JSON (no `secondary_notebook_ids` field), fall back to current behavior — extract UUID from text, treat as single-notebook. This means the feature silently degrades rather than breaking:

```python
async def select_notebook_node(state, *, chat_model, notebook_cache, ...):
    response_text = response.content.strip()

    # Try JSON parse (new cross-notebook format)
    try:
        parsed = json.loads(response_text)
        notebook_id = parsed.get("notebook_id")
        secondary_ids = parsed.get("secondary_notebook_ids", [])
        cross_notebook = parsed.get("cross_notebook", False)
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        # Fallback: extract UUID from free text (backward compatible)
        logger.debug("[ROUTER] JSON parse failed, falling back to UUID extraction")
        notebook_id = _extract_uuid_from_text(response_text, notebooks)
        secondary_ids = []
        cross_notebook = False
        reasoning = f"Selected notebook (fallback): {notebook_id}"
```

---

### 5.8 🟢 Token Limits for Synthesis Prompt

**Risk**: Primary answer (potentially 5000+ words) + multiple secondary answers could exceed the routing LLM's context window (typically 4K-128K tokens depending on model).

**Design — Input truncation pipeline:**

```
                    Primary Answer           Secondary Answers
                    (full length)            (full length each)
                         │                        │
                    ┌────▼──────┐           ┌─────▼────────┐
                    │ Truncate   │           │ Truncate each │
                    │ to 1000    │           │ to max_chars  │
                    │ chars      │           │ (default 2000)│
                    └────┬──────┘           └─────┬────────┘
                         │                        │
                    ┌────▼────────────────────────▼────┐
                    │     Synthesis Prompt               │
                    │     (~500 tokens instruction)      │
                    │     + primary 1000 chars (~250 tok) │
                    │     + N × 2000 chars (~500 tok ea)  │
                    │                                     │
                    │     Total worst case (2 secondary): │
                    │     500 + 250 + 1000 = ~1750 tokens │
                    │     Well within any model's limits   │
                    └─────────────────────────────────────┘
```

**Implementation:**

```python
async def _synthesize_cross_notebook(
    chat_model,
    query: str,
    primary_answer: str,
    secondary_results: list[dict],
    settings: SmartRoutingSettings,
) -> str | None:
    # Truncate primary
    primary_truncated = primary_answer[:1000]
    if len(primary_answer) > 1000:
        primary_truncated += "... (truncated)"

    # Truncate each secondary
    max_chars = settings.cross_notebook_secondary_max_chars
    formatted_secondaries = []
    for result in secondary_results:
        answer = result["answer"][:max_chars]
        if len(result["answer"]) > max_chars:
            answer += "... (truncated)"
        formatted_secondaries.append(
            f"Source: {result['notebook_title']}\nAnswer: {answer}"
        )

    # Build prompt
    prompt = load_prompt("synthesize_cross_notebook").format(
        query=query,
        primary_answer_truncated=primary_truncated,
        secondary_results="\n\n".join(formatted_secondaries),
    )
    # ...
```

**Configuration**: `cross_notebook_secondary_max_chars` is configurable to tune the trade-off between synthesis quality and token usage.

---

### 5.9 🟢 MCP Server Compatibility

**Risk**: `AgentCore` and `RouterState` are shared between OpenAI proxy and MCP server. Changes to these shared types could break MCP server functionality.

**Design — Transport-layer isolation:**

```
                    ┌──────────────────────────────────────────┐
                    │              Shared Layer                  │
                    │                                            │
                    │  RouterState    → gains secondary_*        │
                    │  RoutingDecision → gains secondary_*      │
                    │  AgentCore.route() → populates new fields │
                    │  ResponseCache → gains secondary_*        │
                    │                                            │
                    │  ALL new fields have defaults ── no break  │
                    └──────────────┬───────────────┬────────────┘
                                   │               │
                    ┌──────────────▼──┐    ┌──────▼──────────┐
                    │  OpenAI Proxy    │    │  MCP Server     │
                    │  (server.py)     │    │  (mcp/server.py)│
                    │                  │    │                  │
                    │  ✅ Reads:       │    │  ✅ Reads:       │
                    │  - notebook_id   │    │  - notebook_id   │
                    │  - secondary_*   │    │  (ignores new    │
                    │  - cross_notebook│    │   fields — they  │
                    │                  │    │   default to     │
                    │  ✅ Has cross-   │    │   [] and False)  │
                    │  notebook stream │    │                  │
                    │  logic           │    │  ❌ No cross-    │
                    │                  │    │  notebook logic  │
                    └──────────────────┘    │  needed for V1   │
                                           └──────────────────┘
```

**Verification**: Run existing MCP server tests after changes — they should pass without modification since all new fields have defaults.

---

### 5.10 🟢 RoutingDecision Backward Compatibility

**Risk**: Code that reads `decision.notebook_id`, `decision.request_type`, etc. might break with new fields.

**Design — Additive-only changes:**

```python
# BEFORE:
@dataclass
class RoutingDecision:
    request_type: str
    notebook_id: str | None = None
    reasoning: str = ""
    cache_result: object | None = None
    cache_hit_type: str | None = None
    conversation_id: str | None = None

# AFTER — only additions, no modifications:
@dataclass
class RoutingDecision:
    request_type: str
    notebook_id: str | None = None
    secondary_notebook_ids: list[str] = field(default_factory=list)  # NEW
    cross_notebook: bool = False                                      # NEW
    reasoning: str = ""
    cache_result: object | None = None
    cache_hit_type: str | None = None
    conversation_id: str | None = None
```

**Contract**: All existing code patterns are safe:

```python
# These all continue to work unchanged:
if decision.notebook_id:           # ✅ Still str | None
if decision.request_type == "notebooklm":  # ✅ Still str
decision.reasoning                 # ✅ Still str
decision.cache_result              # ✅ Still object | None
```

**New code checks the flag explicitly:**

```python
# Only the new cross-notebook handler checks the new fields:
if decision.cross_notebook and decision.secondary_notebook_ids:
    return StreamingResponse(stream_cross_notebook_response(...))
else:
    return StreamingResponse(stream_smart_response(...))  # Existing path
```

---

### 5.11 🟢 Connection Exhaustion from Concurrent Cross-Notebook Queries

**Risk**: httpx client has `max_connections=100`, `max_keepalive=20`. With N concurrent users making cross-notebook queries (each spawning 1-3 NLM connections), the pool could be exhausted.

**Design — Global semaphore:**

```
              ┌─────────────────────────────────────────────────┐
              │         Global Cross-Notebook Semaphore           │
              │         (default: 5 concurrent slots)             │
              │                                                   │
              │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
              │  │Slot1│ │Slot2│ │Slot3│ │Slot4│ │Slot5│       │
              │  │ 🔵  │ │ 🔵  │ │ 🔵  │ │ ⚪  │ │ ⚪  │       │
              │  │busy │ │busy │ │busy │ │free │ │free │       │
              │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
              │                                                   │
              │  New secondary query arrives:                     │
              │  ├─ Slot available? → Acquire, run query          │
              │  └─ All full?       → SKIP query (return None)    │
              │                       Log warning, record in trace │
              └─────────────────────────────────────────────────┘
```

**Implementation:**

```python
# Module-level in server.py (initialized in main())
_cross_notebook_semaphore: asyncio.Semaphore | None = None

async def _query_secondary(
    agent_core: AgentCore,
    notebook_id: str,
    notebook_title: str,
    query: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> dict | None:
    """Query a secondary notebook with semaphore guard + timeout."""
    # Try to acquire semaphore without waiting
    acquired = semaphore._value > 0  # Check available slots
    if not acquired:
        # All slots busy — skip this secondary
        logger.warning(
            "[CROSS-NOTEBOOK] Semaphore full, skipping secondary: %s (%s)",
            notebook_title, notebook_id[:12],
        )
        return None

    async with semaphore:
        try:
            result = await asyncio.wait_for(
                agent_core.query(notebook_id, query),
                timeout=timeout,
            )
            answer = result.get("answer", "") if result else ""
            if not answer:
                return None
            return {
                "notebook_id": notebook_id,
                "notebook_title": notebook_title,
                "answer": answer,
                "conversation_id": result.get("conversation_id"),
            }
        except asyncio.TimeoutError:
            logger.warning(
                "[CROSS-NOTEBOOK] Secondary timed out: %s (%s)",
                notebook_title, notebook_id[:12],
            )
            return None
        except Exception as e:
            logger.warning(
                "[CROSS-NOTEBOOK] Secondary failed: %s (%s): %s",
                notebook_title, notebook_id[:12], e,
            )
            return None
```

**Configuration**: `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_CONCURRENCY=5` (default). This limits **all** secondary queries across **all** concurrent requests — not per-request.

**Impact math**: With concurrency=5 and 10 concurrent users, worst case = 5 secondary queries running + 10 primary queries = 15 connections. Well within httpx's 100 max.

---

### 5.12 🟢 Cache Invalidation for Cross-Notebook Entries

**Risk**: A cross-notebook response is cached under primary notebook_id A. Secondary notebook B's sources change → `invalidate_notebook("B")` runs → the cached composite response (under A) is NOT invalidated, even though it contains stale info from notebook B.

**Design:**

```
              NotebookCache detects source change in Notebook B
                                    │
                                    ▼
              ResponseCache.invalidate_notebook("B")
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              Clear entries     Clear aliases    NEW: Clear cross-
              where             pointing to      notebook entries
              notebook_id=B     notebook_id=B    where B ∈
                                                 secondary_notebook_ids
```

**Changes to `invalidate_notebook()`:**

```python
def invalidate_notebook(self, notebook_id: str) -> None:
    with self._lock:
        # [Existing] Clear entries where entry.notebook_id == notebook_id
        entries = self._cache_by_notebook.pop(notebook_id, [])
        for entry in entries:
            self._cache_by_hash.pop(entry.query_hash, None)
            # ... existing cleanup ...

        # [Existing] Clear aliases ...

        # [NEW] Clear cross-notebook entries referencing this notebook as secondary
        cross_notebook_stale = []
        for nb_id, nb_entries in self._cache_by_notebook.items():
            for entry in nb_entries:
                if notebook_id in (entry.secondary_notebook_ids or []):
                    cross_notebook_stale.append((nb_id, entry))

        for nb_id, entry in cross_notebook_stale:
            logger.info(
                "[CACHE] Invalidating cross-notebook entry: primary=%s, "
                "stale_secondary=%s, query='%s'",
                nb_id[:12], notebook_id[:12], entry.query[:60],
            )
            self._remove_entry(entry)
            nb_entries = self._cache_by_notebook.get(nb_id, [])
            if entry in nb_entries:
                nb_entries.remove(entry)

    if entries or cross_notebook_stale:
        logger.info(
            "[CACHE] Invalidated: %d direct + %d cross-notebook entries for %s",
            len(entries), len(cross_notebook_stale), notebook_id[:12],
        )
```

**Performance note**: The cross-notebook scan iterates all entries but is bounded by `max_entries` (default 1000) and runs only during background cache refresh (~every 48 minutes). The overhead is negligible.

---

## 6. Tracing & Observability

### New Span Attributes

**`smart_router.select_notebook` (existing span, new attributes):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `cross_notebook_detected` | `bool` | Whether cross-notebook was triggered |
| `secondary_notebook_ids` | `str` | JSON array of secondary UUIDs |
| `secondary_notebook_count` | `int` | Number of secondaries selected |

**`smart_router.handle_request` (existing span, new attributes):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `cross_notebook` | `bool` | Whether this was a cross-notebook execution |
| `secondary_queries_count` | `int` | Number of secondary queries attempted |
| `secondary_queries_success` | `int` | Number that succeeded |
| `secondary_queries_skipped` | `int` | Number skipped (semaphore full) |
| `synthesis_generated` | `bool` | Whether synthesis was produced |
| `total_notebooks_queried` | `int` | Primary + successful secondaries |

### Grafana Dashboard Additions

- **Cross-notebook trigger rate**: % of requests where `cross_notebook_detected=true`
- **Cross-notebook latency overhead**: Compare `handle_request` duration when `cross_notebook=true` vs `false`
- **Secondary query success rate**: `secondary_queries_success / secondary_queries_count`
- **Synthesis generation rate**: % of cross-notebook requests where `synthesis_generated=true`

---

## 7. Configuration Reference

All settings use the `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_` prefix:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_ENABLED` | `true` | Master switch for cross-notebook feature |
| `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_MAX_SECONDARY` | `2` | Maximum secondary notebooks per query |
| `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_CONCURRENCY` | `5` | Global semaphore limit for background queries |
| `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_SYNTHESIS_ENABLED` | `true` | Whether to run LLM synthesis step |
| `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_TIMEOUT` | `30` | Max seconds to wait for secondaries |
| `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_SECONDARY_MAX_CHARS` | `2000` | Max chars of secondary answer in synthesis prompt |

---

## 8. Files to Modify

| File | Change Type | Description |
|------|------------|-------------|
| `src/nlm_proxy/core/config.py` | MODIFY | Add cross-notebook settings to `SmartRoutingSettings` |
| `src/nlm_proxy/core/routing_graph.py` | MODIFY | Enhance `select_notebook_node` for JSON output with secondaries |
| `src/nlm_proxy/core/agent.py` | MODIFY | Add secondary fields to `RoutingDecision`, update `route()` |
| `src/nlm_proxy/core/response_cache.py` | MODIFY | Add `secondary_notebook_ids` to `CachedResponse`, update `store()` and `invalidate_notebook()` |
| `src/nlm_proxy/openai/server.py` | MODIFY | Add cross-notebook streaming logic, update `handle_smart_routing()` |
| `src/nlm_proxy/openai/prompts/select_notebook.txt` | MODIFY | Add cross-notebook assessment section with JSON output |
| `src/nlm_proxy/openai/prompts/synthesize_cross_notebook.txt` | **NEW** | Synthesis prompt |
| `.env.example` | MODIFY | Add cross-notebook configuration |
| `docs/smart-routing-architecture.md` | MODIFY | Document cross-notebook flow |
| `.agent/memory/smart-routing.md` | MODIFY | Add cross-notebook section |
| `README.md` | MODIFY | Mention cross-notebook feature |
| `GEMINI.md` | MODIFY | Update if needed |
