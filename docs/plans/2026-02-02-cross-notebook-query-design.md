# Cross-Notebook Query Support for OpenAI Proxy

## Summary

Enable smart routing to query multiple notebooks in parallel, streaming the primary response immediately while buffering secondary notebooks, then conditionally synthesizing additional insights.

## Design Decisions

| Decision | Choice |
|----------|--------|
| Combination strategy | LLM Synthesis |
| Notebook selection | Dynamic (LLM decides 1-3 in single call) |
| Fallback strategy | Proactive multi-query |
| Streaming | Stream primary immediately, parallel background queries |
| Response format | Section marker: `📚 **Cross-referenced from other sources:**` |
| **Concurrency** | **Global semaphore across all requests (protects upstream)** |
| **Partial Failures** | **Append warning message to synthesis** |
| **Context Limits** | **Prompt secondary queries for brevity (<300 words)** |
| **Citations** | **Strip/omit citations in synthesis summary** |

## Architecture

```
Time 0ms:     Query arrives
              └─ Combined classify + select (1 LLM call)

Time 700ms:   LLM returns: [notebook_1 (primary), notebook_2, notebook_3]
              ├─ notebook_1 → STREAMING (to user immediately)
              ├─ notebook_2 → buffered query (parallel) [Acquire Semaphore]
              └─ notebook_3 → buffered query (parallel) [Acquire Semaphore]

Time 800ms:   ⚡ FIRST TOKEN streams from notebook_1

Time 3000ms:  Primary complete, secondary already finished
              └─ Evaluate: synthesis needed?

Time 3500ms:  Stream synthesis (if valuable) with section marker
```

**TTFT: ~800ms** (same as current single-notebook)

---

## Implementation Steps

### Step 1: Configuration (`src/nlm_proxy/core/config.py`)

Add to `SmartRoutingSettings`:

```python
cross_notebook_enabled: bool = True
cross_notebook_max_secondary: int = 2
cross_notebook_concurrency: int = 5  # Global limit for background queries
cross_notebook_synthesis_enabled: bool = True
cross_notebook_section_marker: str = "\n\n---\n\n📚 **Cross-referenced from other sources:**\n\n"
```

New env vars: `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_*`

### Step 2: Router Data Models (`src/nlm_proxy/openai/router.py`)

Add new dataclasses:

```python
class NotebookRole(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"

@dataclass
class NotebookSelection:
    notebook_id: str
    role: NotebookRole
    reasoning: str
    title: str = ""

@dataclass
class MultiRoutingDecision:
    request_type: RequestType
    notebooks: list[NotebookSelection]
    reasoning: str = ""

    @property
    def primary_notebook(self) -> NotebookSelection | None: ...

    @property
    def secondary_notebooks(self) -> list[NotebookSelection]: ...
```

### Step 3: Unified Prompt (`src/nlm_proxy/openai/prompts/route_and_select.txt`)

Combined classification + multi-notebook selection returning JSON:

```json
{
  "type": "notebooklm" | "llm_task",
  "notebooks": [
    {"id": "UUID", "role": "primary", "reason": "..."},
    {"id": "UUID", "role": "secondary", "reason": "..."}
  ]
}
```

### Step 4: Synthesis Prompt (`src/nlm_proxy/openai/prompts/synthesize_cross_notebook.txt`)

Conditional synthesis prompt that returns either:
- Synthesized insights (2-4 sentences, **no citations**)
- `NO_SYNTHESIS_NEEDED` if secondary adds no value

### Step 5: Router Method (`src/nlm_proxy/openai/router.py`)

Add `route_multi()` method:
- Single LLM call for classify + select
- Parse JSON response into `MultiRoutingDecision`
- Fallback handling for parse errors

### Step 6: Streaming Implementation (`src/nlm_proxy/openai/server.py`)

Add `stream_cross_notebook_response()`:

1. Stream routing reasoning
2. Start secondary queries as background tasks with `asyncio.create_task()`
   - **Append "Answer in under 300 words" to secondary prompts**
3. Stream primary notebook response immediately (delta conversion)
4. After primary completes, `await asyncio.gather(*secondary_tasks)`
5. **Handle partial failures**: If a secondary failed, append warning text
6. Call synthesis LLM if secondary results have content
7. Stream synthesis with section marker if valuable

Add helper functions:
- `_query_notebook_buffered()` - Background query with **global semaphore**
  - If semaphore full: **Skip query and return None (graceful degradation)**
- `_synthesize_cross_notebook()` - LLM synthesis call

### Step 7: Update Handler (`src/nlm_proxy/openai/server.py`)

Modify `handle_smart_routing()`:
- Check `cross_notebook_enabled` setting
- Call `router.route_multi()` instead of `router.route()`
- Return `StreamingResponse(stream_cross_notebook_response(...))`

### Step 8: Documentation (`.claude/memory/smart-routing.md`)

Add Cross-Notebook Queries section with:
- Configuration table
- Flow diagram
- Logging tags: `[CROSS-NOTEBOOK]`

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/nlm_proxy/core/config.py` | Add cross-notebook settings to `SmartRoutingSettings` |
| `src/nlm_proxy/openai/router.py` | Add dataclasses + `route_multi()` method |
| `src/nlm_proxy/openai/prompts/route_and_select.txt` | **NEW** - unified prompt |
| `src/nlm_proxy/openai/prompts/synthesize_cross_notebook.txt` | **NEW** - synthesis prompt |
| `src/nlm_proxy/openai/server.py` | Add streaming functions + update handler |
| `.claude/memory/smart-routing.md` | Documentation update |

---

## Verification

1. **Unit test routing**:
   ```bash
   uv run pytest tests/ -k "test_route" -v
   ```

2. **Start server with debug**:
   ```bash
   nlm-proxy serve openai --port 8080 --debug
   ```

3. **Test single-notebook query** (should work as before):
   ```bash
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "knowledge-finder", "messages": [{"role": "user", "content": "What is X?"}], "stream": true}'
   ```

4. **Test cross-notebook query** (should trigger multi-select):
   ```bash
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "knowledge-finder", "messages": [{"role": "user", "content": "Compare topic A from docs with topic B from guides"}], "stream": true}'
   ```

5. **Verify in logs**:
   - `[ROUTER] Multi-routing decision: type=notebooklm, notebooks=2`
   - `[CROSS-NOTEBOOK] Streaming primary: ...`
   - `[CROSS-NOTEBOOK] Secondary results: 1/1 successful`
   - `[CROSS-NOTEBOOK] Synthesis generated: N chars`

6. **Verify response format**:
   - Primary response streams immediately
   - Section marker appears if synthesis triggered
   - Cross-referenced content follows marker
