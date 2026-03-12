# Cross-Notebook Query Support — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable the smart router to query multiple notebooks simultaneously for complex cross-domain queries, with LLM synthesis and quoted originals.

**Architecture:** Extended Select Node (Approach 1) — enhance the existing `select_notebook_node` to detect cross-notebook need and select secondary notebooks. Execution logic added to `server.py` streaming handler. Cache, tracing, and config updated with additive-only changes.

**Tech Stack:** LangGraph, LangChain, asyncio, httpx, pytest, pydantic-settings

**Design Doc:** `docs/plans/2026-03-12-cross-notebook-query-design.md`

---

## Task 1: Configuration — Add Cross-Notebook Settings

**Files:**
- Modify: `src/nlm_proxy/core/config.py:137-194` (`SmartRoutingSettings`)
- Modify: `.env.example`

**Step 1: Add settings fields to `SmartRoutingSettings`**

In `src/nlm_proxy/core/config.py`, add these fields to `SmartRoutingSettings` (after line 187, before `model_config`):

```python
    # Cross-notebook query settings
    cross_notebook_enabled: bool = Field(
        default=True,
        description="Enable cross-notebook query support",
    )
    cross_notebook_max_secondary: int = Field(
        default=2,
        description="Maximum secondary notebooks per cross-notebook query",
    )
    cross_notebook_concurrency: int = Field(
        default=5,
        description="Global semaphore limit for background secondary queries",
    )
    cross_notebook_synthesis_enabled: bool = Field(
        default=True,
        description="Enable LLM synthesis of cross-notebook results",
    )
    cross_notebook_timeout: int = Field(
        default=30,
        description="Max seconds to wait for secondary notebook queries",
    )
    cross_notebook_secondary_max_chars: int = Field(
        default=2000,
        description="Max chars of each secondary answer included in synthesis prompt",
    )
```

**Step 2: Verify settings load with defaults**

Run: `uv run python -c "from nlm_proxy.core.config import get_routing_settings; s = get_routing_settings(); print(s.cross_notebook_enabled, s.cross_notebook_max_secondary)"`

Expected: `True 2`

**Step 3: Add to `.env.example`**

```bash
# Cross-notebook query settings
# NLM_PROXY_ROUTING_CROSS_NOTEBOOK_ENABLED=true
# NLM_PROXY_ROUTING_CROSS_NOTEBOOK_MAX_SECONDARY=2
# NLM_PROXY_ROUTING_CROSS_NOTEBOOK_CONCURRENCY=5
# NLM_PROXY_ROUTING_CROSS_NOTEBOOK_SYNTHESIS_ENABLED=true
# NLM_PROXY_ROUTING_CROSS_NOTEBOOK_TIMEOUT=30
# NLM_PROXY_ROUTING_CROSS_NOTEBOOK_SECONDARY_MAX_CHARS=2000
```

**Step 4: Commit**

```bash
git add src/nlm_proxy/core/config.py .env.example
git commit -m "feat(config): add cross-notebook query configuration settings"
```

---

## Task 2: RoutingDecision & RouterState — Add Cross-Notebook Fields

**Files:**
- Modify: `src/nlm_proxy/core/agent.py:31-39` (`RoutingDecision`)
- Modify: `src/nlm_proxy/core/routing_graph.py:33-43` (`RouterState`)
- Test: `tests/core/test_agent.py`

**Step 1: Write the failing test**

Add to `tests/core/test_agent.py`:

```python
# ── Cross-notebook field tests ───────────────────────────────────────────

def test_routing_decision_cross_notebook_defaults():
    """New cross-notebook fields have safe defaults."""
    from nlm_proxy.core.agent import RoutingDecision

    decision = RoutingDecision(request_type="notebooklm", notebook_id="nb-1")
    assert decision.secondary_notebook_ids == []
    assert decision.cross_notebook is False


def test_routing_decision_cross_notebook_populated():
    """RoutingDecision accepts cross-notebook fields."""
    from nlm_proxy.core.agent import RoutingDecision

    decision = RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        secondary_notebook_ids=["nb-2", "nb-3"],
        cross_notebook=True,
        reasoning="Cross-notebook detected",
    )
    assert decision.secondary_notebook_ids == ["nb-2", "nb-3"]
    assert decision.cross_notebook is True
```

Also add an integration test verifying `AgentCore.route()` correctly extracts cross-notebook fields from the LangGraph state:

```python
@pytest.mark.asyncio
async def test_agent_route_extracts_cross_notebook_fields(mock_components):
    """route() correctly extracts cross-notebook fields from graph state."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components
    resp_cache.lookup_global.return_value = (None, None)

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "request_type": "notebooklm",
            "notebook_id": "nb-1",
            "secondary_notebook_ids": ["nb-2"],
            "cross_notebook": True,
            "reasoning": "Cross-notebook detected",
        })
        mock_build.return_value = mock_graph

        agent = AgentCore(
            nlm_client=nlm, notebook_cache=nb_cache,
            response_cache=resp_cache, chat_model=chat_model,
        )
        decision = await agent.route("test", RequestOptions())

    assert decision.secondary_notebook_ids == ["nb-2"]
    assert decision.cross_notebook is True
    assert decision.notebook_id == "nb-1"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_agent.py::test_routing_decision_cross_notebook_defaults tests/core/test_agent.py::test_routing_decision_cross_notebook_populated tests/core/test_agent.py::test_agent_route_extracts_cross_notebook_fields -v`

Expected: FAIL — `RoutingDecision` doesn't have `secondary_notebook_ids` / `cross_notebook` yet.

**Step 3: Update `RoutingDecision`**

In `src/nlm_proxy/core/agent.py`, modify the `RoutingDecision` dataclass:

```python
@dataclass
class RoutingDecision:
    """Result of routing: where to send the query."""
    request_type: str                       # "notebooklm" | "llm_task"
    notebook_id: str | None = None
    secondary_notebook_ids: list[str] = field(default_factory=list)  # Cross-notebook
    cross_notebook: bool = False                                      # Cross-notebook
    reasoning: str = ""
    cache_result: object | None = None      # CachedResponse on cache hit
    cache_hit_type: str | None = None       # "pre_routing_exact" etc.
    conversation_id: str | None = None
```

**Step 4: Update `RouterState`**

In `src/nlm_proxy/core/routing_graph.py`, add new fields to `RouterState`:

```python
class RouterState(TypedDict):
    """Internal state for the routing graph."""
    query: str
    messages: list
    request_type: str | None
    notebook_id: str | None
    secondary_notebook_ids: list[str]  # NEW — cross-notebook secondary UUIDs
    cross_notebook: bool               # NEW — whether cross-notebook was triggered
    reasoning: str
    available_notebooks: list[dict]
    allowed_notebooks: list[str] | None
```

**Step 5: Update `AgentCore.route()` to extract new fields**

In `src/nlm_proxy/core/agent.py`, in the `route()` method, change the `RoutingDecision` creation (around line 100-105):

```python
                decision = RoutingDecision(
                    request_type=state["request_type"],
                    notebook_id=state.get("notebook_id"),
                    secondary_notebook_ids=state.get("secondary_notebook_ids", []),
                    cross_notebook=state.get("cross_notebook", False),
                    reasoning=state.get("reasoning", ""),
                    conversation_id=options.conversation_id,
                )
```

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_agent.py -v`

Expected: ALL PASS (new tests + existing tests unchanged, total 3 new tests)

**Step 7: Commit**

```bash
git add src/nlm_proxy/core/agent.py src/nlm_proxy/core/routing_graph.py tests/core/test_agent.py
git commit -m "feat(agent): add cross-notebook fields to RoutingDecision and RouterState"
```

---

## Task 3: Response Cache — Cross-Notebook Storage & Invalidation

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py:25-38` (`CachedResponse`), `:160-230` (`store`), `:380+` (`invalidate_notebook`)
- Test: `tests/core/test_response_cache.py`

**Step 1: Write failing tests**

Add the following test class to `tests/core/test_response_cache.py`:

```python
class TestCrossNotebookCache:
    """Test cross-notebook cache behavior — ACL safety and invalidation."""

    def _make_cache(self, **kwargs):
        from nlm_proxy.core.response_cache import ResponseCache
        defaults = dict(max_entries=100, ttl_seconds=3600, semantic_enabled=False)
        defaults.update(kwargs)
        return ResponseCache(**defaults)

    def test_cross_notebook_response_has_secondary_ids(self):
        """CachedResponse stores secondary_notebook_ids field."""
        from nlm_proxy.core.response_cache import CachedResponse
        import time
        entry = CachedResponse(
            query="test", query_hash="hash1", notebook_id="nb-1",
            answer="answer", thinking=None, conversation_id="conv-1",
            embedding=None, cached_at=time.time(),
            secondary_notebook_ids=["nb-2", "nb-3"],
        )
        assert entry.secondary_notebook_ids == ["nb-2", "nb-3"]

    def test_cross_notebook_response_default_empty_list(self):
        """CachedResponse.secondary_notebook_ids defaults to empty list."""
        from nlm_proxy.core.response_cache import CachedResponse
        import time
        entry = CachedResponse(
            query="test", query_hash="hash1", notebook_id="nb-1",
            answer="answer", thinking=None, conversation_id="conv-1",
            embedding=None, cached_at=time.time(),
        )
        assert entry.secondary_notebook_ids == []

    def test_cross_notebook_store_skips_global_l1(self):
        """Cross-notebook response NOT stored in global L1 index (ACL safety)."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1", query="cross query", answer="combined answer",
            thinking=None, conversation_id="conv-1",
            secondary_notebook_ids=["nb-2"],
        )
        # Notebook-scoped lookup should work
        result, _ = cache.lookup(notebook_id="nb-1", query="cross query")
        assert result is not None
        assert result.answer == "combined answer"
        # Global lookup should NOT find it (ACL safety)
        result, _ = cache.lookup_global("cross query")
        assert result is None

    def test_single_notebook_store_still_populates_global_l1(self):
        """Single-notebook response still goes to global L1 (unchanged behavior)."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1", query="single query", answer="answer",
            thinking=None, conversation_id="conv-1",
        )
        result, _ = cache.lookup_global("single query")
        assert result is not None

    def test_invalidate_notebook_clears_cross_notebook_entries(self):
        """Invalidating a secondary notebook clears cross-notebook entries referencing it."""
        cache = self._make_cache()
        # Store a cross-notebook response under primary nb-1 with secondary nb-2
        cache.store(
            notebook_id="nb-1", query="cross query", answer="combined",
            thinking=None, conversation_id="conv-1",
            secondary_notebook_ids=["nb-2"],
        )
        # Store a regular entry under nb-2
        cache.store(
            notebook_id="nb-2", query="regular query", answer="regular",
            thinking=None, conversation_id="conv-2",
        )
        # Invalidate nb-2 — should clear BOTH the regular entry AND the cross-notebook entry
        cache.invalidate_notebook("nb-2")
        assert cache.lookup("nb-2", "regular query")[0] is None
        assert cache.lookup("nb-1", "cross query")[0] is None  # Cross-notebook entry also cleared

    def test_invalidate_primary_notebook_clears_cross_notebook(self):
        """Invalidating the primary notebook clears its cross-notebook entries too."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1", query="cross query", answer="combined",
            thinking=None, conversation_id="conv-1",
            secondary_notebook_ids=["nb-2"],
        )
        cache.invalidate_notebook("nb-1")
        assert cache.lookup("nb-1", "cross query")[0] is None

    def test_invalidate_unrelated_notebook_preserves_cross_notebook(self):
        """Invalidating an unrelated notebook preserves cross-notebook entries."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1", query="cross query", answer="combined",
            thinking=None, conversation_id="conv-1",
            secondary_notebook_ids=["nb-2"],
        )
        cache.invalidate_notebook("nb-3")  # Unrelated
        assert cache.lookup("nb-1", "cross query")[0] is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_response_cache.py::TestCrossNotebookCache -v`

Expected: FAIL — `CachedResponse` doesn't have `secondary_notebook_ids`, `store()` doesn't accept it.

**Step 3: Update `CachedResponse`**

In `src/nlm_proxy/core/response_cache.py`, add to the `CachedResponse` dataclass (after `hit_count`):

```python
    secondary_notebook_ids: list[str] = field(default_factory=list)  # Cross-notebook
```

**Step 4: Update `store()` method**

In `src/nlm_proxy/core/response_cache.py`, modify the `store()` method signature to accept `secondary_notebook_ids`:

```python
    def store(
        self,
        notebook_id: str,
        query: str,
        answer: str,
        thinking: str | None,
        conversation_id: str,
        embedding: list[float] | None = None,
        secondary_notebook_ids: list[str] | None = None,
    ) -> None:
```

In the body, pass `secondary_notebook_ids` to `CachedResponse` init, and conditionally skip global index:

- In the "new entry" creation block, add `secondary_notebook_ids=secondary_notebook_ids or []`
- After `self._global_hash_index[global_hash] = entry`, wrap with:

```python
            # Skip global L1 for cross-notebook responses (ACL safety)
            if not secondary_notebook_ids:
                global_hash = self._compute_global_hash(query)
                self._global_hash_index[global_hash] = entry
```

- In the "update existing" block:
  - **Update `secondary_notebook_ids`** on the existing entry: `existing.secondary_notebook_ids = secondary_notebook_ids or []`
  - Conditionally skip global index update if `secondary_notebook_ids` is non-empty
  - If `secondary_notebook_ids` changed from empty to non-empty, **remove** the existing global index entry

```python
            # Update existing entry — update secondary_notebook_ids
            existing.secondary_notebook_ids = secondary_notebook_ids or []
            # ... existing field updates ...

            # Update global index (conditional on cross-notebook)
            global_hash = self._compute_global_hash(query)
            if secondary_notebook_ids:
                # Remove from global if was previously single-notebook
                self._global_hash_index.pop(global_hash, None)
            else:
                self._global_hash_index[global_hash] = existing
```

**Step 5: Update `invalidate_notebook()` method**

Find the `invalidate_notebook()` method. After the existing cleanup logic, add cross-notebook scan:

```python
        # Clear cross-notebook entries referencing this notebook as secondary
        cross_stale = []
        for nb_id, nb_entries in list(self._cache_by_notebook.items()):
            for entry in list(nb_entries):
                if notebook_id in (entry.secondary_notebook_ids or []):
                    cross_stale.append((nb_id, entry))

        for nb_id, entry in cross_stale:
            logger.info(
                "[CACHE] Invalidating cross-notebook entry: primary=%s, stale_secondary=%s",
                nb_id[:12], notebook_id[:12],
            )
            query_hash = entry.query_hash
            self._cache_by_hash.pop(query_hash, None)
            if query_hash in self._lru_order:
                self._lru_order.remove(query_hash)
            global_hash = self._compute_global_hash(entry.query)
            self._global_hash_index.pop(global_hash, None)
            nb_entries = self._cache_by_notebook.get(nb_id, [])
            if entry in nb_entries:
                nb_entries.remove(entry)
```

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_response_cache.py -v`

Expected: ALL PASS (new tests + all existing tests)

**Step 7: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_response_cache.py
git commit -m "feat(cache): cross-notebook storage with ACL-safe global L1 and secondary invalidation"
```

---

## Task 4: Select Notebook Prompt — Cross-Notebook Assessment

> ⚠️ **Important**: This task only creates the **new** prompt files. The `select_notebook.txt` template update is done in **Task 5** alongside the `routing_graph.py` code change to prevent a `KeyError` — the existing code calls `.format()` without the `cross_notebook_instructions` key.

**Files:**
- Create: `src/nlm_proxy/openai/prompts/select_notebook_cross_section.txt`
- Create: `src/nlm_proxy/openai/prompts/synthesize_cross_notebook.txt`

**Step 1: Create the cross-notebook prompt section**

Create `src/nlm_proxy/openai/prompts/select_notebook_cross_section.txt`:

```
CROSS-NOTEBOOK ASSESSMENT:
After selecting the primary notebook, assess whether the query would benefit from supplementary information from OTHER notebooks.

TRIGGER cross-notebook ONLY when:
- The query explicitly spans topics covered by DIFFERENT notebooks
  Example: "How does X (from notebook A) relate to Y (from notebook B)?"
  Example: "Compare the approach in docs A with the approach in docs B"
- The query asks to combine or contrast information across domains
  Example: "What are all the security policies across our docs?"

DO NOT trigger cross-notebook when:
- The primary notebook likely contains the full answer
- The query is simple or single-topic
  Example: "What is X?" — single notebook is sufficient
- Only one notebook is available

IMPORTANT: Respond with JSON format:
{{
  "notebook_id": "PRIMARY-UUID",
  "secondary_notebook_ids": ["UUID-2"],
  "cross_notebook": true,
  "reasoning": "Primary nb covers X, secondary nb-2 covers Y which is also relevant"
}}

If no cross-notebook needed:
{{
  "notebook_id": "PRIMARY-UUID",
  "secondary_notebook_ids": [],
  "cross_notebook": false,
  "reasoning": "Single notebook sufficient for this query"
}}
```

**Step 2: Create synthesis prompt**

Create `src/nlm_proxy/openai/prompts/synthesize_cross_notebook.txt`:

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

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/prompts/
git commit -m "feat(prompts): add cross-notebook assessment and synthesis prompts"
```

---

## Task 5: Routing Graph — Enhanced `select_notebook_node` with JSON Parsing

**Files:**
- Modify: `src/nlm_proxy/core/routing_graph.py:82-186` (`select_notebook_node`)
- Modify: `src/nlm_proxy/openai/prompts/select_notebook.txt` (add `{cross_notebook_instructions}` placeholder — **moved from Task 4**)
- Test: `tests/core/test_routing_graph.py`

**Step 0 (moved from Task 4): Update `select_notebook.txt`**

This MUST happen in the same task as the `routing_graph.py` code change. The existing code calls `prompt_template.format(notebooks_json=..., query=...)` without a `cross_notebook_instructions` key, so updating the prompt template before the code would cause a `KeyError`.

Replace the last line of `select_notebook.txt`:

```
Selection criteria (in order of importance):
1. **Source keywords** - Match query terms to source keywords (e.g., "neural networks" matches keywords ["neural", "deep learning"])
2. **Source summaries** - Match query intent to source descriptions (e.g., "how transformers work" matches summary about attention mechanisms)
3. **Source titles** - If the query mentions a specific document, paper, URL, or file name, prioritize notebooks containing sources with matching titles
4. **Source types** - Match query intent to source types (e.g., "PDF paper" queries should prefer notebooks with PDF sources)
5. **Notebook summary** - Consider how well the notebook's overall topic matches the query
6. **Topics** - Use suggested topics as additional context for relevance

Note: Each notebook's "sources" array contains source metadata. Sources with full descriptions include "title", "keywords", and "summary". Sources beyond the description limit have "title" only.

{cross_notebook_instructions}

Respond with ONLY the notebook_id (UUID) of the most relevant notebook. If none seem relevant, respond with the first notebook's ID.
```

> **Note**: When `cross_notebook_instructions` is empty string (feature disabled), the prompt behaves identically to before — LLM returns a UUID. When populated, the last line is overridden by the JSON format instruction in the cross-section prompt.

**Step 1: Write failing tests**

Add to `tests/core/test_routing_graph.py`:

```python
import json

# --- Cross-notebook tests ---

@pytest.mark.asyncio
async def test_select_notebook_cross_notebook_json_response(mock_chat_model, mock_notebook_cache):
    """JSON response with cross-notebook → returns secondary_notebook_ids."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    json_response = json.dumps({
        "notebook_id": "nb-1",
        "secondary_notebook_ids": ["nb-2"],
        "cross_notebook": True,
        "reasoning": "Query spans AI and project docs",
    })
    mock_chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response(json_response))

    routing_settings = MagicMock()
    routing_settings.cross_notebook_enabled = True
    routing_settings.cross_notebook_max_secondary = 2
    routing_settings.source_descriptions_enabled = False
    routing_settings.max_source_titles = 15

    state = {"query": "How does AI relate to the project?", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache,
        routing_settings=routing_settings,
    )
    assert result["notebook_id"] == "nb-1"
    assert result["secondary_notebook_ids"] == ["nb-2"]
    assert result["cross_notebook"] is True


@pytest.mark.asyncio
async def test_select_notebook_cross_notebook_disabled_fallback(mock_chat_model, mock_notebook_cache):
    """cross_notebook_enabled=False → still parses UUID from plain text (backward compatible)."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("nb-2"))

    routing_settings = MagicMock()
    routing_settings.cross_notebook_enabled = False
    routing_settings.source_descriptions_enabled = False
    routing_settings.max_source_titles = 15

    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache,
        routing_settings=routing_settings,
    )
    assert result["notebook_id"] == "nb-2"
    assert result.get("secondary_notebook_ids", []) == []
    assert result.get("cross_notebook", False) is False


@pytest.mark.asyncio
async def test_select_notebook_json_parse_failure_fallback(mock_chat_model, mock_notebook_cache):
    """Malformed JSON → fallback to UUID extraction (feature silently degrades)."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("nb-1 is the best"))

    routing_settings = MagicMock()
    routing_settings.cross_notebook_enabled = True
    routing_settings.cross_notebook_max_secondary = 2
    routing_settings.source_descriptions_enabled = False
    routing_settings.max_source_titles = 15

    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache,
        routing_settings=routing_settings,
    )
    assert result["notebook_id"] == "nb-1"
    assert result.get("cross_notebook", False) is False


@pytest.mark.asyncio
async def test_select_notebook_secondary_acl_filtered(mock_chat_model, mock_notebook_cache):
    """Secondary notebooks filtered by ACL — nb-3 removed since not in allowed list."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    json_response = json.dumps({
        "notebook_id": "nb-1",
        "secondary_notebook_ids": ["nb-2", "nb-3"],
        "cross_notebook": True,
        "reasoning": "Cross-notebook needed",
    })
    mock_chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response(json_response))

    routing_settings = MagicMock()
    routing_settings.cross_notebook_enabled = True
    routing_settings.cross_notebook_max_secondary = 2
    routing_settings.source_descriptions_enabled = False
    routing_settings.max_source_titles = 15

    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": ["nb-1", "nb-2"]}  # nb-3 NOT allowed
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache,
        routing_settings=routing_settings,
    )
    assert result["notebook_id"] == "nb-1"
    assert result["secondary_notebook_ids"] == ["nb-2"]  # nb-3 filtered out


@pytest.mark.asyncio
async def test_select_notebook_max_secondary_limit(mock_chat_model, mock_notebook_cache):
    """secondary_notebook_ids capped by cross_notebook_max_secondary."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    json_response = json.dumps({
        "notebook_id": "nb-1",
        "secondary_notebook_ids": ["nb-2", "nb-3"],
        "cross_notebook": True,
        "reasoning": "Multiple sources needed",
    })
    mock_chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response(json_response))

    routing_settings = MagicMock()
    routing_settings.cross_notebook_enabled = True
    routing_settings.cross_notebook_max_secondary = 1  # Limit to 1
    routing_settings.source_descriptions_enabled = False
    routing_settings.max_source_titles = 15

    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache,
        routing_settings=routing_settings,
    )
    assert len(result["secondary_notebook_ids"]) <= 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_routing_graph.py -k cross_notebook -v`

Expected: FAIL — current `select_notebook_node` doesn't return `secondary_notebook_ids`.

**Step 3: Implement JSON-aware `select_notebook_node`**

Modify `src/nlm_proxy/core/routing_graph.py` — the `select_notebook_node` function (lines 82-186). Key changes:

1. Conditionally append cross-notebook prompt section
2. Try JSON parsing first, then fallback to UUID extraction
3. Filter secondary IDs by ACL and `max_secondary`
4. Record cross-notebook span attributes

```python
@record_span("smart_router.select_notebook")
async def select_notebook_node(
    state: RouterState,
    *,
    chat_model,
    notebook_cache,
    routing_settings=None,
) -> dict:
    """Select the best notebook for the query, respecting ACL filters."""
    query = state["query"]
    allowed = state.get("allowed_notebooks")
    # ... (existing notebook fetching + ACL filtering — unchanged) ...

    # Build prompt — conditionally include cross-notebook section
    prompt_template = load_prompt("select_notebook")
    cross_notebook_section = ""
    cross_notebook_enabled = False
    if routing_settings and getattr(routing_settings, "cross_notebook_enabled", False):
        cross_notebook_enabled = True
        cross_notebook_section = load_prompt("select_notebook_cross_section")

    prompt = prompt_template.format(
        notebooks_json=json.dumps(notebooks_info, indent=2),
        query=query,
        cross_notebook_instructions=cross_notebook_section,
    )

    response = await chat_model.ainvoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    # Try JSON parse (cross-notebook format)
    if cross_notebook_enabled:
        try:
            parsed = json.loads(response_text)
            notebook_id = parsed.get("notebook_id")
            secondary_ids = parsed.get("secondary_notebook_ids", [])
            cross_notebook = parsed.get("cross_notebook", False)
            reasoning = parsed.get("reasoning", "")

            # Validate primary notebook_id
            valid_ids = {nb.id for nb in notebooks}
            if notebook_id not in valid_ids:
                notebook_id = notebooks[0].id if notebooks else None

            # ACL-filter secondary notebooks
            if allowed is not None:
                secondary_ids = [sid for sid in secondary_ids if sid in allowed]
            # Also filter to valid notebook IDs
            secondary_ids = [sid for sid in secondary_ids if sid in valid_ids and sid != notebook_id]
            # Cap by max_secondary
            max_secondary = getattr(routing_settings, "cross_notebook_max_secondary", 2)
            secondary_ids = secondary_ids[:max_secondary]

            # Record span attributes
            nb_title = next((nb.title for nb in notebooks if nb.id == notebook_id), "Unknown")
            add_span_attributes(
                selected_notebook_id=notebook_id,
                selected_notebook_title=nb_title,
                cross_notebook_detected=cross_notebook and len(secondary_ids) > 0,
                secondary_notebook_ids=json.dumps(secondary_ids),
                secondary_notebook_count=len(secondary_ids),
            )

            return {
                "notebook_id": notebook_id,
                "secondary_notebook_ids": secondary_ids,
                "cross_notebook": cross_notebook and len(secondary_ids) > 0,
                "reasoning": reasoning or f"Selected notebook: {nb_title}",
            }
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.debug("[ROUTER] JSON parse failed, falling back to UUID extraction")

    # Fallback: existing UUID extraction logic (unchanged)
    for nb in notebooks:
        if nb.id in response_text:
            reasoning = f"Selected notebook: {nb.title} (ID: {nb.id})"
            add_span_attributes(
                selected_notebook_id=nb.id,
                selected_notebook_title=nb.title,
                cross_notebook_detected=False,
            )
            return {"notebook_id": nb.id, "reasoning": reasoning}

    # Fallback to first notebook
    if notebooks:
        reasoning = f"Defaulted to notebook: {notebooks[0].title} (ID: {notebooks[0].id})"
        add_span_attributes(
            selected_notebook_id=notebooks[0].id,
            selected_notebook_title=notebooks[0].title,
            selection_fallback=True,
            cross_notebook_detected=False,
        )
        return {"notebook_id": notebooks[0].id, "reasoning": reasoning}

    return {"notebook_id": None, "reasoning": "No suitable notebook found"}
```

**Step 4: Run all routing graph tests**

Run: `uv run pytest tests/core/test_routing_graph.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/routing_graph.py src/nlm_proxy/openai/prompts/select_notebook.txt tests/core/test_routing_graph.py
git commit -m "feat(router): JSON-aware select_notebook_node with cross-notebook detection"
```

---

## Task 6: Server — Cross-Notebook Secondary Query Helper

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Add module-level constants and semaphore**

At the top of `server.py` (after existing imports), add:

```python
import asyncio

# Cross-notebook formatting constant
CROSS_NOTEBOOK_SECTION_MARKER = "\n\n---\n\n📚 **Cross-referenced from additional sources:**\n\n"

# Global semaphore for concurrent secondary queries (initialized in main())
_cross_notebook_semaphore: asyncio.Semaphore | None = None
```

**Step 2: Add `_query_secondary()` function**

```python
async def _query_secondary(
    agent_core: AgentCore,
    notebook_id: str,
    notebook_title: str,
    query: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> dict | None:
    """Query a secondary notebook with semaphore guard + timeout.

    Returns dict with {notebook_id, notebook_title, answer, conversation_id}
    or None if skipped/failed/timed out.
    """
    if semaphore._value <= 0:
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
            logger.warning("[CROSS-NOTEBOOK] Secondary timed out: %s", notebook_title)
            return None
        except Exception as e:
            logger.warning("[CROSS-NOTEBOOK] Secondary failed: %s: %s", notebook_title, e)
            return None
```

**Step 3: Add `_synthesize_cross_notebook()` function**

```python
async def _synthesize_cross_notebook(
    chat_model,
    query: str,
    primary_answer: str,
    secondary_results: list[dict],
    routing_settings,
) -> str | None:
    """LLM synthesis of cross-notebook results.

    Returns synthesis text, or None if synthesis not needed or failed.
    """
    if not routing_settings.cross_notebook_synthesis_enabled:
        return None

    from nlm_proxy.openai.prompts import load_prompt
    from langchain_core.messages import HumanMessage

    max_chars = routing_settings.cross_notebook_secondary_max_chars

    # Truncate primary
    primary_truncated = primary_answer[:1000]
    if len(primary_answer) > 1000:
        primary_truncated += "... (truncated)"

    # Format secondaries
    formatted = []
    for r in secondary_results:
        answer = r["answer"][:max_chars]
        if len(r["answer"]) > max_chars:
            answer += "... (truncated)"
        formatted.append(f"Source: {r['notebook_title']}\nAnswer: {answer}")

    prompt_template = load_prompt("synthesize_cross_notebook")
    prompt = prompt_template.format(
        query=query,
        primary_answer_truncated=primary_truncated,
        secondary_results="\n\n".join(formatted),
    )

    try:
        response = await chat_model.ainvoke([HumanMessage(content=prompt)])
        synthesis = response.content.strip()
        if synthesis == "NO_SYNTHESIS_NEEDED":
            return None
        return synthesis
    except Exception as e:
        logger.warning("[CROSS-NOTEBOOK] Synthesis LLM call failed: %s", e)
        return None
```

**Step 4: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(server): add cross-notebook secondary query and synthesis helpers"
```

---

## Task 7: Server — Cross-Notebook Streaming & `handle_smart_routing` Integration

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Add `stream_cross_notebook_response()` function**

```python
async def stream_cross_notebook_response(
    agent_core: AgentCore,
    decision: RoutingDecision,
    query: str,
    request: ChatCompletionRequest,
    chat_id: str | None,
    tracing_settings=None,
    routing_settings=None,
):
    """Phase 3 streaming handler for cross-notebook queries.

    Streams primary response first, then appends synthesis + quoted originals.
    Degrades gracefully: if secondaries fail → primary-only.
    """
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        if tracing_settings and tracing_settings.request_max_length > 0:
            span.set_attribute("user_query", query[:tracing_settings.request_max_length])
        span.set_attribute("cross_notebook", True)
        span.set_attribute("secondary_queries_count", len(decision.secondary_notebook_ids))

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        conversation_id = None
        accumulated_primary = ""
        previous_thinking = ""
        previous_answer = ""

        # Resolve secondary notebook titles
        notebook_cache = agent_core.notebook_cache
        all_notebooks = notebook_cache.get_all() if notebook_cache else []
        nb_titles = {nb.id: nb.title for nb in all_notebooks}

        # Launch secondary queries as background tasks
        semaphore = _cross_notebook_semaphore or asyncio.Semaphore(
            routing_settings.cross_notebook_concurrency if routing_settings else 5
        )
        timeout = routing_settings.cross_notebook_timeout if routing_settings else 30

        secondary_tasks = []
        for sec_id in decision.secondary_notebook_ids:
            task = asyncio.create_task(
                _query_secondary(
                    agent_core, sec_id, nb_titles.get(sec_id, sec_id[:12]),
                    query, semaphore, timeout,
                )
            )
            secondary_tasks.append(task)

        # [1] Reasoning chunk
        reasoning_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(reasoning_content=decision.reasoning + "\n\n"))],
        )
        yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

        # [2] Stream primary notebook response
        async for chunk in agent_core.query_stream(
            decision.notebook_id, query,
            conversation_id=request.conversation_id,
        ):
            chunk_type = chunk.get("type")
            full_text = chunk.get("text", "")

            new_conv_id = chunk.get("conversation_id")
            if new_conv_id and not conversation_id:
                conversation_id = new_conv_id
                agent_core.save_conversation_id(chat_id, conversation_id)

            if chunk_type == "thinking" and not request.include_thinking:
                previous_thinking = full_text
                continue

            if chunk_type == "thinking":
                delta_text = full_text[len(previous_thinking):]
                previous_thinking = full_text
                if delta_text:
                    delta = DeltaContent(reasoning_content=delta_text)
                    openai_chunk = ChatCompletionChunk(
                        id=chunk_id, created=created, model=request.model,
                        choices=[Choice(delta=delta)],
                        system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
                    )
                    yield f"data: {openai_chunk.model_dump_json()}\n\n"
            else:
                delta_text = full_text[len(previous_answer):]
                previous_answer = full_text
                if delta_text:
                    accumulated_primary += delta_text
                    delta = DeltaContent(content=delta_text)
                    openai_chunk = ChatCompletionChunk(
                        id=chunk_id, created=created, model=request.model,
                        choices=[Choice(delta=delta)],
                        system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
                    )
                    yield f"data: {openai_chunk.model_dump_json()}\n\n"

        # [3] Progress indicator
        progress_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(content="\n\n⏳ *Cross-referencing additional sources...*\n"))],
        )
        yield f"data: {progress_chunk.model_dump_json()}\n\n"

        # [4] Await secondary results
        secondary_results = []
        secondary_failures = 0
        secondary_skipped = 0
        try:
            gathered = await asyncio.wait_for(
                asyncio.gather(*secondary_tasks, return_exceptions=True),
                timeout=timeout,
            )
            for task_result in gathered:
                if isinstance(task_result, Exception):
                    secondary_failures += 1
                elif task_result is None:
                    secondary_skipped += 1
                else:
                    secondary_results.append(task_result)
        except asyncio.TimeoutError:
            secondary_failures = len(secondary_tasks)
            # Stream timeout warning to client (Design §5.4)
            timeout_chunk = ChatCompletionChunk(
                id=chunk_id, created=created, model=request.model,
                choices=[Choice(delta=DeltaContent(content="\n*⚠️ Secondary sources timed out — showing primary answer only.*\n"))],
            )
            yield f"data: {timeout_chunk.model_dump_json()}\n\n"

        span.set_attribute("secondary_queries_success", len(secondary_results))
        span.set_attribute("secondary_queries_skipped", secondary_skipped)

        # [5] Synthesis + quoted originals
        full_composite = accumulated_primary
        if secondary_results:
            synthesis = None
            if routing_settings:
                synthesis = await _synthesize_cross_notebook(
                    agent_core.chat_model, query, accumulated_primary,
                    secondary_results, routing_settings,
                )

            span.set_attribute("synthesis_generated", synthesis is not None)

            # Stream section marker
            marker_chunk = ChatCompletionChunk(
                id=chunk_id, created=created, model=request.model,
                choices=[Choice(delta=DeltaContent(content=CROSS_NOTEBOOK_SECTION_MARKER))],
            )
            yield f"data: {marker_chunk.model_dump_json()}\n\n"
            full_composite += CROSS_NOTEBOOK_SECTION_MARKER

            # Stream synthesis
            if synthesis:
                synth_chunk = ChatCompletionChunk(
                    id=chunk_id, created=created, model=request.model,
                    choices=[Choice(delta=DeltaContent(content=synthesis + "\n\n"))],
                )
                yield f"data: {synth_chunk.model_dump_json()}\n\n"
                full_composite += synthesis + "\n\n"

            # Stream quoted originals
            emoji_icons = ["📗", "📘", "📙", "📕"]
            for i, result in enumerate(secondary_results):
                icon = emoji_icons[i % len(emoji_icons)]
                quoted = f"> {icon} **From \"{result['notebook_title']}\":**\n> {result['answer']}\n\n"
                quote_chunk = ChatCompletionChunk(
                    id=chunk_id, created=created, model=request.model,
                    choices=[Choice(delta=DeltaContent(content=quoted))],
                )
                yield f"data: {quote_chunk.model_dump_json()}\n\n"
                full_composite += quoted
        else:
            span.set_attribute("synthesis_generated", False)
            span.set_attribute("cross_notebook_degraded", True)

        span.set_attribute("total_notebooks_queried", 1 + len(secondary_results))

        # Store composite response in cache
        if (
            agent_core.response_cache
            and full_composite
            and conversation_id
            and decision.notebook_id
        ):
            embedding = None
            if agent_core.response_cache._semantic_enabled:
                emb = agent_core.response_cache._compute_embedding(query)
                if emb is not None:
                    embedding = emb.tolist()
            agent_core.response_cache.store(
                notebook_id=decision.notebook_id,
                query=query,
                answer=full_composite,
                thinking=previous_thinking or None,
                conversation_id=conversation_id,
                embedding=embedding,
                secondary_notebook_ids=decision.secondary_notebook_ids,
            )

        # Trace response
        if tracing_settings and tracing_settings.response_max_length > 0:
            span.set_attribute("response_content", full_composite[:tracing_settings.response_max_length])
            span.set_attribute("response_source", "notebooklm_cross_notebook")

        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
```

**Step 2: Modify `handle_smart_routing()` — Phase 2 bypass + Phase 3 branching**

In `handle_smart_routing()` (around line 532-556), replace the Phase 2 + Phase 3 section:

```python
    # Phase 2: Post-routing cache check — SKIP for cross-notebook
    if (
        decision.request_type == "notebooklm"
        and not decision.cross_notebook
        and not options.bypass_cache
        and agent_core.response_cache
    ):
        cache_result, hit_type = await agent_core.response_cache.lookup_async(
            decision.notebook_id, query
        )
        if cache_result:
            decision.cache_result = cache_result
            decision.cache_hit_type = hit_type
            if request.stream:
                return StreamingResponse(
                    _stream_cached_response(decision, request, tracing_settings),
                    media_type="text/event-stream",
                    headers={"X-Cache-Status": f"HIT_{hit_type.upper()}"},
                )
            else:
                return _json_cached_response(decision, request, tracing_settings)

    # Phase 3: Execute query
    if decision.cross_notebook and decision.secondary_notebook_ids:
        if request.stream:
            return StreamingResponse(
                stream_cross_notebook_response(
                    agent_core, decision, query, request, chat_id,
                    tracing_settings, routing_settings,
                ),
                media_type="text/event-stream",
            )
        else:
            # V1: Non-streaming cross-notebook falls back to primary-only
            logger.info("[CROSS-NOTEBOOK] Non-streaming request — returning primary-only")
            return await _handle_non_streaming(
                agent_core, decision, query, request, chat_id, tracing_settings
            )

    # Single-notebook path (unchanged)
    if request.stream:
        return StreamingResponse(
            stream_smart_response(agent_core, decision, query, request, chat_id, tracing_settings),
            media_type="text/event-stream",
        )
    else:
        return await _handle_non_streaming(
            agent_core, decision, query, request, chat_id, tracing_settings
        )
```

**Step 3: Initialize semaphore in `main()`**

Find the `main()` function in `server.py`. Add after agent_core initialization:

```python
    global _cross_notebook_semaphore
    _cross_notebook_semaphore = asyncio.Semaphore(routing_settings.cross_notebook_concurrency)
```

**Step 4: Run existing tests to verify no regression**

Run: `uv run pytest tests/ -v --ignore=tests/core/test_embedding_models.py`

Expected: ALL PASS (no existing tests should break)

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(server): cross-notebook streaming response with synthesis and graceful degradation"
```

---

## Task 8: Documentation Updates

**Files:**
- Modify: `docs/smart-routing-architecture.md`
- Modify: `.agent/memory/smart-routing.md`
- Modify: `README.md`
- Modify: `GEMINI.md`

**Step 1: Update `docs/smart-routing-architecture.md`**

Add a new section "## Cross-Notebook Query Support" documenting:
- When it triggers (LLM-detected)
- Response format (primary + synthesis + quoted originals)
- Streaming behavior
- Configuration reference
- Known limitations (follow-up context, non-streaming V2)

**Step 2: Update `.agent/memory/smart-routing.md`**

Add cross-notebook section covering config, behavior, and troubleshooting.

**Step 3: Update `README.md`**

Add a brief mention under Features section.

**Step 4: Review `GEMINI.md`**

Verify GEMINI.md is still accurate (Quick Commands, etc.)

**Step 5: Commit**

```bash
git add docs/ .agent/memory/ README.md GEMINI.md
git commit -m "docs: add cross-notebook query support documentation"
```

---

## Verification Plan

### Automated Tests

All tests run via:

```bash
uv run pytest tests/ -v --ignore=tests/core/test_embedding_models.py
```

**New test files/additions:**
- `tests/core/test_agent.py` — 2 new tests for `RoutingDecision` fields
- `tests/core/test_response_cache.py` — 7 new tests in `TestCrossNotebookCache` class
- `tests/core/test_routing_graph.py` — 5 new tests for JSON parsing, ACL filtering, fallback

### Manual Verification

After implementation, manually verify with a running NLM proxy:

1. **Single-notebook (no regression)**: Send a simple query → should work exactly as before
2. **Cross-notebook trigger**: Send a multi-domain query (e.g., "How does X from notebook A relate to Y from notebook B?") → verify synthesis + quoted originals in response
3. **Kill-switch**: Set `NLM_PROXY_ROUTING_CROSS_NOTEBOOK_ENABLED=false` → verify no cross-notebook behavior
4. **Grafana traces**: Check for new `cross_notebook_detected`, `secondary_queries_count` span attributes
