# Stage 4: LangGraph Routing Graph

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a LangGraph `StateGraph` that replaces the `SmartRouter.classify_request()` → `select_notebook()` flow with graph nodes.

**Architecture:** The graph produces a routing **decision only** — it does NOT execute queries or handle streaming. Two nodes: `classify_node` (intent classification) and `select_notebook_node` (notebook selection with ACL filtering). Conditional edge routes LLM_TASK directly to END.

**Inputs:** Stage 1 complete — LangChain `ChatModel` exists for LLM calls.

**Outputs:** `build_routing_graph(chat_model, notebook_cache)` returns a compiled graph. Graph accepts `{"query": str, "allowed_notebooks": list | None}` and returns `RouterState` with `request_type`, `notebook_id`, `reasoning`.

---

## Task 4.1: Create LangGraph routing graph

**Files:**
- Create: `src/nlm_proxy/core/routing_graph.py`
- Test: `tests/core/test_routing_graph.py`

**Step 1: Write failing tests**

Create `tests/core/test_routing_graph.py`:

```python
"""Tests for LangGraph routing graph."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_notebook_cache():
    """Create a mock NotebookCache with test data."""
    from nlm_proxy.core.notebook_cache import NotebookCache
    mock_nlm = MagicMock()
    mock_nlm.list_notebooks = AsyncMock(return_value=[])
    cache = NotebookCache(nlm_client=mock_nlm, ttl_seconds=3600)
    cache.set("nb-1", "AI Research", "Notes about AI", ["AI", "ML"])
    cache.set("nb-2", "Project Docs", "Project documentation", ["Docs"])
    cache.set("nb-3", "Meeting Notes", "Team meetings", ["Meetings"])
    return cache


@pytest.fixture
def mock_chat_model():
    """Create a mock LangChain ChatModel."""
    model = AsyncMock()
    return model


def _mock_llm_response(content: str):
    """Helper to create a mock AIMessage."""
    resp = MagicMock()
    resp.content = content
    return resp


# --- classify_node tests ---

@pytest.mark.asyncio
async def test_classify_notebooklm(mock_chat_model, mock_notebook_cache):
    """Classify knowledge query as NOTEBOOKLM."""
    from nlm_proxy.core.routing_graph import classify_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("notebooklm")
    )
    state = {"query": "What does my AI research say?", "request_type": None,
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await classify_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["request_type"] == "notebooklm"


@pytest.mark.asyncio
async def test_classify_llm_task(mock_chat_model, mock_notebook_cache):
    """Classify meta-operation as LLM_TASK."""
    from nlm_proxy.core.routing_graph import classify_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("llm_task")
    )
    state = {"query": "Write a poem about cats", "request_type": None,
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await classify_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["request_type"] == "llm_task"
    assert "LLM task" in result["reasoning"]


# --- select_notebook_node tests ---

@pytest.mark.asyncio
async def test_select_notebook_picks_correct(mock_chat_model, mock_notebook_cache):
    """Select the notebook whose ID appears in LLM response."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("nb-2")
    )
    state = {"query": "Project status?", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] == "nb-2"
    assert "Project Docs" in result["reasoning"]


@pytest.mark.asyncio
async def test_select_notebook_no_acl(mock_chat_model, mock_notebook_cache):
    """No ACL filter → all notebooks sent to LLM."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("nb-1")
    )
    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    # Verify all 3 notebooks were in the prompt
    call_args = mock_chat_model.ainvoke.call_args[0][0]
    prompt_text = call_args[0].content if hasattr(call_args[0], 'content') else str(call_args)
    assert "nb-1" in prompt_text
    assert "nb-2" in prompt_text
    assert "nb-3" in prompt_text


@pytest.mark.asyncio
async def test_select_notebook_with_acl(mock_chat_model, mock_notebook_cache):
    """ACL filter → only allowed notebooks sent to LLM."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("nb-2")
    )
    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": ["nb-2", "nb-3"]}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] == "nb-2"
    call_args = mock_chat_model.ainvoke.call_args[0][0]
    prompt_text = call_args[0].content if hasattr(call_args[0], 'content') else str(call_args)
    assert "nb-1" not in prompt_text
    assert "nb-2" in prompt_text


@pytest.mark.asyncio
async def test_select_notebook_acl_filters_all(mock_chat_model, mock_notebook_cache):
    """ACL matches no notebooks → error reasoning, no LLM call."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": ["nb-999"]}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] is None
    assert "No accessible notebooks" in result["reasoning"]
    mock_chat_model.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_select_notebook_empty_acl(mock_chat_model, mock_notebook_cache):
    """Empty ACL list → error reasoning, no LLM call."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": []}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] is None
    assert "No accessible notebooks" in result["reasoning"]


@pytest.mark.asyncio
async def test_select_notebook_fallback(mock_chat_model, mock_notebook_cache):
    """LLM returns unrecognized ID → fallback to first notebook."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("some-random-text")
    )
    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] is not None
    assert "Defaulted" in result["reasoning"]


# --- route_after_classify tests ---

def test_route_after_classify_notebooklm():
    """NOTEBOOKLM → route to select_notebook node."""
    from nlm_proxy.core.routing_graph import route_after_classify
    assert route_after_classify({"request_type": "notebooklm"}) == "select_notebook"


def test_route_after_classify_llm_task():
    """LLM_TASK → route to END."""
    from nlm_proxy.core.routing_graph import route_after_classify, END_NODE
    assert route_after_classify({"request_type": "llm_task"}) == END_NODE


# --- Full graph end-to-end ---

@pytest.mark.asyncio
async def test_full_graph_notebooklm(mock_chat_model, mock_notebook_cache):
    """Full graph: classify as NOTEBOOKLM → select notebook."""
    from nlm_proxy.core.routing_graph import build_routing_graph

    mock_chat_model.ainvoke = AsyncMock(
        side_effect=[
            _mock_llm_response("notebooklm"),  # classify
            _mock_llm_response("nb-1"),          # select
        ]
    )
    graph = build_routing_graph(mock_chat_model, mock_notebook_cache)
    result = await graph.ainvoke({
        "query": "What does my research say?",
        "allowed_notebooks": None,
    })
    assert result["request_type"] == "notebooklm"
    assert result["notebook_id"] == "nb-1"


@pytest.mark.asyncio
async def test_full_graph_llm_task(mock_chat_model, mock_notebook_cache):
    """Full graph: classify as LLM_TASK → skip notebook selection."""
    from nlm_proxy.core.routing_graph import build_routing_graph

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("llm_task")
    )
    graph = build_routing_graph(mock_chat_model, mock_notebook_cache)
    result = await graph.ainvoke({
        "query": "Translate this to Spanish",
        "allowed_notebooks": None,
    })
    assert result["request_type"] == "llm_task"
    assert result["notebook_id"] is None
    assert mock_chat_model.ainvoke.call_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_routing_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nlm_proxy.core.routing_graph'`

**Step 3: Implement routing graph**

Create `src/nlm_proxy/core/routing_graph.py`:

```python
"""LangGraph-based routing graph for smart request classification.

Replaces the linear SmartRouter with a LangGraph StateGraph that:
1. Classifies intent (NOTEBOOKLM vs LLM_TASK)
2. Selects notebook (if NOTEBOOKLM) with ACL filtering

The graph produces a routing DECISION only — it does NOT execute queries
or handle streaming. Those are done by the transport layer.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.tracing import record_span, add_span_attributes
from nlm_proxy.openai.prompts import load_prompt

logger = get_logger(__name__)

# Re-export END as a constant for tests
END_NODE = END

DEFAULT_MAX_SOURCE_TITLES = 15


# --- LangGraph State ---

class RouterState(TypedDict):
    """Internal state for the routing graph."""
    query: str
    request_type: str | None        # "notebooklm" | "llm_task"
    notebook_id: str | None          # Selected notebook UUID
    reasoning: str                   # Human-readable explanation
    available_notebooks: list[dict]  # Populated by nodes
    allowed_notebooks: list[str] | None  # Per-request ACL filter


# --- Graph Nodes ---

@record_span("smart_router.classify")
async def classify_node(
    state: RouterState,
    *,
    chat_model,
    notebook_cache,
) -> dict:
    """Classify the request as NOTEBOOKLM or LLM_TASK using LLM."""
    query = state["query"]
    logger.debug("[ROUTER] Classifying: %s...", query[:100])

    prompt_template = load_prompt("classify_request")
    prompt = prompt_template.format(query=query)

    response = await chat_model.ainvoke([HumanMessage(content=prompt)])
    response_lower = response.content.lower().strip()

    if "notebooklm" in response_lower:
        logger.info("[ROUTER] Classified as NOTEBOOKLM")
        add_span_attributes(classification_result="NOTEBOOKLM")
        return {"request_type": "notebooklm"}

    logger.info("[ROUTER] Classified as LLM_TASK")
    add_span_attributes(classification_result="LLM_TASK")
    return {
        "request_type": "llm_task",
        "reasoning": "Classified as LLM task (not a notebook query)",
    }


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

    logger.debug("[ROUTER] Selecting notebook for: %s...", query[:100])

    # Get all cached notebooks
    notebooks = notebook_cache.get_all()
    if not notebooks:
        logger.warning("[ROUTER] No notebooks available")
        add_span_attributes(candidates_count=0)
        return {"notebook_id": None, "reasoning": "No notebooks available"}

    # Apply per-request ACL filtering
    if allowed is not None:
        notebooks = [nb for nb in notebooks if nb.id in allowed]
        add_span_attributes(
            acl_filter_applied=True,
            acl_allowed_count=len(allowed),
            acl_matched_count=len(notebooks),
        )
        if not notebooks:
            logger.warning("[ROUTER] ACL filter matched no notebooks")
            add_span_attributes(candidates_count=0)
            return {
                "notebook_id": None,
                "reasoning": "No accessible notebooks for this user",
            }
    else:
        add_span_attributes(acl_filter_applied=False)

    add_span_attributes(candidates_count=len(notebooks))

    # Build notebook info for LLM prompt
    if routing_settings is None:
        from nlm_proxy.core.config import get_routing_settings
        routing_settings = get_routing_settings()
    max_source_titles = routing_settings.max_source_titles
    source_descriptions_enabled = routing_settings.source_descriptions_enabled
    source_max_keywords = routing_settings.source_max_keywords
    source_summary_max_chars = routing_settings.source_summary_max_chars
    source_descriptions_max_sources = routing_settings.source_descriptions_max_sources

    notebooks_info = []
    for nb in notebooks:
        info: dict = {
            "id": nb.id,
            "title": nb.title,
            "summary": nb.summary[:500] if nb.summary else "",
            "topics": nb.topics[:5] if nb.topics else [],
            "source_count": nb.source_count,
            "source_types": nb.source_types,
        }
        if source_descriptions_enabled:
            info["sources"] = nb.get_source_descriptions(
                max_sources=source_descriptions_max_sources,
                max_keywords=source_max_keywords,
                summary_max_chars=source_summary_max_chars,
            )[:max_source_titles]
        else:
            info["source_titles"] = nb.source_titles[:max_source_titles]
        notebooks_info.append(info)

    # Call LLM to select notebook
    prompt_template = load_prompt("select_notebook")
    prompt = prompt_template.format(
        notebooks_json=json.dumps(notebooks_info, indent=2),
        query=query,
    )

    logger.debug("[ROUTER] Asking LLM to select from %d notebooks", len(notebooks))
    response = await chat_model.ainvoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    # Parse response — expect notebook_id in the response
    for nb in notebooks:
        if nb.id in response_text:
            reasoning = f"Selected notebook: {nb.title} (ID: {nb.id})"
            logger.info("[ROUTER] %s", reasoning)
            add_span_attributes(
                selected_notebook_id=nb.id,
                selected_notebook_title=nb.title,
            )
            return {"notebook_id": nb.id, "reasoning": reasoning}

    # Fallback to first notebook
    if notebooks:
        reasoning = f"Defaulted to notebook: {notebooks[0].title} (ID: {notebooks[0].id})"
        logger.info("[ROUTER] %s", reasoning)
        add_span_attributes(
            selected_notebook_id=notebooks[0].id,
            selected_notebook_title=notebooks[0].title,
            selection_fallback=True,
        )
        return {"notebook_id": notebooks[0].id, "reasoning": reasoning}

    return {"notebook_id": None, "reasoning": "No suitable notebook found"}


# --- Conditional Edge ---

def route_after_classify(state: RouterState) -> str:
    """Route to select_notebook or END based on classification."""
    if state.get("request_type") == "notebooklm":
        return "select_notebook"
    return END


# --- Graph Builder ---

def build_routing_graph(chat_model, notebook_cache, routing_settings=None):
    """Build and compile the LangGraph routing state graph.

    Args:
        chat_model: LangChain ChatModel for LLM calls
        notebook_cache: NotebookCache with cached notebook summaries
        routing_settings: SmartRoutingSettings for notebook display config

    Returns:
        Compiled LangGraph that accepts {"query": str, "allowed_notebooks": ...}
        and returns RouterState with request_type, notebook_id, reasoning.
    """
    # Bind dependencies to node functions via closures
    async def _classify(state):
        return await classify_node(
            state, chat_model=chat_model, notebook_cache=notebook_cache
        )

    async def _select(state):
        return await select_notebook_node(
            state, chat_model=chat_model, notebook_cache=notebook_cache,
            routing_settings=routing_settings,
        )

    graph = StateGraph(RouterState)
    graph.add_node("classify", _classify)
    graph.add_node("select_notebook", _select)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_edge("select_notebook", END)

    return graph.compile()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_routing_graph.py -v`
Expected: ALL PASS (13 tests)

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/routing_graph.py tests/core/test_routing_graph.py
git commit -m "feat: add LangGraph routing graph with classify + select_notebook nodes"
```

---

## Task 4.2: Rewrite old router tests to use new routing graph

**Files:**
- Rewrite: `tests/test_openai_module/test_router.py`
- Rewrite: `tests/test_openai_module/test_router_acl.py`

> [!IMPORTANT]
> The old `SmartRouter` class still exists. Preserve old tests as `_legacy` copies until Stage 8 cleanup.

**Step 1: Rename old tests as legacy**

```bash
cp tests/test_openai_module/test_router.py tests/test_openai_module/test_router_legacy.py
cp tests/test_openai_module/test_router_acl.py tests/test_openai_module/test_router_acl_legacy.py
```

**Step 2: Rewrite `test_router.py`** to test `build_routing_graph` instead of `SmartRouter`

**Step 3: Rewrite `test_router_acl.py`** to test ACL filtering via routing graph

See the original implementation plan for full test code.

**Step 4: Run all router tests**

Run: `uv run pytest tests/test_openai_module/test_router.py tests/test_openai_module/test_router_acl.py tests/core/test_routing_graph.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_openai_module/
git commit -m "test: rewrite router + ACL tests for LangGraph routing graph"
```

---

## 🔒 Stage 4 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS — new routing graph tested, old router tests preserved as legacy.
