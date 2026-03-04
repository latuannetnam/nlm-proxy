# Stage 5: AgentCore Orchestration Layer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the `AgentCore` class that serves as the shared orchestration layer for both OpenAI proxy and MCP server.

**Architecture:** `AgentCore` wires together the routing graph, response cache, NLM client, and session store. It provides `route()`, `query()`, `query_stream()`, and `handle_direct_query()` methods. It also defines the `RequestOptions` and `RoutingDecision` dataclasses.

**Inputs:** Stages 1 (LLM client), 3 (config + NotebookCache), 4 (routing graph) complete.

**Outputs:** `AgentCore` class, `RequestOptions`, `RoutingDecision` importable from `core/agent.py`.

---

## Task 5.1: Create AgentCore

**Files:**
- Create: `src/nlm_proxy/core/agent.py`
- Test: `tests/core/test_agent.py`

**Step 1: Write failing tests**

Create `tests/core/test_agent.py`:

```python
"""Tests for AgentCore — shared agent logic for OpenAI proxy and MCP."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


@pytest.fixture
def mock_components():
    """Create mock components for AgentCore."""
    mock_nlm = AsyncMock()
    mock_notebook_cache = MagicMock()
    mock_response_cache = MagicMock()
    mock_chat_model = AsyncMock()
    return mock_nlm, mock_notebook_cache, mock_response_cache, mock_chat_model


@pytest.mark.asyncio
async def test_agent_route_cache_hit(mock_components):
    """Pre-routing cache hit → return cached RoutingDecision."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components
    cached = MagicMock(notebook_id="nb-1")
    resp_cache.lookup_global.return_value = (cached, "exact")

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    options = RequestOptions()
    decision = await agent.route("test query", options)

    assert decision.cache_result is cached
    assert decision.cache_hit_type == "pre_routing_exact"


@pytest.mark.asyncio
async def test_agent_route_cache_miss_goes_to_graph(mock_components):
    """Cache miss → invoke LangGraph routing graph."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components
    resp_cache.lookup_global.return_value = (None, None)

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "request_type": "notebooklm",
            "notebook_id": "nb-1",
            "reasoning": "Selected notebook",
        })
        mock_build.return_value = mock_graph

        agent = AgentCore(
            nlm_client=nlm, notebook_cache=nb_cache,
            response_cache=resp_cache, chat_model=chat_model,
        )
        options = RequestOptions()
        decision = await agent.route("What is AI?", options)

    assert decision.request_type == "notebooklm"
    assert decision.notebook_id == "nb-1"
    assert decision.cache_result is None


@pytest.mark.asyncio
async def test_agent_route_bypass_cache(mock_components):
    """bypass_cache=True → skip pre-routing cache check."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "request_type": "llm_task",
            "notebook_id": None,
            "reasoning": "LLM task",
        })
        mock_build.return_value = mock_graph

        agent = AgentCore(
            nlm_client=nlm, notebook_cache=nb_cache,
            response_cache=resp_cache, chat_model=chat_model,
        )
        options = RequestOptions(bypass_cache=True)
        decision = await agent.route("test", options)

    resp_cache.lookup_global.assert_not_called()
    assert decision.request_type == "llm_task"


@pytest.mark.asyncio
async def test_agent_route_acl_rejects_cached(mock_components):
    """Cache hit for notebook NOT in allowed list → skip cache, route normally."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components
    cached = MagicMock(notebook_id="nb-1")
    resp_cache.lookup_global.return_value = (cached, "exact")

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "request_type": "notebooklm",
            "notebook_id": "nb-2",
            "reasoning": "Selected nb-2",
        })
        mock_build.return_value = mock_graph

        agent = AgentCore(
            nlm_client=nlm, notebook_cache=nb_cache,
            response_cache=resp_cache, chat_model=chat_model,
        )
        # ACL allows nb-2 only, cache has nb-1
        options = RequestOptions(allowed_notebooks=["nb-2"])
        decision = await agent.route("test", options)

    # Cache hit rejected due to ACL → routed via graph
    assert decision.notebook_id == "nb-2"
    assert decision.cache_result is None


@pytest.mark.asyncio
async def test_agent_query_delegates_to_nlm(mock_components):
    """query() delegates to nlm_client.query()."""
    from nlm_proxy.core.agent import AgentCore

    nlm, nb_cache, resp_cache, chat_model = mock_components
    nlm.query = AsyncMock(return_value={"answer": "42"})

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    result = await agent.query("nb-1", "What is the answer?")

    assert result["answer"] == "42"
    nlm.query.assert_called_once()


@pytest.mark.asyncio
async def test_agent_query_stream_yields_chunks(mock_components):
    """query_stream() yields NLM streaming chunks."""
    from nlm_proxy.core.agent import AgentCore

    nlm, nb_cache, resp_cache, chat_model = mock_components

    async def mock_stream(*args, **kwargs):
        yield {"type": "answer", "text": "Hello"}
        yield {"type": "answer", "text": "Hello World"}

    nlm.query_stream = mock_stream

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    chunks = []
    async for chunk in agent.query_stream("nb-1", "test"):
        chunks.append(chunk)

    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_agent_wires_cache_invalidation(mock_components):
    """AgentCore wires notebook_cache.on_sources_changed → response_cache.invalidate_notebook."""
    from nlm_proxy.core.agent import AgentCore

    nlm, nb_cache, resp_cache, chat_model = mock_components

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    assert nb_cache._on_sources_changed == resp_cache.invalidate_notebook


@pytest.mark.asyncio
async def test_agent_handle_direct_query_cache_hit(mock_components):
    """handle_direct_query() returns cache result on hit."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components
    cached = MagicMock()
    resp_cache.lookup_async = AsyncMock(return_value=(cached, "exact"))

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    options = RequestOptions()
    result, hit_type = await agent.handle_direct_query("nb-1", "test", options)

    assert result is cached
    assert hit_type == "exact"


@pytest.mark.asyncio
async def test_agent_handle_direct_query_cache_miss(mock_components):
    """handle_direct_query() returns (None, None) on miss."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components
    resp_cache.lookup_async = AsyncMock(return_value=(None, None))

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    options = RequestOptions()
    result, hit_type = await agent.handle_direct_query("nb-1", "test", options)

    assert result is None
    assert hit_type is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement AgentCore**

Create `src/nlm_proxy/core/agent.py`:

```python
"""Shared agent core for both OpenAI proxy and MCP server.

Provides routing (via LangGraph), caching, and NLM query delegation.
Transport-specific concerns (SSE streaming, MCP progress) are handled
by the callers, NOT by AgentCore.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.routing_graph import build_routing_graph

logger = get_logger(__name__)


@dataclass
class RequestOptions:
    """Per-request options extracted from HTTP headers / MCP params."""
    bypass_cache: bool = False
    include_thinking: bool = True
    allowed_notebooks: list[str] | None = None
    conversation_id: str | None = None
    chat_id: str | None = None
    source_ids: list[str] | None = None
    timeout: float | None = None


@dataclass
class RoutingDecision:
    """Result of routing: where to send the query."""
    request_type: str                       # "notebooklm" | "llm_task"
    notebook_id: str | None = None
    reasoning: str = ""
    cache_result: object | None = None      # CachedResponse on cache hit
    cache_hit_type: str | None = None       # "pre_routing_exact" etc.
    conversation_id: str | None = None


class AgentCore:
    """Shared agent logic for both OpenAI proxy and MCP server."""

    def __init__(self, nlm_client, notebook_cache, response_cache, chat_model,
                 session_store=None, routing_settings=None):
        self.nlm_client = nlm_client
        self.notebook_cache = notebook_cache
        self.response_cache = response_cache
        self.chat_model = chat_model
        self.session_store = session_store
        self.routing_graph = build_routing_graph(
            chat_model, notebook_cache, routing_settings=routing_settings
        )

        # Wire bidirectional dependencies
        if notebook_cache and response_cache:
            notebook_cache._on_sources_changed = response_cache.invalidate_notebook
        if nlm_client and notebook_cache:
            nlm_client._notebook_cache = notebook_cache

    async def route(self, query: str, options: RequestOptions) -> RoutingDecision:
        """Get routing decision with optional pre-routing cache check.

        Implements agent_fallback_on_error: if the routing graph fails,
        falls back to a simple NOTEBOOKLM decision using the first
        available notebook (preserving existing behavior on error).
        """
        # Phase 0: Pre-routing global L1 cache check
        if not options.bypass_cache and self.response_cache:
            cached, hit_type = self.response_cache.lookup_global(query)
            if cached:
                # ACL check on cached result
                if options.allowed_notebooks is None or cached.notebook_id in options.allowed_notebooks:
                    return RoutingDecision(
                        request_type="notebooklm",
                        notebook_id=cached.notebook_id,
                        reasoning="Pre-routing cache hit",
                        cache_result=cached,
                        cache_hit_type=f"pre_routing_{hit_type}",
                        conversation_id=options.conversation_id,
                    )

        # Phase 1: LangGraph routing (with thread_id for checkpointing)
        try:
            config = {}
            if options.chat_id:
                config = {"configurable": {"thread_id": options.chat_id}}
            state = await self.routing_graph.ainvoke(
                {
                    "query": query,
                    "allowed_notebooks": options.allowed_notebooks,
                },
                config=config,
            )
            return RoutingDecision(
                request_type=state["request_type"],
                notebook_id=state.get("notebook_id"),
                reasoning=state.get("reasoning", ""),
                conversation_id=options.conversation_id,
            )
        except Exception as e:
            # Fallback: if agent_fallback_on_error is enabled, degrade gracefully
            logger.error("Routing graph failed: %s. Falling back to first notebook.", e)
            fallback_notebook = None
            if self.notebook_cache:
                notebooks = self.notebook_cache.get_all()
                if options.allowed_notebooks is not None:
                    notebooks = [nb for nb in notebooks if nb.id in options.allowed_notebooks]
                if notebooks:
                    fallback_notebook = notebooks[0].id
            if fallback_notebook:
                return RoutingDecision(
                    request_type="notebooklm",
                    notebook_id=fallback_notebook,
                    reasoning=f"Routing fallback (error: {e})",
                    conversation_id=options.conversation_id,
                )
            # No notebooks available — re-raise
            raise

    async def query(self, notebook_id, query, conversation_id=None,
                    source_ids=None, timeout=None) -> dict:
        """Non-streaming query from NotebookLM."""
        return await self.nlm_client.query(
            notebook_id, query_text=query,
            conversation_id=conversation_id,
            source_ids=source_ids,
            timeout=timeout,
        )

    async def query_stream(self, notebook_id, query, conversation_id=None,
                           source_ids=None, **kwargs):
        """Streaming query from NotebookLM. Yields raw NLM chunks."""
        async for chunk in self.nlm_client.query_stream(
            notebook_id, query_text=query,
            conversation_id=conversation_id,
            source_ids=source_ids,
            **kwargs
        ):
            yield chunk

    async def handle_direct_query(self, notebook_id, query, options):
        """Handle direct notebook query (model == notebook_id, bypasses routing).

        Returns (cache_result, hit_type) on cache hit, or (None, None) on miss.
        Caller handles the actual NLM query and format-specific response.
        """
        if not options.bypass_cache and self.response_cache:
            cache_result, hit_type = await self.response_cache.lookup_async(
                notebook_id, query
            )
            if cache_result:
                return cache_result, hit_type
        return None, None
```

**Step 4: Run tests**

Run: `uv run pytest tests/core/test_agent.py -v`
Expected: ALL PASS (9 tests)

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/agent.py tests/core/test_agent.py
git commit -m "feat: add AgentCore with routing, query, and cache integration"
```

---

## Task 5.2: Add integration tests

**Files:**
- Create: `tests/core/test_agent_integration.py`

**Step 1: Write integration tests**

Create `tests/core/test_agent_integration.py`:

```python
"""Integration tests for AgentCore — end-to-end: route → cache check → query → cache store."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def integrated_agent():
    """Create AgentCore with mock components wired together."""
    from nlm_proxy.core.agent import AgentCore

    mock_nlm = AsyncMock()
    mock_notebook_cache = MagicMock()
    mock_notebook_cache.get_all.return_value = []
    mock_response_cache = MagicMock()
    mock_response_cache.lookup_global.return_value = (None, None)
    mock_response_cache.lookup_async = AsyncMock(return_value=(None, None))
    mock_chat_model = AsyncMock()

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_build.return_value = mock_graph
        agent = AgentCore(
            nlm_client=mock_nlm,
            notebook_cache=mock_notebook_cache,
            response_cache=mock_response_cache,
            chat_model=mock_chat_model,
        )
    return agent, mock_nlm, mock_notebook_cache, mock_response_cache, mock_graph


@pytest.mark.asyncio
async def test_route_then_query_with_cache_store(integrated_agent):
    """Full flow: route (cache miss) → query NLM → cache store."""
    from nlm_proxy.core.agent import RequestOptions

    agent, mock_nlm, _, mock_resp_cache, mock_graph = integrated_agent

    # Route returns NOTEBOOKLM
    mock_graph.ainvoke = AsyncMock(return_value={
        "request_type": "notebooklm",
        "notebook_id": "nb-1",
        "reasoning": "Selected notebook",
    })

    # Query returns answer
    mock_nlm.query = AsyncMock(return_value={
        "answer": "The answer is 42",
        "conversation_id": "conv-123",
    })

    options = RequestOptions()
    decision = await agent.route("What is the answer?", options)
    assert decision.request_type == "notebooklm"
    assert decision.notebook_id == "nb-1"

    # Execute query
    result = await agent.query(decision.notebook_id, "What is the answer?")
    assert result["answer"] == "The answer is 42"


@pytest.mark.asyncio
async def test_route_pre_routing_cache_hit_skips_graph(integrated_agent):
    """Pre-routing cache hit → no graph invocation."""
    from nlm_proxy.core.agent import RequestOptions

    agent, _, _, mock_resp_cache, mock_graph = integrated_agent

    # Cache returns hit
    cached = MagicMock(notebook_id="nb-1")
    mock_resp_cache.lookup_global.return_value = (cached, "exact")

    options = RequestOptions()
    decision = await agent.route("cached query", options)

    assert decision.cache_result is cached
    assert decision.cache_hit_type == "pre_routing_exact"
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_route_fallback_on_graph_error(integrated_agent):
    """Graph error → falls back to first available notebook."""
    from nlm_proxy.core.agent import RequestOptions

    agent, _, mock_nb_cache, _, mock_graph = integrated_agent

    # Graph raises error
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    # Notebook cache has one notebook
    mock_nb = MagicMock(id="nb-fallback")
    mock_nb_cache.get_all.return_value = [mock_nb]

    options = RequestOptions()
    decision = await agent.route("failing query", options)

    assert decision.request_type == "notebooklm"
    assert decision.notebook_id == "nb-fallback"
    assert "fallback" in decision.reasoning.lower()


@pytest.mark.asyncio
async def test_direct_query_cache_hit_then_miss(integrated_agent):
    """handle_direct_query: first hit, then miss."""
    from nlm_proxy.core.agent import RequestOptions

    agent, _, _, mock_resp_cache, _ = integrated_agent

    # First call: cache hit
    cached = MagicMock()
    mock_resp_cache.lookup_async = AsyncMock(return_value=(cached, "exact"))
    result, hit_type = await agent.handle_direct_query("nb-1", "q1", RequestOptions())
    assert result is cached

    # Second call: cache miss
    mock_resp_cache.lookup_async = AsyncMock(return_value=(None, None))
    result, hit_type = await agent.handle_direct_query("nb-1", "q2", RequestOptions())
    assert result is None
```

**Step 2: Run tests**

Run: `uv run pytest tests/core/test_agent_integration.py -v`
Expected: ALL PASS (4 tests)

**Step 3: Commit**

```bash
git add tests/core/test_agent_integration.py
git commit -m "test: add AgentCore integration tests (route→query→cache flow)"
```

---

## 🔒 Stage 5 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS — AgentCore tested independently with unit + integration tests. Old SmartRouter still used by server (updated in Stage 6).
