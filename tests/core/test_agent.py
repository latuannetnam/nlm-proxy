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


# ── Session helper tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_conversation_id_returns_stored():
    """session_store.get() returns stored conversation_id."""
    from nlm_proxy.core.agent import AgentCore

    mock_session = MagicMock()
    mock_session.get.return_value = "conv-123"

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=mock_session,
        )
    assert agent.get_conversation_id("chat-1") == "conv-123"
    mock_session.get.assert_called_once_with("chat-1")


@pytest.mark.asyncio
async def test_get_conversation_id_no_session_store():
    """session_store=None → returns None."""
    from nlm_proxy.core.agent import AgentCore

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=None,
        )
    assert agent.get_conversation_id("chat-1") is None


@pytest.mark.asyncio
async def test_get_conversation_id_empty_chat_id():
    """chat_id='' → returns None."""
    from nlm_proxy.core.agent import AgentCore

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=MagicMock(),
        )
    assert agent.get_conversation_id("") is None


@pytest.mark.asyncio
async def test_save_conversation_id_calls_session_store():
    """session_store.set() called with correct args."""
    from nlm_proxy.core.agent import AgentCore

    mock_session = MagicMock()

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=mock_session,
        )
    agent.save_conversation_id("chat-1", "conv-123")
    mock_session.set.assert_called_once_with("chat-1", "conv-123")


@pytest.mark.asyncio
async def test_save_conversation_id_noop_when_no_store():
    """session_store=None → no error."""
    from nlm_proxy.core.agent import AgentCore

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=None,
        )
    # Should not raise
    agent.save_conversation_id("chat-1", "conv-123")


@pytest.mark.asyncio
async def test_save_conversation_id_noop_when_empty_conv_id():
    """conversation_id='' → not saved."""
    from nlm_proxy.core.agent import AgentCore

    mock_session = MagicMock()

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=mock_session,
        )
    agent.save_conversation_id("chat-1", "")
    mock_session.set.assert_not_called()


# ── Edge case tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_no_response_cache():
    """response_cache=None → skip cache, go straight to graph."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "request_type": "notebooklm",
            "notebook_id": "nb-1",
            "reasoning": "Selected",
        })
        mock_build.return_value = mock_graph

        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=None, chat_model=AsyncMock(),
        )
        decision = await agent.route("test", RequestOptions())

    assert decision.request_type == "notebooklm"
    assert decision.cache_result is None


@pytest.mark.asyncio
async def test_route_fallback_empty_notebooks_reraises():
    """Graph error + no notebooks → exception propagated."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))
        mock_build.return_value = mock_graph

        mock_nb_cache = MagicMock()
        mock_nb_cache.get_all.return_value = []  # No notebooks

        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=mock_nb_cache,
            response_cache=MagicMock(), chat_model=AsyncMock(),
        )
        agent.response_cache.lookup_global.return_value = (None, None)

        with pytest.raises(RuntimeError, match="LLM error"):
            await agent.route("test", RequestOptions())


@pytest.mark.asyncio
async def test_route_fallback_with_acl_filters_notebooks():
    """Graph error + ACL filter → fallback uses only allowed notebooks."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_build.return_value = mock_graph

        nb1 = MagicMock(id="nb-1")
        nb2 = MagicMock(id="nb-2")
        mock_nb_cache = MagicMock()
        mock_nb_cache.get_all.return_value = [nb1, nb2]

        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=mock_nb_cache,
            response_cache=MagicMock(), chat_model=AsyncMock(),
        )
        agent.response_cache.lookup_global.return_value = (None, None)

        options = RequestOptions(allowed_notebooks=["nb-2"])
        decision = await agent.route("test", options)

    assert decision.notebook_id == "nb-2"
    assert "fallback" in decision.reasoning.lower()
