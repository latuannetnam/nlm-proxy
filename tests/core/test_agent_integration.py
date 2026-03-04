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
