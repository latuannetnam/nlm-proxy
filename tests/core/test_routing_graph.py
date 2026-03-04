"""Tests for LangGraph routing graph."""

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
    assert result.get("notebook_id") is None
    assert mock_chat_model.ainvoke.call_count == 1
