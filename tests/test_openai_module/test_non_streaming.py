"""Tests for non-streaming (stream=false) response path."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nlm_proxy.core.agent import AgentCore, RoutingDecision


@pytest.mark.asyncio
async def test_non_streaming_notebooklm():
    """Non-streaming NOTEBOOKLM query returns ChatCompletionResponse JSON."""
    from nlm_proxy.openai.server import _handle_non_streaming, app
    from nlm_proxy.openai.types import ChatCompletionRequest, Message

    mock_agent = MagicMock(spec=AgentCore)
    mock_agent.query = AsyncMock(return_value={
        "answer": "The answer is 42.",
        "conversation_id": "conv-123",
    })
    mock_agent.response_cache = None
    mock_agent.chat_model = AsyncMock()

    decision = RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Selected notebook",
    )
    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="What is the meaning of life?")],
        stream=False,
    )

    app.state.session_store = MagicMock()
    response = await _handle_non_streaming(mock_agent, decision, "What is the meaning of life?", request, "chat-1")

    assert response.choices[0].message.content == "The answer is 42."
    assert response.choices[0].message.reasoning_content == "Selected notebook"


@pytest.mark.asyncio
async def test_non_streaming_llm_task():
    """Non-streaming LLM_TASK returns ChatCompletionResponse from ainvoke."""
    from nlm_proxy.openai.server import _handle_non_streaming
    from nlm_proxy.openai.types import ChatCompletionRequest, Message

    mock_response = MagicMock()
    mock_response.content = "Here is your poem about cats..."

    mock_agent = MagicMock(spec=AgentCore)
    mock_agent.chat_model = AsyncMock()
    mock_agent.chat_model.ainvoke = AsyncMock(return_value=mock_response)
    mock_agent.response_cache = None

    decision = RoutingDecision(
        request_type="llm_task",
        reasoning="Classified as LLM task",
    )
    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="Write a poem about cats")],
        stream=False,
    )

    response = await _handle_non_streaming(mock_agent, decision, "Write a poem about cats", request)

    assert response.choices[0].message.content == "Here is your poem about cats..."
    assert "LLM task" in response.choices[0].message.reasoning_content


@pytest.mark.asyncio
async def test_non_streaming_cache_hit():
    """Non-streaming cache hit returns cached ChatCompletionResponse."""
    from nlm_proxy.openai.server import _json_cached_response
    from nlm_proxy.openai.types import ChatCompletionRequest, Message

    cached = MagicMock()
    cached.answer = "Cached answer"
    cached.conversation_id = "conv-1"

    decision = RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Cache hit",
        cache_result=cached,
        cache_hit_type="exact",
    )
    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="test")],
        stream=False,
    )

    response = _json_cached_response(decision, request)
    assert response.status_code == 200
    assert response.headers.get("X-Cache-Status") == "HIT_EXACT"
