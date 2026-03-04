"""Tests for stream_smart_response() — Phase 3a of four-phase pipeline."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nlm_proxy.core.agent import AgentCore, RoutingDecision
from nlm_proxy.openai.types import ChatCompletionRequest, Message


def _make_request(**kwargs):
    defaults = dict(
        model="knowledge-finder",
        messages=[Message(role="user", content="test query")],
        stream=True,
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _make_mock_agent(query_stream_chunks=None, chat_model_chunks=None):
    """Create a mock AgentCore."""
    agent = MagicMock(spec=AgentCore)
    agent.response_cache = None
    agent.save_conversation_id = MagicMock()

    if query_stream_chunks is not None:
        async def mock_stream(*args, **kwargs):
            for c in query_stream_chunks:
                yield c
        agent.query_stream = mock_stream

    if chat_model_chunks is not None:
        async def mock_astream(messages):
            for c in chat_model_chunks:
                yield c
        agent.chat_model = MagicMock()
        agent.chat_model.astream = mock_astream

    return agent


def _parse_sse_chunks(raw_chunks):
    """Parse SSE data lines into JSON objects."""
    parsed = []
    for chunk in raw_chunks:
        if chunk.startswith("data: {"):
            parsed.append(json.loads(chunk[6:].strip()))
        elif chunk == "data: [DONE]\n\n":
            parsed.append("[DONE]")
    return parsed


@pytest.mark.asyncio
async def test_llm_task_streaming_sse_format():
    """LLM_TASK → chat_model.astream() → correct SSE format."""
    from nlm_proxy.openai.server import stream_smart_response

    chunk1 = MagicMock()
    chunk1.content = "Hello"
    chunk2 = MagicMock()
    chunk2.content = " World"

    agent = _make_mock_agent(chat_model_chunks=[chunk1, chunk2])
    decision = RoutingDecision(request_type="llm_task", reasoning="LLM task")
    request = _make_request()

    chunks = []
    async for chunk in stream_smart_response(agent, decision, "test", request):
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    # Should have: reasoning, content1, content2, stop, [DONE]
    content_chunks = [p for p in parsed if isinstance(p, dict) and p["choices"][0]["delta"].get("content")]
    assert len(content_chunks) == 2
    assert content_chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert content_chunks[1]["choices"][0]["delta"]["content"] == " World"


@pytest.mark.asyncio
async def test_notebooklm_cache_store_after_stream():
    """NLM stream completes → response_cache.store() called."""
    from nlm_proxy.openai.server import stream_smart_response

    nlm_chunks = [
        {"type": "answer", "text": "The answer", "conversation_id": "conv-1"},
        {"type": "answer", "text": "The answer is 42", "conversation_id": "conv-1"},
    ]
    agent = _make_mock_agent(query_stream_chunks=nlm_chunks)
    mock_cache = MagicMock()
    mock_cache._semantic_enabled = False
    agent.response_cache = mock_cache

    decision = RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="Selected",
    )
    request = _make_request()
    request.conversation_id = None

    chunks = []
    async for chunk in stream_smart_response(agent, decision, "test", request):
        chunks.append(chunk)

    mock_cache.store.assert_called_once()
    call_kwargs = mock_cache.store.call_args[1]
    assert call_kwargs["notebook_id"] == "nb-1"
    assert call_kwargs["answer"] == "The answer is 42"  # fully accumulated
    assert call_kwargs["conversation_id"] == "conv-1"


@pytest.mark.asyncio
async def test_include_thinking_false_filters_thinking():
    """include_thinking=False → thinking chunks not in output."""
    from nlm_proxy.openai.server import stream_smart_response

    nlm_chunks = [
        {"type": "thinking", "text": "Let me think...", "conversation_id": "conv-1"},
        {"type": "answer", "text": "The answer", "conversation_id": "conv-1"},
    ]
    agent = _make_mock_agent(query_stream_chunks=nlm_chunks)
    agent.response_cache = None

    decision = RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="Selected",
    )
    request = _make_request(include_thinking=False)
    request.conversation_id = None

    chunks = []
    async for chunk in stream_smart_response(agent, decision, "test", request):
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    # No reasoning_content from thinking chunks should be present
    thinking_chunks = [
        p for p in parsed
        if isinstance(p, dict) and p["choices"][0]["delta"].get("reasoning_content")
        and "Selected" not in p["choices"][0]["delta"].get("reasoning_content", "")
    ]
    assert len(thinking_chunks) == 0


@pytest.mark.asyncio
async def test_conversation_id_extracted_and_saved():
    """NLM chunk has conversation_id → agent.save_conversation_id() called."""
    from nlm_proxy.openai.server import stream_smart_response

    nlm_chunks = [
        {"type": "answer", "text": "Hello", "conversation_id": "conv-new"},
    ]
    agent = _make_mock_agent(query_stream_chunks=nlm_chunks)
    agent.response_cache = None

    decision = RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="Selected",
    )
    request = _make_request()
    request.conversation_id = None

    async for _ in stream_smart_response(agent, decision, "test", request, chat_id="chat-1"):
        pass

    agent.save_conversation_id.assert_called_once_with("chat-1", "conv-new")


@pytest.mark.asyncio
async def test_stream_ends_with_done():
    """All paths → final output is [DONE]."""
    from nlm_proxy.openai.server import stream_smart_response

    agent = _make_mock_agent(chat_model_chunks=[])
    decision = RoutingDecision(request_type="llm_task", reasoning="LLM task")
    request = _make_request()

    chunks = []
    async for chunk in stream_smart_response(agent, decision, "test", request):
        chunks.append(chunk)

    assert chunks[-1] == "data: [DONE]\n\n"
    # Second-to-last should be stop chunk
    stop = json.loads(chunks[-2][6:].strip())
    assert stop["choices"][0]["finish_reason"] == "stop"
