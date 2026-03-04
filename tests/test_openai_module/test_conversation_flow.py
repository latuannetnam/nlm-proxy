"""Tests for conversation flow: system_fingerprint emission and structured logging."""

import json
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nlm_proxy.core.agent import AgentCore, RoutingDecision, RequestOptions
from nlm_proxy.openai.session import SessionStore
from nlm_proxy.openai.types import ChatCompletionRequest, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nlm_chunks(conversation_id="conv-abc123"):
    """Create a sequence of NotebookLM streaming chunks."""
    return [
        {"type": "thinking", "text": "Let me think...", "conversation_id": conversation_id},
        {"type": "answer", "text": "The answer is 42.", "conversation_id": conversation_id},
    ]


async def _async_iter(items):
    """Convert a list to an async iterator."""
    for item in items:
        yield item


def _collect_sse_chunks(raw_output: str) -> list[dict]:
    """Parse SSE output into a list of parsed JSON chunks."""
    chunks = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def _make_mock_agent_core(query_stream_chunks=None):
    """Create a mock AgentCore with optional query_stream chunks."""
    mock_agent = MagicMock(spec=AgentCore)
    mock_agent.response_cache = None
    mock_agent.chat_model = AsyncMock()

    if query_stream_chunks is not None:
        async def mock_query_stream(*args, **kwargs):
            for chunk in query_stream_chunks:
                yield chunk
        mock_agent.query_stream = mock_query_stream

    return mock_agent


# ---------------------------------------------------------------------------
# Tests: system_fingerprint in stream_smart_response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_smart_response_emits_system_fingerprint():
    """stream_smart_response() should include system_fingerprint in all chunks
    when NotebookLM returns a conversation_id."""
    from nlm_proxy.openai.server import stream_smart_response, app

    # Setup
    app.state.session_store = SessionStore(ttl_seconds=3600)

    mock_agent = _make_mock_agent_core(_make_nlm_chunks("conv-xyz"))

    decision = RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-123",
        reasoning="Selected notebook: Test",
    )

    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="What is 42?")],
        stream=True,
        include_thinking=True,
    )

    # Collect SSE output
    raw_output = ""
    async for chunk_str in stream_smart_response(
        mock_agent, decision, "What is 42?", request, chat_id="test-chat"
    ):
        raw_output += chunk_str

    chunks = _collect_sse_chunks(raw_output)

    # First chunk is routing reasoning — no system_fingerprint expected
    # Subsequent chunks from NotebookLM should have system_fingerprint
    nlm_chunks = [c for c in chunks if c.get("system_fingerprint")]
    assert len(nlm_chunks) >= 1, f"Expected system_fingerprint in NLM chunks, got: {[c.get('system_fingerprint') for c in chunks]}"

    for chunk in nlm_chunks:
        assert chunk["system_fingerprint"] == "conv_conv-xyz"

    # Final chunk should also have system_fingerprint
    final_chunk = chunks[-1]
    assert final_chunk["choices"][0].get("finish_reason") == "stop"
    assert final_chunk["system_fingerprint"] == "conv_conv-xyz"

    app.state.session_store.shutdown()


@pytest.mark.asyncio
async def test_stream_smart_response_no_fingerprint_without_conversation_id():
    """stream_smart_response() should emit system_fingerprint=null when NLM
    returns no conversation_id."""
    from nlm_proxy.openai.server import stream_smart_response, app

    app.state.session_store = SessionStore(ttl_seconds=3600)

    # Chunks without conversation_id
    chunks_no_conv = [
        {"type": "answer", "text": "Hello"},
    ]
    mock_agent = _make_mock_agent_core(chunks_no_conv)

    decision = RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-123",
        reasoning="Test routing",
    )

    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="Hi")],
        stream=True,
    )

    raw_output = ""
    async for chunk_str in stream_smart_response(
        mock_agent, decision, "Hi", request, chat_id="test-chat"
    ):
        raw_output += chunk_str

    chunks = _collect_sse_chunks(raw_output)

    # All chunks should have system_fingerprint=None (serialized as null or absent)
    for chunk in chunks:
        assert chunk.get("system_fingerprint") is None

    app.state.session_store.shutdown()


# ---------------------------------------------------------------------------
# Tests: system_fingerprint in handle_smart_routing (non-streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_smart_routing_non_streaming_emits_system_fingerprint():
    """handle_smart_routing() non-streaming should include system_fingerprint
    in the ChatCompletionResponse when conversation_id is stored in session."""
    from nlm_proxy.openai.server import handle_smart_routing, app

    app.state.session_store = SessionStore(ttl_seconds=3600)
    # Pre-store a conversation_id
    app.state.session_store.set("test-chat-id", "conv-stored-123")

    # Build a mock AgentCore that routes to NOTEBOOKLM
    mock_agent = MagicMock(spec=AgentCore)
    mock_agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-456",
        reasoning="Selected notebook",
    ))
    mock_agent.response_cache = MagicMock()
    mock_agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    mock_agent.query = AsyncMock(return_value={
        "answer": "Test answer",
        "conversation_id": "conv-stored-123",
    })
    mock_agent.chat_model = AsyncMock()
    app.state.agent_core = mock_agent

    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="Test query")],
        stream=False,
    )

    mock_http_request = MagicMock()
    mock_http_request.headers = {"X-OpenWebUI-Chat-Id": "test-chat-id"}

    with patch("nlm_proxy.openai.server.get_routing_settings") as mock_routing, \
         patch("nlm_proxy.openai.server.get_tracing_settings") as mock_tracing:

        mock_routing.return_value = MagicMock(
            router_model_name="knowledge-finder",
        )
        mock_tracing.return_value = MagicMock(
            request_max_length=100,
            response_max_length=100,
        )

        response = await handle_smart_routing(request, mock_http_request)

        assert response.system_fingerprint == "conv_conv-stored-123"

    app.state.session_store.shutdown()


# ---------------------------------------------------------------------------
# Tests: structured logging events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_saved_logged_on_stream(caplog):
    """session_saved log event should fire when conversation_id is saved during streaming."""
    from nlm_proxy.openai.server import stream_smart_response, app

    app.state.session_store = SessionStore(ttl_seconds=3600)

    mock_agent = _make_mock_agent_core(_make_nlm_chunks("conv-log-test"))

    decision = RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-log",
        reasoning="Test",
    )

    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="Log test")],
        stream=True,
    )

    with caplog.at_level(logging.INFO, logger="nlm_proxy"):
        async for _ in stream_smart_response(
            mock_agent, decision, "Log test", request, chat_id="log-chat-id"
        ):
            pass

    # Check for structured log events
    log_messages = [r.message for r in caplog.records]
    assert any("session_saved" in msg and "log-chat-id" in msg for msg in log_messages), \
        f"Expected 'session_saved' log with 'log-chat-id', got: {log_messages}"
    assert any("conversation_id_from_nlm" in msg and "conv-log-test" in msg for msg in log_messages), \
        f"Expected 'conversation_id_from_nlm' log, got: {log_messages}"

    app.state.session_store.shutdown()


@pytest.mark.asyncio
async def test_session_not_saved_logged_when_no_conv_id(caplog):
    """session_not_saved log event should fire when NLM returns no conversation_id (non-streaming)."""
    from nlm_proxy.openai.server import chat_completions, app

    app.state.session_store = SessionStore(ttl_seconds=3600)

    mock_client = AsyncMock()
    mock_client.query = AsyncMock(return_value={
        "answer": "Hello!",
        "conversation_id": "",  # Empty — no conversation_id
    })
    mock_client.close = AsyncMock()

    mock_http_request = MagicMock()
    mock_http_request.headers = {}  # No chat_id from header

    request = ChatCompletionRequest(
        model="nb-direct",
        messages=[Message(role="user", content="Hi")],
        stream=False,
        metadata={"chat_id": "no-conv-chat"},
    )

    with patch("nlm_proxy.openai.server.get_client", return_value=mock_client), \
         patch("nlm_proxy.openai.server.get_routing_settings") as mock_routing:

        mock_routing.return_value = MagicMock(
            router_model_name="knowledge-finder",
        )

        with caplog.at_level(logging.INFO, logger="nlm_proxy"):
            response = await chat_completions(request, mock_http_request)

    log_messages = [r.message for r in caplog.records]
    assert any("session_not_saved" in msg and "no-conv-chat" in msg for msg in log_messages), \
        f"Expected 'session_not_saved' log, got: {log_messages}"

    app.state.session_store.shutdown()
