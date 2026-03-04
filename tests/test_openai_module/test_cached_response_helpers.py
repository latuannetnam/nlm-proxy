"""Tests for _stream_cached_response and _json_cached_response helpers."""

import json
import pytest
from unittest.mock import MagicMock

from nlm_proxy.core.agent import RoutingDecision
from nlm_proxy.openai.types import ChatCompletionRequest, Message


def _make_decision(thinking=None, hit_type="exact"):
    """Create a RoutingDecision with cache result."""
    cached = MagicMock()
    cached.answer = "Cached answer text"
    cached.thinking = thinking
    cached.conversation_id = "conv-123"
    return RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Cache hit — returning cached response.",
        cache_result=cached,
        cache_hit_type=hit_type,
    )


def _make_request():
    return ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="test query")],
        stream=True,
    )


@pytest.mark.asyncio
async def test_stream_cached_produces_correct_sse_sequence():
    """Reasoning → thinking → answer → stop → [DONE]."""
    from nlm_proxy.openai.server import _stream_cached_response

    decision = _make_decision(thinking="Thinking text...")
    request = _make_request()
    request.include_thinking = True

    chunks = []
    async for chunk in _stream_cached_response(decision, request):
        chunks.append(chunk)

    # Should have: reasoning, thinking, answer, stop, [DONE]
    assert len(chunks) == 5
    assert chunks[-1] == "data: [DONE]\n\n"

    # Parse each data chunk
    parsed = []
    for c in chunks[:-1]:
        assert c.startswith("data: ")
        parsed.append(json.loads(c[6:].strip()))

    # First chunk: reasoning
    assert "reasoning_content" in str(parsed[0]["choices"][0]["delta"])
    # Last data chunk: finish_reason=stop
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_stream_cached_without_thinking():
    """When cache_result.thinking=None → no thinking chunk emitted."""
    from nlm_proxy.openai.server import _stream_cached_response

    decision = _make_decision(thinking=None)
    request = _make_request()

    chunks = []
    async for chunk in _stream_cached_response(decision, request):
        chunks.append(chunk)

    # Should have: reasoning, answer, stop, [DONE] (no thinking)
    assert len(chunks) == 4


@pytest.mark.asyncio
async def test_stream_cached_system_fingerprint_format():
    """system_fingerprint = cache_{hit_type}_conv_{conversation_id}."""
    from nlm_proxy.openai.server import _stream_cached_response

    decision = _make_decision(hit_type="semantic")
    request = _make_request()

    chunks = []
    async for chunk in _stream_cached_response(decision, request):
        if chunk.startswith("data: {"):
            parsed = json.loads(chunk[6:].strip())
            fp = parsed.get("system_fingerprint")
            if fp:
                chunks.append(fp)

    # Answer and stop chunks should have fingerprint
    assert any("cache_semantic_conv_conv-123" in fp for fp in chunks)


def test_json_cached_x_cache_status_header():
    """X-Cache-Status header set correctly."""
    from nlm_proxy.openai.server import _json_cached_response

    for hit_type, expected_header in [
        ("exact", "HIT_EXACT"),
        ("semantic", "HIT_SEMANTIC"),
        ("pre_routing_exact", "HIT_PRE_ROUTING_EXACT"),
    ]:
        decision = _make_decision(hit_type=hit_type)
        request = _make_request()
        response = _json_cached_response(decision, request)
        assert response.headers.get("X-Cache-Status") == expected_header


def test_json_cached_response_content():
    """JSON body has correct content, reasoning_content, system_fingerprint."""
    from nlm_proxy.openai.server import _json_cached_response

    decision = _make_decision(hit_type="exact")
    request = _make_request()
    response = _json_cached_response(decision, request)

    body = json.loads(response.body.decode())
    assert body["choices"][0]["message"]["content"] == "Cached answer text"
    assert "Cache hit" in body["choices"][0]["message"]["reasoning_content"]
    assert body["system_fingerprint"] == "cache_exact_conv_conv-123"
