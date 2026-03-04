"""Tests for direct notebook path (model ≠ router_model_name)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_settings():
    with patch("nlm_proxy.openai.server.get_openai_settings") as m1:
        m1.return_value = MagicMock(api_key="test-key")
        with patch("nlm_proxy.openai.server.get_routing_settings") as m2:
            m2.return_value = MagicMock(
                router_model_name="knowledge-finder",
                llm_base_url="https://api.test.com/v1",
                llm_api_key="test-key",
                llm_model="gpt-4o-mini",
            )
            yield


def _setup_app(agent_core=None):
    from nlm_proxy.openai.server import app
    app.state.agent_core = agent_core
    app.state.session_store = MagicMock()
    return app


def test_direct_cache_hit_non_streaming(mock_settings):
    """Direct path cache hit → JSON with X-Cache-Status."""
    from nlm_proxy.core.agent import AgentCore

    cached = MagicMock()
    cached.answer = "Cached direct answer"
    cached.thinking = None
    cached.conversation_id = "conv-direct"

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(cached, "exact"))
    agent.get_conversation_id = MagicMock(return_value=None)

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "nb-direct-123",
            "messages": [{"role": "user", "content": "What are key points?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("X-Cache-Status") == "HIT_EXACT"
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Cached direct answer"


def test_direct_cache_miss_non_streaming(mock_settings):
    """Direct path cache miss → agent.query() → response + cache store."""
    from nlm_proxy.core.agent import AgentCore

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={
        "answer": "Live direct answer",
        "conversation_id": "conv-live",
    })
    agent.response_cache = MagicMock()
    agent.response_cache._semantic_enabled = False
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "nb-direct-123",
            "messages": [{"role": "user", "content": "What are key points?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Live direct answer"
    # Verify cache store was called
    agent.response_cache.store.assert_called_once()


def test_direct_cache_hit_streaming(mock_settings):
    """Direct path cache hit → SSE stream with X-Cache-Status."""
    from nlm_proxy.core.agent import AgentCore

    cached = MagicMock()
    cached.answer = "Cached streaming answer"
    cached.thinking = None
    cached.conversation_id = "conv-stream"

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(cached, "semantic"))
    agent.get_conversation_id = MagicMock(return_value=None)

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "nb-direct-123",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("X-Cache-Status") == "HIT_SEMANTIC"
    assert "data: [DONE]" in response.text


def test_direct_session_lookup_and_save(mock_settings):
    """Direct path loads conversation_id from session, saves new one."""
    from nlm_proxy.core.agent import AgentCore

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={
        "answer": "Answer",
        "conversation_id": "conv-new",
    })
    agent.response_cache = MagicMock()
    agent.response_cache._semantic_enabled = False
    agent.get_conversation_id = MagicMock(return_value="conv-old")
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-key",
            "X-OpenWebUI-Chat-Id": "chat-123",
        },
        json={
            "model": "nb-direct-123",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    agent.get_conversation_id.assert_called_with("chat-123")
    agent.save_conversation_id.assert_called_with("chat-123", "conv-new")


def test_direct_cache_miss_streaming(mock_settings):
    """Direct cache miss + streaming → stream_response() called."""
    from nlm_proxy.core.agent import AgentCore

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(None, None))
    agent.get_conversation_id = MagicMock(return_value=None)

    app = _setup_app(agent_core=agent)

    # Mock get_client to return a client with query_stream
    async def mock_stream(*args, **kwargs):
        yield {"type": "answer", "text": "Hi", "conversation_id": "conv-1"}

    with patch("nlm_proxy.openai.server.get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.query_stream = mock_stream
        mock_client.close = AsyncMock()
        mock_get.return_value = mock_client

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "nb-direct-123",
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
