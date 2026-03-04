"""Tests for handle_smart_routing() — four-phase pipeline integration."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from nlm_proxy.core.agent import AgentCore, RoutingDecision


def _setup_app(agent_core=None, session_store=None):
    """Configure app.state for tests."""
    from nlm_proxy.openai.server import app
    app.state.agent_core = agent_core
    app.state.session_store = session_store or MagicMock()
    return app


def _make_agent_with_cache_hit(hit_type="exact"):
    """Create mock AgentCore that returns a pre-routing cache hit."""
    agent = MagicMock(spec=AgentCore)
    cached = MagicMock()
    cached.answer = "Cached answer"
    cached.thinking = None
    cached.conversation_id = "conv-cached"

    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Pre-routing cache hit",
        cache_result=cached,
        cache_hit_type=f"pre_routing_{hit_type}",
    ))
    agent.response_cache = MagicMock()
    agent.chat_model = AsyncMock()
    return agent


def _make_agent_with_post_routing_cache_hit():
    """AgentCore routes (no pre-routing hit), but Phase 2 has a hit."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Selected notebook",
    ))

    cached = MagicMock()
    cached.answer = "Post-routing cached"
    cached.thinking = None
    cached.conversation_id = "conv-post"

    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(cached, "semantic"))
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    return agent


@pytest.fixture
def mock_settings():
    with patch("nlm_proxy.openai.server.get_openai_settings") as mock_openai:
        mock_openai.return_value = MagicMock(api_key="test-key")
        with patch("nlm_proxy.openai.server.get_routing_settings") as mock_routing:
            mock_routing.return_value = MagicMock(
                router_model_name="knowledge-finder",
                llm_base_url="https://api.test.com/v1",
                llm_api_key="test-key",
                llm_model="gpt-4o-mini",
            )
            with patch("nlm_proxy.openai.server.get_tracing_settings") as mock_tracing:
                mock_tracing.return_value = MagicMock(
                    request_max_length=100,
                    response_max_length=100,
                )
                yield


def test_phase0_cache_hit_non_streaming(mock_settings):
    """Pre-routing cache hit → JSON response with X-Cache-Status."""
    agent = _make_agent_with_cache_hit("exact")
    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert "HIT" in response.headers.get("X-Cache-Status", "")
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Cached answer"


def test_phase0_cache_hit_streaming(mock_settings):
    """Pre-routing cache hit → SSE stream with X-Cache-Status."""
    agent = _make_agent_with_cache_hit("exact")
    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "HIT" in response.headers.get("X-Cache-Status", "")
    assert "data: [DONE]" in response.text


def test_phase2_post_routing_cache_hit_non_streaming(mock_settings):
    """Route (cache miss) → notebook-scoped cache hit → JSON."""
    agent = _make_agent_with_post_routing_cache_hit()
    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("X-Cache-Status") == "HIT_SEMANTIC"


def test_bypass_cache_skips_all_cache_checks(mock_settings):
    """bypass_cache=True → no Phase 0/2, goes to Phase 3."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Selected",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "Live answer", "conversation_id": "conv-1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "bypass_cache": True,
        },
    )

    assert response.status_code == 200
    # route() should have been called with bypass_cache=True in options
    call_args = agent.route.call_args
    options = call_args[0][1]
    assert options.bypass_cache is True


def test_acl_wildcard_allows_all(mock_settings):
    """metadata.allowed_notebooks=['*'] → treated as None."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="OK",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "ans", "conversation_id": "c1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "metadata": {"allowed_notebooks": ["*"]},
        },
    )

    assert response.status_code == 200
    options = agent.route.call_args[0][1]
    assert options.allowed_notebooks is None  # Wildcard → None


def test_acl_empty_blocks_all(mock_settings):
    """metadata.allowed_notebooks=[] → empty list passed."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="OK",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "ans", "conversation_id": "c1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "metadata": {"allowed_notebooks": []},
        },
    )

    assert response.status_code == 200
    options = agent.route.call_args[0][1]
    assert options.allowed_notebooks == []


def test_chat_id_from_header(mock_settings):
    """X-OpenWebUI-Chat-Id header → used as chat_id."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="OK",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "ans", "conversation_id": "c1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-key",
            "X-OpenWebUI-Chat-Id": "chat-from-header",
        },
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    agent.get_conversation_id.assert_called_with("chat-from-header")


def test_chat_id_from_metadata(mock_settings):
    """No header → falls back to metadata.chat_id."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="OK",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "ans", "conversation_id": "c1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "metadata": {"chat_id": "chat-from-meta"},
        },
    )

    assert response.status_code == 200
    agent.get_conversation_id.assert_called_with("chat-from-meta")


def test_agent_core_not_initialized_503(mock_settings):
    """app.state.agent_core=None → HTTP 503."""
    app = _setup_app(agent_core=None)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 503


def test_no_user_message_400(mock_settings):
    """Request with no user messages → HTTP 400."""
    agent = MagicMock(spec=AgentCore)
    agent.get_conversation_id = MagicMock(return_value=None)
    app = _setup_app(agent_core=agent)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "system", "content": "Be helpful"}],
            "stream": False,
        },
    )

    assert response.status_code == 400
