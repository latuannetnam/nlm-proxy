"""Tests for OpenAI proxy server."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("nlm_proxy.openai.server.get_openai_settings") as mock_openai:
        mock_openai.return_value = MagicMock(api_key="test-api-key")
        with patch("nlm_proxy.openai.server.get_routing_settings") as mock_routing:
            mock_routing.return_value = MagicMock(
                router_model_name="knowledge-finder",
                llm_base_url="https://api.test.com/v1",
                llm_api_key="test-key",
                llm_model="gpt-4o-mini"
            )
            yield


def test_list_models_includes_smart_router(mock_settings):
    """Test that /v1/models includes the knowledge finder model."""
    from nlm_proxy.openai.server import app

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.list_notebooks = AsyncMock(return_value=[
            MagicMock(id="nb-123", title="Test Notebook", source_count=3)
        ])
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-api-key"}
        )

        assert response.status_code == 200
        data = response.json()
        model_ids = [m["id"] for m in data["data"]]

        assert "knowledge-finder" in model_ids
        assert "nb-123" in model_ids


def test_chat_completions_smart_routing_notebooklm(mock_settings):
    """Test smart routing to NotebookLM via AgentCore."""
    from nlm_proxy.openai.server import app
    from nlm_proxy.core.agent import AgentCore, RoutingDecision

    mock_agent = MagicMock(spec=AgentCore)
    mock_agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-123",
        reasoning="Selected notebook: Research",
    ))
    mock_agent.response_cache = MagicMock()
    mock_agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    mock_agent.query = AsyncMock(return_value={
        "answer": "The research says...",
        "conversation_id": "conv-123",
    })
    mock_agent.chat_model = AsyncMock()

    app.state.agent_core = mock_agent
    app.state.session_store = MagicMock()

    with patch("nlm_proxy.openai.server.get_tracing_settings") as mock_tracing:
        mock_tracing.return_value = MagicMock(
            request_max_length=100,
            response_max_length=100,
        )

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-api-key"},
            json={
                "model": "knowledge-finder",
                "messages": [{"role": "user", "content": "What does my research say?"}],
                "stream": False
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "Research" in data["choices"][0]["message"].get("reasoning_content", "")
