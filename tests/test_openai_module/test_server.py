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


@pytest.mark.skip(reason="DEFERRED TO STAGE 6: Will be rewritten when server is rewired to use AgentCore + LangGraph routing")
def test_chat_completions_smart_routing_notebooklm(mock_settings):
    """Test smart routing to NotebookLM."""
    from nlm_proxy.openai.server import app

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.list_notebooks = AsyncMock(return_value=[
            MagicMock(id="nb-123", title="Research", source_count=3)
        ])
        mock_client.get_notebook_summary = AsyncMock(return_value={
            "summary": "Research notes",
            "suggested_topics": ["AI"]
        })
        mock_client.query = AsyncMock(return_value={
            "answer": "The research says...",
            "conversation_id": "conv-123"
        })
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        with patch("nlm_proxy.openai.server.SmartRouter") as mock_router_class:
            from nlm_proxy.openai.router import RequestType, RoutingDecision
            mock_router = AsyncMock()
            mock_router.route = AsyncMock(return_value=RoutingDecision(
                request_type=RequestType.NOTEBOOKLM,
                notebook_id="nb-123",
                reasoning="Selected notebook: Research"
            ))
            mock_router.close = AsyncMock()
            mock_router_class.return_value = mock_router

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
