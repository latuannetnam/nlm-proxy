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
                router_model_name="smart-router",
                llm_base_url="https://api.test.com/v1",
                llm_api_key="test-key",
                llm_model="gpt-4o-mini"
            )
            yield


def test_list_models_includes_smart_router(mock_settings):
    """Test that /v1/models includes the smart router model."""
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

        assert "smart-router" in model_ids
        assert "nb-123" in model_ids
