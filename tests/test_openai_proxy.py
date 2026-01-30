# tests/test_openai_proxy.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock


def test_health_endpoint():
    from notebooklm_mcp.openai_proxy import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embeddings_returns_501():
    from notebooklm_mcp.openai_proxy import app
    client = TestClient(app)
    response = client.post("/v1/embeddings", json={"input": "test", "model": "x"})
    assert response.status_code == 501
    assert "not supported" in response.json()["detail"].lower()


def test_models_list_returns_notebooks():
    from notebooklm_mcp.openai_proxy import app
    from notebooklm_mcp.api_client import Notebook

    mock_notebooks = [
        Notebook(id="nb-123", title="Research Notes", source_count=3, sources=[]),
        Notebook(id="nb-456", title="Project Docs", source_count=1, sources=[]),
    ]

    with patch("notebooklm_mcp.openai_proxy.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.list_notebooks = AsyncMock(return_value=mock_notebooks)
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "nb-123"
        assert data["data"][0]["owned_by"] == "notebooklm"
