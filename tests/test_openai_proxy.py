# tests/test_openai_proxy.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import subprocess
import sys


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


def test_chat_completions_non_streaming():
    from notebooklm_mcp.openai_proxy import app

    mock_query_result = {
        "answer": "Based on your sources, the answer is 42.",
        "conversation_id": "conv-789",
        "turn_number": 1,
        "is_follow_up": False,
    }

    with patch("notebooklm_mcp.openai_proxy.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.query = AsyncMock(return_value=mock_query_result)
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "model": "nb-123",
            "messages": [{"role": "user", "content": "What is the answer?"}],
            "stream": False
        })

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Based on your sources, the answer is 42."
        assert data["system_fingerprint"] == "conv_conv-789"


def test_chat_completions_streaming():
    from notebooklm_mcp.openai_proxy import app

    async def mock_stream():
        yield {"type": "thinking", "text": "Reading sources...", "conversation_id": "conv-123"}
        yield {"type": "answer", "text": "The answer is ", "conversation_id": "conv-123"}
        yield {"type": "answer", "text": "42.", "conversation_id": "conv-123"}

    with patch("notebooklm_mcp.openai_proxy.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.query_stream = MagicMock(return_value=mock_stream())
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "model": "nb-123",
            "messages": [{"role": "user", "content": "What?"}],
            "stream": True
        })

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Parse SSE chunks
        chunks = [line for line in response.text.split("\n") if line.startswith("data:")]
        # Should have answer chunks (thinking filtered by default)
        assert len(chunks) >= 2  # At least 2 answer chunks + [DONE]
        assert "data: [DONE]" in response.text


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "notebooklm_mcp.openai_proxy", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "--port" in result.stdout
    assert "--host" in result.stdout
