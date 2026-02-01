# tests/test_openai_proxy.py
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import subprocess
import sys

# Test API key used across all tests
TEST_API_KEY = "test-api-key-for-tests"


@pytest.fixture(autouse=True)
def setup_test_api_key():
    """Set up test API key for all tests."""
    with patch.dict(os.environ, {"NLM_PROXY_OPENAI_API_KEY": TEST_API_KEY}, clear=False):
        import nlm_proxy.core.config as config
        config._openai = None  # Reset singleton
        yield


@pytest.mark.openai
def test_health_endpoint():
    from nlm_proxy.openai.server import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.openai
def test_embeddings_returns_501():
    from nlm_proxy.openai.server import app
    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={"input": "test", "model": "x"},
        headers={"Authorization": f"Bearer {TEST_API_KEY}"}
    )
    assert response.status_code == 501
    assert "not supported" in response.json()["detail"].lower()


@pytest.mark.openai
def test_models_list_returns_notebooks():
    from nlm_proxy.openai.server import app
    from nlm_proxy.core import Notebook

    mock_notebooks = [
        Notebook(id="nb-123", title="Research Notes", source_count=3, sources=[]),
        Notebook(id="nb-456", title="Project Docs", source_count=1, sources=[]),
    ]

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.list_notebooks = AsyncMock(return_value=mock_notebooks)
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "nb-123"
        assert data["data"][0]["owned_by"] == "notebooklm"


@pytest.mark.openai
def test_chat_completions_non_streaming():
    from nlm_proxy.openai.server import app

    mock_query_result = {
        "answer": "Based on your sources, the answer is 42.",
        "conversation_id": "conv-789",
        "turn_number": 1,
        "is_follow_up": False,
    }

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.query = AsyncMock(return_value=mock_query_result)
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nb-123",
                "messages": [{"role": "user", "content": "What is the answer?"}],
                "stream": False
            },
            headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Based on your sources, the answer is 42."
        assert data["system_fingerprint"] == "conv_conv-789"


@pytest.mark.openai
def test_chat_completions_streaming():
    from nlm_proxy.openai.server import app

    async def mock_stream():
        yield {"type": "thinking", "text": "Reading sources...", "conversation_id": "conv-123"}
        yield {"type": "answer", "text": "The answer is ", "conversation_id": "conv-123"}
        yield {"type": "answer", "text": "42.", "conversation_id": "conv-123"}

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.query_stream = MagicMock(return_value=mock_stream())
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nb-123",
                "messages": [{"role": "user", "content": "What?"}],
                "stream": True
            },
            headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Parse SSE chunks
        chunks = [line for line in response.text.split("\n") if line.startswith("data:")]
        # Should have answer chunks (thinking filtered by default)
        assert len(chunks) >= 2  # At least 2 answer chunks + [DONE]
        assert "data: [DONE]" in response.text


@pytest.mark.openai
def test_chat_completions_streaming_with_thinking():
    from nlm_proxy.openai.server import app

    async def mock_stream():
        yield {"type": "thinking", "text": "Reading sources...", "conversation_id": "conv-123"}
        yield {"type": "answer", "text": "42", "conversation_id": "conv-123"}

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.query_stream = MagicMock(return_value=mock_stream())
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nb-123",
                "messages": [{"role": "user", "content": "What?"}],
                "stream": True,
                "include_thinking": True  # Should include thinking chunks
            },
            headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
        # Thinking chunk should be included
        assert "Reading sources" in response.text


@pytest.mark.openai
@pytest.mark.skip(reason="Server module doesn't support --help flag without starting uvicorn")
def test_cli_help():
    # CLI help test needs api_key in env for module to load
    env = os.environ.copy()
    env["NLM_PROXY_OPENAI_API_KEY"] = TEST_API_KEY
    result = subprocess.run(
        [sys.executable, "-m", "nlm_proxy.openai.server", "--help"],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0
    assert "--port" in result.stdout
    assert "--host" in result.stdout


@pytest.mark.openai
def test_missing_auth_header_returns_401():
    """Request without Authorization header should return 401."""
    from nlm_proxy.openai.server import app
    client = TestClient(app)
    response = client.get("/v1/models")

    assert response.status_code == 401
    error = response.json()["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_api_key"


@pytest.mark.openai
def test_invalid_api_key_returns_401():
    """Request with wrong API key should return 401."""
    from nlm_proxy.openai.server import app
    client = TestClient(app)
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer wrong-key"}
    )

    assert response.status_code == 401


@pytest.mark.openai
def test_valid_api_key_allows_request():
    """Request with correct API key should succeed."""
    from nlm_proxy.openai.server import app
    from nlm_proxy.core import Notebook

    mock_notebooks = [
        Notebook(id="nb-123", title="Test", source_count=1, sources=[]),
    ]

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.list_notebooks = AsyncMock(return_value=mock_notebooks)
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"}
        )

        assert response.status_code == 200
