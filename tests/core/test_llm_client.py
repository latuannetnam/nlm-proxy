"""Tests for external LLM client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_external_llm_client_complete():
    """Test non-streaming completion."""
    from nlm_proxy.core.llm_client import ExternalLLMClient

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="notebooklm"))]

    with patch("nlm_proxy.core.llm_client.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        client = ExternalLLMClient(
            base_url="https://api.test.com/v1",
            api_key="test-key",
            model="gpt-4o-mini"
        )

        result = await client.complete("Classify this request")

        assert result == "notebooklm"
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_external_llm_client_stream():
    """Test streaming completion."""
    from nlm_proxy.core.llm_client import ExternalLLMClient

    # Mock streaming chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content=" World"))]

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch("nlm_proxy.core.llm_client.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_client_class.return_value = mock_client

        client = ExternalLLMClient(
            base_url="https://api.test.com/v1",
            api_key="test-key",
            model="gpt-4o-mini"
        )

        chunks = []
        async for chunk in await client.stream([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert len(chunks) == 2


@pytest.mark.asyncio
async def test_external_llm_client_close():
    """Test client cleanup."""
    from nlm_proxy.core.llm_client import ExternalLLMClient

    with patch("nlm_proxy.core.llm_client.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        client = ExternalLLMClient(
            base_url="https://api.test.com/v1",
            api_key="test-key",
            model="gpt-4o-mini"
        )
        # Access client property to initialize it (triggers lazy initialization)
        _ = client.client
        await client.close()

        mock_client.close.assert_called_once()
