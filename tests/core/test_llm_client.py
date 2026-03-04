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


# ── LangChainLLMClient Tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_langchain_llm_client_complete():
    """Test non-streaming completion via ChatModel.ainvoke()."""
    from nlm_proxy.core.llm_client import LangChainLLMClient

    mock_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "notebooklm"
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    client = LangChainLLMClient(chat_model=mock_model)
    result = await client.complete("Classify this request")

    assert result == "notebooklm"
    mock_model.ainvoke.assert_called_once()
    # Verify message format
    call_args = mock_model.ainvoke.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].content == "Classify this request"


@pytest.mark.asyncio
async def test_langchain_llm_client_stream():
    """Test streaming via ChatModel.astream()."""
    from nlm_proxy.core.llm_client import LangChainLLMClient

    chunk1 = MagicMock()
    chunk1.content = "Hello"
    chunk2 = MagicMock()
    chunk2.content = " World"

    async def mock_astream(messages):
        yield chunk1
        yield chunk2

    mock_model = MagicMock()
    mock_model.astream = mock_astream

    client = LangChainLLMClient(chat_model=mock_model)
    chunks = []
    async for chunk in client.astream([{"role": "user", "content": "Hi"}]):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " World"


@pytest.mark.asyncio
async def test_langchain_llm_client_ainvoke():
    """Test non-streaming ainvoke with messages list."""
    from nlm_proxy.core.llm_client import LangChainLLMClient

    mock_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "The answer is 42"
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    client = LangChainLLMClient(chat_model=mock_model)
    result = await client.ainvoke([{"role": "user", "content": "What is 6*7?"}])

    assert result.content == "The answer is 42"


def test_init_chat_model_factory():
    """Test chat model factory with different providers."""
    from nlm_proxy.core.llm_client import create_chat_model

    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        model = create_chat_model(
            model="gpt-4o-mini",
            provider="openai",
            base_url="https://api.test.com/v1",
            api_key="test-key",
            temperature=0.0,
        )
        mock_init.assert_called_once()
        assert model is not None

