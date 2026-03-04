"""Tests for LangChain LLM client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


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

