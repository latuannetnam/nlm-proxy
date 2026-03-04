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


# ── _convert_messages tests ──────────────────────────────────────────────


def test_convert_messages_system_role():
    """System role → SystemMessage."""
    from nlm_proxy.core.llm_client import _convert_messages
    from langchain_core.messages import SystemMessage

    result = _convert_messages([{"role": "system", "content": "You are a helpful assistant"}])
    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are a helpful assistant"


def test_convert_messages_assistant_role():
    """Assistant role → AIMessage."""
    from nlm_proxy.core.llm_client import _convert_messages
    from langchain_core.messages import AIMessage

    result = _convert_messages([{"role": "assistant", "content": "Hello!"}])
    assert len(result) == 1
    assert isinstance(result[0], AIMessage)


def test_convert_messages_mixed_sequence():
    """Multi-role conversation → correct order and types."""
    from nlm_proxy.core.llm_client import _convert_messages
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is AI?"},
    ]
    result = _convert_messages(messages)
    assert len(result) == 4
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], HumanMessage)


def test_convert_messages_pydantic_objects():
    """Objects with .role/.content attributes → correct conversion."""
    from nlm_proxy.core.llm_client import _convert_messages
    from nlm_proxy.openai.types import Message

    messages = [Message(role="user", content="Hello")]
    result = _convert_messages(messages)
    assert len(result) == 1
    assert result[0].content == "Hello"


def test_create_chat_model_anthropic():
    """provider='anthropic' → correct kwargs passed to init_chat_model."""
    from nlm_proxy.core.llm_client import create_chat_model

    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        create_chat_model(
            model="claude-3-5-sonnet",
            provider="anthropic",
            api_key="test-key",
        )
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "anthropic"
        assert call_kwargs["api_key"] == "test-key"
        assert "base_url" not in call_kwargs


def test_create_chat_model_ollama():
    """provider='ollama' → base_url used, no api_key."""
    from nlm_proxy.core.llm_client import create_chat_model

    with patch("langchain.chat_models.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        create_chat_model(
            model="llama3",
            provider="ollama",
            base_url="http://localhost:11434",
        )
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "ollama"
        assert call_kwargs["base_url"] == "http://localhost:11434"
        assert "api_key" not in call_kwargs
