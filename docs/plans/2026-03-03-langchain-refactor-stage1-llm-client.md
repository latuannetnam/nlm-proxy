# Stage 1: Replace ExternalLLMClient → LangChain ChatModel

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LangChain/LangGraph dependencies and create `LangChainLLMClient` as a drop-in replacement for `ExternalLLMClient`.

**Architecture:** Create `create_chat_model()` factory using `init_chat_model()` for multi-provider support. Wrap in `LangChainLLMClient` class with `complete()`, `ainvoke()`, and `astream()` matching the existing interface.

**Inputs:** None — this is a foundation stage.

**Outputs:** `LangChainLLMClient` class and `create_chat_model()` factory in `core/llm_client.py`. Old `ExternalLLMClient` kept for now (removed in Stage 8).

---

## Task 1.1: Add LangChain dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

```toml
requires-python = ">=3.10"  # Required by LangChain/LangGraph v1.0+

dependencies = [
    # Core dependencies
    "httpx>=0.27.0",
    "websocket-client>=1.6.0",
    "pydantic-settings>=2.0.0",
    "typer>=0.9.0",
    # OpenAI proxy dependencies
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "openai>=1.0.0",              # KEEP: OpenAI Pydantic types (ChatCompletionChunk, ChatCompletionResponse) used for SSE formatting
    "fastmcp>=0.1.0",
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp-proto-grpc>=1.20.0",
    "opentelemetry-exporter-otlp-proto-http>=1.20.0",
    "opentelemetry-instrumentation-fastapi>=0.41b0",
    "opentelemetry-instrumentation-httpx>=0.41b0",
    # LangChain + LangGraph (NEW — latest as of March 2026)
    "langchain>=1.2",
    "langchain-openai>=1.1",
    "langchain-huggingface>=1.2",  # HuggingFaceEmbeddings (replaces fastembed)
    "langgraph>=1.0",
    "langgraph-checkpoint>=4.0",
    # Response cache (embedding + similarity)
    "numpy>=1.24",
    # NOTE: fastembed REMOVED — replaced by langchain-huggingface
    # NOTE: langchain-community NOT needed — we only use langchain-huggingface for embeddings
]

[project.optional-dependencies]
cache-gpu = ["torch>=2.0"]  # GPU acceleration for HuggingFaceEmbeddings
anthropic = ["langchain-anthropic>=1.0"]  # Multi-provider: Anthropic support
ollama = ["langchain-ollama>=1.0"]        # Multi-provider: Ollama support
```

**Step 2: Install dependencies**

Run: `uv pip install -e ".[all]"`
Expected: Clean install, no conflicts

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add langchain/langgraph dependencies, remove fastembed"
```

---

## Task 1.2: Rewrite LLM client

**Files:**
- Modify: `src/nlm_proxy/core/llm_client.py`
- Test: `tests/core/test_llm_client.py`

**Step 1: Write failing tests for new LangChainLLMClient**

Replace entire contents of `tests/core/test_llm_client.py`:

```python
"""Tests for LangChain LLM client wrapper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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

    with patch("nlm_proxy.core.llm_client.init_chat_model") as mock_init:
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_llm_client.py -v`
Expected: FAIL — `LangChainLLMClient` not defined

**Step 3: Implement LangChainLLMClient**

Replace `src/nlm_proxy/core/llm_client.py`:

```python
"""LangChain-based LLM client for multi-provider support.

Wraps LangChain's ChatModel abstraction with a simplified interface
used by smart routing, cache L3 verification, and LLM task passthrough.
"""

from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

from nlm_proxy.core.logging import get_logger

logger = get_logger(__name__)


def create_chat_model(
    model: str,
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
):
    """Factory to create a LangChain ChatModel for any provider.

    Supports: openai, anthropic, ollama, azure via LangChain provider packages.
    """
    kwargs = {"model": model, "temperature": temperature}
    if provider == "openai":
        kwargs["model_provider"] = "openai"
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
    elif provider == "anthropic":
        kwargs["model_provider"] = "anthropic"
        if api_key:
            kwargs["api_key"] = api_key
    elif provider == "ollama":
        kwargs["model_provider"] = "ollama"
        if base_url:
            kwargs["base_url"] = base_url
    else:
        kwargs["model_provider"] = provider
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

    logger.info(
        "Initializing ChatModel: provider=%s, model=%s", provider, model
    )
    return init_chat_model(**kwargs)


class LangChainLLMClient:
    """Wrapper around LangChain ChatModel with simplified interface.

    Used by:
    - SmartRouter / LangGraph nodes: classify_request(), select_notebook()
    - ResponseCache: L3 semantic verification
    - OpenAI proxy: LLM_TASK passthrough (streaming + non-streaming)
    """

    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def complete(self, prompt: str, max_tokens: int = 50) -> str:
        """Simple completion (non-streaming). Used by L3 cache verification."""
        logger.debug("[LLM] complete: prompt=%s...", prompt[:200])
        response = await self.chat_model.ainvoke([HumanMessage(content=prompt)])
        result = response.content.strip()
        logger.debug("[LLM] complete result: %s...", result[:200])
        return result

    async def ainvoke(self, messages: list[dict]):
        """Non-streaming invoke with messages list. Returns AIMessage."""
        lc_messages = _convert_messages(messages)
        return await self.chat_model.ainvoke(lc_messages)

    async def astream(self, messages: list[dict]) -> AsyncIterator:
        """Stream completion. Yields AIMessageChunk objects."""
        lc_messages = _convert_messages(messages)
        async for chunk in self.chat_model.astream(lc_messages):
            yield chunk


def _convert_messages(messages: list[dict]) -> list:
    """Convert OpenAI-style message dicts to LangChain message objects."""
    from langchain_core.messages import (
        AIMessage, HumanMessage, SystemMessage,
    )
    mapping = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    result = []
    for msg in messages:
        role = msg.get("role", "user") if isinstance(msg, dict) else msg.role
        content = msg.get("content", "") if isinstance(msg, dict) else msg.content
        cls = mapping.get(role, HumanMessage)
        result.append(cls(content=content))
    return result
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_llm_client.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/llm_client.py tests/core/test_llm_client.py
git commit -m "refactor: replace ExternalLLMClient with LangChain ChatModel"
```

---

> [!NOTE]
> **Task 1.3 (LangChain prompt templates) — REMOVED.** The routing graph nodes in Stage 4 use `load_prompt()` + `str.format()` directly with `HumanMessage`, matching the current pattern.

## 🔒 Stage 1 Checkpoint

Run: `uv run pytest -v`
Expected: ALL existing tests pass. The router and server still use the old `ExternalLLMClient` — they won't be updated until Stages 5/6.
