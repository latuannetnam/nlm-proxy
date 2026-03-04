# LangChain/LangGraph Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor nlm-proxy to use LangChain (LLM abstraction, embeddings) and LangGraph (stateful routing graph) while preserving all existing features through staged migration with full test coverage.

**Architecture:** Six-stage migration: (1) foundation layer swaps, (2) cache thread-safety fix, (3) config + notebook_cache move, (4) routing graph rewrite, (5) server pipeline rewire, (6) MCP unification. Each stage has a validation checkpoint — all existing tests must pass before proceeding.

**Tech Stack:** LangChain ≥1.2, LangGraph ≥1.0, langchain-openai ≥1.1, langchain-huggingface ≥1.2 (HuggingFaceEmbeddings), langgraph-checkpoint ≥4.0. **Requires Python ≥3.10.**

**Design Document:** [2026-03-03-langchain-refactor-design.md](file:///d:/latuan/Programming/nlm-proxy/docs/plans/2026-03-03-langchain-refactor-design.md)

---

## Stage Overview

| Stage | Component | Risk | Depends On |
|-------|-----------|------|------------|
| **1** | Replace `ExternalLLMClient` → LangChain `ChatModel` | 🟢 Low | — |
| **2** | Replace `fastembed` → LangChain `Embeddings` + fix `_last_hit_type` | 🟢 Low | Stage 1 |
| **3** | Add `AgentSettings` + move `NotebookCache` to `core/` | 🟢 Low | — |
| **4** | Rewrite `SmartRouter` → LangGraph `StateGraph` | 🟡 Medium | Stages 1, 3 |
| **5** | Rewire OpenAI proxy server (four-phase pipeline) | 🔴 High | Stages 1–4 |
| **6** | MCP server unification (shared `AgentCore`) | 🟡 Medium | Stage 5 |

> [!IMPORTANT]
> **Validation checkpoint** after each stage: `uv run pytest` must pass. If any test fails, fix before proceeding.

---

## Stage 1: Replace ExternalLLMClient → LangChain ChatModel

### Task 1.1: Add LangChain dependencies

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
    "openai>=1.0.0",              # KEEP: OpenAI types used for streaming SSE formatting
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

### Task 1.2: Rewrite LLM client

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
> **Task 1.3 (LangChain prompt templates) — REMOVED.** The routing graph nodes in Stage 4 use `load_prompt()` + `str.format()` directly with `HumanMessage`, matching the current pattern. Creating `ChatPromptTemplate` wrappers would be dead code. If we later want pipeline chains (prompt | model | parser), we can add templates at that point.

### 🔒 Stage 1 Checkpoint

Run: `uv run pytest -v`
Expected: ALL existing tests pass. The router and server still use the old `ExternalLLMClient` — they won't be updated until Stage 4/5.

---

## Stage 2: Replace fastembed → LangChain Embeddings + Fix `_last_hit_type`

### Task 2.1: Replace fastembed with LangChain HuggingFaceEmbeddings

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py` (methods `_load_embedding_model`, `_compute_embedding`, `__init__` signature)
- Test: `tests/core/test_response_cache_semantic.py`, `tests/core/test_embedding_models.py`

**Step 1: Update `_load_embedding_model()` and `_compute_embedding()` in response_cache.py**

Replace the fastembed import and model loading with:

```python
# In __init__, change the fastembed loading block:
def _load_embedding_model(self):
    """Load the LangChain HuggingFace embedding model."""
    try:
        # langchain-huggingface >= 1.2 (NOT langchain-community)
        from langchain_huggingface import HuggingFaceEmbeddings
        import numpy as np
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._embedding_model_name
        )
        self._np = np
        logger.info(
            "Loaded embedding model: %s", self._embedding_model_name
        )
    except Exception as e:
        logger.warning("Failed to load embedding model: %s", e)
        self._embedding_model = None

def _compute_embedding(self, query: str):
    """Compute query embedding using LangChain HuggingFace model."""
    if self._embedding_model is None:
        self._load_embedding_model()
    if self._embedding_model is None:
        return None
    try:
        embedding = self._embedding_model.embed_query(query)
        vec = self._np.array(embedding)
        norm = self._np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        logger.warning("Embedding computation failed: %s", e)
        return None
```

**Step 2: Update test_embedding_models.py**

Replace fastembed imports with LangChain equivalents:

```python
"""Test embedding model performance for Vietnamese and multilingual queries."""

import pytest

# Skip if sentence-transformers not installed
st = pytest.importorskip("sentence_transformers")

# langchain-huggingface >= 1.2 (NOT langchain-community)
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import time

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="module")
def model():
    """Load embedding model once for all tests."""
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def cosine_sim(model: HuggingFaceEmbeddings, text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    embeddings = model.embed_documents([text_a, text_b])
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ... rest of test classes unchanged, just use new cosine_sim signature
```

**Step 3: Run tests**

Run: `uv run pytest tests/core/test_response_cache_semantic.py tests/core/test_embedding_models.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_embedding_models.py
git commit -m "refactor: replace fastembed with LangChain HuggingFaceEmbeddings"
```

---

### Task 2.2: Fix `_last_hit_type` thread safety — return tuple from lookup

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py` (`lookup()`, `lookup_async()`, `lookup_global()`)
- Modify: `tests/core/test_response_cache.py`, `tests/core/test_response_cache_integration.py`
- Modify: `src/nlm_proxy/openai/server.py` (all callers of `_last_hit_type`)

**Step 1: Update test to expect tuple return**

In `tests/core/test_response_cache_integration.py`, update `test_lookup_returns_cache_hit_type`:

```python
def test_lookup_returns_cache_hit_type(self):
    """Lookup result should return (CachedResponse, hit_type) tuple."""
    from nlm_proxy.core.response_cache import ResponseCache

    cache = ResponseCache(max_entries=100, ttl_seconds=3600, semantic_enabled=False)
    cache.store("nb-1", "key points?", "answer", None, "conv-1")

    result, hit_type = cache.lookup("nb-1", "key points?")
    assert result is not None
    assert hit_type == "exact"

    result, hit_type = cache.lookup("nb-1", "nonexistent")
    assert result is None
    assert hit_type is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_response_cache_integration.py::TestFullCacheLookup::test_lookup_returns_cache_hit_type -v`
Expected: FAIL — `ValueError: too many values to unpack`

**Step 3: Update `lookup()`, `lookup_async()`, `lookup_global()` to return tuples**

In `response_cache.py`, change returns:
- `lookup()`: return `(CachedResponse, "exact")` or `(None, None)`
- `lookup_async()`: return `(CachedResponse, hit_type)` or `(None, None)`
- `lookup_global()`: return `(CachedResponse, "exact")` or `(None, None)`
- Remove `self._last_hit_type` field entirely

**Step 4: Update all callers in `server.py`**

Replace every `cache_result = ...; hit_type = app.state.response_cache._last_hit_type or "exact"` with:
```python
cache_result, hit_type = ...
```

There are **6 locations** in `server.py` that read `_last_hit_type`:
- Lines ~376, ~438, ~509, ~566, ~818 (search for `_last_hit_type`)

**Step 5: Run ALL tests**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py src/nlm_proxy/openai/server.py tests/
git commit -m "fix: return (result, hit_type) tuple from cache lookup — thread safety"
```

---

### Task 2.3: Update L3 verification to use LangChain ChatModel

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py` (`_verify_semantic_match`)
- Update: `tests/core/test_response_cache_llm.py`

The L3 verification calls `self._llm_client.complete(prompt)`. The new `LangChainLLMClient` also exposes `.complete()` with the same signature, so **no changes needed to `_verify_semantic_match()`** — it works transparently.

Only update the server init in `main()` to pass a `LangChainLLMClient` instead of `ExternalLLMClient`:

```python
# In server.py main(), change:
# from nlm_proxy.core.llm_client import ExternalLLMClient
# llm_client = ExternalLLMClient(base_url=..., api_key=..., model=...)
# TO:
from nlm_proxy.core.llm_client import LangChainLLMClient, create_chat_model
chat_model = create_chat_model(
    model=routing_settings.llm_model,
    base_url=routing_settings.llm_base_url,
    api_key=routing_settings.llm_api_key,
)
llm_client = LangChainLLMClient(chat_model=chat_model)
```

No test changes needed — the mock `llm_client` in `test_response_cache_llm.py` already mocks `.complete()`.

**Step 1: Run tests**

Run: `uv run pytest tests/core/test_response_cache_llm.py -v`
Expected: PASS

**Step 2: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "refactor: use LangChainLLMClient for L3 cache verification"
```

---

### 🔒 Stage 2 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS

---

## Stage 3: Config + NotebookCache Move

### Task 3.1: Add AgentSettings to config.py

**Files:**
- Modify: `src/nlm_proxy/core/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
def test_agent_settings_defaults():
    """Test AgentSettings has expected defaults."""
    from nlm_proxy.core.config import get_agent_settings
    settings = get_agent_settings()
    assert settings.llm_provider == "openai"
    assert settings.embedding_provider == "huggingface"
    assert settings.memory_backend == "memory"
    assert settings.agent_max_iterations == 10
    assert settings.agent_fallback_on_error is True
```

**Step 2: Implement AgentSettings**

Add to `config.py` after `CacheSettings`:

```python
class AgentSettings(BaseSettings):
    """LangChain/LangGraph agent configuration (additive — does not replace existing)."""
    llm_provider: str = Field(default="openai", description="LLM provider")
    embedding_provider: str = Field(default="huggingface", description="Embedding provider")
    memory_backend: str = Field(default="memory", description="memory | sqlite | postgres")
    memory_db_path: str = Field(default="~/.nlm-proxy/memory.db")
    agent_max_iterations: int = Field(default=10)
    agent_verbose: bool = Field(default=False)
    agent_fallback_on_error: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_AGENT_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

# Add singleton:
_agent: AgentSettings | None = None

def get_agent_settings() -> AgentSettings:
    """Get the agent settings instance."""
    global _agent
    if _agent is None:
        _agent = AgentSettings()
    return _agent
```

**Step 3: Run test, commit**

Run: `uv run pytest tests/test_config.py -v`

```bash
git commit -m "feat: add AgentSettings with NLM_PROXY_AGENT_ prefix"
```

---

### Task 3.2: Move NotebookCache to core/

**Files:**
- Move: `src/nlm_proxy/openai/notebook_cache.py` → `src/nlm_proxy/core/notebook_cache.py`
- Modify: All imports referencing `nlm_proxy.openai.notebook_cache`
- Test: `tests/test_openai_module/test_notebook_cache.py`

**Step 1: Copy file and add re-export**

```bash
# Copy the file
cp src/nlm_proxy/openai/notebook_cache.py src/nlm_proxy/core/notebook_cache.py

# Add re-export in openai/notebook_cache.py for backward compat:
```

Replace `src/nlm_proxy/openai/notebook_cache.py` with:
```python
"""Backward-compatible re-export. Actual implementation moved to core/."""
from nlm_proxy.core.notebook_cache import (  # noqa: F401
    NotebookCache, NotebookInfo, SourceInfo, _extract_first_sentence,
)
```

**Step 2: Update `core/__init__.py`** to export `NotebookCache`

**Step 3: Run ALL tests**

Run: `uv run pytest -v`
Expected: ALL PASS — re-export maintains backward compatibility

**Step 4: Commit**

```bash
git commit -m "refactor: move NotebookCache to core/ with backward-compat re-export"
```

---

### 🔒 Stage 3 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS

---

## Stage 4: Rewrite SmartRouter → LangGraph StateGraph

### Task 4.1: Create LangGraph routing graph

**Files:**
- Create: `src/nlm_proxy/core/routing_graph.py`
- Test: `tests/core/test_routing_graph.py`

**Step 1: Write failing tests**

Create `tests/core/test_routing_graph.py`:

```python
"""Tests for LangGraph routing graph."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_notebook_cache():
    """Create a mock NotebookCache with test data."""
    from nlm_proxy.core.notebook_cache import NotebookCache
    mock_nlm = MagicMock()
    mock_nlm.list_notebooks = AsyncMock(return_value=[])
    cache = NotebookCache(nlm_client=mock_nlm, ttl_seconds=3600)
    cache.set("nb-1", "AI Research", "Notes about AI", ["AI", "ML"])
    cache.set("nb-2", "Project Docs", "Project documentation", ["Docs"])
    cache.set("nb-3", "Meeting Notes", "Team meetings", ["Meetings"])
    return cache


@pytest.fixture
def mock_chat_model():
    """Create a mock LangChain ChatModel."""
    model = AsyncMock()
    return model


def _mock_llm_response(content: str):
    """Helper to create a mock AIMessage."""
    resp = MagicMock()
    resp.content = content
    return resp


# --- classify_node tests ---

@pytest.mark.asyncio
async def test_classify_notebooklm(mock_chat_model, mock_notebook_cache):
    """Classify knowledge query as NOTEBOOKLM."""
    from nlm_proxy.core.routing_graph import classify_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("notebooklm")
    )
    state = {"query": "What does my AI research say?", "request_type": None,
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await classify_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["request_type"] == "notebooklm"


@pytest.mark.asyncio
async def test_classify_llm_task(mock_chat_model, mock_notebook_cache):
    """Classify meta-operation as LLM_TASK."""
    from nlm_proxy.core.routing_graph import classify_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("llm_task")
    )
    state = {"query": "Write a poem about cats", "request_type": None,
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await classify_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["request_type"] == "llm_task"
    assert "LLM task" in result["reasoning"]


# --- select_notebook_node tests ---

@pytest.mark.asyncio
async def test_select_notebook_picks_correct(mock_chat_model, mock_notebook_cache):
    """Select the notebook whose ID appears in LLM response."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("nb-2")
    )
    state = {"query": "Project status?", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] == "nb-2"
    assert "Project Docs" in result["reasoning"]


@pytest.mark.asyncio
async def test_select_notebook_no_acl(mock_chat_model, mock_notebook_cache):
    """No ACL filter → all notebooks sent to LLM."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("nb-1")
    )
    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    # Verify all 3 notebooks were in the prompt
    call_args = mock_chat_model.ainvoke.call_args[0][0]
    prompt_text = call_args[0].content if hasattr(call_args[0], 'content') else str(call_args)
    assert "nb-1" in prompt_text
    assert "nb-2" in prompt_text
    assert "nb-3" in prompt_text


@pytest.mark.asyncio
async def test_select_notebook_with_acl(mock_chat_model, mock_notebook_cache):
    """ACL filter → only allowed notebooks sent to LLM."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("nb-2")
    )
    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": ["nb-2", "nb-3"]}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] == "nb-2"
    # Verify nb-1 was filtered out of prompt
    call_args = mock_chat_model.ainvoke.call_args[0][0]
    prompt_text = call_args[0].content if hasattr(call_args[0], 'content') else str(call_args)
    assert "nb-1" not in prompt_text
    assert "nb-2" in prompt_text


@pytest.mark.asyncio
async def test_select_notebook_acl_filters_all(mock_chat_model, mock_notebook_cache):
    """ACL matches no notebooks → error reasoning, no LLM call."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": ["nb-999"]}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] is None
    assert "No accessible notebooks" in result["reasoning"]
    mock_chat_model.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_select_notebook_empty_acl(mock_chat_model, mock_notebook_cache):
    """Empty ACL list → error reasoning, no LLM call."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": []}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    assert result["notebook_id"] is None
    assert "No accessible notebooks" in result["reasoning"]


@pytest.mark.asyncio
async def test_select_notebook_fallback(mock_chat_model, mock_notebook_cache):
    """LLM returns unrecognized ID → fallback to first notebook."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("some-random-text")
    )
    state = {"query": "test", "request_type": "notebooklm",
             "notebook_id": None, "reasoning": "", "available_notebooks": [],
             "allowed_notebooks": None}
    result = await select_notebook_node(
        state, chat_model=mock_chat_model, notebook_cache=mock_notebook_cache
    )
    # Should fall back to first notebook
    assert result["notebook_id"] is not None
    assert "Defaulted" in result["reasoning"]


# --- route_after_classify tests ---

def test_route_after_classify_notebooklm():
    """NOTEBOOKLM → route to select_notebook node."""
    from nlm_proxy.core.routing_graph import route_after_classify
    assert route_after_classify({"request_type": "notebooklm"}) == "select_notebook"


def test_route_after_classify_llm_task():
    """LLM_TASK → route to END."""
    from nlm_proxy.core.routing_graph import route_after_classify, END_NODE
    assert route_after_classify({"request_type": "llm_task"}) == END_NODE


# --- Full graph end-to-end ---

@pytest.mark.asyncio
async def test_full_graph_notebooklm(mock_chat_model, mock_notebook_cache):
    """Full graph: classify as NOTEBOOKLM → select notebook."""
    from nlm_proxy.core.routing_graph import build_routing_graph

    mock_chat_model.ainvoke = AsyncMock(
        side_effect=[
            _mock_llm_response("notebooklm"),  # classify
            _mock_llm_response("nb-1"),          # select
        ]
    )
    graph = build_routing_graph(mock_chat_model, mock_notebook_cache)
    result = await graph.ainvoke({
        "query": "What does my research say?",
        "allowed_notebooks": None,
    })
    assert result["request_type"] == "notebooklm"
    assert result["notebook_id"] == "nb-1"


@pytest.mark.asyncio
async def test_full_graph_llm_task(mock_chat_model, mock_notebook_cache):
    """Full graph: classify as LLM_TASK → skip notebook selection."""
    from nlm_proxy.core.routing_graph import build_routing_graph

    mock_chat_model.ainvoke = AsyncMock(
        return_value=_mock_llm_response("llm_task")
    )
    graph = build_routing_graph(mock_chat_model, mock_notebook_cache)
    result = await graph.ainvoke({
        "query": "Translate this to Spanish",
        "allowed_notebooks": None,
    })
    assert result["request_type"] == "llm_task"
    assert result["notebook_id"] is None
    # LLM called only once (classify), not twice
    assert mock_chat_model.ainvoke.call_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_routing_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nlm_proxy.core.routing_graph'`

**Step 3: Implement routing graph**

Create `src/nlm_proxy/core/routing_graph.py`:

```python
"""LangGraph-based routing graph for smart request classification.

Replaces the linear SmartRouter with a LangGraph StateGraph that:
1. Classifies intent (NOTEBOOKLM vs LLM_TASK)
2. Selects notebook (if NOTEBOOKLM) with ACL filtering

The graph produces a routing DECISION only — it does NOT execute queries
or handle streaming. Those are done by the transport layer.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.tracing import record_span, add_span_attributes
from nlm_proxy.openai.prompts import load_prompt

logger = get_logger(__name__)

# Re-export END as a constant for tests
END_NODE = END

DEFAULT_MAX_SOURCE_TITLES = 15


# --- LangGraph State ---

class RouterState(TypedDict):
    """Internal state for the routing graph."""
    query: str
    request_type: str | None        # "notebooklm" | "llm_task"
    notebook_id: str | None          # Selected notebook UUID
    reasoning: str                   # Human-readable explanation
    available_notebooks: list[dict]  # Populated by nodes
    allowed_notebooks: list[str] | None  # Per-request ACL filter


# --- Graph Nodes ---

@record_span("smart_router.classify")
async def classify_node(
    state: RouterState,
    *,
    chat_model,
    notebook_cache,
) -> dict:
    """Classify the request as NOTEBOOKLM or LLM_TASK using LLM."""
    query = state["query"]
    logger.debug("[ROUTER] Classifying: %s...", query[:100])

    prompt_template = load_prompt("classify_request")
    prompt = prompt_template.format(query=query)

    response = await chat_model.ainvoke([HumanMessage(content=prompt)])
    response_lower = response.content.lower().strip()

    if "notebooklm" in response_lower:
        logger.info("[ROUTER] Classified as NOTEBOOKLM")
        add_span_attributes(classification_result="NOTEBOOKLM")
        return {"request_type": "notebooklm"}

    logger.info("[ROUTER] Classified as LLM_TASK")
    add_span_attributes(classification_result="LLM_TASK")
    return {
        "request_type": "llm_task",
        "reasoning": "Classified as LLM task (not a notebook query)",
    }


@record_span("smart_router.select_notebook")
async def select_notebook_node(
    state: RouterState,
    *,
    chat_model,
    notebook_cache,
    routing_settings=None,
) -> dict:
    """Select the best notebook for the query, respecting ACL filters."""
    query = state["query"]
    allowed = state.get("allowed_notebooks")

    logger.debug("[ROUTER] Selecting notebook for: %s...", query[:100])

    # Get all cached notebooks
    notebooks = notebook_cache.get_all()
    if not notebooks:
        logger.warning("[ROUTER] No notebooks available")
        add_span_attributes(candidates_count=0)
        return {"notebook_id": None, "reasoning": "No notebooks available"}

    # Apply per-request ACL filtering
    if allowed is not None:
        notebooks = [nb for nb in notebooks if nb.id in allowed]
        add_span_attributes(
            acl_filter_applied=True,
            acl_allowed_count=len(allowed),
            acl_matched_count=len(notebooks),
        )
        if not notebooks:
            logger.warning("[ROUTER] ACL filter matched no notebooks")
            add_span_attributes(candidates_count=0)
            return {
                "notebook_id": None,
                "reasoning": "No accessible notebooks for this user",
            }
    else:
        add_span_attributes(acl_filter_applied=False)

    add_span_attributes(candidates_count=len(notebooks))

    # Build notebook info for LLM prompt (use routing_settings, NOT os.environ)
    if routing_settings is None:
        from nlm_proxy.core.config import get_routing_settings
        routing_settings = get_routing_settings()
    max_source_titles = routing_settings.max_source_titles
    source_descriptions_enabled = routing_settings.source_descriptions_enabled
    source_max_keywords = routing_settings.source_max_keywords
    source_summary_max_chars = routing_settings.source_summary_max_chars
    source_descriptions_max_sources = routing_settings.source_descriptions_max_sources

    notebooks_info = []
    for nb in notebooks:
        info: dict = {
            "id": nb.id,
            "title": nb.title,
            "summary": nb.summary[:500] if nb.summary else "",
            "topics": nb.topics[:5] if nb.topics else [],
            "source_count": nb.source_count,
            "source_types": nb.source_types,
        }
        if source_descriptions_enabled:
            info["sources"] = nb.get_source_descriptions(
                max_sources=source_descriptions_max_sources,
                max_keywords=source_max_keywords,
                summary_max_chars=source_summary_max_chars,
            )[:max_source_titles]
        else:
            info["source_titles"] = nb.source_titles[:max_source_titles]
        notebooks_info.append(info)

    # Call LLM to select notebook
    prompt_template = load_prompt("select_notebook")
    prompt = prompt_template.format(
        notebooks_json=json.dumps(notebooks_info, indent=2),
        query=query,
    )

    logger.debug("[ROUTER] Asking LLM to select from %d notebooks", len(notebooks))
    response = await chat_model.ainvoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    # Parse response — expect notebook_id in the response
    for nb in notebooks:
        if nb.id in response_text:
            reasoning = f"Selected notebook: {nb.title} (ID: {nb.id})"
            logger.info("[ROUTER] %s", reasoning)
            add_span_attributes(
                selected_notebook_id=nb.id,
                selected_notebook_title=nb.title,
            )
            return {"notebook_id": nb.id, "reasoning": reasoning}

    # Fallback to first notebook
    if notebooks:
        reasoning = f"Defaulted to notebook: {notebooks[0].title} (ID: {notebooks[0].id})"
        logger.info("[ROUTER] %s", reasoning)
        add_span_attributes(
            selected_notebook_id=notebooks[0].id,
            selected_notebook_title=notebooks[0].title,
            selection_fallback=True,
        )
        return {"notebook_id": notebooks[0].id, "reasoning": reasoning}

    return {"notebook_id": None, "reasoning": "No suitable notebook found"}


# --- Conditional Edge ---

def route_after_classify(state: RouterState) -> str:
    """Route to select_notebook or END based on classification."""
    if state.get("request_type") == "notebooklm":
        return "select_notebook"
    return END


# --- Graph Builder ---

def build_routing_graph(chat_model, notebook_cache, routing_settings=None):
    """Build and compile the LangGraph routing state graph.

    Args:
        chat_model: LangChain ChatModel for LLM calls
        notebook_cache: NotebookCache with cached notebook summaries
        routing_settings: SmartRoutingSettings for notebook display config

    Returns:
        Compiled LangGraph that accepts {"query": str, "allowed_notebooks": ...}
        and returns RouterState with request_type, notebook_id, reasoning.
    """
    # Bind dependencies to node functions via closures
    async def _classify(state):
        return await classify_node(
            state, chat_model=chat_model, notebook_cache=notebook_cache
        )

    async def _select(state):
        return await select_notebook_node(
            state, chat_model=chat_model, notebook_cache=notebook_cache,
            routing_settings=routing_settings,
        )

    graph = StateGraph(RouterState)
    graph.add_node("classify", _classify)
    graph.add_node("select_notebook", _select)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_edge("select_notebook", END)

    return graph.compile()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_routing_graph.py -v`
Expected: ALL PASS (13 tests)

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/routing_graph.py tests/core/test_routing_graph.py
git commit -m "feat: add LangGraph routing graph with classify + select_notebook nodes"
```

---

### Task 4.2: Rewrite old router tests to use new routing graph

**Files:**
- Rewrite: `tests/test_openai_module/test_router.py`
- Rewrite: `tests/test_openai_module/test_router_acl.py`

> [!IMPORTANT]
> The old `SmartRouter` class still exists at this point (it will be removed in Stage 5).
> These test files are rewritten to test the **new** routing graph while the old router
> tests are preserved as `test_router_legacy.py` temporarily until Stage 5 removes `SmartRouter`.

**Step 1: Rename old tests as legacy**

```bash
cp tests/test_openai_module/test_router.py tests/test_openai_module/test_router_legacy.py
cp tests/test_openai_module/test_router_acl.py tests/test_openai_module/test_router_acl_legacy.py
```

**Step 2: Rewrite `test_router.py`**

```python
"""Tests for smart routing via LangGraph routing graph."""

import pytest
from unittest.mock import AsyncMock, MagicMock


def _mock_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


@pytest.mark.asyncio
async def test_router_classify_notebooklm():
    """Test classification of NotebookLM queries."""
    from nlm_proxy.core.routing_graph import build_routing_graph
    from nlm_proxy.core.notebook_cache import NotebookCache

    mock_nlm = MagicMock()
    mock_nlm.list_notebooks = AsyncMock(return_value=[])
    cache = NotebookCache(nlm_client=mock_nlm, ttl_seconds=3600)
    cache.set("nb-123", "Research Notes", "AI research", ["AI"])

    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(side_effect=[
        _mock_response("notebooklm"),
        _mock_response("nb-123"),
    ])

    graph = build_routing_graph(mock_model, cache)
    result = await graph.ainvoke({"query": "What does my research say?", "allowed_notebooks": None})

    assert result["request_type"] == "notebooklm"
    assert result["notebook_id"] == "nb-123"
    assert "Research Notes" in result["reasoning"]


@pytest.mark.asyncio
async def test_router_classify_llm_task():
    """Test classification of LLM tasks."""
    from nlm_proxy.core.routing_graph import build_routing_graph
    from nlm_proxy.core.notebook_cache import NotebookCache

    mock_nlm = MagicMock()
    mock_nlm.list_notebooks = AsyncMock(return_value=[])
    cache = NotebookCache(nlm_client=mock_nlm, ttl_seconds=3600)

    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value=_mock_response("llm_task"))

    graph = build_routing_graph(mock_model, cache)
    result = await graph.ainvoke({"query": "Write a poem", "allowed_notebooks": None})

    assert result["request_type"] == "llm_task"
    assert result["notebook_id"] is None
```

**Step 3: Rewrite `test_router_acl.py`**

```python
"""Tests for per-request ACL filtering in LangGraph routing graph."""

import pytest
from unittest.mock import AsyncMock, MagicMock


def _mock_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


@pytest.fixture
def acl_setup():
    """Common setup for ACL tests."""
    from nlm_proxy.core.notebook_cache import NotebookCache
    mock_nlm = MagicMock()
    mock_nlm.list_notebooks = AsyncMock(return_value=[])
    cache = NotebookCache(nlm_client=mock_nlm, ttl_seconds=3600)
    cache.set("nb-1", "Notebook 1", "Summary 1", ["AI"])
    cache.set("nb-2", "Notebook 2", "Summary 2", ["ML"])
    cache.set("nb-3", "Notebook 3", "Summary 3", ["Data"])
    return cache


@pytest.mark.asyncio
async def test_select_notebook_no_acl_filter(acl_setup):
    """All notebooks considered when no ACL filter."""
    from nlm_proxy.core.routing_graph import build_routing_graph
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(side_effect=[
        _mock_response("notebooklm"), _mock_response("nb-2"),
    ])
    graph = build_routing_graph(mock_model, acl_setup)
    result = await graph.ainvoke({"query": "test", "allowed_notebooks": None})
    assert result["notebook_id"] == "nb-2"
    # Verify all 3 in prompt (second call = select)
    select_call = mock_model.ainvoke.call_args_list[1][0][0]
    prompt = select_call[0].content
    assert "nb-1" in prompt and "nb-2" in prompt and "nb-3" in prompt


@pytest.mark.asyncio
async def test_select_notebook_with_acl_filter(acl_setup):
    """Only allowed notebooks in LLM prompt."""
    from nlm_proxy.core.routing_graph import build_routing_graph
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(side_effect=[
        _mock_response("notebooklm"), _mock_response("nb-2"),
    ])
    graph = build_routing_graph(mock_model, acl_setup)
    result = await graph.ainvoke({"query": "test", "allowed_notebooks": ["nb-2", "nb-3"]})
    assert result["notebook_id"] == "nb-2"
    select_call = mock_model.ainvoke.call_args_list[1][0][0]
    prompt = select_call[0].content
    assert "nb-1" not in prompt
    assert "nb-2" in prompt


@pytest.mark.asyncio
async def test_select_notebook_acl_filters_all(acl_setup):
    """ACL matches no notebooks → error."""
    from nlm_proxy.core.routing_graph import build_routing_graph
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(side_effect=[
        _mock_response("notebooklm"),
    ])
    graph = build_routing_graph(mock_model, acl_setup)
    result = await graph.ainvoke({"query": "test", "allowed_notebooks": ["nb-999"]})
    assert result["notebook_id"] is None
    assert "No accessible notebooks" in result["reasoning"]


@pytest.mark.asyncio
async def test_select_notebook_empty_acl_list(acl_setup):
    """Empty ACL list → error."""
    from nlm_proxy.core.routing_graph import build_routing_graph
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(side_effect=[
        _mock_response("notebooklm"),
    ])
    graph = build_routing_graph(mock_model, acl_setup)
    result = await graph.ainvoke({"query": "test", "allowed_notebooks": []})
    assert result["notebook_id"] is None
    assert "No accessible notebooks" in result["reasoning"]


@pytest.mark.asyncio
async def test_route_llm_task_ignores_acl(acl_setup):
    """LLM_TASK bypasses ACL — select_notebook never called."""
    from nlm_proxy.core.routing_graph import build_routing_graph
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value=_mock_response("llm_task"))
    graph = build_routing_graph(mock_model, acl_setup)
    result = await graph.ainvoke({"query": "Summarize this", "allowed_notebooks": []})
    assert result["request_type"] == "llm_task"
    assert result["notebook_id"] is None
    assert mock_model.ainvoke.call_count == 1  # Only classify, no select
```

**Step 4: Run all router tests**

Run: `uv run pytest tests/test_openai_module/test_router.py tests/test_openai_module/test_router_acl.py tests/core/test_routing_graph.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_openai_module/
git commit -m "test: rewrite router + ACL tests for LangGraph routing graph"
```

---

### Task 4.3: Create AgentCore

**Files:**
- Create: `src/nlm_proxy/core/agent.py`
- Test: `tests/core/test_agent.py`

**Step 1: Write failing tests**

Create `tests/core/test_agent.py`:

```python
"""Tests for AgentCore — shared agent logic for OpenAI proxy and MCP."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


@pytest.fixture
def mock_components():
    """Create mock components for AgentCore."""
    mock_nlm = AsyncMock()
    mock_notebook_cache = MagicMock()
    mock_response_cache = MagicMock()
    mock_chat_model = AsyncMock()
    return mock_nlm, mock_notebook_cache, mock_response_cache, mock_chat_model


@pytest.mark.asyncio
async def test_agent_route_cache_hit(mock_components):
    """Pre-routing cache hit → return cached RoutingDecision."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components
    cached = MagicMock(notebook_id="nb-1")
    resp_cache.lookup_global.return_value = (cached, "exact")

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    options = RequestOptions()
    decision = await agent.route("test query", options)

    assert decision.cache_result is cached
    assert decision.cache_hit_type == "pre_routing_exact"


@pytest.mark.asyncio
async def test_agent_route_cache_miss_goes_to_graph(mock_components):
    """Cache miss → invoke LangGraph routing graph."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components
    resp_cache.lookup_global.return_value = (None, None)

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "request_type": "notebooklm",
            "notebook_id": "nb-1",
            "reasoning": "Selected notebook",
        })
        mock_build.return_value = mock_graph

        agent = AgentCore(
            nlm_client=nlm, notebook_cache=nb_cache,
            response_cache=resp_cache, chat_model=chat_model,
        )
        options = RequestOptions()
        decision = await agent.route("What is AI?", options)

    assert decision.request_type == "notebooklm"
    assert decision.notebook_id == "nb-1"
    assert decision.cache_result is None


@pytest.mark.asyncio
async def test_agent_route_bypass_cache(mock_components):
    """bypass_cache=True → skip pre-routing cache check."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    nlm, nb_cache, resp_cache, chat_model = mock_components

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "request_type": "llm_task",
            "notebook_id": None,
            "reasoning": "LLM task",
        })
        mock_build.return_value = mock_graph

        agent = AgentCore(
            nlm_client=nlm, notebook_cache=nb_cache,
            response_cache=resp_cache, chat_model=chat_model,
        )
        options = RequestOptions(bypass_cache=True)
        decision = await agent.route("test", options)

    resp_cache.lookup_global.assert_not_called()
    assert decision.request_type == "llm_task"


@pytest.mark.asyncio
async def test_agent_query_delegates_to_nlm(mock_components):
    """query() delegates to nlm_client.query()."""
    from nlm_proxy.core.agent import AgentCore

    nlm, nb_cache, resp_cache, chat_model = mock_components
    nlm.query = AsyncMock(return_value={"answer": "42"})

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    result = await agent.query("nb-1", "What is the answer?")

    assert result["answer"] == "42"
    nlm.query.assert_called_once()


@pytest.mark.asyncio
async def test_agent_query_stream_yields_chunks(mock_components):
    """query_stream() yields NLM streaming chunks."""
    from nlm_proxy.core.agent import AgentCore

    nlm, nb_cache, resp_cache, chat_model = mock_components

    async def mock_stream(*args, **kwargs):
        yield {"type": "answer", "text": "Hello"}
        yield {"type": "answer", "text": "Hello World"}

    nlm.query_stream = mock_stream

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    chunks = []
    async for chunk in agent.query_stream("nb-1", "test"):
        chunks.append(chunk)

    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_agent_wires_cache_invalidation(mock_components):
    """AgentCore wires notebook_cache.on_sources_changed → response_cache.invalidate_notebook."""
    from nlm_proxy.core.agent import AgentCore

    nlm, nb_cache, resp_cache, chat_model = mock_components

    agent = AgentCore(
        nlm_client=nlm, notebook_cache=nb_cache,
        response_cache=resp_cache, chat_model=chat_model,
    )
    assert nb_cache._on_sources_changed == resp_cache.invalidate_notebook
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement AgentCore**

Create `src/nlm_proxy/core/agent.py`:

```python
"""Shared agent core for both OpenAI proxy and MCP server.

Provides routing (via LangGraph), caching, and NLM query delegation.
Transport-specific concerns (SSE streaming, MCP progress) are handled
by the callers, NOT by AgentCore.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.routing_graph import build_routing_graph

logger = get_logger(__name__)


@dataclass
class RequestOptions:
    """Per-request options extracted from HTTP headers / MCP params."""
    bypass_cache: bool = False
    include_thinking: bool = True
    allowed_notebooks: list[str] | None = None
    conversation_id: str | None = None
    chat_id: str | None = None
    source_ids: list[str] | None = None
    timeout: float | None = None


@dataclass
class RoutingDecision:
    """Result of routing: where to send the query."""
    request_type: str                       # "notebooklm" | "llm_task"
    notebook_id: str | None = None
    reasoning: str = ""
    cache_result: object | None = None      # CachedResponse on cache hit
    cache_hit_type: str | None = None       # "pre_routing_exact" etc.
    conversation_id: str | None = None


class AgentCore:
    """Shared agent logic for both OpenAI proxy and MCP server."""

    def __init__(self, nlm_client, notebook_cache, response_cache, chat_model,
                 session_store=None, routing_settings=None):
        self.nlm_client = nlm_client
        self.notebook_cache = notebook_cache
        self.response_cache = response_cache
        self.chat_model = chat_model
        self.session_store = session_store  # For conversation_id persistence
        self.routing_graph = build_routing_graph(
            chat_model, notebook_cache, routing_settings=routing_settings
        )

        # Wire bidirectional dependencies
        if notebook_cache and response_cache:
            notebook_cache._on_sources_changed = response_cache.invalidate_notebook
        if nlm_client and notebook_cache:
            nlm_client._notebook_cache = notebook_cache

    async def route(self, query: str, options: RequestOptions) -> RoutingDecision:
        """Get routing decision with optional pre-routing cache check."""
        # Phase 0: Pre-routing global L1 cache check
        if not options.bypass_cache and self.response_cache:
            cached, hit_type = self.response_cache.lookup_global(query)
            if cached:
                # ACL check on cached result
                if options.allowed_notebooks is None or cached.notebook_id in options.allowed_notebooks:
                    return RoutingDecision(
                        request_type="notebooklm",
                        notebook_id=cached.notebook_id,
                        reasoning="Pre-routing cache hit",
                        cache_result=cached,
                        cache_hit_type=f"pre_routing_{hit_type}",
                        conversation_id=options.conversation_id,
                    )

        # Phase 1: LangGraph routing (with thread_id for checkpointing)
        config = {}
        if options.chat_id:
            config = {"configurable": {"thread_id": options.chat_id}}
        state = await self.routing_graph.ainvoke(
            {
                "query": query,
                "allowed_notebooks": options.allowed_notebooks,
            },
            config=config,
        )
        return RoutingDecision(
            request_type=state["request_type"],
            notebook_id=state.get("notebook_id"),
            reasoning=state.get("reasoning", ""),
            conversation_id=options.conversation_id,
        )

    async def query(self, notebook_id, query, conversation_id=None,
                    source_ids=None, timeout=None) -> dict:
        """Non-streaming query from NotebookLM."""
        return await self.nlm_client.query(
            notebook_id, query_text=query,
            conversation_id=conversation_id,
            source_ids=source_ids,
            timeout=timeout,
        )

    async def query_stream(self, notebook_id, query, conversation_id=None,
                           source_ids=None, **kwargs):
        """Streaming query from NotebookLM. Yields raw NLM chunks."""
        async for chunk in self.nlm_client.query_stream(
            notebook_id, query_text=query,
            conversation_id=conversation_id,
            source_ids=source_ids,
            **kwargs
        ):
            yield chunk

    async def handle_direct_query(self, notebook_id, query, options):
        """Handle direct notebook query (model == notebook_id, bypasses routing).

        Returns (cache_result, hit_type) on cache hit, or (None, None) on miss.
        Caller handles the actual NLM query and format-specific response.
        """
        if not options.bypass_cache and self.response_cache:
            cache_result, hit_type = await self.response_cache.lookup_async(
                notebook_id, query
            )
            if cache_result:
                return cache_result, hit_type
        return None, None
```

**Step 4: Run tests**

Run: `uv run pytest tests/core/test_agent.py -v`
Expected: ALL PASS (7 tests)

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/agent.py tests/core/test_agent.py
git commit -m "feat: add AgentCore with routing, query, and cache integration"
```

---

### 🔒 Stage 4 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS — new routing graph + agent core tested, old router tests preserved as legacy

---

## Stage 5: Rewire OpenAI Proxy Server

> [!CAUTION]
> This is the **highest-risk stage** (🔴 High, 3-4 days). Test every change incrementally.

### Task 5.1: Update `main()` — singleton AgentCore

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Replace per-request SmartRouter with singleton AgentCore**

Replace the `main()` initialization section. Key changes:
- Remove per-request `SmartRouter` creation pattern
- Create `AgentCore` singleton at startup, store in `app.state.agent_core`
- Use `LangChainLLMClient` instead of `ExternalLLMClient`

```python
# In main(), replace the llm_client and notebook_cache initialization:

from nlm_proxy.core.llm_client import LangChainLLMClient, create_chat_model
from nlm_proxy.core.agent import AgentCore
from nlm_proxy.core.config import get_agent_settings

# Create shared ChatModel (used for routing, L3 verification, and LLM_TASK)
routing_settings = get_routing_settings()
agent_settings = get_agent_settings()
chat_model = create_chat_model(
    model=routing_settings.llm_model,
    provider=agent_settings.llm_provider,
    base_url=routing_settings.llm_base_url,
    api_key=routing_settings.llm_api_key,
)
llm_client = LangChainLLMClient(chat_model=chat_model)

# ... (response cache init same as before, but use llm_client) ...

# Create AgentCore singleton (replaces per-request SmartRouter)
if routing_settings.llm_api_key:
    tokens = load_cached_tokens()
    if tokens and tokens.cookies:
        nlm_client = NotebookLMClient(
            cookies=tokens.cookies,
            csrf_token=tokens.csrf_token or "",
            session_id=tokens.session_id or "",
            notebook_cache=None,
        )
        on_sources_changed = None
        if app.state.response_cache:
            on_sources_changed = app.state.response_cache.invalidate_notebook

        notebook_cache = NotebookCache(
            nlm_client=nlm_client,
            ttl_seconds=routing_settings.summary_cache_ttl,
            allowed_notebooks=routing_settings.allowed_notebooks,
            on_sources_changed=on_sources_changed,
        )
        app.state.notebook_cache = notebook_cache

        # AgentCore handles bidirectional wiring internally
        app.state.agent_core = AgentCore(
            nlm_client=nlm_client,
            notebook_cache=notebook_cache,
            response_cache=app.state.response_cache,
            chat_model=chat_model,
            session_store=app.state.session_store,
            routing_settings=routing_settings,
        )
        logger.info("AgentCore initialized (singleton)")
```

**Step 2: Remove old imports**

Remove:
```python
# REMOVE:
from nlm_proxy.openai.router import SmartRouter, RequestType
```

Add:
```python
# ADD:
from nlm_proxy.core.agent import AgentCore, RequestOptions, RoutingDecision
```

---

### Task 5.2: Rewrite `handle_smart_routing()` — four-phase pipeline

**Step 1: Replace `handle_smart_routing()` with AgentCore-based pipeline**

Key structural change: no per-request `SmartRouter()` creation, no `router.close()`. Instead use `app.state.agent_core.route()`.

```python
async def handle_smart_routing(request: ChatCompletionRequest, http_request: Request):
    """Handle requests to the smart router model — four-phase pipeline."""
    routing_settings = get_routing_settings()
    tracing_settings = get_tracing_settings()
    agent_core: AgentCore = app.state.agent_core

    if not agent_core:
        raise HTTPException(status_code=503, detail="Agent core not initialized.")

    # Extract chat_id from headers or request metadata
    chat_id = http_request.headers.get("X-OpenWebUI-Chat-Id")
    if not chat_id and hasattr(request, 'metadata') and request.metadata:
        chat_id = request.metadata.get("chat_id")

    # Extract allowed_notebooks from request metadata (per-request ACL)
    request_allowed_notebooks = None
    if hasattr(request, 'metadata') and request.metadata:
        raw = request.metadata.get("allowed_notebooks")
        if raw is not None:
            if raw == ["*"]:
                request_allowed_notebooks = None
            else:
                request_allowed_notebooks = raw

    # Load conversation_id from session store
    conversation_id = None
    if chat_id and app.state.session_store:
        conversation_id = app.state.session_store.get(chat_id)
        if conversation_id:
            request.conversation_id = conversation_id

    # Build RequestOptions (used by all phases)
    options = RequestOptions(
        bypass_cache=request.bypass_cache,
        include_thinking=request.include_thinking,
        allowed_notebooks=request_allowed_notebooks,
        conversation_id=conversation_id,
        chat_id=chat_id,
    )

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    query = user_messages[-1].content

    # Phase 0+1: Route (includes pre-routing cache check)
    decision = await agent_core.route(query, options)

    # Phase 0 hit → return cached response
    if decision.cache_result:
        if request.stream:
            return StreamingResponse(
                _stream_cached(decision, request),
                media_type="text/event-stream",
                headers={"X-Cache-Status": f"HIT_{decision.cache_hit_type.upper()}"},
            )
        else:
            return _json_cached(decision, request)

    # Phase 2: Post-routing cache check (notebook-scoped, NOTEBOOKLM only)
    if decision.request_type == "notebooklm" and not options.bypass_cache and agent_core.response_cache:
        cache_result, hit_type = await agent_core.response_cache.lookup_async(
            decision.notebook_id, query
        )
        if cache_result:
            decision.cache_result = cache_result
            decision.cache_hit_type = hit_type
            if request.stream:
                return StreamingResponse(
                    _stream_cached(decision, request),
                    media_type="text/event-stream",
                    headers={"X-Cache-Status": f"HIT_{hit_type.upper()}"},
                )
            else:
                return _json_cached(decision, request)

    # Phase 3: Execute query (streaming or non-streaming)
    if request.stream:
        return StreamingResponse(
            stream_smart_response(agent_core, decision, query, request, chat_id, tracing_settings),
            media_type="text/event-stream",
        )
    else:
        return await _handle_non_streaming(agent_core, decision, query, request, chat_id, tracing_settings)
```

**Step 2: Update `stream_smart_response()` — LLM_TASK uses `agent_core.chat_model`**

Key change: LLM_TASK path now uses `agent_core.chat_model.astream()` → yields `AIMessageChunk` instead of `ChatCompletionChunk`.

```python
async def stream_smart_response(agent_core: AgentCore, decision, query, request, chat_id, tracing_settings):
    """Stream response — Phase 3a."""
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        conversation_id = None
        accumulated_response = ""

        # Reasoning chunk
        reasoning_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(reasoning_content=decision.reasoning + "\n\n"))],
        )
        yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

        if decision.request_type == "llm_task":
            # LLM_TASK: stream via LangChain ChatModel.astream()
            # NOTE: yields AIMessageChunk (chunk.content), NOT OpenAI ChatCompletionChunk
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            async for chunk in agent_core.chat_model.astream(messages):
                delta_content = chunk.content if chunk.content else ""
                if delta_content:
                    accumulated_response += delta_content
                    openai_chunk = ChatCompletionChunk(
                        id=chunk_id, created=created, model=request.model,
                        choices=[Choice(delta=DeltaContent(content=delta_content))],
                    )
                    yield f"data: {openai_chunk.model_dump_json()}\n\n"
        else:
            # NOTEBOOKLM: stream via query_stream (same direct pipe as before)
            previous_thinking = ""
            previous_answer = ""

            async for chunk in agent_core.query_stream(
                decision.notebook_id, query,
                conversation_id=request.conversation_id,
            ):
                # ... (same delta conversion logic as current stream_smart_response)
                # Extract conversation_id, save to session store, etc.
                # This section is preserved EXACTLY from current code.
                pass

            # Store in response cache after stream completes
            if agent_core.response_cache and accumulated_response and conversation_id and decision.notebook_id:
                embedding = None
                if agent_core.response_cache._semantic_enabled:
                    emb = agent_core.response_cache._compute_embedding(query)
                    if emb is not None:
                        embedding = emb.tolist()
                agent_core.response_cache.store(
                    notebook_id=decision.notebook_id,
                    query=query, answer=accumulated_response,
                    thinking=previous_thinking or None,
                    conversation_id=conversation_id,
                    embedding=embedding,
                )

        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
```

**Step 3: Add `_handle_non_streaming()` helper**

```python
async def _handle_non_streaming(agent_core, decision, query, request, chat_id, tracing_settings):
    """Phase 3b: Non-streaming response (both NOTEBOOKLM and LLM_TASK)."""
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        if decision.request_type == "llm_task":
            # LLM_TASK: invoke via LangChain ChatModel
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            result = await agent_core.chat_model.ainvoke(messages)
            response_text = result.content
        else:
            # NOTEBOOKLM: query directly
            client = await get_client()
            try:
                result = await client.query(
                    notebook_id=decision.notebook_id,
                    query_text=query,
                    conversation_id=request.conversation_id,
                )
                response_text = result.get("answer", "") if result else ""
                conv_id = result.get("conversation_id", "") if result else ""

                # Save session + cache store (same logic as current non-streaming path)
                if chat_id and conv_id and app.state.session_store:
                    app.state.session_store.set(chat_id, conv_id)

                if agent_core.response_cache and response_text and conv_id:
                    embedding = None
                    if agent_core.response_cache._semantic_enabled:
                        emb = agent_core.response_cache._compute_embedding(query)
                        if emb is not None:
                            embedding = emb.tolist()
                    agent_core.response_cache.store(
                        notebook_id=decision.notebook_id, query=query,
                        answer=response_text, thinking=None,
                        conversation_id=conv_id, embedding=embedding,
                    )
            finally:
                await client.close()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ResponseChoice(
                index=0,
                message=ResponseMessage(role="assistant", content=response_text, reasoning_content=decision.reasoning),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=len(query), completion_tokens=len(response_text), total_tokens=len(query) + len(response_text)),
        )
```

**Step 4: Update `chat_completions()` — direct notebook path uses `agent_core.handle_direct_query()`**

```python
# In chat_completions(), for model != router_model_name:
# Replace direct cache lookup with:
if agent_core:
    cache_result, hit_type = await agent_core.handle_direct_query(
        request.model, query_text, options
    )
    if cache_result:
        # Return cached response (stream or non-stream) — same as current
        ...
```

**Step 5: Remove `SmartRouter` import and per-request creation**

Delete all references to:
- `from nlm_proxy.openai.router import SmartRouter, RequestType`
- `router = SmartRouter(...)` in `handle_smart_routing()`
- `await router.close()` — no longer needed (singleton)

**Step 6: Run tests**

Run: `uv run pytest -v`
Expected: Need to update `test_server.py` and `test_conversation_flow.py` mocks

---

### Task 5.3: Update server tests

**Files:**
- Modify: `tests/test_openai_module/test_server.py`
- Modify: `tests/test_openai_module/test_conversation_flow.py`

**Key mock changes:**
- Replace `SmartRouter` mocks with `AgentCore` mocks
- Replace `router.route()` → `agent_core.route()`
- Replace `ExternalLLMClient` mocks → `LangChainLLMClient` mocks
- Update `app.state.agent_core` instead of per-request router creation

For `test_conversation_flow.py`:
- `SessionStore` is **KEPT** in this refactor (see "Deferred" section below)
- Tests should verify `session_store.set()` is called with `conversation_id` from NLM responses
- Update mocks to patch `app.state.agent_core.query_stream()` instead of `client.query_stream()`

---

### Task 5.4: Add non-streaming tests

**Files:**
- Create: `tests/test_openai_module/test_non_streaming.py`

```python
"""Tests for non-streaming (stream=false) response path."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_non_streaming_notebooklm():
    """Non-streaming NOTEBOOKLM query returns ChatCompletionResponse JSON."""
    # Mock agent_core.route() → RoutingDecision(request_type="notebooklm")
    # Mock client.query() → {"answer": "test", "conversation_id": "conv-1"}
    # Assert response has correct structure and reasoning_content
    pass


@pytest.mark.asyncio
async def test_non_streaming_llm_task():
    """Non-streaming LLM_TASK returns ChatCompletionResponse from ainvoke."""
    # Mock agent_core.route() → RoutingDecision(request_type="llm_task")
    # Mock agent_core.chat_model.ainvoke() → AIMessage(content="response")
    # Assert response is ChatCompletionResponse JSON
    pass


@pytest.mark.asyncio
async def test_non_streaming_cache_hit():
    """Non-streaming cache hit returns cached ChatCompletionResponse."""
    # Mock agent_core.route() with cache_result pre-filled
    # Assert X-Cache-Status header present
    pass
```

**Step 1: Run tests**

Run: `uv run pytest tests/test_openai_module/test_non_streaming.py -v`

**Step 2: Commit**

```bash
git commit -m "feat: rewire OpenAI proxy to use AgentCore + four-phase pipeline"
```

---

### 🔒 Stage 5 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS

---

## Stage 6: MCP Server Unification

### Task 6.1: Add `_agent_core` singleton to MCP server

**Files:**
- Modify: `src/nlm_proxy/mcp/server.py`

**Step 1: Add agent core initialization**

Add alongside existing `_client` singleton:

```python
from nlm_proxy.core.agent import AgentCore, RequestOptions

_agent_core: AgentCore | None = None

async def get_agent_core() -> AgentCore:
    """Get or create the shared AgentCore singleton for MCP query tools."""
    global _agent_core
    if _agent_core is None:
        client = await get_client()

        # Import config lazily
        from nlm_proxy.core.config import get_routing_settings, get_cache_settings, get_agent_settings
        from nlm_proxy.core.llm_client import LangChainLLMClient, create_chat_model

        routing_settings = get_routing_settings()
        agent_settings = get_agent_settings()

        # Only create agent if LLM is configured (same guard as OpenAI proxy)
        if routing_settings.llm_api_key:
            chat_model = create_chat_model(
                model=routing_settings.llm_model,
                provider=agent_settings.llm_provider,
                base_url=routing_settings.llm_base_url,
                api_key=routing_settings.llm_api_key,
            )

            # Response cache (optional)
            cache_settings = get_cache_settings()
            response_cache = None
            if cache_settings.response_cache_enabled:
                from nlm_proxy.core.response_cache import ResponseCache
                llm_client = LangChainLLMClient(chat_model=chat_model)
                response_cache = ResponseCache(
                    max_entries=cache_settings.response_cache_max_entries,
                    ttl_seconds=cache_settings.response_cache_ttl,
                    semantic_enabled=cache_settings.semantic_match_enabled,
                    llm_client=llm_client,
                    embedding_model=cache_settings.embedding_model,
                    similarity_threshold=cache_settings.similarity_threshold,
                )

            # NotebookCache (optional)
            from nlm_proxy.core.notebook_cache import NotebookCache
            notebook_cache = NotebookCache(
                nlm_client=client,
                ttl_seconds=routing_settings.summary_cache_ttl,
                allowed_notebooks=routing_settings.allowed_notebooks,
                on_sources_changed=response_cache.invalidate_notebook if response_cache else None,
            )

            _agent_core = AgentCore(
                nlm_client=client,
                notebook_cache=notebook_cache,
                response_cache=response_cache,
                chat_model=chat_model,
                routing_settings=routing_settings,
            )
        else:
            # No LLM configured — AgentCore without routing (queries pass through directly)
            _agent_core = AgentCore(
                nlm_client=client,
                notebook_cache=None,
                response_cache=None,
                chat_model=None,
            )

    return _agent_core
```

---

### Task 6.2: Update `notebook_query` tool

```python
@logged_tool()
async def notebook_query(
    notebook_id: str,
    query: str,
    source_ids: list[str] | str | None = None,
    conversation_id: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Ask AI about EXISTING sources already in notebook."""
    try:
        # Handle JSON string source_ids (same as current)
        if isinstance(source_ids, str):
            import json
            try:
                source_ids = json.loads(source_ids)
            except json.JSONDecodeError:
                source_ids = [source_ids]

        effective_timeout = timeout if timeout is not None else _query_timeout

        agent = await get_agent_core()
        # Use agent.query() which delegates to nlm_client.query()
        result = await agent.query(
            notebook_id, query,
            conversation_id=conversation_id,
            source_ids=source_ids,
            timeout=effective_timeout,
        )

        if result:
            return {
                "status": "success",
                "answer": result.get("answer", ""),
                "conversation_id": result.get("conversation_id"),
            }
        return {"status": "error", "error": "Failed to query notebook"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

---

### Task 6.3: Update `notebook_query_stream` tool

```python
@logged_tool()
async def notebook_query_stream(
    notebook_id: str,
    query: str,
    source_ids: list[str] | str | None = None,
    conversation_id: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Ask AI with real-time streaming."""
    try:
        if isinstance(source_ids, str):
            import json as json_module
            try:
                source_ids = json_module.loads(source_ids)
            except json_module.JSONDecodeError:
                source_ids = [source_ids]

        agent = await get_agent_core()

        thinking_steps: list[str] = []
        answer_chunks: list[str] = []
        final_conversation_id: str | None = None
        chunk_count = 0

        # Use agent.query_stream() which delegates to nlm_client.query_stream()
        async for chunk in agent.query_stream(
            notebook_id, query,
            source_ids=source_ids,
            conversation_id=conversation_id,
        ):
            chunk_count += 1
            final_conversation_id = chunk.get("conversation_id")

            if chunk["type"] == "thinking":
                thinking_steps.append(chunk["text"])
                # MCP progress reporting — transport-layer concern, PRESERVED
                if ctx:
                    preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                    await ctx.report_progress(
                        progress=chunk_count,
                        total=chunk_count + 5,
                        message=f"Thinking: {preview}",
                    )
            else:
                answer_chunks.append(chunk["text"])
                if ctx:
                    await ctx.report_progress(
                        progress=chunk_count,
                        total=chunk_count + 1,
                        message=f"Answer:{chunk['text']}",
                    )

        final_answer = max(answer_chunks, key=len) if answer_chunks else ""

        return {
            "status": "success",
            "answer": final_answer,
            "conversation_id": final_conversation_id,
            "thinking_steps": thinking_steps,
            "chunk_count": chunk_count,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

> [!NOTE]
> Only `notebook_query` and `notebook_query_stream` use `AgentCore`. All other MCP tools (`notebook_list`, `notebook_create`, `source_describe`, etc.) continue using `get_client()` directly — they are CRUD operations, not knowledge queries.

**Step 1: Manual verification**

Test with MCP client:
```
# notebook_query should work identically
# notebook_query_stream should stream with progress reporting
```

**Step 2: Commit**

```bash
git commit -m "feat: MCP server uses shared AgentCore for query tools"
```

---

### 🔒 Stage 6 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS (MCP tools not auto-tested, verify manually)

---

## Stage 7: Documentation & Cleanup

### Task 7.1: Update documentation

**Files:**
- Modify: `README.md` — update architecture description, dependency list
- Modify: `GEMINI.md` — update Architecture section, add AgentCore references
- Modify: `.env.example` — add `NLM_PROXY_AGENT_*` variables
- Modify: `docs/smart-routing-architecture.md` — update flow diagram with LangGraph
- Modify: `docs/TRACING.md` — document LangGraph span interaction with OTEL

### Task 7.2: Remove dead code

- Delete `ExternalLLMClient` class from `core/llm_client.py` (replaced by `LangChainLLMClient`)
- Remove `openai/notebook_cache.py` re-export (keep for 1 release, then remove)
- Remove all `fastembed` imports and references
- Delete `tests/test_openai_module/test_router_legacy.py` and `test_router_acl_legacy.py`
- Clean up `openai/router.py` — can be deleted entirely after Stage 5 proves stable

**Step 1: Commit**

```bash
git commit -m "docs: update README, GEMINI.md, TRACING.md for LangChain refactor"
git commit -m "chore: remove dead ExternalLLMClient, fastembed refs, legacy router tests"
```

---

## Deferred Items

> [!IMPORTANT]
> The following items from the design document are **explicitly deferred** from this implementation plan:

### SessionStore → LangGraph Memory (Design Section 5) — DEFERRED

**Rationale**: The current `SessionStore` is simple, works correctly, and has no bugs. Replacing it with LangGraph `MemorySaver`/`SqliteSaver` adds complexity without immediate benefit. The `SessionStore` is kept as-is in this refactor.

**Future**: When we need persistent conversation memory (cross-restart), full message history, or multi-turn agentic reasoning, we can add LangGraph checkpointing. The `AgentCore` already passes `thread_id` to `routing_graph.ainvoke()` to enable this later.

### Tool-Calling Agent (Design Section 6) — DEFERRED

**Rationale**: The tool-calling agent (wrapping `NotebookLMClient` methods as LangGraph tools for autonomous agent use) is a **new feature**, not a refactor. This implementation plan focuses on replacing existing components with LangChain equivalents while preserving all current behavior.

**Future**: When we want autonomous multi-step reasoning (e.g., agent decides to list notebooks, describe each, then query the best one), we can add tool definitions to the routing graph.

### Cross-Notebook Query (Design Section 7) — DEFERRED

Already marked as deferred in the design document.

---

## Verification Plan

### Automated Tests

After EACH stage:
```bash
uv run pytest -v
```

Full regression after all stages:
```bash
uv run pytest -v --tb=long
```

### Manual Verification

> [!IMPORTANT]
> Manual testing requires a configured `.env` with valid auth tokens and LLM API keys. Run after Stage 5.

1. **Smart routing streaming**: `curl` or Open WebUI — send query to `knowledge-finder` model, verify SSE stream works
2. **Smart routing non-streaming**: Send `stream=false` request, verify JSON response
3. **Direct notebook query**: Send query with `model=<notebook-id>`, verify bypass
4. **Cache hit**: Send same query twice, verify `X-Cache-Status` header on second
5. **LLM_TASK**: Send "write a poem" to `knowledge-finder`, verify LLM passthrough
6. **MCP query**: Use MCP client to call `notebook_query_stream`, verify progress reporting

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Stages 4-5 break streaming | Keep `query_stream()` direct pipe unchanged |
| LangChain version conflicts | Pin `langchain>=1.2,<2.0` — v1.x is stable since Oct 2025 |
| fastembed → langchain-huggingface breaks embedding perf | Run `test_embedding_models.py` before/after — same underlying `sentence-transformers` model |
| `_last_hit_type` fix breaks callers | Fix in Stage 2 before later stages touch server |
| Python 3.10+ requirement | LangChain/LangGraph v1.0+ requires Python ≥3.10 — verify `pyproject.toml` `requires-python` |
| `AIMessageChunk` format change | LLM_TASK streaming reads `chunk.content` instead of `chunk.choices[0].delta.content` |
| Singleton lifecycle | AgentCore created at startup, no per-request teardown — verify no resource leaks |

