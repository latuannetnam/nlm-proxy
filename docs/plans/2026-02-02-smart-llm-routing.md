# Smart LLM Request Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add intelligent request routing that uses an external LLM to classify requests and auto-select the best notebook or route to external LLM.

**Architecture:** Requests to "smart-router" model trigger LLM classification. "NotebookLM queries" get routed to auto-selected notebook based on cached summaries. "LLM tasks" (summarize, follow-up questions) route to external OpenAI-compatible endpoint. Routing decisions appear as `reasoning_content` in the stream.

**Tech Stack:** Python 3.11+, FastAPI, httpx, pydantic-settings

---

## Task 1: Add SmartRoutingSettings Configuration

**Files:**
- Modify: `src/nlm_proxy/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config.py`:

```python
def test_smart_routing_settings_defaults():
    """Test SmartRoutingSettings has correct defaults."""
    from nlm_proxy.core.config import SmartRoutingSettings

    settings = SmartRoutingSettings(llm_api_key="test-key")

    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.router_model_name == "smart-router"
    assert settings.allowed_notebooks == []
    assert settings.summary_cache_ttl == 3600


def test_smart_routing_settings_from_env(monkeypatch):
    """Test SmartRoutingSettings loads from environment."""
    monkeypatch.setenv("NLM_PROXY_ROUTING_LLM_BASE_URL", "https://custom.api/v1")
    monkeypatch.setenv("NLM_PROXY_ROUTING_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("NLM_PROXY_ROUTING_LLM_MODEL", "gpt-4o")

    from nlm_proxy.core.config import SmartRoutingSettings
    settings = SmartRoutingSettings()

    assert settings.llm_base_url == "https://custom.api/v1"
    assert settings.llm_api_key == "sk-test"
    assert settings.llm_model == "gpt-4o"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_config.py::test_smart_routing_settings_defaults -v`
Expected: FAIL with "cannot import name 'SmartRoutingSettings'"

**Step 3: Write minimal implementation**

Add to `src/nlm_proxy/core/config.py` after `AuthSettings`:

```python
class SmartRoutingSettings(BaseSettings):
    """Smart routing configuration for LLM-based request classification."""

    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for external OpenAI-compatible LLM"
    )
    llm_api_key: str = Field(
        default="",
        description="API key for external LLM"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for classification and routing"
    )
    router_model_name: str = Field(
        default="smart-router",
        description="Model name that triggers smart routing"
    )
    allowed_notebooks: list[str] = Field(
        default_factory=list,
        description="List of notebook IDs to include (empty = all)"
    )
    summary_cache_ttl: int = Field(
        default=3600,
        description="TTL for notebook summary cache in seconds"
    )

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_ROUTING_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


_routing: SmartRoutingSettings | None = None


def get_routing_settings() -> SmartRoutingSettings:
    """Get the smart routing settings instance."""
    global _routing
    if _routing is None:
        _routing = SmartRoutingSettings()
    return _routing
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_config.py::test_smart_routing_settings_defaults tests/core/test_config.py::test_smart_routing_settings_from_env -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/config.py tests/core/test_config.py
git commit -m "feat(config): add SmartRoutingSettings for LLM routing"
```

---

## Task 2: Create NotebookCache Module

**Files:**
- Create: `src/nlm_proxy/openai/notebook_cache.py`
- Test: `tests/openai/test_notebook_cache.py`

**Step 1: Write the failing test**

Create `tests/openai/test_notebook_cache.py`:

```python
"""Tests for notebook summary cache."""

import time
import pytest


def test_notebook_cache_set_and_get():
    """Test basic set and get operations."""
    from nlm_proxy.openai.notebook_cache import NotebookCache

    cache = NotebookCache(ttl_seconds=3600)
    cache.set(
        notebook_id="nb-123",
        title="Research Notes",
        summary="Notes about AI research",
        topics=["AI", "ML"]
    )

    info = cache.get("nb-123")
    assert info is not None
    assert info.id == "nb-123"
    assert info.title == "Research Notes"
    assert info.summary == "Notes about AI research"
    assert info.topics == ["AI", "ML"]


def test_notebook_cache_expiration():
    """Test that entries expire after TTL."""
    from nlm_proxy.openai.notebook_cache import NotebookCache

    cache = NotebookCache(ttl_seconds=0.1)  # 100ms TTL
    cache.set("nb-123", "Test", "Summary", [])

    assert cache.get("nb-123") is not None
    time.sleep(0.15)
    assert cache.get("nb-123") is None


def test_notebook_cache_get_all():
    """Test getting all non-expired entries."""
    from nlm_proxy.openai.notebook_cache import NotebookCache

    cache = NotebookCache(ttl_seconds=3600)
    cache.set("nb-1", "First", "Summary 1", ["topic1"])
    cache.set("nb-2", "Second", "Summary 2", ["topic2"])

    all_notebooks = cache.get_all()
    assert len(all_notebooks) == 2
    ids = {nb.id for nb in all_notebooks}
    assert ids == {"nb-1", "nb-2"}


def test_notebook_cache_clear():
    """Test clearing the cache."""
    from nlm_proxy.openai.notebook_cache import NotebookCache

    cache = NotebookCache(ttl_seconds=3600)
    cache.set("nb-123", "Test", "Summary", [])
    cache.clear()

    assert cache.get("nb-123") is None
    assert cache.get_all() == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/openai/test_notebook_cache.py -v`
Expected: FAIL with "No module named 'nlm_proxy.openai.notebook_cache'"

**Step 3: Write minimal implementation**

Create `src/nlm_proxy/openai/notebook_cache.py`:

```python
"""Notebook summary cache for smart routing."""

import threading
import time
from dataclasses import dataclass


@dataclass
class NotebookInfo:
    """Cached notebook information."""
    id: str
    title: str
    summary: str
    topics: list[str]
    cached_at: float


class NotebookCache:
    """Thread-safe cache for notebook summaries with TTL expiration."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, NotebookInfo] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, notebook_id: str) -> NotebookInfo | None:
        """Get cached notebook info if not expired."""
        with self._lock:
            info = self._cache.get(notebook_id)
            if info is None:
                return None
            if time.time() - info.cached_at > self._ttl:
                del self._cache[notebook_id]
                return None
            return info

    def set(self, notebook_id: str, title: str, summary: str, topics: list[str]) -> None:
        """Cache notebook info."""
        with self._lock:
            self._cache[notebook_id] = NotebookInfo(
                id=notebook_id,
                title=title,
                summary=summary,
                topics=topics,
                cached_at=time.time()
            )

    def get_all(self) -> list[NotebookInfo]:
        """Get all non-expired cached notebooks."""
        with self._lock:
            current_time = time.time()
            valid = []
            expired = []
            for nb_id, info in self._cache.items():
                if current_time - info.cached_at > self._ttl:
                    expired.append(nb_id)
                else:
                    valid.append(info)
            for nb_id in expired:
                del self._cache[nb_id]
            return valid

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/openai/test_notebook_cache.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/notebook_cache.py tests/openai/test_notebook_cache.py
git commit -m "feat(openai): add NotebookCache for summary caching"
```

---

## Task 3: Create ExternalLLMClient Module (in core for reuse)

**Files:**
- Create: `src/nlm_proxy/core/llm_client.py`
- Modify: `src/nlm_proxy/core/__init__.py`
- Modify: `pyproject.toml` (add openai dependency)
- Test: `tests/core/test_llm_client.py`

**Step 1: Add openai dependency**

Add to `pyproject.toml` dependencies:
```toml
"openai>=1.0.0",
```

Run: `uv sync`

**Step 2: Write the failing test**

Create `tests/core/test_llm_client.py`:

```python
"""Tests for external LLM client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_external_llm_client_complete():
    """Test non-streaming completion."""
    from nlm_proxy.core.llm_client import ExternalLLMClient

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="notebooklm"))]

    with patch("openai.AsyncOpenAI") as mock_client_class:
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

    with patch("openai.AsyncOpenAI") as mock_client_class:
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

    with patch("openai.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        client = ExternalLLMClient(
            base_url="https://api.test.com/v1",
            api_key="test-key",
            model="gpt-4o-mini"
        )
        # Access client to initialize it
        _ = client._client
        await client.close()

        mock_client.close.assert_called_once()
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/core/test_llm_client.py -v`
Expected: FAIL with "No module named 'nlm_proxy.core.llm_client'"

**Step 4: Write minimal implementation**

Create `src/nlm_proxy/core/llm_client.py`:

```python
"""External LLM client for OpenAI-compatible endpoints.

This module is in core/ for reuse across multiple features:
- Smart routing (openai proxy)
- Future MCP tools
- Any feature needing external LLM calls

Uses the official OpenAI SDK for better compatibility and maintainability.
"""

from typing import AsyncIterator

from openai import AsyncOpenAI

from nlm_proxy.core.logging import get_logger

logger = get_logger(__name__)


class ExternalLLMClient:
    """Client for calling external OpenAI-compatible LLM using OpenAI SDK."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client (lazy initialization)."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=30.0
            )
        return self._client

    async def complete(self, prompt: str, max_tokens: int = 50) -> str:
        """Get a simple completion (non-streaming)."""
        logger.debug(f"[LLM] Calling complete: model={self.model}, max_tokens={max_tokens}")
        logger.debug(f"[LLM] Request prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()

        logger.debug(f"[LLM] Response: {result[:200]}{'...' if len(result) > 200 else ''}")
        return result

    async def stream(self, messages: list[dict]) -> AsyncIterator:
        """Stream a completion for LLM task passthrough.

        Returns an async iterator that yields ChatCompletionChunk objects.
        Each chunk has: chunk.choices[0].delta.content
        """
        logger.debug(f"[LLM] Starting stream: model={self.model}, messages={len(messages)}")
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        logger.debug("[LLM] Stream started")
        return stream

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            logger.debug("[LLM] Closing client")
            await self._client.close()
            self._client = None
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/core/test_llm_client.py -v`
Expected: PASS (3 tests)

**Step 6: Export from core module**

Add to `src/nlm_proxy/core/__init__.py`:
```python
from nlm_proxy.core.llm_client import ExternalLLMClient

__all__ = [..., "ExternalLLMClient"]
```

**Step 7: Commit**

```bash
git add pyproject.toml src/nlm_proxy/core/llm_client.py src/nlm_proxy/core/__init__.py tests/core/test_llm_client.py
git commit -m "feat(core): add ExternalLLMClient using OpenAI SDK"
```

---

## Task 4: Create Prompt Templates

**Files:**
- Create: `src/nlm_proxy/openai/prompts/__init__.py`
- Create: `src/nlm_proxy/openai/prompts/classify_request.txt`
- Create: `src/nlm_proxy/openai/prompts/select_notebook.txt`
- Test: `tests/openai/test_prompts.py`

**Step 1: Write the failing test**

Create `tests/openai/test_prompts.py`:

```python
"""Tests for prompt template loading."""

from pathlib import Path


def test_prompt_templates_exist():
    """Test that prompt template files exist."""
    prompts_dir = Path(__file__).parent.parent.parent / "src" / "nlm_proxy" / "openai" / "prompts"

    assert (prompts_dir / "classify_request.txt").exists()
    assert (prompts_dir / "select_notebook.txt").exists()


def test_classify_request_prompt_content():
    """Test classify_request prompt has required placeholders."""
    prompts_dir = Path(__file__).parent.parent.parent / "src" / "nlm_proxy" / "openai" / "prompts"
    content = (prompts_dir / "classify_request.txt").read_text()

    assert "{query}" in content
    assert "notebooklm" in content.lower()
    assert "llm_task" in content.lower()


def test_select_notebook_prompt_content():
    """Test select_notebook prompt has required placeholders."""
    prompts_dir = Path(__file__).parent.parent.parent / "src" / "nlm_proxy" / "openai" / "prompts"
    content = (prompts_dir / "select_notebook.txt").read_text()

    assert "{query}" in content
    assert "{notebooks_json}" in content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/openai/test_prompts.py -v`
Expected: FAIL with "AssertionError" (files don't exist)

**Step 3: Write minimal implementation**

Create `src/nlm_proxy/openai/prompts/__init__.py`:

```python
"""Prompt templates for smart routing."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name."""
    prompt_path = PROMPTS_DIR / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")
```

Create `src/nlm_proxy/openai/prompts/classify_request.txt`:

```
You are a request classifier. Analyze the user's request and determine its type.

Request types:
1. "notebooklm" - Questions seeking information, facts, or knowledge that would be found in documents/notebooks. Examples: "What does the report say about X?", "Find information about Y", "What are the key points in my notes?"
2. "llm_task" - Meta-tasks that don't require document knowledge. Examples: summarizing conversation history, generating follow-up questions, rephrasing text, translation, creative writing, general chat.

User request:
{query}

Respond with ONLY one word: "notebooklm" or "llm_task"
```

Create `src/nlm_proxy/openai/prompts/select_notebook.txt`:

```
You are a notebook selector. Given the user's query and available notebooks with their summaries, select the most relevant notebook that can answer the query.

Available notebooks:
{notebooks_json}

User query:
{query}

Respond with ONLY the notebook_id (UUID) of the most relevant notebook. If none seem relevant, respond with the first notebook's ID.
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/openai/test_prompts.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/prompts/
git add tests/openai/test_prompts.py
git commit -m "feat(openai): add prompt templates for request classification"
```

---

## Task 5: Create SmartRouter Module

**Files:**
- Create: `src/nlm_proxy/openai/router.py`
- Test: `tests/openai/test_router.py`

**Step 1: Write the failing test**

Create `tests/openai/test_router.py`:

```python
"""Tests for smart router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_router_classify_notebooklm():
    """Test classification of NotebookLM queries."""
    from nlm_proxy.openai.router import SmartRouter, RequestType

    mock_nlm_client = MagicMock()

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="notebooklm")
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        result = await router.classify_request("What does my research notebook say about AI?")

        assert result == RequestType.NOTEBOOKLM


@pytest.mark.asyncio
async def test_router_classify_llm_task():
    """Test classification of LLM tasks."""
    from nlm_proxy.openai.router import SmartRouter, RequestType

    mock_nlm_client = MagicMock()

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="llm_task")
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        result = await router.classify_request("Summarize our conversation so far")

        assert result == RequestType.LLM_TASK


@pytest.mark.asyncio
async def test_router_route_decision():
    """Test full routing decision."""
    from nlm_proxy.openai.router import SmartRouter, RequestType

    mock_nlm_client = AsyncMock()
    mock_nlm_client.list_notebooks = AsyncMock(return_value=[
        MagicMock(id="nb-123", title="Research Notes", source_count=5)
    ])
    mock_nlm_client.get_notebook_summary = AsyncMock(return_value={
        "summary": "AI research notes",
        "suggested_topics": ["AI", "ML"]
    })

    with patch("nlm_proxy.openai.router.ExternalLLMClient") as mock_llm_class:
        mock_llm = AsyncMock()
        # First call: classify as notebooklm
        # Second call: select notebook
        mock_llm.complete = AsyncMock(side_effect=["notebooklm", "nb-123"])
        mock_llm_class.return_value = mock_llm

        router = SmartRouter(
            nlm_client=mock_nlm_client,
            llm_base_url="https://api.test.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini"
        )

        decision = await router.route("What does my research say?")

        assert decision.request_type == RequestType.NOTEBOOKLM
        assert decision.notebook_id == "nb-123"
        assert "Research Notes" in decision.reasoning
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/openai/test_router.py -v`
Expected: FAIL with "No module named 'nlm_proxy.openai.router'"

**Step 3: Write minimal implementation**

Create `src/nlm_proxy/openai/router.py`:

```python
"""Smart request router using LLM classification."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.llm_client import ExternalLLMClient
from nlm_proxy.openai.notebook_cache import NotebookCache
from nlm_proxy.openai.prompts import load_prompt

if TYPE_CHECKING:
    from nlm_proxy.core import NotebookLMClient

logger = get_logger(__name__)


class RequestType(Enum):
    """Type of request after classification."""
    NOTEBOOKLM = "notebooklm"
    LLM_TASK = "llm_task"


@dataclass
class RoutingDecision:
    """Result of request classification and routing."""
    request_type: RequestType
    notebook_id: str | None = None
    reasoning: str = ""


class SmartRouter:
    """Classifies requests and routes to appropriate backend."""

    def __init__(
        self,
        nlm_client: "NotebookLMClient",
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        allowed_notebooks: list[str] | None = None,
        cache_ttl: int = 3600
    ):
        self.nlm_client = nlm_client
        self.llm_client = ExternalLLMClient(llm_base_url, llm_api_key, llm_model)
        self.notebook_cache = NotebookCache(ttl_seconds=cache_ttl)
        self.allowed_notebooks = allowed_notebooks or []

    async def _ensure_notebooks_cached(self) -> list:
        """Ensure notebook summaries are cached, refresh if needed."""
        cached = self.notebook_cache.get_all()
        if cached:
            logger.debug(f"[ROUTER] Using {len(cached)} cached notebooks")
            return cached

        logger.debug("[ROUTER] Cache empty, fetching notebooks from NotebookLM")
        notebooks = await self.nlm_client.list_notebooks()
        logger.debug(f"[ROUTER] Found {len(notebooks)} notebooks")

        # Filter if configured
        if self.allowed_notebooks:
            notebooks = [nb for nb in notebooks if nb.id in self.allowed_notebooks]
            logger.debug(f"[ROUTER] Filtered to {len(notebooks)} allowed notebooks")

        # Get summaries for each notebook
        for nb in notebooks:
            try:
                logger.debug(f"[ROUTER] Fetching summary for notebook: {nb.title} ({nb.id})")
                summary_data = await self.nlm_client.get_notebook_summary(nb.id)
                self.notebook_cache.set(
                    notebook_id=nb.id,
                    title=nb.title,
                    summary=summary_data.get("summary", ""),
                    topics=summary_data.get("suggested_topics", [])
                )
            except Exception as e:
                logger.warning(f"[ROUTER] Failed to get summary for notebook {nb.id}: {e}")
                # Cache with just the title
                self.notebook_cache.set(
                    notebook_id=nb.id,
                    title=nb.title,
                    summary="",
                    topics=[]
                )

        return self.notebook_cache.get_all()

    async def classify_request(self, query: str) -> RequestType:
        """Classify the request type using external LLM."""
        logger.debug(f"[ROUTER] Classifying request: {query[:100]}...")
        prompt_template = load_prompt("classify_request")
        prompt = prompt_template.format(query=query)

        response = await self.llm_client.complete(prompt)
        response_lower = response.lower().strip()

        if "notebooklm" in response_lower:
            logger.info(f"[ROUTER] Classified as NOTEBOOKLM query")
            return RequestType.NOTEBOOKLM
        logger.info(f"[ROUTER] Classified as LLM_TASK")
        return RequestType.LLM_TASK

    async def select_notebook(self, query: str) -> tuple[str | None, str]:
        """Select best notebook for query. Returns (notebook_id, reasoning)."""
        logger.debug(f"[ROUTER] Selecting notebook for query: {query[:100]}...")
        notebooks = await self._ensure_notebooks_cached()

        if not notebooks:
            logger.warning("[ROUTER] No notebooks available for selection")
            return None, "No notebooks available"

        # Build notebook info for LLM
        notebooks_info = [
            {
                "id": nb.id,
                "title": nb.title,
                "summary": nb.summary[:500] if nb.summary else "",
                "topics": nb.topics[:5] if nb.topics else []
            }
            for nb in notebooks
        ]

        prompt_template = load_prompt("select_notebook")
        prompt = prompt_template.format(
            notebooks_json=json.dumps(notebooks_info, indent=2),
            query=query
        )

        logger.debug(f"[ROUTER] Asking LLM to select from {len(notebooks)} notebooks")
        response = await self.llm_client.complete(prompt, max_tokens=100)

        # Parse response - expect notebook_id
        for nb in notebooks:
            if nb.id in response:
                reasoning = f"Selected notebook: {nb.title} (ID: {nb.id})"
                logger.info(f"[ROUTER] {reasoning}")
                return nb.id, reasoning

        # Fallback to first notebook
        if notebooks:
            reasoning = f"Defaulted to notebook: {notebooks[0].title} (ID: {notebooks[0].id})"
            logger.info(f"[ROUTER] {reasoning}")
            return notebooks[0].id, reasoning

        return None, "No suitable notebook found"

    async def route(self, query: str) -> RoutingDecision:
        """Classify and route the request."""
        logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")
        request_type = await self.classify_request(query)

        if request_type == RequestType.LLM_TASK:
            logger.info("[ROUTER] Routing to external LLM")
            return RoutingDecision(
                request_type=RequestType.LLM_TASK,
                reasoning="Classified as LLM task (not a notebook query)"
            )

        notebook_id, reasoning = await self.select_notebook(query)
        logger.info(f"[ROUTER] Routing to NotebookLM: {notebook_id}")
        return RoutingDecision(
            request_type=RequestType.NOTEBOOKLM,
            notebook_id=notebook_id,
            reasoning=reasoning
        )

    async def close(self) -> None:
        """Cleanup resources."""
        logger.debug("[ROUTER] Closing router resources")
        await self.llm_client.close()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/openai/test_router.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/router.py tests/openai/test_router.py
git commit -m "feat(openai): add SmartRouter for request classification and routing"
```

---

## Task 6: Add Smart Router Model to /v1/models Endpoint

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`
- Test: `tests/openai/test_server.py`

**Step 1: Write the failing test**

Add to `tests/openai/test_server.py` (create if doesn't exist):

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/openai/test_server.py::test_list_models_includes_smart_router -v`
Expected: FAIL (smart-router not in model list)

**Step 3: Write minimal implementation**

Modify `src/nlm_proxy/openai/server.py`:

Add import at top:
```python
from nlm_proxy.core.config import get_openai_settings, get_routing_settings
```

Modify the `list_models` function:
```python
@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """List notebooks as available models."""
    logger.debug("[PROXY] Received request: GET /v1/models")

    routing_settings = get_routing_settings()

    client = await get_client()
    try:
        logger.debug("[NOTEBOOKLM] Calling list_notebooks()")
        notebooks = await client.list_notebooks()
        logger.debug(f"[NOTEBOOKLM] Response: {len(notebooks)} notebooks found")

        # Smart router model
        smart_router_model = {
            "id": routing_settings.router_model_name,
            "object": "model",
            "created": 0,
            "owned_by": "nlm-proxy",
            "name": "Smart Router",
            "description": "AI-powered routing to best notebook or external LLM",
        }

        notebook_models = [
            {
                "id": nb.id,
                "object": "model",
                "created": 0,
                "owned_by": "notebooklm",
                "name": nb.title,
                "source_count": nb.source_count,
            }
            for nb in notebooks
        ]

        response = {
            "object": "list",
            "data": [smart_router_model] + notebook_models
        }
        logger.debug(f"[PROXY] Returning {len(response['data'])} models")
        return response
    finally:
        await client.close()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/openai/test_server.py::test_list_models_includes_smart_router -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/server.py tests/openai/test_server.py
git commit -m "feat(openai): add smart-router model to /v1/models"
```

---

## Task 7: Implement Smart Routing Handler in /v1/chat/completions

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`
- Test: `tests/openai/test_server.py`

**Step 1: Write the failing test**

Add to `tests/openai/test_server.py`:

```python
def test_chat_completions_smart_routing_notebooklm(mock_settings):
    """Test smart routing to NotebookLM."""
    from nlm_proxy.openai.server import app

    with patch("nlm_proxy.openai.server.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.list_notebooks = AsyncMock(return_value=[
            MagicMock(id="nb-123", title="Research", source_count=3)
        ])
        mock_client.get_notebook_summary = AsyncMock(return_value={
            "summary": "Research notes",
            "suggested_topics": ["AI"]
        })
        mock_client.query = AsyncMock(return_value={
            "answer": "The research says...",
            "conversation_id": "conv-123"
        })
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        with patch("nlm_proxy.openai.server.SmartRouter") as mock_router_class:
            from nlm_proxy.openai.router import RequestType, RoutingDecision
            mock_router = AsyncMock()
            mock_router.route = AsyncMock(return_value=RoutingDecision(
                request_type=RequestType.NOTEBOOKLM,
                notebook_id="nb-123",
                reasoning="Selected notebook: Research"
            ))
            mock_router.close = AsyncMock()
            mock_router_class.return_value = mock_router

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-api-key"},
                json={
                    "model": "smart-router",
                    "messages": [{"role": "user", "content": "What does my research say?"}],
                    "stream": False
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert "Research" in data["choices"][0]["message"].get("reasoning_content", "")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/openai/test_server.py::test_chat_completions_smart_routing_notebooklm -v`
Expected: FAIL (smart routing not implemented)

**Step 3: Write minimal implementation**

Add imports to `src/nlm_proxy/openai/server.py`:
```python
from nlm_proxy.openai.router import SmartRouter, RequestType
```

Add new handler function before `chat_completions`:
```python
async def handle_smart_routing(request: ChatCompletionRequest, http_request: Request):
    """Handle requests to the smart router model."""
    routing_settings = get_routing_settings()
    client = await get_client()

    router = SmartRouter(
        nlm_client=client,
        llm_base_url=routing_settings.llm_base_url,
        llm_api_key=routing_settings.llm_api_key,
        llm_model=routing_settings.llm_model,
        allowed_notebooks=routing_settings.allowed_notebooks,
        cache_ttl=routing_settings.summary_cache_ttl
    )

    try:
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        query = user_messages[-1].content
        decision = await router.route(query)

        logger.info(f"[SMART-ROUTER] Decision: {decision.request_type.value}, notebook={decision.notebook_id}")

        if request.stream:
            return StreamingResponse(
                stream_smart_response(client, router, decision, query, request),
                media_type="text/event-stream"
            )

        # Non-streaming path
        if decision.request_type == RequestType.LLM_TASK:
            # Call external LLM
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            response_text = await router.llm_client.complete(
                messages[-1]["content"],
                max_tokens=4096
            )
        else:
            # Call NotebookLM
            result = await client.query(
                notebook_id=decision.notebook_id,
                query_text=query
            )
            response_text = result.get("answer", "") if result else ""

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ResponseChoice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=response_text,
                    reasoning_content=decision.reasoning
                ),
                finish_reason="stop"
            )],
            usage=Usage(
                prompt_tokens=len(query),
                completion_tokens=len(response_text),
                total_tokens=len(query) + len(response_text)
            )
        )
    finally:
        await router.close()
        await client.close()
```

Add streaming handler:
```python
async def stream_smart_response(client, router: SmartRouter, decision, query: str, request: ChatCompletionRequest):
    """Stream response with routing reasoning as reasoning_content."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # First, stream the routing decision as reasoning_content
    reasoning_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[Choice(delta=DeltaContent(reasoning_content=decision.reasoning + "\n\n"))]
    )
    yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

    if decision.request_type == RequestType.LLM_TASK:
        # Stream from external LLM
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        async for chunk in router.llm_client.stream(messages):
            delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta_content:
                openai_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=request.model,
                    choices=[Choice(delta=DeltaContent(content=delta_content))]
                )
                yield f"data: {openai_chunk.model_dump_json()}\n\n"
    else:
        # Stream from NotebookLM - reuse existing logic
        previous_thinking = ""
        previous_answer = ""

        async for chunk in client.query_stream(
            notebook_id=decision.notebook_id,
            query_text=query
        ):
            chunk_type = chunk.get("type")
            full_text = chunk.get("text", "")

            if chunk_type == "thinking" and not request.include_thinking:
                previous_thinking = full_text
                continue

            if chunk_type == "thinking":
                delta_text = full_text[len(previous_thinking):]
                previous_thinking = full_text
                if delta_text:
                    delta = DeltaContent(reasoning_content=delta_text)
            else:
                delta_text = full_text[len(previous_answer):]
                previous_answer = full_text
                if delta_text:
                    delta = DeltaContent(content=delta_text)

            if delta_text:
                openai_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=request.model,
                    choices=[Choice(delta=delta)]
                )
                yield f"data: {openai_chunk.model_dump_json()}\n\n"

    # Final chunk
    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[Choice(delta=DeltaContent(), finish_reason="stop")]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```

Modify `chat_completions` to route smart requests:
```python
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    logger.debug(f"[PROXY] Received request: POST /v1/chat/completions")

    # Check if using smart router
    routing_settings = get_routing_settings()
    if request.model == routing_settings.router_model_name:
        return await handle_smart_routing(request, http_request)

    # ... rest of existing code unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/openai/test_server.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/server.py tests/openai/test_server.py
git commit -m "feat(openai): implement smart routing in chat completions endpoint"
```

---

## Task 8: Update .env.example with Routing Configuration

**Files:**
- Modify: `.env.example`

**Step 1: Add routing configuration**

Append to `.env.example`:

```bash
# Smart Routing Configuration
# NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
# NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-openai-key
# NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
# NLM_PROXY_ROUTING_ROUTER_MODEL_NAME=smart-router
# NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS=  # comma-separated notebook IDs, empty=all
# NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL=3600
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add smart routing configuration to .env.example"
```

---

## Task 9: Manual End-to-End Verification

**Step 1: Configure routing**

Add to `~/.nlm-proxy/.env`:
```bash
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
```

**Step 2: Start the server**

```bash
nlm-proxy serve openai --port 9999
```

**Step 3: Verify smart-router in models list**

```bash
curl http://localhost:8080/v1/models -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" | jq '.data[].id'
```

Expected: Should include "smart-router" at the top

**Step 4: Test smart routing with streaming**

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart-router",
    "messages": [{"role": "user", "content": "What information is in my notebooks?"}],
    "stream": true
  }'
```

Expected: First chunk should have `reasoning_content` with "Selected notebook: ..."

**Step 5: Test LLM task routing**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart-router",
    "messages": [{"role": "user", "content": "Summarize what we discussed"}],
    "stream": false
  }'
```

Expected: `reasoning_content` should say "Classified as LLM task"

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `config.py` | Add SmartRoutingSettings |
| 2 | `notebook_cache.py` | Create TTL cache for notebooks |
| 3 | `llm_client.py` | Create external LLM client |
| 4 | `prompts/` | Create prompt templates |
| 5 | `router.py` | Create SmartRouter class |
| 6 | `server.py` | Add smart-router to /v1/models |
| 7 | `server.py` | Implement routing in chat completions |
| 8 | `.env.example` | Document configuration |
| 9 | Manual | End-to-end verification |
| 10 | `docs/`, `README.md`, `.claude/memory/` | Documentation and architecture |

---

## Task 10: Documentation - Architecture, README, and Claude Memory

**Files:**
- Create: `docs/smart-routing-architecture.md`
- Modify: `README.md`
- Create: `.claude/memory/smart-routing.md`

**Step 1: Create detailed architecture document**

Create `docs/smart-routing-architecture.md`:

```markdown
# Smart LLM Request Routing Architecture

## Overview

The Smart Router is an intelligent request routing system that uses an external LLM to classify incoming requests and automatically route them to the appropriate backend:

- **NotebookLM queries** → Auto-selected notebook based on content summaries
- **LLM tasks** → External OpenAI-compatible endpoint

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OpenAI Proxy Server                             │
│                            (FastAPI Application)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POST /v1/chat/completions                                                   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Model Router                                  │    │
│  │                                                                      │    │
│  │   model == "smart-router"?                                          │    │
│  │         │                                                            │    │
│  │    ┌────┴────┐                                                       │    │
│  │    │         │                                                       │    │
│  │   Yes        No ──────────────────────────────────────┐              │    │
│  │    │                                                   │              │    │
│  │    ▼                                                   ▼              │    │
│  │  SmartRouter                                    Direct NotebookLM    │    │
│  │                                                 (existing flow)       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Request Classification                           │    │
│  │                                                                      │    │
│  │   ┌──────────────────┐      ┌─────────────────────────────────┐     │    │
│  │   │  External LLM    │◄─────│  classify_request.txt prompt    │     │    │
│  │   │  (OpenAI SDK)    │      │                                  │     │    │
│  │   └────────┬─────────┘      │  "Is this a notebook query or   │     │    │
│  │            │                │   an LLM task?"                  │     │    │
│  │            ▼                └─────────────────────────────────┘     │    │
│  │   ┌────────────────┐                                                │    │
│  │   │ RequestType    │                                                │    │
│  │   │                │                                                │    │
│  │   │ • NOTEBOOKLM   │                                                │    │
│  │   │ • LLM_TASK     │                                                │    │
│  │   └───────┬────────┘                                                │    │
│  │           │                                                          │    │
│  └───────────┼──────────────────────────────────────────────────────────┘    │
│              │                                                               │
│       ┌──────┴──────┐                                                        │
│       │             │                                                        │
│       ▼             ▼                                                        │
│  ┌─────────┐   ┌─────────────────────────────────────────────────────────┐  │
│  │LLM_TASK │   │                    NOTEBOOKLM                            │  │
│  └────┬────┘   │                                                          │  │
│       │        │  ┌────────────────┐     ┌──────────────────────────┐    │  │
│       │        │  │ NotebookCache  │◄────│ NotebookLM API           │    │  │
│       │        │  │                │     │ • list_notebooks()       │    │  │
│       │        │  │ TTL-based      │     │ • get_notebook_summary() │    │  │
│       │        │  │ In-memory      │     └──────────────────────────┘    │  │
│       │        │  └───────┬────────┘                                      │  │
│       │        │          │                                               │  │
│       │        │          ▼                                               │  │
│       │        │  ┌────────────────┐     ┌──────────────────────────┐    │  │
│       │        │  │ External LLM   │◄────│ select_notebook.txt      │    │  │
│       │        │  │ (notebook      │     │                          │    │  │
│       │        │  │  selection)    │     │ "Which notebook best     │    │  │
│       │        │  └───────┬────────┘     │  answers this query?"    │    │  │
│       │        │          │              └──────────────────────────┘    │  │
│       │        │          ▼                                               │  │
│       │        │  ┌────────────────┐                                      │  │
│       │        │  │ notebook_id    │                                      │  │
│       │        │  └───────┬────────┘                                      │  │
│       │        └──────────┼───────────────────────────────────────────────┘  │
│       │                   │                                                  │
│       ▼                   ▼                                                  │
│  ┌─────────────┐    ┌─────────────────┐                                     │
│  │ External    │    │ NotebookLM      │                                     │
│  │ LLM API     │    │ query_stream()  │                                     │
│  │ (streaming) │    │                 │                                     │
│  └──────┬──────┘    └────────┬────────┘                                     │
│         │                    │                                               │
│         └────────┬───────────┘                                               │
│                  │                                                           │
│                  ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Response Streaming                              │    │
│  │                                                                      │    │
│  │   1. reasoning_content: routing decision                            │    │
│  │      "Selected notebook: Research Notes (ID: abc-123)"              │    │
│  │                                                                      │    │
│  │   2. reasoning_content: NotebookLM thinking (if enabled)            │    │
│  │      "Looking at sources..."                                         │    │
│  │                                                                      │    │
│  │   3. content: actual response                                        │    │
│  │      "Based on your research notes..."                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                  │                                                           │
│                  ▼                                                           │
│            SSE Stream to Client                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. ExternalLLMClient (`src/nlm_proxy/core/llm_client.py`)

Reusable client for calling OpenAI-compatible LLM endpoints.

```python
from nlm_proxy.core import ExternalLLMClient

client = ExternalLLMClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o-mini"
)

# Non-streaming (for classification)
result = await client.complete("Classify this request")

# Streaming (for LLM task passthrough)
async for chunk in await client.stream(messages):
    print(chunk.choices[0].delta.content)
```

**Key features:**
- Uses official OpenAI SDK for compatibility
- Lazy client initialization
- Supports any OpenAI-compatible endpoint (OpenRouter, Ollama, etc.)

### 2. NotebookCache (`src/nlm_proxy/openai/notebook_cache.py`)

Thread-safe, TTL-based cache for notebook summaries.

```python
cache = NotebookCache(ttl_seconds=3600)

# Cache notebook info
cache.set(
    notebook_id="abc-123",
    title="Research Notes",
    summary="AI and ML research...",
    topics=["AI", "ML", "Neural Networks"]
)

# Retrieve (returns None if expired)
info = cache.get("abc-123")

# Get all valid entries
all_notebooks = cache.get_all()
```

**Purpose:** Avoid repeated API calls to NotebookLM for summaries.

### 3. SmartRouter (`src/nlm_proxy/openai/router.py`)

Core routing logic with two-stage LLM classification.

```python
router = SmartRouter(
    nlm_client=notebooklm_client,
    llm_base_url="https://api.openai.com/v1",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    allowed_notebooks=["nb-1", "nb-2"],  # Optional filter
    cache_ttl=3600
)

decision = await router.route("What does my research say about AI?")
# decision.request_type = RequestType.NOTEBOOKLM
# decision.notebook_id = "nb-123"
# decision.reasoning = "Selected notebook: Research Notes (ID: nb-123)"
```

### 4. Prompt Templates (`src/nlm_proxy/openai/prompts/`)

Customizable LLM prompts for classification and selection.

| Template | Purpose |
|----------|---------|
| `classify_request.txt` | Determine if request is NotebookLM query or LLM task |
| `select_notebook.txt` | Choose best notebook from available summaries |

## Data Flow

### Flow 1: NotebookLM Query

```
User: "What does my research say about neural networks?"
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 1. Classify Request           │
    │    LLM: "notebooklm"          │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 2. Load Notebook Summaries    │
    │    (from cache or API)        │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 3. Select Best Notebook       │
    │    LLM picks: "Research Notes"│
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 4. Query NotebookLM           │
    │    query_stream(notebook_id)  │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 5. Stream Response            │
    │    • reasoning: notebook info │
    │    • reasoning: thinking      │
    │    • content: answer          │
    └───────────────────────────────┘
```

### Flow 2: LLM Task

```
User: "Summarize our conversation so far"
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 1. Classify Request           │
    │    LLM: "llm_task"            │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 2. Route to External LLM      │
    │    (skip notebook selection)  │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ 3. Stream from External LLM   │
    │    • reasoning: "LLM task"    │
    │    • content: summary         │
    └───────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# External LLM for classification/routing
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini

# Router settings
NLM_PROXY_ROUTING_ROUTER_MODEL_NAME=smart-router
NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS=  # comma-separated, empty=all
NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL=3600
```

### Using Different LLM Providers

The router works with any OpenAI-compatible endpoint:

```bash
# OpenRouter
NLM_PROXY_ROUTING_LLM_BASE_URL=https://openrouter.ai/api/v1
NLM_PROXY_ROUTING_LLM_MODEL=anthropic/claude-3-haiku

# Ollama (local)
NLM_PROXY_ROUTING_LLM_BASE_URL=http://localhost:11434/v1
NLM_PROXY_ROUTING_LLM_MODEL=llama3.2

# Azure OpenAI
NLM_PROXY_ROUTING_LLM_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/gpt-4o-mini
```

## Logging

The smart router uses consistent logging tags for debugging:

```
[ROUTER] Starting routing for query: What does my research...
[LLM] Calling complete: model=gpt-4o-mini, max_tokens=50
[LLM] Request prompt: You are a request classifier...
[LLM] Response: notebooklm
[ROUTER] Classified as NOTEBOOKLM query
[ROUTER] Using 5 cached notebooks
[ROUTER] Asking LLM to select from 5 notebooks
[ROUTER] Selected notebook: Research Notes (ID: nb-123)
[ROUTER] Routing to NotebookLM: nb-123
```

Enable debug logging:
```bash
NLM_PROXY_DEBUG=true nlm-proxy serve openai
```

## API Usage

### List Models (includes smart-router)

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer $API_KEY"
```

Response:
```json
{
  "data": [
    {"id": "smart-router", "name": "Smart Router", "owned_by": "nlm-proxy"},
    {"id": "nb-123", "name": "Research Notes", "owned_by": "notebooklm"},
    {"id": "nb-456", "name": "Project Docs", "owned_by": "notebooklm"}
  ]
}
```

### Smart Routing Request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart-router",
    "messages": [{"role": "user", "content": "What is in my research notes?"}],
    "stream": true
  }'
```

### Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-api-key"
)

# Smart routing - auto-selects notebook
response = client.chat.completions.create(
    model="smart-router",
    messages=[{"role": "user", "content": "What does my research say?"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        print(f"[Routing] {chunk.choices[0].delta.reasoning_content}")
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| External LLM unavailable | Returns 503 with error message |
| Classification fails | Defaults to NotebookLM query |
| No notebooks available | Returns error in response |
| Notebook selection fails | Falls back to first notebook |
| Summary fetch fails | Uses notebook title only |

## Performance Considerations

1. **Cache TTL**: Default 1 hour. Adjust based on how often notebooks change.
2. **LLM Latency**: Classification adds ~200-500ms per request.
3. **Parallel Requests**: Cache is thread-safe for concurrent access.
4. **Token Usage**: Classification uses ~50 tokens, selection uses ~100 tokens.
```

**Step 2: Update README.md**

Add to the Features section in `README.md`:

```markdown
### Smart Request Routing

Automatically route requests to the best notebook or external LLM:

```bash
# Configure external LLM for routing
export NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
export NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-key
export NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini

# Start proxy
nlm-proxy serve openai --port 8080
```

Use `smart-router` as the model to enable automatic routing:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="your-key")

# Automatically selects the best notebook
response = client.chat.completions.create(
    model="smart-router",  # Uses LLM to pick notebook
    messages=[{"role": "user", "content": "What's in my research notes?"}]
)
```

See [Smart Routing Architecture](docs/smart-routing-architecture.md) for details.
```

**Step 3: Create Claude memory file**

Create `.claude/memory/smart-routing.md`:

```markdown
# Smart Routing

## Overview

Smart routing uses an external LLM to classify requests and auto-select notebooks.

## Key Files

| File | Purpose |
|------|---------|
| `src/nlm_proxy/core/llm_client.py` | External LLM client (OpenAI SDK) |
| `src/nlm_proxy/openai/notebook_cache.py` | TTL cache for notebook summaries |
| `src/nlm_proxy/openai/router.py` | SmartRouter class |
| `src/nlm_proxy/openai/prompts/` | Prompt templates |
| `src/nlm_proxy/core/config.py` | SmartRoutingSettings |

## Configuration

```bash
NLM_PROXY_ROUTING_LLM_BASE_URL=https://api.openai.com/v1
NLM_PROXY_ROUTING_LLM_API_KEY=sk-xxx
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
NLM_PROXY_ROUTING_ROUTER_MODEL_NAME=smart-router
NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS=  # empty=all
NLM_PROXY_ROUTING_SUMMARY_CACHE_TTL=3600
```

## Logging Tags

- `[LLM]` - External LLM client calls
- `[ROUTER]` - Smart routing decisions

## Architecture

See `docs/smart-routing-architecture.md` for detailed diagrams.
```

**Step 4: Update memory modules table in CLAUDE.md**

Add to the table in `CLAUDE.md`:

```markdown
| `smart-routing.md` | Smart routing configuration, LLM client, router |
```

**Step 5: Commit**

```bash
git add docs/smart-routing-architecture.md README.md .claude/memory/smart-routing.md CLAUDE.md
git commit -m "docs: add smart routing architecture documentation"
```
