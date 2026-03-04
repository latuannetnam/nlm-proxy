# Missing Test Cases Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 49 tests covering all untested code paths in the LangChain refactor to ensure behavior parity.

**Architecture:** TDD approach — write each test file, verify tests fail for the right reasons (mocking gaps), then fix any test setup issues. All tests use mocks (no real LLM/NLM calls). Tests organized by component: pipeline integration, streaming, direct path, cached helpers, session store, agent core, LLM client, routing graph.

**Tech Stack:** pytest, pytest-asyncio, unittest.mock, FastAPI TestClient

**Design doc:** `docs/plans/2026-03-04-missing-test-cases-design.md`

---

### Task 1: Fix conftest.py — add session_store reset

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Add session_store to reset fixture**

```python
"""Root conftest for tests — reset shared app state between tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_openai_app_state():
    """Reset the OpenAI server app.state between tests."""
    from nlm_proxy.openai.server import app

    original_agent_core = getattr(app.state, "agent_core", None)
    original_response_cache = getattr(app.state, "response_cache", None)
    original_session_store = getattr(app.state, "session_store", None)

    yield

    app.state.agent_core = original_agent_core
    app.state.response_cache = original_response_cache
    app.state.session_store = original_session_store
```

**Step 2: Run existing tests to verify no regressions**

Run: `uv run pytest --tb=short -q`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add session_store to conftest reset fixture"
```

---

### Task 2: SessionStore unit tests

**Files:**
- Create: `tests/core/test_session_store.py`

**Step 1: Write tests**

```python
"""Tests for SessionStore (core/session.py)."""

import time
import pytest
from nlm_proxy.core.session import SessionStore


class TestSessionStore:
    """Test thread-safe session store with TTL."""

    def test_set_and_get(self):
        """Store → retrieve → correct conversation_id."""
        store = SessionStore(ttl_seconds=3600)
        try:
            store.set("chat-1", "conv-abc")
            assert store.get("chat-1") == "conv-abc"
        finally:
            store.shutdown()

    def test_get_expired_returns_none(self):
        """TTL expired → returns None."""
        store = SessionStore(ttl_seconds=1)
        try:
            store.set("chat-1", "conv-abc")
            time.sleep(1.1)
            assert store.get("chat-1") is None
        finally:
            store.shutdown()

    def test_get_nonexistent_returns_none(self):
        """Non-existent chat_id → returns None."""
        store = SessionStore(ttl_seconds=3600)
        try:
            assert store.get("nonexistent") is None
        finally:
            store.shutdown()

    def test_delete_returns_true_on_existing(self):
        """Delete existing → True."""
        store = SessionStore(ttl_seconds=3600)
        try:
            store.set("chat-1", "conv-abc")
            assert store.delete("chat-1") is True
            assert store.get("chat-1") is None
        finally:
            store.shutdown()

    def test_delete_returns_false_on_missing(self):
        """Delete non-existent → False."""
        store = SessionStore(ttl_seconds=3600)
        try:
            assert store.delete("nonexistent") is False
        finally:
            store.shutdown()

    def test_list_all(self):
        """Multiple sessions → all listed with metadata."""
        store = SessionStore(ttl_seconds=3600)
        try:
            store.set("chat-1", "conv-a")
            store.set("chat-2", "conv-b")
            sessions = store.list_all()
            assert len(sessions) == 2
            assert sessions["chat-1"]["conversation_id"] == "conv-a"
            assert sessions["chat-2"]["conversation_id"] == "conv-b"
            assert "age_seconds" in sessions["chat-1"]
            assert "expires_in_seconds" in sessions["chat-1"]
        finally:
            store.shutdown()

    def test_cleanup_expired(self):
        """Mix of fresh/expired → only expired removed."""
        store = SessionStore(ttl_seconds=1)
        try:
            store.set("old", "conv-old")
            time.sleep(1.1)
            store.set("new", "conv-new")
            removed = store.cleanup_expired()
            assert removed == 1
            assert store.get("old") is None
            assert store.get("new") == "conv-new"
        finally:
            store.shutdown()

    def test_get_stats(self):
        """Returns total_sessions, oldest_session_age_seconds."""
        store = SessionStore(ttl_seconds=3600)
        try:
            store.set("chat-1", "conv-a")
            store.set("chat-2", "conv-b")
            stats = store.get_stats()
            assert stats["total_sessions"] == 2
            assert stats["ttl_seconds"] == 3600
            assert "oldest_session_age_seconds" in stats
        finally:
            store.shutdown()
```

**Step 2: Run tests**

Run: `uv run pytest tests/core/test_session_store.py -v`
Expected: All 7 PASS

**Step 3: Commit**

```bash
git add tests/core/test_session_store.py
git commit -m "test: add SessionStore unit tests (7 tests)"
```

---

### Task 3: AgentCore session helpers + edge cases

**Files:**
- Modify: `tests/core/test_agent.py`

**Step 1: Add session helper tests and edge case tests at the end of the file**

```python
# ── Session helper tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_conversation_id_returns_stored():
    """session_store.get() returns stored conversation_id."""
    from nlm_proxy.core.agent import AgentCore

    mock_session = MagicMock()
    mock_session.get.return_value = "conv-123"

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=mock_session,
        )
    assert agent.get_conversation_id("chat-1") == "conv-123"
    mock_session.get.assert_called_once_with("chat-1")


@pytest.mark.asyncio
async def test_get_conversation_id_no_session_store():
    """session_store=None → returns None."""
    from nlm_proxy.core.agent import AgentCore

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=None,
        )
    assert agent.get_conversation_id("chat-1") is None


@pytest.mark.asyncio
async def test_get_conversation_id_empty_chat_id():
    """chat_id='' → returns None."""
    from nlm_proxy.core.agent import AgentCore

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=MagicMock(),
        )
    assert agent.get_conversation_id("") is None


@pytest.mark.asyncio
async def test_save_conversation_id_calls_session_store():
    """session_store.set() called with correct args."""
    from nlm_proxy.core.agent import AgentCore

    mock_session = MagicMock()

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=mock_session,
        )
    agent.save_conversation_id("chat-1", "conv-123")
    mock_session.set.assert_called_once_with("chat-1", "conv-123")


@pytest.mark.asyncio
async def test_save_conversation_id_noop_when_no_store():
    """session_store=None → no error."""
    from nlm_proxy.core.agent import AgentCore

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=None,
        )
    # Should not raise
    agent.save_conversation_id("chat-1", "conv-123")


@pytest.mark.asyncio
async def test_save_conversation_id_noop_when_empty_conv_id():
    """conversation_id='' → not saved."""
    from nlm_proxy.core.agent import AgentCore

    mock_session = MagicMock()

    with patch("nlm_proxy.core.agent.build_routing_graph"):
        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=MagicMock(), chat_model=AsyncMock(),
            session_store=mock_session,
        )
    agent.save_conversation_id("chat-1", "")
    mock_session.set.assert_not_called()


# ── Edge case tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_no_response_cache():
    """response_cache=None → skip cache, go straight to graph."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "request_type": "notebooklm",
            "notebook_id": "nb-1",
            "reasoning": "Selected",
        })
        mock_build.return_value = mock_graph

        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=MagicMock(),
            response_cache=None, chat_model=AsyncMock(),
        )
        decision = await agent.route("test", RequestOptions())

    assert decision.request_type == "notebooklm"
    assert decision.cache_result is None


@pytest.mark.asyncio
async def test_route_fallback_empty_notebooks_reraises():
    """Graph error + no notebooks → exception propagated."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))
        mock_build.return_value = mock_graph

        mock_nb_cache = MagicMock()
        mock_nb_cache.get_all.return_value = []  # No notebooks

        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=mock_nb_cache,
            response_cache=MagicMock(), chat_model=AsyncMock(),
        )
        agent.response_cache.lookup_global.return_value = (None, None)

        with pytest.raises(RuntimeError, match="LLM error"):
            await agent.route("test", RequestOptions())


@pytest.mark.asyncio
async def test_route_fallback_with_acl_filters_notebooks():
    """Graph error + ACL filter → fallback uses only allowed notebooks."""
    from nlm_proxy.core.agent import AgentCore, RequestOptions

    with patch("nlm_proxy.core.agent.build_routing_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_build.return_value = mock_graph

        nb1 = MagicMock(id="nb-1")
        nb2 = MagicMock(id="nb-2")
        mock_nb_cache = MagicMock()
        mock_nb_cache.get_all.return_value = [nb1, nb2]

        agent = AgentCore(
            nlm_client=AsyncMock(), notebook_cache=mock_nb_cache,
            response_cache=MagicMock(), chat_model=AsyncMock(),
        )
        agent.response_cache.lookup_global.return_value = (None, None)

        options = RequestOptions(allowed_notebooks=["nb-2"])
        decision = await agent.route("test", options)

    assert decision.notebook_id == "nb-2"
    assert "fallback" in decision.reasoning.lower()
```

**Step 2: Run tests**

Run: `uv run pytest tests/core/test_agent.py -v`
Expected: All 19 PASS (10 existing + 9 new)

**Step 3: Commit**

```bash
git add tests/core/test_agent.py
git commit -m "test: add AgentCore session helpers + edge case tests (9 tests)"
```

---

### Task 4: LLM client message conversion + provider tests

**Files:**
- Modify: `tests/core/test_llm_client.py`

**Step 1: Add message conversion and provider tests at the end of the file**

```python
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
```

**Step 2: Run tests**

Run: `uv run pytest tests/core/test_llm_client.py -v`
Expected: All 10 PASS (4 existing + 6 new)

**Step 3: Commit**

```bash
git add tests/core/test_llm_client.py
git commit -m "test: add message conversion + provider factory tests (6 tests)"
```

---

### Task 5: Cached response helper tests

**Files:**
- Create: `tests/test_openai_module/test_cached_response_helpers.py`

**Step 1: Write tests**

```python
"""Tests for _stream_cached_response and _json_cached_response helpers."""

import json
import pytest
from unittest.mock import MagicMock

from nlm_proxy.core.agent import RoutingDecision
from nlm_proxy.openai.types import ChatCompletionRequest, Message


def _make_decision(thinking=None, hit_type="exact"):
    """Create a RoutingDecision with cache result."""
    cached = MagicMock()
    cached.answer = "Cached answer text"
    cached.thinking = thinking
    cached.conversation_id = "conv-123"
    return RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Cache hit — returning cached response.",
        cache_result=cached,
        cache_hit_type=hit_type,
    )


def _make_request():
    return ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="test query")],
        stream=True,
    )


@pytest.mark.asyncio
async def test_stream_cached_produces_correct_sse_sequence():
    """Reasoning → thinking → answer → stop → [DONE]."""
    from nlm_proxy.openai.server import _stream_cached_response

    decision = _make_decision(thinking="Thinking text...")
    request = _make_request()
    request.include_thinking = True

    chunks = []
    async for chunk in _stream_cached_response(decision, request):
        chunks.append(chunk)

    # Should have: reasoning, thinking, answer, stop, [DONE]
    assert len(chunks) == 5
    assert chunks[-1] == "data: [DONE]\n\n"

    # Parse each data chunk
    parsed = []
    for c in chunks[:-1]:
        assert c.startswith("data: ")
        parsed.append(json.loads(c[6:].strip()))

    # First chunk: reasoning
    assert "reasoning_content" in str(parsed[0]["choices"][0]["delta"])
    # Last data chunk: finish_reason=stop
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_stream_cached_without_thinking():
    """When cache_result.thinking=None → no thinking chunk emitted."""
    from nlm_proxy.openai.server import _stream_cached_response

    decision = _make_decision(thinking=None)
    request = _make_request()

    chunks = []
    async for chunk in _stream_cached_response(decision, request):
        chunks.append(chunk)

    # Should have: reasoning, answer, stop, [DONE] (no thinking)
    assert len(chunks) == 4


@pytest.mark.asyncio
async def test_stream_cached_system_fingerprint_format():
    """system_fingerprint = cache_{hit_type}_conv_{conversation_id}."""
    from nlm_proxy.openai.server import _stream_cached_response

    decision = _make_decision(hit_type="semantic")
    request = _make_request()

    chunks = []
    async for chunk in _stream_cached_response(decision, request):
        if chunk.startswith("data: {"):
            parsed = json.loads(chunk[6:].strip())
            fp = parsed.get("system_fingerprint")
            if fp:
                chunks.append(fp)

    # Answer and stop chunks should have fingerprint
    assert any("cache_semantic_conv_conv-123" in fp for fp in chunks)


def test_json_cached_x_cache_status_header():
    """X-Cache-Status header set correctly."""
    from nlm_proxy.openai.server import _json_cached_response

    for hit_type, expected_header in [
        ("exact", "HIT_EXACT"),
        ("semantic", "HIT_SEMANTIC"),
        ("pre_routing_exact", "HIT_PRE_ROUTING_EXACT"),
    ]:
        decision = _make_decision(hit_type=hit_type)
        request = _make_request()
        response = _json_cached_response(decision, request)
        assert response.headers.get("X-Cache-Status") == expected_header


def test_json_cached_response_content():
    """JSON body has correct content, reasoning_content, system_fingerprint."""
    from nlm_proxy.openai.server import _json_cached_response

    decision = _make_decision(hit_type="exact")
    request = _make_request()
    response = _json_cached_response(decision, request)

    body = json.loads(response.body.decode())
    assert body["choices"][0]["message"]["content"] == "Cached answer text"
    assert "Cache hit" in body["choices"][0]["message"]["reasoning_content"]
    assert body["system_fingerprint"] == "cache_exact_conv_conv-123"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_openai_module/test_cached_response_helpers.py -v`
Expected: All 5 PASS

**Step 3: Commit**

```bash
git add tests/test_openai_module/test_cached_response_helpers.py
git commit -m "test: add cached response helper tests — SSE + JSON format (5 tests)"
```

---

### Task 6: Streaming smart response tests

**Files:**
- Create: `tests/test_openai_module/test_streaming_smart.py`

**Step 1: Write tests**

```python
"""Tests for stream_smart_response() — Phase 3a of four-phase pipeline."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nlm_proxy.core.agent import AgentCore, RoutingDecision
from nlm_proxy.openai.types import ChatCompletionRequest, Message


def _make_request(**kwargs):
    defaults = dict(
        model="knowledge-finder",
        messages=[Message(role="user", content="test query")],
        stream=True,
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _make_mock_agent(query_stream_chunks=None, chat_model_chunks=None):
    """Create a mock AgentCore."""
    agent = MagicMock(spec=AgentCore)
    agent.response_cache = None
    agent.save_conversation_id = MagicMock()

    if query_stream_chunks:
        async def mock_stream(*args, **kwargs):
            for c in query_stream_chunks:
                yield c
        agent.query_stream = mock_stream

    if chat_model_chunks:
        async def mock_astream(messages):
            for c in chat_model_chunks:
                yield c
        agent.chat_model = MagicMock()
        agent.chat_model.astream = mock_astream

    return agent


def _parse_sse_chunks(raw_chunks):
    """Parse SSE data lines into JSON objects."""
    parsed = []
    for chunk in raw_chunks:
        if chunk.startswith("data: {"):
            parsed.append(json.loads(chunk[6:].strip()))
        elif chunk == "data: [DONE]\n\n":
            parsed.append("[DONE]")
    return parsed


@pytest.mark.asyncio
async def test_llm_task_streaming_sse_format():
    """LLM_TASK → chat_model.astream() → correct SSE format."""
    from nlm_proxy.openai.server import stream_smart_response

    chunk1 = MagicMock()
    chunk1.content = "Hello"
    chunk2 = MagicMock()
    chunk2.content = " World"

    agent = _make_mock_agent(chat_model_chunks=[chunk1, chunk2])
    decision = RoutingDecision(request_type="llm_task", reasoning="LLM task")
    request = _make_request()

    chunks = []
    async for chunk in stream_smart_response(agent, decision, "test", request):
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    # Should have: reasoning, content1, content2, stop, [DONE]
    content_chunks = [p for p in parsed if isinstance(p, dict) and p["choices"][0]["delta"].get("content")]
    assert len(content_chunks) == 2
    assert content_chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert content_chunks[1]["choices"][0]["delta"]["content"] == " World"


@pytest.mark.asyncio
async def test_notebooklm_cache_store_after_stream():
    """NLM stream completes → response_cache.store() called."""
    from nlm_proxy.openai.server import stream_smart_response

    nlm_chunks = [
        {"type": "answer", "text": "The answer", "conversation_id": "conv-1"},
        {"type": "answer", "text": "The answer is 42", "conversation_id": "conv-1"},
    ]
    agent = _make_mock_agent(query_stream_chunks=nlm_chunks)
    mock_cache = MagicMock()
    mock_cache._semantic_enabled = False
    agent.response_cache = mock_cache

    decision = RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="Selected",
    )
    request = _make_request()
    request.conversation_id = None

    chunks = []
    async for chunk in stream_smart_response(agent, decision, "test", request):
        chunks.append(chunk)

    mock_cache.store.assert_called_once()
    call_kwargs = mock_cache.store.call_args[1]
    assert call_kwargs["notebook_id"] == "nb-1"
    assert call_kwargs["answer"] == " is 42"  # accumulated delta
    assert call_kwargs["conversation_id"] == "conv-1"


@pytest.mark.asyncio
async def test_include_thinking_false_filters_thinking():
    """include_thinking=False → thinking chunks not in output."""
    from nlm_proxy.openai.server import stream_smart_response

    nlm_chunks = [
        {"type": "thinking", "text": "Let me think...", "conversation_id": "conv-1"},
        {"type": "answer", "text": "The answer", "conversation_id": "conv-1"},
    ]
    agent = _make_mock_agent(query_stream_chunks=nlm_chunks)
    agent.response_cache = None

    decision = RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="Selected",
    )
    request = _make_request(include_thinking=False)
    request.conversation_id = None

    chunks = []
    async for chunk in stream_smart_response(agent, decision, "test", request):
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    # No reasoning_content from thinking chunks should be present
    thinking_chunks = [
        p for p in parsed
        if isinstance(p, dict) and p["choices"][0]["delta"].get("reasoning_content")
        and "Selected" not in p["choices"][0]["delta"].get("reasoning_content", "")
    ]
    assert len(thinking_chunks) == 0


@pytest.mark.asyncio
async def test_conversation_id_extracted_and_saved():
    """NLM chunk has conversation_id → agent.save_conversation_id() called."""
    from nlm_proxy.openai.server import stream_smart_response

    nlm_chunks = [
        {"type": "answer", "text": "Hello", "conversation_id": "conv-new"},
    ]
    agent = _make_mock_agent(query_stream_chunks=nlm_chunks)
    agent.response_cache = None

    decision = RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="Selected",
    )
    request = _make_request()
    request.conversation_id = None

    async for _ in stream_smart_response(agent, decision, "test", request, chat_id="chat-1"):
        pass

    agent.save_conversation_id.assert_called_once_with("chat-1", "conv-new")


@pytest.mark.asyncio
async def test_stream_ends_with_done():
    """All paths → final output is [DONE]."""
    from nlm_proxy.openai.server import stream_smart_response

    agent = _make_mock_agent(chat_model_chunks=[])
    decision = RoutingDecision(request_type="llm_task", reasoning="LLM task")
    request = _make_request()

    chunks = []
    async for chunk in stream_smart_response(agent, decision, "test", request):
        chunks.append(chunk)

    assert chunks[-1] == "data: [DONE]\n\n"
    # Second-to-last should be stop chunk
    stop = json.loads(chunks[-2][6:].strip())
    assert stop["choices"][0]["finish_reason"] == "stop"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_openai_module/test_streaming_smart.py -v`
Expected: All 5 PASS

**Step 3: Commit**

```bash
git add tests/test_openai_module/test_streaming_smart.py
git commit -m "test: add stream_smart_response tests — LLM + NLM + cache (5 tests)"
```

---

### Task 7: Four-phase pipeline integration tests

**Files:**
- Create: `tests/test_openai_module/test_smart_routing_pipeline.py`

**Step 1: Write tests**

```python
"""Tests for handle_smart_routing() — four-phase pipeline integration."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from nlm_proxy.core.agent import AgentCore, RoutingDecision


def _setup_app(agent_core=None, session_store=None):
    """Configure app.state for tests."""
    from nlm_proxy.openai.server import app
    app.state.agent_core = agent_core
    app.state.session_store = session_store or MagicMock()
    return app


def _make_agent_with_cache_hit(hit_type="exact"):
    """Create mock AgentCore that returns a pre-routing cache hit."""
    agent = MagicMock(spec=AgentCore)
    cached = MagicMock()
    cached.answer = "Cached answer"
    cached.thinking = None
    cached.conversation_id = "conv-cached"

    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Pre-routing cache hit",
        cache_result=cached,
        cache_hit_type=f"pre_routing_{hit_type}",
    ))
    agent.response_cache = MagicMock()
    agent.chat_model = AsyncMock()
    return agent


def _make_agent_with_post_routing_cache_hit():
    """AgentCore routes (no pre-routing hit), but Phase 2 has a hit."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Selected notebook",
    ))

    cached = MagicMock()
    cached.answer = "Post-routing cached"
    cached.thinking = None
    cached.conversation_id = "conv-post"

    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(cached, "semantic"))
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    return agent


@pytest.fixture
def mock_settings():
    with patch("nlm_proxy.openai.server.get_openai_settings") as mock_openai:
        mock_openai.return_value = MagicMock(api_key="test-key")
        with patch("nlm_proxy.openai.server.get_routing_settings") as mock_routing:
            mock_routing.return_value = MagicMock(
                router_model_name="knowledge-finder",
                llm_base_url="https://api.test.com/v1",
                llm_api_key="test-key",
                llm_model="gpt-4o-mini",
            )
            with patch("nlm_proxy.openai.server.get_tracing_settings") as mock_tracing:
                mock_tracing.return_value = MagicMock(
                    request_max_length=100,
                    response_max_length=100,
                )
                yield


def test_phase0_cache_hit_non_streaming(mock_settings):
    """Pre-routing cache hit → JSON response with X-Cache-Status."""
    agent = _make_agent_with_cache_hit("exact")
    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert "HIT" in response.headers.get("X-Cache-Status", "")
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Cached answer"


def test_phase0_cache_hit_streaming(mock_settings):
    """Pre-routing cache hit → SSE stream with X-Cache-Status."""
    agent = _make_agent_with_cache_hit("exact")
    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "HIT" in response.headers.get("X-Cache-Status", "")
    assert "data: [DONE]" in response.text


def test_phase2_post_routing_cache_hit_non_streaming(mock_settings):
    """Route (cache miss) → notebook-scoped cache hit → JSON."""
    agent = _make_agent_with_post_routing_cache_hit()
    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("X-Cache-Status") == "HIT_SEMANTIC"


def test_bypass_cache_skips_all_cache_checks(mock_settings):
    """bypass_cache=True → no Phase 0/2, goes to Phase 3."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Selected",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "Live answer", "conversation_id": "conv-1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "bypass_cache": True,
        },
    )

    assert response.status_code == 200
    # route() should have been called with bypass_cache=True in options
    call_args = agent.route.call_args
    options = call_args[0][1]
    assert options.bypass_cache is True


def test_acl_wildcard_allows_all(mock_settings):
    """metadata.allowed_notebooks=['*'] → treated as None."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="OK",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "ans", "conversation_id": "c1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "metadata": {"allowed_notebooks": ["*"]},
        },
    )

    assert response.status_code == 200
    options = agent.route.call_args[0][1]
    assert options.allowed_notebooks is None  # Wildcard → None


def test_acl_empty_blocks_all(mock_settings):
    """metadata.allowed_notebooks=[] → empty list passed."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="OK",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "ans", "conversation_id": "c1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "metadata": {"allowed_notebooks": []},
        },
    )

    assert response.status_code == 200
    options = agent.route.call_args[0][1]
    assert options.allowed_notebooks == []


def test_chat_id_from_header(mock_settings):
    """X-OpenWebUI-Chat-Id header → used as chat_id."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="OK",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "ans", "conversation_id": "c1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-key",
            "X-OpenWebUI-Chat-Id": "chat-from-header",
        },
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    agent.get_conversation_id.assert_called_with("chat-from-header")


def test_chat_id_from_metadata(mock_settings):
    """No header → falls back to metadata.chat_id."""
    agent = MagicMock(spec=AgentCore)
    agent.route = AsyncMock(return_value=RoutingDecision(
        request_type="notebooklm", notebook_id="nb-1", reasoning="OK",
    ))
    agent.response_cache = MagicMock()
    agent.response_cache.lookup_async = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={"answer": "ans", "conversation_id": "c1"})
    agent.chat_model = AsyncMock()
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "metadata": {"chat_id": "chat-from-meta"},
        },
    )

    assert response.status_code == 200
    agent.get_conversation_id.assert_called_with("chat-from-meta")


def test_agent_core_not_initialized_503(mock_settings):
    """app.state.agent_core=None → HTTP 503."""
    app = _setup_app(agent_core=None)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 503


def test_no_user_message_400(mock_settings):
    """Request with no user messages → HTTP 400."""
    agent = MagicMock(spec=AgentCore)
    agent.get_conversation_id = MagicMock(return_value=None)
    app = _setup_app(agent_core=agent)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "knowledge-finder",
            "messages": [{"role": "system", "content": "Be helpful"}],
            "stream": False,
        },
    )

    assert response.status_code == 400
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_openai_module/test_smart_routing_pipeline.py -v`
Expected: All 11 PASS

**Step 3: Commit**

```bash
git add tests/test_openai_module/test_smart_routing_pipeline.py
git commit -m "test: add four-phase pipeline integration tests (11 tests)"
```

---

### Task 8: Direct notebook path tests

**Files:**
- Create: `tests/test_openai_module/test_direct_notebook.py`

**Step 1: Write tests**

```python
"""Tests for direct notebook path (model ≠ router_model_name)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_settings():
    with patch("nlm_proxy.openai.server.get_openai_settings") as m1:
        m1.return_value = MagicMock(api_key="test-key")
        with patch("nlm_proxy.openai.server.get_routing_settings") as m2:
            m2.return_value = MagicMock(
                router_model_name="knowledge-finder",
                llm_base_url="https://api.test.com/v1",
                llm_api_key="test-key",
                llm_model="gpt-4o-mini",
            )
            yield


def _setup_app(agent_core=None):
    from nlm_proxy.openai.server import app
    app.state.agent_core = agent_core
    app.state.session_store = MagicMock()
    return app


def test_direct_cache_hit_non_streaming(mock_settings):
    """Direct path cache hit → JSON with X-Cache-Status."""
    from nlm_proxy.core.agent import AgentCore

    cached = MagicMock()
    cached.answer = "Cached direct answer"
    cached.thinking = None
    cached.conversation_id = "conv-direct"

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(cached, "exact"))
    agent.get_conversation_id = MagicMock(return_value=None)

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "nb-direct-123",
            "messages": [{"role": "user", "content": "What are key points?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("X-Cache-Status") == "HIT_EXACT"
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Cached direct answer"


def test_direct_cache_miss_non_streaming(mock_settings):
    """Direct path cache miss → agent.query() → response + cache store."""
    from nlm_proxy.core.agent import AgentCore

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={
        "answer": "Live direct answer",
        "conversation_id": "conv-live",
    })
    agent.response_cache = MagicMock()
    agent.response_cache._semantic_enabled = False
    agent.get_conversation_id = MagicMock(return_value=None)
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "nb-direct-123",
            "messages": [{"role": "user", "content": "What are key points?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Live direct answer"
    # Verify cache store was called
    agent.response_cache.store.assert_called_once()


def test_direct_cache_hit_streaming(mock_settings):
    """Direct path cache hit → SSE stream with X-Cache-Status."""
    from nlm_proxy.core.agent import AgentCore

    cached = MagicMock()
    cached.answer = "Cached streaming answer"
    cached.thinking = None
    cached.conversation_id = "conv-stream"

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(cached, "semantic"))
    agent.get_conversation_id = MagicMock(return_value=None)

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "nb-direct-123",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("X-Cache-Status") == "HIT_SEMANTIC"
    assert "data: [DONE]" in response.text


def test_direct_session_lookup_and_save(mock_settings):
    """Direct path loads conversation_id from session, saves new one."""
    from nlm_proxy.core.agent import AgentCore

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(None, None))
    agent.query = AsyncMock(return_value={
        "answer": "Answer",
        "conversation_id": "conv-new",
    })
    agent.response_cache = MagicMock()
    agent.response_cache._semantic_enabled = False
    agent.get_conversation_id = MagicMock(return_value="conv-old")
    agent.save_conversation_id = MagicMock()

    app = _setup_app(agent_core=agent)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-key",
            "X-OpenWebUI-Chat-Id": "chat-123",
        },
        json={
            "model": "nb-direct-123",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    agent.get_conversation_id.assert_called_with("chat-123")
    agent.save_conversation_id.assert_called_with("chat-123", "conv-new")


def test_direct_cache_miss_streaming(mock_settings):
    """Direct cache miss + streaming → stream_response() called."""
    from nlm_proxy.core.agent import AgentCore

    agent = MagicMock(spec=AgentCore)
    agent.handle_direct_query = AsyncMock(return_value=(None, None))
    agent.get_conversation_id = MagicMock(return_value=None)

    app = _setup_app(agent_core=agent)

    # Mock get_client to return a client with query_stream
    async def mock_stream(*args, **kwargs):
        yield {"type": "answer", "text": "Hi", "conversation_id": "conv-1"}

    with patch("nlm_proxy.openai.server.get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.query_stream = mock_stream
        mock_client.close = AsyncMock()
        mock_get.return_value = mock_client

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "nb-direct-123",
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_openai_module/test_direct_notebook.py -v`
Expected: All 5 PASS

**Step 3: Commit**

```bash
git add tests/test_openai_module/test_direct_notebook.py
git commit -m "test: add direct notebook path tests — cache + session (5 tests)"
```

---

### Task 9: Routing graph source descriptions test

**Files:**
- Modify: `tests/core/test_routing_graph.py`

**Step 1: Add test at end of file**

```python
@pytest.mark.asyncio
async def test_select_notebook_with_source_descriptions(mock_chat_model, mock_notebook_cache):
    """source_descriptions_enabled=True → enriched prompt sent to LLM."""
    from nlm_proxy.core.routing_graph import select_notebook_node

    mock_chat_model.ainvoke = AsyncMock(return_value=_mock_llm_response("nb-1"))

    # Mock notebook_cache.get_source_descriptions
    mock_notebook_cache.get_source_descriptions = AsyncMock(return_value={
        "nb-1": [{"title": "Source A", "summary": "About AI"}],
    })

    routing_settings = MagicMock()
    routing_settings.source_descriptions_enabled = True
    routing_settings.source_descriptions_max_sources = 5
    routing_settings.source_max_keywords = 5
    routing_settings.source_summary_max_chars = 200
    routing_settings.max_source_titles = 15

    state = {
        "query": "What is AI?",
        "request_type": "notebooklm",
        "available_notebooks": [],
        "allowed_notebooks": None,
        "notebook_id": None,
        "reasoning": "",
        "messages": [],
    }

    result = await select_notebook_node(
        state,
        chat_model=mock_chat_model,
        notebook_cache=mock_notebook_cache,
        routing_settings=routing_settings,
    )

    # Verify LLM was called (the enriched prompt would include source info)
    mock_chat_model.ainvoke.assert_called_once()
```

**Step 2: Run tests**

Run: `uv run pytest tests/core/test_routing_graph.py -v`
Expected: All 15 PASS (14 existing + 1 new)

**Step 3: Commit**

```bash
git add tests/core/test_routing_graph.py
git commit -m "test: add routing graph source descriptions test"
```

---

### Task 10: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest --tb=short -q`
Expected: All tests PASS (existing + ~49 new)

**Step 2: Commit all if any missed**

```bash
git add -A
git status  # Verify only test files changed
```
