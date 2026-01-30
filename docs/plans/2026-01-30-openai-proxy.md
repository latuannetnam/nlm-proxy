# OpenAI-Compatible Proxy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a FastAPI server exposing OpenAI-compatible API endpoints that translate requests to NotebookLM's `query_stream` API.

**Architecture:** Standalone FastAPI app reusing existing `NotebookLMClient` and auth system. Notebook UUID passed via `model` field, conversation continuity via `system_fingerprint`. Stateless design - no server-side session tracking.

**Tech Stack:** FastAPI, Pydantic, uvicorn, existing httpx-based NotebookLMClient

---

## Task 1: Create Pydantic Models for OpenAI API

**Files:**
- Create: `src/notebooklm_mcp/openai_types.py`
- Test: `tests/test_openai_types.py`

**Step 1: Write the failing test for Message model**

```python
# tests/test_openai_types.py
import pytest
from pydantic import ValidationError


def test_message_valid_user_role():
    from notebooklm_mcp.openai_types import Message
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_message_invalid_role_rejected():
    from notebooklm_mcp.openai_types import Message
    with pytest.raises(ValidationError):
        Message(role="invalid", content="Hello")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_types.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'notebooklm_mcp.openai_types'"

**Step 3: Write minimal implementation**

```python
# src/notebooklm_mcp/openai_types.py
"""Pydantic models for OpenAI-compatible API."""

from typing import Literal
from pydantic import BaseModel


class Message(BaseModel):
    """A single message in the conversation."""
    role: Literal["system", "user", "assistant"]
    content: str
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_types.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_types.py tests/test_openai_types.py
git commit -m "feat(openai-proxy): add Message model"
```

---

## Task 2: Add ChatCompletionRequest Model

**Files:**
- Modify: `src/notebooklm_mcp/openai_types.py`
- Test: `tests/test_openai_types.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_types.py

def test_chat_completion_request_minimal():
    from notebooklm_mcp.openai_types import ChatCompletionRequest, Message
    req = ChatCompletionRequest(
        model="notebook-uuid",
        messages=[Message(role="user", content="Hello")]
    )
    assert req.model == "notebook-uuid"
    assert req.stream is False  # Default
    assert req.conversation_id is None
    assert req.include_thinking is False


def test_chat_completion_request_with_extras():
    from notebooklm_mcp.openai_types import ChatCompletionRequest, Message
    req = ChatCompletionRequest(
        model="nb-123",
        messages=[Message(role="user", content="Hi")],
        stream=True,
        conversation_id="conv-456",
        include_thinking=True
    )
    assert req.stream is True
    assert req.conversation_id == "conv-456"
    assert req.include_thinking is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_types.py::test_chat_completion_request_minimal -v`
Expected: FAIL with "cannot import name 'ChatCompletionRequest'"

**Step 3: Write minimal implementation**

```python
# Add to src/notebooklm_mcp/openai_types.py

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str  # notebook_id
    messages: list[Message]
    stream: bool = False
    # Ignored by NotebookLM but accepted for compatibility
    temperature: float | None = None
    max_tokens: int | None = None
    # Custom extensions
    conversation_id: str | None = None
    include_thinking: bool = False
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_types.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_types.py tests/test_openai_types.py
git commit -m "feat(openai-proxy): add ChatCompletionRequest model"
```

---

## Task 3: Add Response Models (Choice, Chunk, Completion)

**Files:**
- Modify: `src/notebooklm_mcp/openai_types.py`
- Test: `tests/test_openai_types.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_types.py
import json


def test_chat_completion_chunk_serialization():
    from notebooklm_mcp.openai_types import ChatCompletionChunk, Choice, DeltaContent

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        created=1700000000,
        model="nb-uuid",
        choices=[Choice(index=0, delta=DeltaContent(content="Hello"))],
        system_fingerprint="conv_abc123"
    )

    data = json.loads(chunk.model_dump_json())
    assert data["object"] == "chat.completion.chunk"
    assert data["choices"][0]["delta"]["content"] == "Hello"
    assert data["system_fingerprint"] == "conv_abc123"


def test_chat_completion_chunk_final():
    from notebooklm_mcp.openai_types import ChatCompletionChunk, Choice, DeltaContent

    chunk = ChatCompletionChunk(
        id="chatcmpl-123",
        created=1700000000,
        model="nb-uuid",
        choices=[Choice(index=0, delta=DeltaContent(), finish_reason="stop")]
    )

    assert chunk.choices[0].finish_reason == "stop"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_types.py::test_chat_completion_chunk_serialization -v`
Expected: FAIL with "cannot import name 'ChatCompletionChunk'"

**Step 3: Write minimal implementation**

```python
# Add to src/notebooklm_mcp/openai_types.py

class DeltaContent(BaseModel):
    """Delta content in streaming response."""
    role: str | None = None
    content: str | None = None


class Choice(BaseModel):
    """A single choice in chat completion response."""
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[Choice]
    system_fingerprint: str | None = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_types.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_types.py tests/test_openai_types.py
git commit -m "feat(openai-proxy): add response models (Choice, DeltaContent, ChatCompletionChunk)"
```

---

## Task 4: Add Non-Streaming Response Model

**Files:**
- Modify: `src/notebooklm_mcp/openai_types.py`
- Test: `tests/test_openai_types.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_types.py

def test_chat_completion_response_non_streaming():
    from notebooklm_mcp.openai_types import (
        ChatCompletionResponse, ResponseChoice, ResponseMessage, Usage
    )

    response = ChatCompletionResponse(
        id="chatcmpl-123",
        created=1700000000,
        model="nb-uuid",
        choices=[ResponseChoice(
            index=0,
            message=ResponseMessage(role="assistant", content="Hello!"),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        system_fingerprint="conv_abc123"
    )

    data = response.model_dump()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello!"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_types.py::test_chat_completion_response_non_streaming -v`
Expected: FAIL with "cannot import name 'ChatCompletionResponse'"

**Step 3: Write minimal implementation**

```python
# Add to src/notebooklm_mcp/openai_types.py

class ResponseMessage(BaseModel):
    """Message in non-streaming response."""
    role: str = "assistant"
    content: str


class ResponseChoice(BaseModel):
    """Choice in non-streaming response."""
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible non-streaming response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ResponseChoice]
    usage: Usage
    system_fingerprint: str | None = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_types.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_types.py tests/test_openai_types.py
git commit -m "feat(openai-proxy): add non-streaming response models"
```

---

## Task 5: Create FastAPI App Skeleton with Health Endpoint

**Files:**
- Create: `src/notebooklm_mcp/openai_proxy.py`
- Test: `tests/test_openai_proxy.py`

**Step 1: Write the failing test**

```python
# tests/test_openai_proxy.py
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint():
    from notebooklm_mcp.openai_proxy import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_proxy.py::test_health_endpoint -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'notebooklm_mcp.openai_proxy'"

**Step 3: Write minimal implementation**

```python
# src/notebooklm_mcp/openai_proxy.py
"""OpenAI-compatible proxy server for NotebookLM."""

from fastapi import FastAPI

app = FastAPI(
    title="NotebookLM OpenAI Proxy",
    description="OpenAI-compatible API for NotebookLM",
    version="0.1.0"
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_proxy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_proxy.py tests/test_openai_proxy.py
git commit -m "feat(openai-proxy): add FastAPI app skeleton with health endpoint"
```

---

## Task 6: Add Embeddings Stub (501 Not Implemented)

**Files:**
- Modify: `src/notebooklm_mcp/openai_proxy.py`
- Test: `tests/test_openai_proxy.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_proxy.py

def test_embeddings_returns_501():
    from notebooklm_mcp.openai_proxy import app
    client = TestClient(app)
    response = client.post("/v1/embeddings", json={"input": "test", "model": "x"})
    assert response.status_code == 501
    assert "not supported" in response.json()["detail"].lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_proxy.py::test_embeddings_returns_501 -v`
Expected: FAIL with 404 (endpoint not found)

**Step 3: Write minimal implementation**

```python
# Add to src/notebooklm_mcp/openai_proxy.py
from fastapi import HTTPException


@app.post("/v1/embeddings")
async def embeddings():
    """Embeddings endpoint - not supported by NotebookLM."""
    raise HTTPException(
        status_code=501,
        detail="Embeddings not supported. NotebookLM does not provide embedding generation."
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_proxy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_proxy.py tests/test_openai_proxy.py
git commit -m "feat(openai-proxy): add embeddings stub returning 501"
```

---

## Task 7: Add Models List Endpoint

**Files:**
- Modify: `src/notebooklm_mcp/openai_proxy.py`
- Test: `tests/test_openai_proxy.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_proxy.py
from unittest.mock import AsyncMock, patch, MagicMock


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
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "nb-123"
        assert data["data"][0]["owned_by"] == "notebooklm"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_proxy.py::test_models_list_returns_notebooks -v`
Expected: FAIL with 404 (endpoint not found)

**Step 3: Write minimal implementation**

```python
# Add to src/notebooklm_mcp/openai_proxy.py
from .api_client import NotebookLMClient
from .auth import load_cached_tokens


async def get_client() -> NotebookLMClient:
    """Get authenticated NotebookLM client."""
    tokens = load_cached_tokens()
    if not tokens or not tokens.cookies:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Run 'notebooklm-mcp-auth' first."
        )
    client = NotebookLMClient(
        cookies=tokens.cookies,
        csrf_token=tokens.csrf_token or "",
        session_id=tokens.session_id or ""
    )
    await client._ensure_initialized()
    return client


@app.get("/v1/models")
async def list_models():
    """List notebooks as available models."""
    client = await get_client()
    try:
        notebooks = await client.list_notebooks()
        return {
            "object": "list",
            "data": [
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
        }
    finally:
        await client.close()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_proxy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_proxy.py tests/test_openai_proxy.py
git commit -m "feat(openai-proxy): add /v1/models endpoint listing notebooks"
```

---

## Task 8: Add Chat Completions Endpoint (Non-Streaming)

**Files:**
- Modify: `src/notebooklm_mcp/openai_proxy.py`
- Test: `tests/test_openai_proxy.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_proxy.py

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_proxy.py::test_chat_completions_non_streaming -v`
Expected: FAIL with 404 (endpoint not found)

**Step 3: Write minimal implementation**

```python
# Add to src/notebooklm_mcp/openai_proxy.py
import time
import uuid

from .openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseChoice,
    ResponseMessage,
    Usage,
)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    # Extract last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    query_text = user_messages[-1].content

    client = await get_client()
    try:
        if request.stream:
            # Streaming handled in next task
            raise HTTPException(status_code=501, detail="Streaming not yet implemented")

        # Non-streaming: use query() method
        result = await client.query(
            notebook_id=request.model,
            query_text=query_text,
            conversation_id=request.conversation_id,
        )

        answer = result.get("answer", "") if result else ""
        conv_id = result.get("conversation_id", "") if result else ""

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ResponseChoice(
                index=0,
                message=ResponseMessage(role="assistant", content=answer),
                finish_reason="stop"
            )],
            usage=Usage(prompt_tokens=len(query_text), completion_tokens=len(answer), total_tokens=len(query_text) + len(answer)),
            system_fingerprint=f"conv_{conv_id}" if conv_id else None
        )
    finally:
        await client.close()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_proxy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_proxy.py tests/test_openai_proxy.py
git commit -m "feat(openai-proxy): add /v1/chat/completions non-streaming"
```

---

## Task 9: Add Chat Completions Streaming Support

**Files:**
- Modify: `src/notebooklm_mcp/openai_proxy.py`
- Test: `tests/test_openai_proxy.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_proxy.py

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_proxy.py::test_chat_completions_streaming -v`
Expected: FAIL with 501 "Streaming not yet implemented"

**Step 3: Write minimal implementation**

```python
# Add imports at top of src/notebooklm_mcp/openai_proxy.py
from fastapi.responses import StreamingResponse

from .openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    DeltaContent,
    ResponseChoice,
    ResponseMessage,
    Usage,
)


# Replace the streaming placeholder in chat_completions with:
async def stream_response(client, notebook_id: str, query_text: str, request: ChatCompletionRequest):
    """Generate OpenAI-compatible SSE stream from NotebookLM query_stream."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    conversation_id = None

    try:
        async for chunk in client.query_stream(
            notebook_id=notebook_id,
            query_text=query_text,
            conversation_id=request.conversation_id
        ):
            # Filter thinking unless requested
            if chunk["type"] == "thinking" and not request.include_thinking:
                continue

            conversation_id = chunk.get("conversation_id", conversation_id)

            openai_chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=notebook_id,
                choices=[Choice(delta=DeltaContent(content=chunk["text"]))],
                system_fingerprint=f"conv_{conversation_id}" if conversation_id else None
            )
            yield f"data: {openai_chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=notebook_id,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await client.close()


# Update chat_completions to use streaming:
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    query_text = user_messages[-1].content

    client = await get_client()

    if request.stream:
        return StreamingResponse(
            stream_response(client, request.model, query_text, request),
            media_type="text/event-stream"
        )

    # Non-streaming path (existing code)
    try:
        result = await client.query(
            notebook_id=request.model,
            query_text=query_text,
            conversation_id=request.conversation_id,
        )

        answer = result.get("answer", "") if result else ""
        conv_id = result.get("conversation_id", "") if result else ""

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ResponseChoice(
                index=0,
                message=ResponseMessage(role="assistant", content=answer),
                finish_reason="stop"
            )],
            usage=Usage(prompt_tokens=len(query_text), completion_tokens=len(answer), total_tokens=len(query_text) + len(answer)),
            system_fingerprint=f"conv_{conv_id}" if conv_id else None
        )
    finally:
        await client.close()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_proxy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_proxy.py tests/test_openai_proxy.py
git commit -m "feat(openai-proxy): add streaming support for chat completions"
```

---

## Task 10: Add CLI Entry Point

**Files:**
- Modify: `src/notebooklm_mcp/openai_proxy.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_proxy.py
import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "notebooklm_mcp.openai_proxy", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "--port" in result.stdout
    assert "--host" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_proxy.py::test_cli_help -v`
Expected: FAIL (module not runnable)

**Step 3: Write minimal implementation**

```python
# Add to end of src/notebooklm_mcp/openai_proxy.py
import argparse


def main():
    """CLI entry point for OpenAI-compatible proxy."""
    parser = argparse.ArgumentParser(description="NotebookLM OpenAI-compatible proxy server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

Update pyproject.toml:

```toml
[project.scripts]
notebooklm-mcp = "notebooklm_mcp.server:main"
notebooklm-mcp-auth = "notebooklm_mcp.auth_cli:main"
notebooklm-openai = "notebooklm_mcp.openai_proxy:main"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_proxy.py::test_cli_help -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/notebooklm_mcp/openai_proxy.py pyproject.toml
git commit -m "feat(openai-proxy): add CLI entry point with --host and --port"
```

---

## Task 11: Add Include Thinking Test

**Files:**
- Test: `tests/test_openai_proxy.py`

**Step 1: Write the test for include_thinking**

```python
# Add to tests/test_openai_proxy.py

def test_chat_completions_streaming_with_thinking():
    from notebooklm_mcp.openai_proxy import app

    async def mock_stream():
        yield {"type": "thinking", "text": "Reading sources...", "conversation_id": "conv-123"}
        yield {"type": "answer", "text": "42", "conversation_id": "conv-123"}

    with patch("notebooklm_mcp.openai_proxy.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.query_stream = MagicMock(return_value=mock_stream())
        mock_client.close = AsyncMock()
        mock_get_client.return_value = mock_client

        client = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "model": "nb-123",
            "messages": [{"role": "user", "content": "What?"}],
            "stream": True,
            "include_thinking": True  # Should include thinking chunks
        })

        assert response.status_code == 200
        # Thinking chunk should be included
        assert "Reading sources" in response.text
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_proxy.py::test_chat_completions_streaming_with_thinking -v`
Expected: PASS (already implemented)

**Step 3: Commit**

```bash
git add tests/test_openai_proxy.py
git commit -m "test(openai-proxy): add test for include_thinking parameter"
```

---

## Task 12: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add OpenAI Proxy section to CLAUDE.md**

Add this section after the existing documentation:

```markdown
## OpenAI-Compatible Proxy

An OpenAI-compatible proxy server that allows connecting any OpenAI client to NotebookLM.

### Usage

```bash
# Start the proxy server
notebooklm-openai --port 8080

# Or with custom host
notebooklm-openai --host 127.0.0.1 --port 8000
```

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Chat with NotebookLM (streaming + non-streaming) |
| `GET /v1/models` | List notebooks as available models |
| `GET /v1/embeddings` | Returns 501 (not supported) |
| `GET /health` | Health check |

### Example: OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

# List available notebooks
models = client.models.list()
for model in models:
    print(f"{model.id}: {model.name}")

# Chat with a notebook
response = client.chat.completions.create(
    model="<notebook-uuid>",  # Use notebook ID from models list
    messages=[{"role": "user", "content": "Summarize the key points"}],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Custom Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | string | null | For multi-turn conversations |
| `include_thinking` | bool | false | Include NotebookLM's thinking steps |

Pass via `extra_body`:
```python
response = client.chat.completions.create(
    model="notebook-id",
    messages=[...],
    extra_body={"conversation_id": "prev-conv-id", "include_thinking": True}
)
```
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add OpenAI-compatible proxy documentation"
```

---

## Task 13: Final Integration Test

**Step 1: Reinstall package**

```bash
uv cache clean && uv tool install --force .
```

**Step 2: Run all tests**

```bash
uv run pytest tests/test_openai_types.py tests/test_openai_proxy.py -v
```
Expected: All PASS

**Step 3: Manual verification (requires auth)**

```bash
# Start proxy
notebooklm-openai --port 8080 &

# Test health
curl http://localhost:8080/health

# Test models list
curl http://localhost:8080/v1/models

# Test chat (replace with real notebook ID)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"YOUR-NOTEBOOK-ID","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(openai-proxy): complete OpenAI-compatible proxy implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Message model | openai_types.py |
| 2 | ChatCompletionRequest model | openai_types.py |
| 3 | Response models (Chunk, Choice) | openai_types.py |
| 4 | Non-streaming response models | openai_types.py |
| 5 | FastAPI app + health endpoint | openai_proxy.py |
| 6 | Embeddings stub (501) | openai_proxy.py |
| 7 | Models list endpoint | openai_proxy.py |
| 8 | Chat completions (non-streaming) | openai_proxy.py |
| 9 | Chat completions (streaming) | openai_proxy.py |
| 10 | CLI entry point | openai_proxy.py, pyproject.toml |
| 11 | Include thinking test | tests |
| 12 | Documentation | CLAUDE.md |
| 13 | Integration test | - |
