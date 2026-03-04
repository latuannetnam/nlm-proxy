# Stage 6: Rewire OpenAI Proxy Server

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the per-request `SmartRouter` pattern in `server.py` with a singleton `AgentCore` and rewrite `handle_smart_routing()` to use the four-phase pipeline.

**Architecture:** `AgentCore` is created once in `main()` and stored in `app.state.agent_core`. `handle_smart_routing()` calls `agent_core.route()` for Phase 0+1, then does Phase 2 (post-routing cache) and Phase 3 (streaming/non-streaming response) inline. No per-request `SmartRouter()` creation or `router.close()`.

**Inputs:** All stages 0-5 complete. Specifically:
- Stage 0: Cache lookups return `(result, hit_type)` tuples
- Stage 1: `LangChainLLMClient` and `create_chat_model()` exist
- Stage 5: `AgentCore`, `RequestOptions`, `RoutingDecision` exist

**Outputs:** `server.py` uses `AgentCore` singleton. Streaming, non-streaming, cache, session, tracing all work.

> [!CAUTION]
> This is the **highest-risk stage** (🔴 High). Test every task incrementally. Do NOT skip commits.

---

## Task 6.1: Update `main()` — singleton AgentCore

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Add `app.state.agent_core` initialization**

Add after existing `app.state` declarations (~line 48):
```python
app.state.agent_core = None
```

**Step 2: Update imports**

Replace:
```python
from nlm_proxy.openai.router import SmartRouter, RequestType
```
Add:
```python
from nlm_proxy.core.agent import AgentCore, RequestOptions, RoutingDecision
from nlm_proxy.core.llm_client import LangChainLLMClient, create_chat_model
from nlm_proxy.core.config import get_agent_settings
```

**Step 3: Update `main()` initialization**

In `main()`, find where `ExternalLLMClient` / `SmartRouter` dependencies are set up. Replace the LLM client creation with:

```python
# Create shared ChatModel (used for routing, L3 verification, and LLM_TASK)
agent_settings = get_agent_settings()
chat_model = create_chat_model(
    model=routing_settings.llm_model,
    provider=agent_settings.llm_provider,
    base_url=routing_settings.llm_base_url,
    api_key=routing_settings.llm_api_key,
)
llm_client = LangChainLLMClient(chat_model=chat_model)
```

After the notebook cache and response cache are created, add:

```python
# Create AgentCore singleton (replaces per-request SmartRouter)
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

**Step 4: Add AgentCore teardown in `main()` finally block**

In `main()`, add cleanup logic to the finally block (or signal handler):

```python
# In main() finally block:
try:
    ...
finally:
    # Graceful shutdown — clean up AgentCore resources
    if app.state.agent_core and hasattr(app.state.agent_core, 'chat_model'):
        logger.info("Shutting down AgentCore...")
        # LangChain ChatModel and HuggingFace embeddings may hold connections
        # No explicit close needed for most providers, but log for observability
    logger.info("AgentCore shutdown complete")
```

> [!NOTE]
> Most LangChain providers (OpenAI, Anthropic) don't require explicit teardown. The
> `HuggingFaceEmbeddings` model is loaded in-process and GC'd automatically.
> This step is primarily for logging and future extensibility.

**Step 5: Run tests**

Run: `uv run pytest -v`
Expected: PASS — existing code paths still work since we haven't changed `handle_smart_routing` yet.

**Step 6: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "refactor: initialize AgentCore singleton in main() with teardown"
```

---

## Task 6.2: Rewrite `handle_smart_routing()` — four-phase pipeline

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Replace `handle_smart_routing()` entirely**

Replace the current function (lines ~290-660) with:

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
    chat_id_source = "header" if chat_id else None
    if not chat_id and hasattr(request, 'metadata') and request.metadata:
        chat_id = request.metadata.get("chat_id")
        if chat_id:
            chat_id_source = "metadata"

    # Extract allowed_notebooks from request metadata (per-request ACL)
    request_allowed_notebooks = None
    if hasattr(request, 'metadata') and request.metadata:
        raw = request.metadata.get("allowed_notebooks")
        if raw is not None:
            if raw == ["*"]:
                request_allowed_notebooks = None
                logger.debug("[SMART-ROUTER] ACL wildcard ['*'] - all notebooks allowed")
            elif raw == []:
                request_allowed_notebooks = []
                logger.debug("[SMART-ROUTER] ACL empty list [] - no notebooks allowed")
            else:
                request_allowed_notebooks = raw
                logger.debug("[SMART-ROUTER] ACL filter: %d allowed notebooks", len(raw))
        else:
            logger.debug("[SMART-ROUTER] No ACL metadata - all notebooks allowed")
    else:
        logger.debug("[SMART-ROUTER] No metadata - all notebooks allowed")

    # Load conversation_id from session store
    conversation_id = None
    if chat_id and app.state.session_store:
        stored_conv_id = app.state.session_store.get(chat_id)
        if stored_conv_id:
            logger.info("session_lookup: chat_id=%s, conversation_id=%s, source=%s", chat_id, stored_conv_id, chat_id_source)
            request.conversation_id = stored_conv_id
            conversation_id = stored_conv_id
        else:
            logger.info("session_lookup: chat_id=%s, conversation_id=None, source=%s", chat_id, chat_id_source)

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    query = user_messages[-1].content

    # Build RequestOptions
    options = RequestOptions(
        bypass_cache=request.bypass_cache,
        include_thinking=request.include_thinking,
        allowed_notebooks=request_allowed_notebooks,
        conversation_id=conversation_id,
        chat_id=chat_id,
    )

    # Phase 0+1: Route (includes pre-routing cache check)
    decision = await agent_core.route(query, options)

    # Phase 0 hit → return cached response
    if decision.cache_result:
        if request.stream:
            return StreamingResponse(
                _stream_cached_response(decision, request),
                media_type="text/event-stream",
                headers={"X-Cache-Status": f"HIT_{decision.cache_hit_type.upper()}"},
            )
        else:
            return _json_cached_response(decision, request)

    logger.info("[SMART-ROUTER] Decision: %s, notebook=%s", decision.request_type, decision.notebook_id)

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
                    _stream_cached_response(decision, request),
                    media_type="text/event-stream",
                    headers={"X-Cache-Status": f"HIT_{hit_type.upper()}"},
                )
            else:
                return _json_cached_response(decision, request)

    # Phase 3: Execute query (streaming or non-streaming)
    if request.stream:
        return StreamingResponse(
            stream_smart_response(agent_core, decision, query, request, chat_id, tracing_settings),
            media_type="text/event-stream",
        )
    else:
        return await _handle_non_streaming(agent_core, decision, query, request, chat_id, tracing_settings)
```

**Step 2: Add `_stream_cached_response()` helper**

```python
async def _stream_cached_response(decision: RoutingDecision, request: ChatCompletionRequest):
    """Stream a cached response as SSE (used by both Phase 0 and Phase 2 cache hits)."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    cache_result = decision.cache_result
    hit_type = decision.cache_hit_type or "exact"

    # Reasoning chunk
    reasoning = decision.reasoning or "Cache hit — returning cached response."
    reasoning_chunk = ChatCompletionChunk(
        id=chunk_id, created=created, model=request.model,
        choices=[Choice(delta=DeltaContent(reasoning_content=reasoning + "\n\n"))],
    )
    yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

    # Thinking chunk (if available and requested)
    if cache_result.thinking and request.include_thinking:
        thinking_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(reasoning_content=cache_result.thinking))],
            system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
        )
        yield f"data: {thinking_chunk.model_dump_json()}\n\n"

    # Answer chunk
    answer_chunk = ChatCompletionChunk(
        id=chunk_id, created=created, model=request.model,
        choices=[Choice(delta=DeltaContent(content=cache_result.answer))],
        system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
    )
    yield f"data: {answer_chunk.model_dump_json()}\n\n"

    # Final chunk
    final_chunk = ChatCompletionChunk(
        id=chunk_id, created=created, model=request.model,
        choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
        system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```

**Step 3: Add `_json_cached_response()` helper**

```python
def _json_cached_response(decision: RoutingDecision, request: ChatCompletionRequest):
    """Return a cached response as JSON (non-streaming cache hit)."""
    from fastapi.responses import JSONResponse
    cache_result = decision.cache_result
    hit_type = decision.cache_hit_type or "exact"

    resp = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ResponseChoice(
            index=0,
            message=ResponseMessage(
                role="assistant",
                content=cache_result.answer,
                reasoning_content=decision.reasoning or "Cache hit — returning cached response.",
            ),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=len(request.messages[-1].content) if request.messages else 0,
            completion_tokens=len(cache_result.answer),
            total_tokens=(len(request.messages[-1].content) if request.messages else 0) + len(cache_result.answer),
        ),
        system_fingerprint=f"cache_{hit_type}_conv_{cache_result.conversation_id}",
    )
    return JSONResponse(
        content=resp.model_dump(),
        headers={"X-Cache-Status": f"HIT_{hit_type.upper()}"},
    )
```

**Step 4: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "refactor: rewrite handle_smart_routing with four-phase pipeline"
```

---

## Task 6.3: Rewrite `stream_smart_response()` — use AgentCore

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

Replace the existing `stream_smart_response()` function entirely. Key changes:
- Accepts `agent_core: AgentCore` instead of `client` + `router: SmartRouter`
- LLM_TASK path uses `agent_core.chat_model.astream()` (yields `AIMessageChunk` with `chunk.content`)
- NOTEBOOKLM path uses `agent_core.query_stream()` (same NLM streaming as before)
- No `router.close()` or `client.close()` — singleton, no cleanup needed

```python
async def stream_smart_response(agent_core: AgentCore, decision: RoutingDecision, query: str,
                                 request: ChatCompletionRequest, chat_id: str = None,
                                 tracing_settings=None):
    """Stream response — Phase 3a of the four-phase pipeline.

    Note: This function creates its own span because the span must live for the full
    streaming duration. The parent function does NOT create a span for streaming.
    """
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        # Add user_query to span
        if tracing_settings and tracing_settings.request_max_length > 0:
            span.set_attribute("user_query", query[:tracing_settings.request_max_length])

        response_source = "llm" if decision.request_type == "llm_task" else "notebooklm"

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
            from nlm_proxy.core.llm_client import _convert_messages
            lc_messages = _convert_messages(
                [{"role": m.role, "content": m.content} for m in request.messages]
            )
            async for chunk in agent_core.chat_model.astream(lc_messages):
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
                chunk_type = chunk.get("type")
                full_text = chunk.get("text", "")

                # Extract conversation_id from first chunk
                new_conv_id = chunk.get("conversation_id")
                if new_conv_id and not conversation_id:
                    conversation_id = new_conv_id
                    logger.info("conversation_id_from_nlm: conversation_id=%s, notebook_id=%s", conversation_id, decision.notebook_id)
                    if chat_id and app.state.session_store:
                        app.state.session_store.set(chat_id, conversation_id)
                        logger.info("session_saved: chat_id=%s, conversation_id=%s", chat_id, conversation_id)

                # Filter thinking unless requested
                if chunk_type == "thinking" and not request.include_thinking:
                    previous_thinking = full_text
                    continue

                if chunk_type == "thinking":
                    delta_text = full_text[len(previous_thinking):]
                    previous_thinking = full_text
                    if delta_text:
                        delta = DeltaContent(reasoning_content=delta_text)
                        openai_chunk = ChatCompletionChunk(
                            id=chunk_id, created=created, model=request.model,
                            choices=[Choice(delta=delta)],
                            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"
                else:
                    delta_text = full_text[len(previous_answer):]
                    previous_answer = full_text
                    if delta_text:
                        accumulated_response += delta_text
                        delta = DeltaContent(content=delta_text)
                        openai_chunk = ChatCompletionChunk(
                            id=chunk_id, created=created, model=request.model,
                            choices=[Choice(delta=delta)],
                            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"

            # Store in response cache after stream completes
            if (
                agent_core.response_cache
                and accumulated_response
                and conversation_id
                and decision.notebook_id
            ):
                embedding = None
                if agent_core.response_cache._semantic_enabled:
                    emb = agent_core.response_cache._compute_embedding(query)
                    if emb is not None:
                        embedding = emb.tolist()
                agent_core.response_cache.store(
                    notebook_id=decision.notebook_id,
                    query=query,
                    answer=accumulated_response,
                    thinking=previous_thinking or None,
                    conversation_id=conversation_id,
                    embedding=embedding,
                )

        # Add response to trace BEFORE final yield (span still open)
        if tracing_settings and tracing_settings.response_max_length > 0:
            span.set_attribute("response_content", accumulated_response[:tracing_settings.response_max_length])
            span.set_attribute("response_source", response_source)

        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[Choice(delta=DeltaContent(), finish_reason="stop")],
            system_fingerprint=f"conv_{conversation_id}" if conversation_id else None,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
```

**Step 2: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "refactor: rewrite stream_smart_response to use AgentCore"
```

---

## Task 6.4: Add `_handle_non_streaming()` — non-streaming pipeline

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

```python
async def _handle_non_streaming(agent_core: AgentCore, decision: RoutingDecision, query: str,
                                 request: ChatCompletionRequest, chat_id: str = None,
                                 tracing_settings=None):
    """Phase 3b: Non-streaming response (both NOTEBOOKLM and LLM_TASK)."""
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        if tracing_settings and tracing_settings.request_max_length > 0:
            span.set_attribute("user_query", query[:tracing_settings.request_max_length])

        if decision.request_type == "llm_task":
            # LLM_TASK: invoke via LangChain ChatModel
            from nlm_proxy.core.llm_client import _convert_messages
            lc_messages = _convert_messages(
                [{"role": m.role, "content": m.content} for m in request.messages]
            )
            result = await agent_core.chat_model.ainvoke(lc_messages)
            response_text = result.content
            response_source = "llm"
        else:
            # NOTEBOOKLM: query via agent_core (uses singleton nlm_client)
            result = await agent_core.query(
                notebook_id=decision.notebook_id,
                query=query,
                conversation_id=request.conversation_id,
            )
            response_text = result.get("answer", "") if result else ""
            response_source = "notebooklm"

            # Save conversation_id to session store
            conv_id = result.get("conversation_id", "") if result else ""
            if chat_id and conv_id and app.state.session_store:
                app.state.session_store.set(chat_id, conv_id)
                logger.info("session_saved: chat_id=%s, conversation_id=%s", chat_id, conv_id)
            elif chat_id and not conv_id:
                logger.info("session_not_saved: chat_id=%s, reason=no_conversation_id_from_nlm", chat_id)

            # Store in response cache
            if agent_core.response_cache and response_text and conv_id:
                embedding = None
                if agent_core.response_cache._semantic_enabled:
                    emb = agent_core.response_cache._compute_embedding(query)
                    if emb is not None:
                        embedding = emb.tolist()
                agent_core.response_cache.store(
                    notebook_id=decision.notebook_id,
                    query=query,
                    answer=response_text,
                    thinking=None,
                    conversation_id=conv_id,
                    embedding=embedding,
                )

        # Trace response
        if tracing_settings and tracing_settings.response_max_length > 0:
            add_span_attributes(
                response_content=response_text[:tracing_settings.response_max_length],
                response_source=response_source,
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ResponseChoice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=response_text,
                    reasoning_content=decision.reasoning,
                ),
                finish_reason="stop",
            )],
            usage=Usage(
                prompt_tokens=len(query),
                completion_tokens=len(response_text),
                total_tokens=len(query) + len(response_text),
            ),
            system_fingerprint=f"conv_{request.conversation_id}" if request.conversation_id else None,
        )
```

**Step 2: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat: add _handle_non_streaming for Phase 3b pipeline"
```

---

## Task 6.5: Remove old SmartRouter usage + update `chat_completions()` direct path

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Update `chat_completions()` direct notebook path**

In `chat_completions()`, for `model != router_model_name`, replace direct cache lookups with `agent_core.handle_direct_query()`:

```python
# Replace the direct cache lookup in chat_completions()
agent_core = app.state.agent_core
if agent_core:
    options = RequestOptions(
        bypass_cache=request.bypass_cache,
        conversation_id=request.conversation_id,
    )
    cache_result, hit_type = await agent_core.handle_direct_query(
        request.model, query_text, options
    )
    if cache_result:
        # Return cached response (same format as current)
        ...
```

**Step 2: Remove old SmartRouter import**

Delete:
```python
from nlm_proxy.openai.router import SmartRouter, RequestType
```

**Step 3: Run ALL tests**

Run: `uv run pytest -v`
Expected: Some server tests may need mock updates (Task 6.6)

**Step 4: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "refactor: remove SmartRouter from server, use AgentCore for all paths"
```

---

## Task 6.6: Update server tests

**Files:**
- Modify: `tests/test_openai_module/test_server.py`
- Modify: `tests/test_openai_module/test_conversation_flow.py`

**Key mock changes (detailed):**

#### `test_server.py` — mock updates

1. **Remove per-request `SmartRouter` mock**:
   ```python
   # BEFORE: mocking SmartRouter creation in each test
   with patch("nlm_proxy.openai.server.SmartRouter") as MockRouter:
       mock_router = MockRouter.return_value
       mock_router.route.return_value = {...}
   
   # AFTER: set app.state.agent_core before tests
   from nlm_proxy.core.agent import AgentCore, RoutingDecision, RequestOptions
   mock_agent = MagicMock(spec=AgentCore)
   mock_agent.route = AsyncMock(return_value=RoutingDecision(
       request_type="notebooklm", notebook_id="nb-1", reasoning="test",
   ))
   mock_agent.response_cache = MagicMock()
   mock_agent.chat_model = AsyncMock()
   app.state.agent_core = mock_agent
   ```

2. **Replace `ExternalLLMClient` mocks → `LangChainLLMClient` mocks**:
   ```python
   # BEFORE:
   mock_llm = MagicMock(spec=ExternalLLMClient)
   mock_llm.stream.return_value = async_iter([...])
   
   # AFTER:
   mock_agent.chat_model.astream = mock_async_iter([chunk1, chunk2])
   # Where chunk.content = "text" (AIMessageChunk format)
   ```

3. **Patch streaming via `agent_core.query_stream()`**:
   ```python
   # BEFORE:
   with patch.object(client, "query_stream", return_value=async_iter([...])):
   
   # AFTER:
   mock_agent.query_stream = mock_async_generator([
       {"type": "thinking", "text": "..."},
       {"type": "answer", "text": "...", "conversation_id": "conv-1"},
   ])
   ```

4. **Remove `router.close()` assertions** — no per-request cleanup.

#### `test_conversation_flow.py` — minimal changes

- `SessionStore` is **KEPT** — no changes to session semantics
- Replace `SmartRouter` route mock → `agent_core.route()` mock
- Verify `session_store.set()` is called with `conversation_id` from NLM responses
- Verify `session_store.get()` returns stored `conversation_id` on subsequent requests

**Step 1: Run tests, fix failures iteratively**

Run: `uv run pytest tests/test_openai_module/ -v`

**Step 2: Commit**

```bash
git add tests/test_openai_module/
git commit -m "test: update server + conversation tests for AgentCore"
```

---

## Task 6.7: Add non-streaming tests

**Files:**
- Create: `tests/test_openai_module/test_non_streaming.py`

```python
"""Tests for non-streaming (stream=false) response path."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nlm_proxy.core.agent import RoutingDecision


@pytest.mark.asyncio
async def test_non_streaming_notebooklm():
    """Non-streaming NOTEBOOKLM query returns ChatCompletionResponse JSON."""
    from nlm_proxy.openai.server import _handle_non_streaming, app
    from nlm_proxy.core.agent import AgentCore
    from nlm_proxy.openai.types import ChatCompletionRequest, Message

    mock_agent = MagicMock(spec=AgentCore)
    mock_agent.query = AsyncMock(return_value={
        "answer": "The answer is 42.",
        "conversation_id": "conv-123",
    })
    mock_agent.response_cache = None  # Skip cache store
    mock_agent.chat_model = AsyncMock()

    decision = RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Selected notebook",
    )
    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="What is the meaning of life?")],
        stream=False,
    )

    app.state.session_store = MagicMock()
    response = await _handle_non_streaming(mock_agent, decision, "What is the meaning of life?", request, "chat-1")

    assert response.choices[0].message.content == "The answer is 42."
    assert response.choices[0].message.reasoning_content == "Selected notebook"


@pytest.mark.asyncio
async def test_non_streaming_llm_task():
    """Non-streaming LLM_TASK returns ChatCompletionResponse from ainvoke."""
    from nlm_proxy.openai.server import _handle_non_streaming
    from nlm_proxy.core.agent import AgentCore
    from nlm_proxy.openai.types import ChatCompletionRequest, Message

    mock_response = MagicMock()
    mock_response.content = "Here is your poem about cats..."

    mock_agent = MagicMock(spec=AgentCore)
    mock_agent.chat_model = AsyncMock()
    mock_agent.chat_model.ainvoke = AsyncMock(return_value=mock_response)
    mock_agent.response_cache = None

    decision = RoutingDecision(
        request_type="llm_task",
        reasoning="Classified as LLM task",
    )
    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="Write a poem about cats")],
        stream=False,
    )

    response = await _handle_non_streaming(mock_agent, decision, "Write a poem about cats", request)

    assert response.choices[0].message.content == "Here is your poem about cats..."
    assert "LLM task" in response.choices[0].message.reasoning_content


@pytest.mark.asyncio
async def test_non_streaming_cache_hit():
    """Non-streaming cache hit returns cached ChatCompletionResponse."""
    from nlm_proxy.openai.server import _json_cached_response
    from nlm_proxy.openai.types import ChatCompletionRequest, Message

    cached = MagicMock()
    cached.answer = "Cached answer"
    cached.conversation_id = "conv-1"

    decision = RoutingDecision(
        request_type="notebooklm",
        notebook_id="nb-1",
        reasoning="Cache hit",
        cache_result=cached,
        cache_hit_type="exact",
    )
    request = ChatCompletionRequest(
        model="knowledge-finder",
        messages=[Message(role="user", content="test")],
        stream=False,
    )

    response = _json_cached_response(decision, request)
    assert response.status_code == 200
    assert response.headers.get("X-Cache-Status") == "HIT_EXACT"
```

**Step 1: Run tests**

Run: `uv run pytest tests/test_openai_module/test_non_streaming.py -v`
Expected: ALL PASS

**Step 2: Commit**

```bash
git add tests/test_openai_module/test_non_streaming.py
git commit -m "test: add non-streaming path tests for NOTEBOOKLM, LLM_TASK, and cache hit"
```

---

## 🔒 Stage 6 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS

> [!IMPORTANT]
> **Manual smoke test recommended** before proceeding to Stage 7:
> 1. Start server: `nlm-proxy serve openai`
> 2. Send streaming query to `knowledge-finder`
> 3. Send non-streaming query (`stream=false`)
> 4. Send direct notebook query (`model=<notebook-id>`)
> 5. Verify cache hits on repeat queries
