# Trace Response Content Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LLM and NotebookLM response content to OpenTelemetry traces for debugging and analytics.

**Architecture:** Create a new `smart_router.handle_request` span that wraps the entire request/response lifecycle. Accumulate streaming responses while yielding to maintain zero client latency. Store truncated response content as span attributes.

**Tech Stack:** OpenTelemetry Python SDK, pydantic-settings, FastAPI async generators

---

## Task 1: Add request_max_length and response_max_length to TracingSettings

**Files:**
- Modify: `src/nlm_proxy/core/config.py:162-180`
- Test: `tests/test_tracing.py`

**Step 1: Write the failing tests**

Add to `tests/test_tracing.py`:

```python
def test_tracing_settings_request_max_length_default():
    """Test TracingSettings has correct default for request_max_length."""
    from nlm_proxy.core.config import TracingSettings

    settings = TracingSettings()

    assert settings.request_max_length == 500


def test_tracing_settings_request_max_length_from_env(monkeypatch):
    """Test request_max_length reads from environment."""
    monkeypatch.setenv("NLM_PROXY_OTEL_REQUEST_MAX_LENGTH", "1000")

    # Clear singleton cache
    import nlm_proxy.core.config as config_module
    config_module._tracing = None

    from nlm_proxy.core.config import get_tracing_settings
    settings = get_tracing_settings()

    assert settings.request_max_length == 1000


def test_tracing_settings_response_max_length_default():
    """Test TracingSettings has correct default for response_max_length."""
    from nlm_proxy.core.config import TracingSettings

    settings = TracingSettings()

    assert settings.response_max_length == 1000


def test_tracing_settings_response_max_length_from_env(monkeypatch):
    """Test response_max_length reads from environment."""
    monkeypatch.setenv("NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH", "5000")

    # Clear singleton cache
    import nlm_proxy.core.config as config_module
    config_module._tracing = None

    from nlm_proxy.core.config import get_tracing_settings
    settings = get_tracing_settings()

    assert settings.response_max_length == 5000


def test_tracing_settings_response_max_length_disabled(monkeypatch):
    """Test response_max_length can be set to 0 to disable."""
    monkeypatch.setenv("NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH", "0")

    # Clear singleton cache
    import nlm_proxy.core.config as config_module
    config_module._tracing = None

    from nlm_proxy.core.config import get_tracing_settings
    settings = get_tracing_settings()

    assert settings.response_max_length == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_tracing_settings_request_max_length_default -v`

Expected: FAIL with `AttributeError: 'TracingSettings' object has no attribute 'request_max_length'`

**Step 3: Write minimal implementation**

In `src/nlm_proxy/core/config.py`, update `TracingSettings` class (around line 162):

```python
class TracingSettings(BaseSettings):
    """OpenTelemetry tracing configuration."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP collector endpoint (gRPC)"
    )
    service_name: str = Field(
        default="nlm-proxy",
        description="Service name in traces"
    )
    request_max_length: int = Field(
        default=500,
        description="Max chars of user query to store in trace (0 to disable)"
    )
    response_max_length: int = Field(
        default=1000,
        description="Max chars of response to store in trace (0 to disable)"
    )

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_OTEL_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing.py -v -k "max_length"`

Expected: All 5 new tests PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/config.py tests/test_tracing.py
git commit -m "feat(tracing): add request_max_length and response_max_length settings"
```

---

## Task 2: Remove user_query from router.route() span

**Files:**
- Modify: `src/nlm_proxy/openai/router.py:165-170`
- Test: `tests/test_openai_module/test_router.py`

**Step 1: Identify the line to remove**

In `src/nlm_proxy/openai/router.py`, the `route()` method has:

```python
@record_span("smart_router.route")
async def route(self, query: str) -> RoutingDecision:
    """Classify and route the request."""
    logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")
    add_span_attributes(user_query=query[:500])  # <-- REMOVE THIS LINE
```

**Step 2: Remove the user_query attribute**

Update `src/nlm_proxy/openai/router.py` route method:

```python
@record_span("smart_router.route")
async def route(self, query: str) -> RoutingDecision:
    """Classify and route the request."""
    logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")
    # user_query attribute moved to smart_router.handle_request span

    request_type = await self.classify_request(query)
```

**Step 3: Run existing router tests to ensure no breakage**

Run: `uv run pytest tests/test_openai_module/test_router.py -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/nlm_proxy/openai/router.py
git commit -m "refactor(tracing): remove user_query from route span (moved to handle_request)"
```

---

## Task 3: Add handle_request span to non-streaming path

**Files:**
- Modify: `src/nlm_proxy/openai/server.py:234-330`
- Test: Manual integration test

**Step 1: Add imports at top of server.py**

Add to imports section (around line 16):

```python
from nlm_proxy.core.tracing import init_tracing, shutdown_tracing, instrument_fastapi, instrument_httpx, get_tracer, add_span_attributes, get_tracing_settings
```

**Step 2: Wrap handle_smart_routing with span and add response tracing**

Replace the `handle_smart_routing` function (lines 234-330) with:

```python
async def handle_smart_routing(request: ChatCompletionRequest, http_request: Request):
    """Handle requests to the smart router model."""
    routing_settings = get_routing_settings()
    tracing_settings = get_tracing_settings()
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        client = await get_client()

        # Use shared notebook cache from app.state
        if not app.state.notebook_cache:
            raise HTTPException(
                status_code=503,
                detail="Notebook cache not initialized. Smart routing is not available."
            )

        router = SmartRouter(
            nlm_client=client,
            notebook_cache=app.state.notebook_cache,
            llm_base_url=routing_settings.llm_base_url,
            llm_api_key=routing_settings.llm_api_key,
            llm_model=routing_settings.llm_model,
            allowed_notebooks=routing_settings.allowed_notebooks
        )

        # Extract chat_id from headers or request metadata
        chat_id = http_request.headers.get("X-OpenWebUI-Chat-Id")
        if not chat_id and hasattr(request, 'metadata') and request.metadata:
            chat_id = request.metadata.get("chat_id")

        logger.debug(f"[SMART-ROUTER] Extracted chat_id: {chat_id}")

        # Load existing conversation_id from session store if chat_id exists
        if chat_id and app.state.session_store:
            stored_conv_id = app.state.session_store.get(chat_id)
            if stored_conv_id:
                logger.info(f"[SMART-ROUTER] Reusing conversation: chat_id={chat_id}, conversation_id={stored_conv_id}")
                request.conversation_id = stored_conv_id
            else:
                logger.info(f"[SMART-ROUTER] New conversation for chat_id={chat_id}")

        try:
            user_messages = [m for m in request.messages if m.role == "user"]
            if not user_messages:
                raise HTTPException(status_code=400, detail="No user message found")

            query = user_messages[-1].content

            # Add user_query to span (using configurable max length)
            if tracing_settings.request_max_length > 0:
                add_span_attributes(user_query=query[:tracing_settings.request_max_length])

            decision = await router.route(query)

            logger.info(f"[SMART-ROUTER] Decision: {decision.request_type.value}, notebook={decision.notebook_id}")

            if request.stream:
                # Pass tracing settings to streaming function
                return StreamingResponse(
                    stream_smart_response(client, router, decision, query, request, chat_id, tracing_settings),
                    media_type="text/event-stream"
                )

            # Non-streaming path
            if decision.request_type == RequestType.LLM_TASK:
                # Call external LLM
                response_text = await router.llm_client.complete(
                    query,
                    max_tokens=4096
                )
                response_source = "llm"
            else:
                # Call NotebookLM with conversation_id for continuity
                result = await client.query(
                    notebook_id=decision.notebook_id,
                    query_text=query,
                    conversation_id=request.conversation_id
                )
                response_text = result.get("answer", "") if result else ""
                response_source = "notebooklm"

                # Save conversation_id to session store if we have a chat_id
                conv_id = result.get("conversation_id", "") if result else ""
                if chat_id and conv_id and app.state.session_store:
                    app.state.session_store.set(chat_id, conv_id)
                    logger.debug(f"[SMART-ROUTER] Saved session: chat_id={chat_id}, conversation_id={conv_id}")

            # Add response to trace
            if tracing_settings.response_max_length > 0:
                add_span_attributes(
                    response_content=response_text[:tracing_settings.response_max_length],
                    response_source=response_source
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

**Step 3: Update imports if needed**

Ensure `get_tracer` and `get_tracing_settings` are imported from tracing module.

**Step 4: Run the server manually to verify no import errors**

Run: `uv run python -c "from nlm_proxy.openai.server import app; print('OK')"`

Expected: Prints `OK` with no errors

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(tracing): add handle_request span with response tracing (non-streaming)"
```

---

## Task 4: Add response tracing to streaming path

**Files:**
- Modify: `src/nlm_proxy/openai/server.py:136-232`

**Step 1: Update stream_smart_response signature and add tracing**

Replace the `stream_smart_response` function with:

```python
async def stream_smart_response(client, router: SmartRouter, decision, query: str, request: ChatCompletionRequest, chat_id: str = None, tracing_settings=None):
    """Stream response with routing reasoning as reasoning_content and response tracing."""
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        # Add user_query to span (using configurable max length)
        if tracing_settings and tracing_settings.request_max_length > 0:
            span.set_attribute("user_query", query[:tracing_settings.request_max_length])

        # Determine response source
        response_source = "llm" if decision.request_type == RequestType.LLM_TASK else "notebooklm"

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        conversation_id = None
        accumulated_response = ""

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
            stream = await router.llm_client.stream(messages)
            async for chunk in stream:
                # Safety check: some chunks may have empty choices array
                if not chunk.choices:
                    continue

                # Safety check: delta and content may be None
                delta = chunk.choices[0].delta
                delta_content = delta.content if delta and delta.content else ""

                if delta_content:
                    accumulated_response += delta_content  # Accumulate for tracing
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
                query_text=query,
                conversation_id=request.conversation_id
            ):
                chunk_type = chunk.get("type")
                full_text = chunk.get("text", "")

                # Extract conversation_id from first chunk
                new_conv_id = chunk.get("conversation_id")
                if new_conv_id and not conversation_id:
                    conversation_id = new_conv_id
                    # Save to session store if we have a chat_id
                    if chat_id and app.state.session_store:
                        app.state.session_store.set(chat_id, conversation_id)
                        logger.debug(f"[SMART-ROUTER] Saved session: chat_id={chat_id}, conversation_id={conversation_id}")

                if chunk_type == "thinking" and not request.include_thinking:
                    previous_thinking = full_text
                    continue

                if chunk_type == "thinking":
                    delta_text = full_text[len(previous_thinking):]
                    previous_thinking = full_text
                    if delta_text:
                        delta = DeltaContent(reasoning_content=delta_text)
                        openai_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=request.model,
                            choices=[Choice(delta=delta)]
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"
                else:
                    delta_text = full_text[len(previous_answer):]
                    previous_answer = full_text
                    if delta_text:
                        accumulated_response += delta_text  # Accumulate for tracing
                        delta = DeltaContent(content=delta_text)
                        openai_chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=request.model,
                            choices=[Choice(delta=delta)]
                        )
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"

        # Add response to trace BEFORE final yield
        if tracing_settings and tracing_settings.response_max_length > 0:
            span.set_attribute("response_content", accumulated_response[:tracing_settings.response_max_length])
            span.set_attribute("response_source", response_source)

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

**Step 2: Run import check**

Run: `uv run python -c "from nlm_proxy.openai.server import stream_smart_response; print('OK')"`

Expected: Prints `OK`

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(tracing): add response tracing to streaming path"
```

---

## Task 5: Update Grafana dashboard queries

**Files:**
- Modify: `docker/grafana/provisioning/dashboards/routing-analytics.json`

**Step 1: Update Total Requests query (panel id 1)**

Find `"id": 1` and update `rawSql`:

```json
"rawSql": "SELECT count() as value\nFROM nlm_traces.otel_traces\nWHERE SpanName = 'smart_router.handle_request'\n  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)"
```

**Step 2: Update Average Request Time query (panel id 2)**

Find `"id": 2` and update `rawSql` and `title`:

```json
"rawSql": "SELECT round(avg(Duration)/1000000, 2) as value\nFROM nlm_traces.otel_traces\nWHERE SpanName = 'smart_router.handle_request'\n  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)"
```

Also update `"title": "Average Request Time"` (was "Average Routing Time")

**Step 3: Update Error Rate query (panel id 3)**

Find `"id": 3` and update `rawSql`:

```json
"rawSql": "SELECT\n  round(countIf(StatusCode = 'ERROR') * 100.0 / count(), 2) as value\nFROM nlm_traces.otel_traces\nWHERE SpanName = 'smart_router.handle_request'\n  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)"
```

**Step 4: Update P95 Latency query (panel id 4)**

Find `"id": 4` and update `rawSql`:

```json
"rawSql": "SELECT round(quantile(0.95)(Duration)/1000000, 2) as value\nFROM nlm_traces.otel_traces\nWHERE SpanName = 'smart_router.handle_request'\n  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)"
```

**Step 5: Update Request Volume Over Time query (panel id 5)**

Find `"id": 5` and update `rawSql`:

```json
"rawSql": "SELECT\n  toStartOfMinute(Timestamp) as time,\n  count() as requests\nFROM nlm_traces.otel_traces\nWHERE SpanName = 'smart_router.handle_request'\n  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)\nGROUP BY time\nORDER BY time"
```

**Step 6: Update Recent Requests query (panel id 8)**

Find `"id": 8` and update `rawSql`:

```json
"rawSql": "SELECT\n  formatDateTime(\n    argMax(Timestamp, SpanName = 'smart_router.handle_request'),\n    '%Y-%m-%d %H:%i:%S'\n  ) as Time,\n  substring(\n    argMax(SpanAttributes['user_query'], SpanName = 'smart_router.handle_request'),\n    1, 50\n  ) as Query,\n  argMax(SpanAttributes['classification_result'], SpanName = 'smart_router.classify') as Classification,\n  argMax(SpanAttributes['selected_notebook_title'], SpanName = 'smart_router.select_notebook') as Notebook,\n  argMax(SpanAttributes['response_source'], SpanName = 'smart_router.handle_request') as Source,\n  substring(\n    argMax(SpanAttributes['response_content'], SpanName = 'smart_router.handle_request'),\n    1, 100\n  ) as Response_Preview,\n  round(\n    argMax(Duration, SpanName = 'smart_router.handle_request') / 1000000,\n    2\n  ) as Duration_ms,\n  argMax(StatusCode, SpanName = 'smart_router.handle_request') as Status,\n  TraceId\nFROM nlm_traces.otel_traces\nWHERE SpanName LIKE 'smart_router%'\n  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)\nGROUP BY TraceId\nORDER BY Time DESC\nLIMIT 20"
```

**Step 7: Commit**

```bash
git add docker/grafana/provisioning/dashboards/routing-analytics.json
git commit -m "feat(dashboard): update queries for handle_request span and add response columns"
```

---

## Task 6: Update TRACING.md documentation

**Files:**
- Modify: `docs/TRACING.md`

**Step 1: Add new configuration option to Configuration section**

After line 108 (after `NLM_PROXY_OTEL_SERVICE_NAME`), add:

```markdown
| `NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH` | `1000` | Max chars of response to store in trace (0 to disable) |
```

**Step 2: Update Span Hierarchy section**

Update the Understanding Traces section to show new hierarchy:

```markdown
### Span Hierarchy

Each request creates a trace with nested spans:

```
smart_router.handle_request (parent - full request lifecycle)
├── user_query
├── response_content
├── response_source
│
└── smart_router.route (child - routing decision)
    ├── smart_router.classify (grandchild)
    └── smart_router.select_notebook (grandchild, if NotebookLM)
```
```

**Step 3: Add new attributes to smart_router.handle_request table**

Add new section after smart_router.route attributes:

```markdown
#### smart_router.handle_request
| Attribute | Type | Description |
|-----------|------|-------------|
| `user_query` | string | User's query (truncated to 500 chars) |
| `response_content` | string | Response text (truncated per `RESPONSE_MAX_LENGTH`) |
| `response_source` | string | "llm" or "notebooklm" |
```

**Step 4: Commit**

```bash
git add docs/TRACING.md
git commit -m "docs: update TRACING.md with response tracing configuration"
```

---

## Task 7: Run full test suite

**Step 1: Run all tests**

Run: `uv run pytest -v`

Expected: All tests PASS

**Step 2: If any failures, fix and re-run**

**Step 3: Final commit if any fixes were needed**

---

## Task 8: Integration test with live tracing

**Step 1: Start tracing infrastructure**

Run: `docker compose -f docker-compose.otel.yml up -d`

**Step 2: Start the proxy with tracing enabled**

```bash
export NLM_PROXY_OTEL_ENABLED=true
export NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH=1000
uv run nlm-proxy serve openai --port 8080
```

**Step 3: Send a test request**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NLM_PROXY_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "knowledge-finder",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }'
```

**Step 4: Verify trace in ClickHouse**

```bash
docker exec nlm-clickhouse clickhouse-client --query "
SELECT
  SpanName,
  SpanAttributes['user_query'] as query,
  SpanAttributes['response_content'] as response,
  SpanAttributes['response_source'] as source
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.handle_request'
ORDER BY Timestamp DESC
LIMIT 1
FORMAT Pretty
"
```

Expected: Shows `response_content` and `response_source` populated

**Step 5: Verify Grafana dashboard**

Open: `http://localhost:3000`

Navigate to: **Dashboards > NLM Proxy - Routing Analytics**

Verify:
- Recent Requests table shows Source and Response_Preview columns
- All stat panels show data

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat(tracing): complete response content tracing implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add response_max_length setting | config.py, test_tracing.py |
| 2 | Remove user_query from route span | router.py |
| 3 | Add handle_request span (non-streaming) | server.py |
| 4 | Add response tracing (streaming) | server.py |
| 5 | Update Grafana dashboard queries | routing-analytics.json |
| 6 | Update documentation | TRACING.md |
| 7 | Run full test suite | - |
| 8 | Integration test | - |

**Total estimated time:** 45-60 minutes
