# Design: Store LLM and NotebookLM Responses in Traces

## Overview

Add response content tracing to capture LLM and NotebookLM responses in OpenTelemetry spans for debugging and analytics.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Response length | Configurable via env var, default 1000 chars | Balance storage vs. detail |
| Span location | New `smart_router.handle_request` span | Wraps entire request/response lifecycle |
| Streaming | Accumulate while yielding (zero latency) | No impact on client experience |
| Attributes | `response_content` + `response_source` | Flexible filtering and analytics |

## Configuration

### New Environment Variable

```bash
NLM_PROXY_TRACE_RESPONSE_MAX_LENGTH=1000  # Default: store first 1000 chars
NLM_PROXY_TRACE_RESPONSE_MAX_LENGTH=5000  # For debugging: more detail
NLM_PROXY_TRACE_RESPONSE_MAX_LENGTH=0     # Disable response tracing
```

### Updated TracingSettings

**File:** `src/nlm_proxy/core/config.py`

```python
class TracingSettings(BaseSettings):
    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    service_name: str = "nlm-proxy"
    response_max_length: int = 1000  # NEW

    model_config = SettingsConfigDict(env_prefix="NLM_PROXY_OTEL_")
```

Note: Use `NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH` as the env var (follows existing prefix pattern).

## New Span Hierarchy

```
smart_router.handle_request (NEW - wraps entire request/response)
├── user_query              ← Moved from route
├── response_content        ← NEW
├── response_source         ← NEW ("llm" or "notebooklm")
│
└── smart_router.route (child - routing decision only)
    ├── request_type
    ├── notebook_id
    ├── routing_reasoning
    │
    ├── smart_router.classify (grandchild)
    │   ├── classification_result
    │   └── llm_model
    │
    └── smart_router.select_notebook (grandchild)
        ├── selected_notebook_id
        ├── selected_notebook_title
        └── candidates_count
```

## Implementation

### 1. Non-Streaming Path

**File:** `src/nlm_proxy/openai/server.py`

Wrap `handle_smart_routing()` with a new span and add response attributes:

```python
from nlm_proxy.core.tracing import get_tracer, add_span_attributes, get_tracing_settings

async def handle_smart_routing(request: ChatCompletionRequest, http_request: Request):
    """Handle requests to the smart router model."""
    tracer = get_tracer(__name__)
    settings = get_tracing_settings()

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        # ... existing setup code ...

        query = user_messages[-1].content
        add_span_attributes(user_query=query[:500])

        decision = await router.route(query)

        if request.stream:
            # Streaming handled separately (see below)
            return StreamingResponse(
                stream_smart_response(client, router, decision, query, request, chat_id, span),
                media_type="text/event-stream"
            )

        # Non-streaming path
        if decision.request_type == RequestType.LLM_TASK:
            response_text = await router.llm_client.complete(query, max_tokens=4096)
            response_source = "llm"
        else:
            result = await client.query(
                notebook_id=decision.notebook_id,
                query_text=query,
                conversation_id=request.conversation_id
            )
            response_text = result.get("answer", "") if result else ""
            response_source = "notebooklm"

        # Add response to trace
        if settings.response_max_length > 0:
            add_span_attributes(
                response_content=response_text[:settings.response_max_length],
                response_source=response_source
            )

        return ChatCompletionResponse(...)
```

### 2. Streaming Path

**File:** `src/nlm_proxy/openai/server.py`

Move span lifecycle inside the generator to keep it open until streaming completes:

```python
async def stream_smart_response(
    client,
    router: SmartRouter,
    decision,
    query: str,
    request: ChatCompletionRequest,
    chat_id: str = None
):
    """Stream response with response content tracing."""
    tracer = get_tracer(__name__)
    settings = get_tracing_settings()

    with tracer.start_as_current_span("smart_router.handle_request") as span:
        span.set_attribute("user_query", query[:500])

        # Determine response source
        response_source = "llm" if decision.request_type == RequestType.LLM_TASK else "notebooklm"

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        accumulated_response = ""

        # First, stream the routing decision as reasoning_content
        reasoning_chunk = ChatCompletionChunk(...)
        yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

        if decision.request_type == RequestType.LLM_TASK:
            # Stream from external LLM
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            stream = await router.llm_client.stream(messages)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                delta_content = delta.content if delta and delta.content else ""
                if delta_content:
                    accumulated_response += delta_content  # Accumulate
                    openai_chunk = ChatCompletionChunk(...)
                    yield f"data: {openai_chunk.model_dump_json()}\n\n"  # Yield immediately
        else:
            # Stream from NotebookLM
            previous_answer = ""
            async for chunk in client.query_stream(...):
                chunk_type = chunk.get("type")
                full_text = chunk.get("text", "")

                if chunk_type == "answer":
                    delta_text = full_text[len(previous_answer):]
                    previous_answer = full_text
                    if delta_text:
                        accumulated_response += delta_text  # Accumulate
                        openai_chunk = ChatCompletionChunk(...)
                        yield f"data: {openai_chunk.model_dump_json()}\n\n"  # Yield immediately

        # Add response to trace BEFORE final yield
        if settings.response_max_length > 0:
            span.set_attribute("response_content", accumulated_response[:settings.response_max_length])
            span.set_attribute("response_source", response_source)

        # Final chunk
        final_chunk = ChatCompletionChunk(...)
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        # Span closes here after generator exhausts
```

### 3. Remove @record_span from route()

**File:** `src/nlm_proxy/openai/router.py`

Remove `user_query` attribute from `route()` since it moves to `handle_request`:

```python
@record_span("smart_router.route")
async def route(self, query: str) -> RoutingDecision:
    """Classify and route the request."""
    logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")
    # REMOVED: add_span_attributes(user_query=query[:500])

    request_type = await self.classify_request(query)
    # ... rest unchanged ...
```

## Dashboard Updates

### Panels to Update (change span from `smart_router.route` to `smart_router.handle_request`)

1. **Total Requests** (panel id 1)
2. **Average Routing Time** → rename to **Average Request Time** (panel id 2)
3. **Error Rate** (panel id 3)
4. **P95 Latency** (panel id 4)
5. **Request Volume Over Time** (panel id 5)
6. **Recent Requests** (panel id 8)

### Panels Unchanged

- **Classification Distribution** (panel id 6) - uses `smart_router.classify`
- **Routing Time by Type** (panel id 7) - keep using `smart_router.route` for routing-only timing
- **Top 10 Notebooks** (panel id 9) - uses `smart_router.select_notebook`

### Updated "Recent Requests" Query

```sql
SELECT
  formatDateTime(
    argMax(Timestamp, SpanName = 'smart_router.handle_request'),
    '%Y-%m-%d %H:%i:%S'
  ) as Time,
  substring(
    argMax(SpanAttributes['user_query'], SpanName = 'smart_router.handle_request'),
    1, 50
  ) as Query,
  argMax(SpanAttributes['classification_result'], SpanName = 'smart_router.classify') as Classification,
  argMax(SpanAttributes['selected_notebook_title'], SpanName = 'smart_router.select_notebook') as Notebook,
  argMax(SpanAttributes['response_source'], SpanName = 'smart_router.handle_request') as Source,
  substring(
    argMax(SpanAttributes['response_content'], SpanName = 'smart_router.handle_request'),
    1, 100
  ) as Response_Preview,
  round(
    argMax(Duration, SpanName = 'smart_router.handle_request') / 1000000,
    2
  ) as Duration_ms,
  argMax(StatusCode, SpanName = 'smart_router.handle_request') as Status,
  TraceId
FROM nlm_traces.otel_traces
WHERE SpanName LIKE 'smart_router%'
  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)
GROUP BY TraceId
ORDER BY Time DESC
LIMIT 20
```

### Updated Stat Panel Queries

**Total Requests:**
```sql
SELECT count() as value
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.handle_request'
  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)
```

**Average Request Time:**
```sql
SELECT round(avg(Duration)/1000000, 2) as value
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.handle_request'
  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)
```

**Error Rate:**
```sql
SELECT
  round(countIf(StatusCode = 'ERROR') * 100.0 / count(), 2) as value
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.handle_request'
  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)
```

**P95 Latency:**
```sql
SELECT round(quantile(0.95)(Duration)/1000000, 2) as value
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.handle_request'
  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)
```

**Request Volume Over Time:**
```sql
SELECT
  toStartOfMinute(Timestamp) as time,
  count() as requests
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.handle_request'
  AND Timestamp >= toDateTime($__fromTime) AND Timestamp <= toDateTime($__toTime)
GROUP BY time
ORDER BY time
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/nlm_proxy/core/config.py` | Add `response_max_length` to `TracingSettings` |
| `src/nlm_proxy/openai/server.py` | Add `smart_router.handle_request` span, accumulate streaming responses |
| `src/nlm_proxy/openai/router.py` | Remove `user_query` from `route()` span |
| `docker/grafana/provisioning/dashboards/routing-analytics.json` | Update queries for new span name and add response columns |
| `docs/TRACING.md` | Document new attributes and configuration |

## Testing Plan

1. **Unit test:** Verify `response_max_length` setting is read correctly
2. **Non-streaming test:** Send non-streaming request, verify `response_content` in trace
3. **Streaming test:** Send streaming request, verify response accumulated correctly
4. **Truncation test:** Send request with response > max_length, verify truncation
5. **Disabled test:** Set `response_max_length=0`, verify no response attribute
6. **Dashboard test:** Verify all panels load with updated queries

## Rollback Plan

If issues arise:
1. Set `NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH=0` to disable response tracing
2. Revert dashboard JSON to previous version
3. Response tracing is additive - existing traces remain queryable
