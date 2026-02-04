# OpenTelemetry Tracing

## Overview

NLM Proxy supports distributed tracing via OpenTelemetry to monitor routing decisions, timing, and request/response content.

## Configuration

```bash
# Enable tracing
NLM_PROXY_OTEL_ENABLED=true
NLM_PROXY_OTEL_ENDPOINT=http://localhost:4317
NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy

# Content capture (configurable truncation)
NLM_PROXY_OTEL_REQUEST_MAX_LENGTH=500   # User query max chars (0=disable)
NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH=1000 # Response max chars (0=disable)
```

## Span Hierarchy

```
smart_router.handle_request (parent - full request lifecycle)
├── user_query (attribute)
├── response_content (attribute) - truncated per RESPONSE_MAX_LENGTH
├── response_source (attribute) - "llm" or "notebooklm"
│
└── smart_router.route (child - routing decision)
    ├── request_type: "NOTEBOOKLM" or "LLM_TASK"
    ├── notebook_id: selected notebook UUID
    │
    ├── smart_router.classify (grandchild)
    │   ├── classification_result
    │   └── llm_model
    │
    └── smart_router.select_notebook (grandchild, if NotebookLM)
        ├── candidates_count
        ├── selected_notebook_id
        └── selected_notebook_title
```

## Implementation Details

### Response Tracing Architecture

**Critical Design Principle: Separate Span Ownership**

Streaming and non-streaming requests require different span ownership patterns due to how FastAPI/Starlette handles `StreamingResponse`:

**Non-streaming path** (`server.py:handle_smart_routing`):
- Creates span with `tracer.start_as_current_span("smart_router.handle_request")`
- Adds `user_query` after extracting from messages
- Calls `router.route(query)` (creates child spans)
- Executes LLM or NotebookLM request
- Adds `response_content` and `response_source` before returning
- Span closes when function returns

**Streaming path** (`server.py:stream_smart_response`):
- **Creates its own span** inside the generator function
- The generator owns the span because it must live for the full streaming duration
- `handle_smart_routing` does NOT create a span for streaming requests
- Accumulates response chunks in `accumulated_response` variable while yielding
- Adds `response_content` and `response_source` BEFORE final `[DONE]` chunk
- **Zero-latency design**: chunks yielded immediately, accumulation happens in parallel

### Key Files

- `src/nlm_proxy/core/config.py` - TracingSettings with request/response max lengths
- `src/nlm_proxy/openai/server.py` - handle_smart_routing and stream_smart_response
- `docker/grafana/provisioning/dashboards/routing-analytics.json` - Updated queries

## Quick Start

```bash
# Start infrastructure
docker compose -f docker-compose.otel.yml up -d

# Enable and start
export NLM_PROXY_OTEL_ENABLED=true
nlm-proxy serve openai --port 8080

# View traces
docker exec nlm-clickhouse clickhouse-client --query "
SELECT
  SpanName,
  SpanAttributes['user_query'] as query,
  SpanAttributes['response_source'] as source,
  substring(SpanAttributes['response_content'], 1, 50) as response
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.handle_request'
ORDER BY Timestamp DESC LIMIT 10"
```

## Grafana Dashboard

Navigate to `http://localhost:3000` → **Dashboards → NLM Proxy - Routing Analytics**

**Panels updated to use `smart_router.handle_request` span:**
- Total Requests
- Average Request Time (renamed from "Routing Time")
- Error Rate
- P95 Latency
- Request Volume Over Time
- Recent Requests (now includes Source and Response_Preview columns)

## Reference

Full documentation: `docs/TRACING.md`
Architecture details: `docs/smart-routing-architecture.md`

## Known Issues & Lessons Learned

### Duplicate Span Anti-Pattern (Fixed)

**Issue:** Creating multiple spans with the same name in a single trace causes `argMax()` aggregation queries to return non-deterministic results. This manifested as Grafana dashboard data appearing/disappearing on refresh.

**Root cause:** Both `handle_smart_routing` and `stream_smart_response` were creating `smart_router.handle_request` spans, resulting in TWO spans with the same name per streaming request.

**Why passing parent span doesn't work for streaming:**
When you return a `StreamingResponse`, the parent function's `with` block exits immediately (closing the span), but the generator hasn't been consumed yet. The span closes before any response data is generated.

```python
# WRONG - span closes before streaming starts
async def handle_smart_routing(...):
    with tracer.start_as_current_span("smart_router.handle_request") as span:
        return StreamingResponse(stream_smart_response(..., span))
        # <-- span closes HERE, before generator runs!
```

**Correct Fix: Separate Span Ownership**

```python
# CORRECT - generator owns its span
async def stream_smart_response(...):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("smart_router.handle_request") as span:
        # Span lives for full streaming duration
        async for chunk in ...:
            accumulated_response += chunk
            yield chunk
        span.set_attribute("response_content", accumulated_response)

async def handle_smart_routing(...):
    if request.stream:
        # NO span here - generator owns it
        return StreamingResponse(stream_smart_response(...))

    # Non-streaming: create span here
    with tracer.start_as_current_span("smart_router.handle_request") as span:
        response = await get_response()
        span.set_attribute("response_content", response)
        return response
```

**Best Practice:** For streaming responses, the generator must own the span. For non-streaming, the parent function owns it. Never create the same span name in both places for the same request type.
