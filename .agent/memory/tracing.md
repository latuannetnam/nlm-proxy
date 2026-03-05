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
└── AgentCore.route() → LangGraph StateGraph
    ├── request_type: "notebooklm" or "llm_task"
    ├── notebook_id: selected notebook UUID
    │
    ├── classify_node (LangGraph node)
    │   ├── classification_result
    │   └── llm_model
    │
    └── select_notebook_node (LangGraph node, if notebooklm)
        ├── acl_filter_applied, acl_matched_count
        ├── candidates_count
        ├── selected_notebook_id
        └── selected_notebook_title
```

## Implementation Details

### Response Tracing Architecture

**Critical Design Principle: Separate Span Ownership**

Streaming and non-streaming requests require different span ownership patterns due to how FastAPI/Starlette handles `StreamingResponse`:

**Non-streaming path** (`server.py:_handle_non_streaming`):
- Creates span with `tracer.start_as_current_span("smart_router.handle_request")`
- Adds `user_query` after extracting from messages
- Calls `agent_core.route(query)` → LangGraph routing
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
- `src/nlm_proxy/core/agent.py` - AgentCore.route() with fallback-on-error
- `src/nlm_proxy/core/routing_graph.py` - LangGraph routing nodes (classify, select)
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

## Development vs Production

**Development Setup (No Security):**
```bash
docker compose -f docker-compose.otel.yml up -d
```
- Plain HTTP/gRPC without TLS or authentication
- Quick start, minimal configuration

**Production Setup (Secure):**
```bash
bash docker/otel/generate-certs.sh
openssl rand -base64 32  # Generate bearer token
docker compose -f docker-compose.otel-secure.yml up -d
```
- TLS + bearer token authentication
- Configure both collector and client with matching tokens
- See `docs/TRACING.md` for complete setup guide

**Key Configuration Files:**
- `docker/otel/config.yaml` - Basic collector config (no auth/TLS)
- `docker/otel/config-secure.yaml` - Secure collector config (TLS + bearertokenauth)
- `docker/otel/generate-certs.sh` - Self-signed certificate generator
- `docker/otel/.env.example` - Environment variables template

## Known Issues & Lessons Learned

### Asyncio + Threading Pitfalls

**Issue:** "Event loop is closed" errors when mixing asyncio and threading.

**Root cause:** Asyncio objects (Lock, httpx.AsyncClient) bind to their creation event loop. When reused in a different thread's event loop, they fail.

**Fix:** Close async clients before switching event loops.

**Debugging checklist:**
1. Alternating success/failure patterns → `asyncio.run()` misuse
2. First operation fails, rest succeed → async object bound to wrong loop
3. Always close async clients before event loop closes

See `docs/ASYNCIO_THREADING_PITFALLS.md` for details.

### OpenTelemetry Connection Issues

**Generic `StatusCode.UNAVAILABLE` error masks:**
- Auth failures
- TLS handshake errors
- Network issues

**Debugging checklist:**
1. Check collector logs first: `docker logs nlm-otel-collector` or `sudo journalctl -u otelcol-contrib`
2. Verify bearer token matches: `NLM_PROXY_OTEL_API_KEY` == collector's `OTEL_BEARER_TOKEN`
3. Check TLS consistency: If collector has TLS, client needs `INSECURE=false` + CA cert
4. Protocol matters: gRPC doesn't support skip-verify; use HTTP for self-signed certs

### TLS Certificate Hostname vs IP Address

**Problem:** `tls: bad record MAC` error when connecting to remote OTEL collector via IP address.

**Root Cause:** TLS certificates validate against Subject Alternative Names (SAN). If the certificate only contains DNS hostnames, connecting via IP will fail.

**Solution (Recommended): Use Hostname**

Add hostname to local hosts file:

**Linux/Mac** (`/etc/hosts`):
```bash
sudo bash -c 'echo "10.60.5.76    ai-analytics" >> /etc/hosts'
```

**Windows** (`C:\Windows\System32\drivers\etc\hosts` - requires Administrator):
```
10.60.5.76    ai-analytics
```

Then update client `.env`:
```bash
NLM_PROXY_OTEL_ENDPOINT=ai-analytics:4317  # Use hostname instead of IP
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_CA_CERT_PATH=/path/to/ca.crt
NLM_PROXY_OTEL_API_KEY=your-bearer-token
```

**Alternative Solutions:**
1. Regenerate certificate with server IP in SAN (edit `docker/otel/generate-certs.sh`)
2. Use `VERIFY_CERT=false` with HTTP protocol (dev/testing only)

**Common Errors:**
- `bad record MAC` → Hostname/IP mismatch in certificate SAN
- `UNAVAILABLE` → Check collector logs for real error
- `authentication failed` → Bearer token mismatch
- `x509: unknown authority` → Missing or wrong CA cert path

### Duplicate Span Anti-Pattern

**Issue:** Creating multiple spans with the same name in a single trace causes non-deterministic results in Grafana dashboard queries.

**Root cause:** Both `handle_smart_routing` and `stream_smart_response` were creating `smart_router.handle_request` spans, resulting in TWO spans with the same name per streaming request.

**Why passing parent span doesn't work for streaming:** When you return a `StreamingResponse`, the parent function's `with` block exits immediately (closing the span), but the generator hasn't been consumed yet.

**Correct Fix: Separate Span Ownership**
- **Streaming path**: Generator owns the span (lives for full streaming duration)
- **Non-streaming path**: Parent function owns the span

See `docs/TRACING.md` for implementation details.

## Reference

Full documentation: `docs/TRACING.md`
Architecture details: `docs/smart-routing-architecture.md`
