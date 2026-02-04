# OpenTelemetry Tracing Guide

This guide explains how to set up and operate the OpenTelemetry tracing feature for NLM Proxy. Tracing provides visibility into request routing decisions, helping you understand how queries are classified and which notebooks are selected.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Infrastructure Setup](#infrastructure-setup)
- [Understanding Traces](#understanding-traces)
- [Querying Trace Data](#querying-trace-data)
- [Troubleshooting](#troubleshooting)

## Overview

The tracing feature instruments the Smart Router to capture:

- **Request classification**: Whether a query is routed to NotebookLM or an external LLM
- **Notebook selection**: Which notebook was chosen and why
- **Timing data**: How long each operation takes
- **Request attributes**: Query text, notebook IDs, and routing decisions

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│   NLM Proxy     │────▶│  OTel Collector  │────▶│  ClickHouse │
│  (OTLP export)  │     │  (batch + route) │     │  (storage)  │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

**Components:**
- **NLM Proxy**: Exports spans via OTLP gRPC
- **OpenTelemetry Collector**: Receives, batches, and exports traces
- **ClickHouse**: Stores traces with 90-day retention

## Quick Start

### 1. Start the Tracing Infrastructure

```bash
# From the nlm-proxy directory
docker compose -f docker-compose.otel.yml up -d
```

This starts:
- ClickHouse on ports 8123 (HTTP) and 9000 (native)
- OTel Collector on ports 4317 (gRPC) and 4318 (HTTP)

### 2. Enable Tracing

Add to your `.env` file or export as environment variables:

```bash
NLM_PROXY_OTEL_ENABLED=true
NLM_PROXY_OTEL_ENDPOINT=http://localhost:4317
NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy
```

### 3. Start NLM Proxy

```bash
nlm-proxy serve openai --port 8080
```

You should see in the logs:
```
[TRACING] OpenTelemetry initialized: endpoint=http://localhost:4317, service=nlm-proxy
[TRACING] FastAPI instrumentation enabled
[TRACING] httpx instrumentation enabled
```

### 4. Send a Test Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "knowledge-finder",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }'
```

### 5. View Traces

```bash
docker exec -it nlm-clickhouse /bin/bash
docker exec -it nlm-clickhouse clickhouse-client --query \
  "SELECT SpanName, Duration/1000000 as duration_ms
   FROM nlm_traces.otel_traces
   ORDER BY Timestamp DESC
   LIMIT 10 FORMAT Pretty"
```

**Note:** The ClickHouse exporter automatically creates the `otel_traces` table with an optimized schema for OpenTelemetry data.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NLM_PROXY_OTEL_ENABLED` | `false` | Enable/disable tracing |
| `NLM_PROXY_OTEL_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint (gRPC) |
| `NLM_PROXY_OTEL_SERVICE_NAME` | `nlm-proxy` | Service name in traces |

### Configuration File

Add to `.env` or `~/.nlm-proxy/.env`:

```bash
# OpenTelemetry Tracing
NLM_PROXY_OTEL_ENABLED=true
NLM_PROXY_OTEL_ENDPOINT=http://localhost:4317
NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy
```

### Programmatic Configuration

```python
from nlm_proxy.core.config import TracingSettings

settings = TracingSettings(
    enabled=True,
    endpoint="http://collector:4317",
    service_name="my-nlm-proxy"
)
```

## Infrastructure Setup

### Docker Compose (Recommended)

The provided `docker-compose.otel.yml` sets up everything:

```bash
# Start services
docker compose -f docker-compose.otel.yml up -d

# Check status
docker compose -f docker-compose.otel.yml ps

# View collector logs
docker logs nlm-otel-collector

# View ClickHouse logs
docker logs nlm-clickhouse

# Stop services
docker compose -f docker-compose.otel.yml down

# Stop and remove data
docker compose -f docker-compose.otel.yml down -v
```

### Manual Setup

#### ClickHouse

1. Start ClickHouse:
```bash
docker run -d --name clickhouse \
  -p 8123:8123 -p 9000:9000 \
  -v clickhouse_data:/var/lib/clickhouse \
  clickhouse/clickhouse-server:24.1
```

2. Create the schema:
```bash
docker exec -i clickhouse clickhouse-client < docker/clickhouse/init.sql
```

#### OTel Collector

1. Create config file (see `docker/otel/config.yaml`)

2. Start collector:
```bash
docker run -d --name otel-collector \
  -p 4317:4317 -p 4318:4318 \
  -v $(pwd)/docker/otel/config.yaml:/etc/otelcol-contrib/config.yaml:ro \
  otel/opentelemetry-collector-contrib:0.96.0 \
  --config=/etc/otelcol-contrib/config.yaml
```

### Cloud Deployment

For production, consider:

- **ClickHouse Cloud**: Managed ClickHouse service
- **Grafana Tempo**: Alternative trace storage
- **Jaeger**: Open-source tracing backend
- **AWS X-Ray**: AWS-native tracing

Update the collector config to export to your chosen backend.

## Understanding Traces

### Span Hierarchy

Each request creates a trace with nested spans:

```
smart_router.route (parent)
├── smart_router.classify (child)
└── smart_router.select_notebook (child, if NotebookLM)
```

### Span Attributes

#### smart_router.route
| Attribute | Type | Description |
|-----------|------|-------------|
| `user_query` | string | User's query (truncated to 500 chars) |
| `request_type` | string | "LLM_TASK" or "NOTEBOOKLM" |
| `notebook_id` | string | Selected notebook ID (if applicable) |
| `routing_reasoning` | string | Explanation of routing decision |

#### smart_router.classify
| Attribute | Type | Description |
|-----------|------|-------------|
| `classification_result` | string | "LLM_TASK" or "NOTEBOOKLM" |
| `llm_model` | string | Model used for classification |

#### smart_router.select_notebook
| Attribute | Type | Description |
|-----------|------|-------------|
| `candidates_count` | int | Number of notebooks considered |
| `selected_notebook_id` | string | Chosen notebook ID |
| `selected_notebook_title` | string | Chosen notebook title |
| `selection_fallback` | bool | True if fell back to first notebook |

### Auto-Instrumented Spans

With FastAPI and httpx instrumentation enabled, you'll also see:

- **HTTP server spans**: Incoming API requests
- **HTTP client spans**: Outgoing calls to NotebookLM and external LLMs

## Querying Trace Data

### ClickHouse Queries

Connect to ClickHouse:
```bash
docker exec -it nlm-clickhouse clickhouse-client
```

#### End-to-End Request Flow

View the complete flow for a single request, showing all spans from HTTP request through routing to response:

```sql
-- Complete request flow grouped by TraceId
SELECT
    TraceId,
    formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') as time,
    SpanName,
    round(Duration/1000000, 2) as duration_ms,
    SpanAttributes['user_query'] as query,
    SpanAttributes['classification_result'] as classification,
    SpanAttributes['selected_notebook_title'] as selected_notebook,
    SpanAttributes['candidates_count'] as candidates,
    StatusCode as status
FROM nlm_traces.otel_traces
WHERE TraceId IN (
    SELECT DISTINCT TraceId
    FROM nlm_traces.otel_traces
    WHERE SpanName = 'smart_router.route'
    ORDER BY Timestamp DESC
    LIMIT 1
)
ORDER BY Timestamp
FORMAT Pretty;
```

**Example Output:**
```
┌─TraceId──────────────────────────┬─time─────────────────┬─SpanName────────────────────────┬─duration_ms─┬─query─────────────────────┬─classification─┬─selected_notebook─┬─candidates─┬─status─┐
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:25:21  │ POST /v1/chat/completions       │    51018.59 │                           │                │                   │            │ Unset  │
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:25:21  │ smart_router.classify           │     3766.18 │                           │ NOTEBOOKLM     │                   │            │ Ok     │
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:25:21  │ smart_router.route              │     4699.97 │ What is machine learning? │                │                   │            │ Ok     │
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:25:24  │ smart_router.select_notebook    │      933.79 │                           │                │ ML Research       │ 4          │ Ok     │
│ 6e0b1e50c2c8d6a4d1f4b74767284148 │ 2026-02-04 04:26:12  │ POST /v1/chat/completions send  │           0 │                           │                │                   │            │ Unset  │
└──────────────────────────────────┴──────────────────────┴─────────────────────────────────┴─────────────┴───────────────────────────┴────────────────┴───────────────────┴────────────┴────────┘
```

This shows the complete request lifecycle:
1. **POST /v1/chat/completions** - HTTP request received (51 seconds total)
2. **smart_router.classify** - LLM classification (3.8 seconds) → Result: NOTEBOOKLM
3. **smart_router.route** - Main routing logic (4.7 seconds)
4. **smart_router.select_notebook** - Notebook selection (0.9 seconds) → Selected: ML Research (4 candidates)
5. **POST /v1/chat/completions send** - HTTP response sent

#### Recent Requests Summary

View a summary of recent requests showing key routing decisions:

```sql
-- Summary view of recent requests
SELECT
    substring(TraceId, 1, 8) as trace,
    formatDateTime(min(Timestamp), '%H:%M:%S') as time,
    any(SpanAttributes['user_query']) as user_query,
    any(SpanAttributes['classification_result']) as classification,
    any(SpanAttributes['selected_notebook_title']) as notebook,
    round(sum(Duration)/1000000, 2) as total_ms
FROM nlm_traces.otel_traces
WHERE SpanName LIKE 'smart_router%'
  AND Timestamp > now() - INTERVAL 1 HOUR
GROUP BY TraceId
ORDER BY min(Timestamp) DESC
LIMIT 10
FORMAT Pretty;
```

This provides a high-level overview of routing activity with one row per request.

#### Recent Traces
```sql
SELECT
    TraceId,
    SpanName,
    Duration/1000000 as duration_ms,
    SpanAttributes['request_type'] as request_type,
    SpanAttributes['notebook_id'] as notebook_id
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.route'
ORDER BY Timestamp DESC
LIMIT 20;
```

#### Average Routing Time by Type
```sql
SELECT
    SpanAttributes['request_type'] as request_type,
    count() as count,
    avg(Duration)/1000000 as avg_duration_ms,
    quantile(0.95)(Duration)/1000000 as p95_duration_ms
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.route'
  AND Timestamp > now() - INTERVAL 1 HOUR
GROUP BY request_type;
```

#### Most Selected Notebooks
```sql
SELECT
    SpanAttributes['selected_notebook_id'] as notebook_id,
    SpanAttributes['selected_notebook_title'] as notebook_title,
    count() as selection_count
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.select_notebook'
  AND Timestamp > now() - INTERVAL 24 HOUR
GROUP BY notebook_id, notebook_title
ORDER BY selection_count DESC
LIMIT 10;
```

#### Fallback Rate
```sql
SELECT
    countIf(SpanAttributes['selection_fallback'] = 'true') as fallback_count,
    count() as total_count,
    fallback_count / total_count * 100 as fallback_rate_pct
FROM nlm_traces.otel_traces
WHERE SpanName = 'smart_router.select_notebook'
  AND Timestamp > now() - INTERVAL 24 HOUR;
```

#### Error Rate
```sql
SELECT
    SpanName,
    countIf(StatusCode = 'ERROR') as error_count,
    count() as total_count,
    error_count / total_count * 100 as error_rate_pct
FROM nlm_traces.otel_traces
WHERE Timestamp > now() - INTERVAL 1 HOUR
GROUP BY SpanName;
```

#### Trace Details by TraceId
```sql
SELECT
    SpanName,
    Duration/1000000 as duration_ms,
    SpanAttributes,
    StatusCode
FROM nlm_traces.otel_traces
WHERE TraceId = 'your-trace-id-here'
ORDER BY Timestamp;
```

### Grafana Integration

To visualize traces in Grafana:

1. Add ClickHouse as a data source
2. Create dashboards with the queries above
3. Set up alerts for error rates or slow responses

## Troubleshooting

### Tracing Not Working

**Check if tracing is enabled:**
```bash
echo $NLM_PROXY_OTEL_ENABLED
# Should output: true
```

**Check logs for initialization:**
```bash
nlm-proxy serve openai --debug 2>&1 | grep TRACING
```

Expected output:
```
[TRACING] OpenTelemetry initialized: endpoint=http://localhost:4317, service=nlm-proxy
```

### No Traces in ClickHouse

**Check collector is receiving spans:**
```bash
docker logs nlm-otel-collector 2>&1 | tail -20
```

**Check collector can reach ClickHouse:**
```bash
docker exec nlm-otel-collector wget -q -O- http://clickhouse:8123/ping
```

**Verify ClickHouse table exists:**
```bash
docker exec nlm-clickhouse clickhouse-client --query \
  "SHOW TABLES FROM nlm_traces"
```

### High Memory Usage

The BatchSpanProcessor buffers spans before export. If you're seeing high memory:

1. Reduce batch size in `docker/otel/config.yaml`:
```yaml
processors:
  batch:
    timeout: 500ms
    send_batch_size: 512
```

2. Restart the collector:
```bash
docker compose -f docker-compose.otel.yml restart otel-collector
```

### Slow Exports

If traces are delayed:

1. Check network connectivity between proxy and collector
2. Reduce batch timeout for faster exports
3. Consider using HTTP instead of gRPC if behind a proxy

### Disk Space

ClickHouse data is retained for 90 days by default. To check usage:

```bash
docker exec nlm-clickhouse clickhouse-client --query \
  "SELECT
     formatReadableSize(sum(bytes_on_disk)) as size,
     count() as parts
   FROM system.parts
   WHERE database = 'nlm_traces'"
```

To manually clean old data:
```sql
ALTER TABLE nlm_traces.routing_traces
DELETE WHERE Timestamp < now() - INTERVAL 30 DAY;
```

### Disabling Tracing

To disable tracing without removing configuration:

```bash
export NLM_PROXY_OTEL_ENABLED=false
```

Or remove the environment variables entirely. The proxy will start without tracing overhead.

## Best Practices

1. **Use sampling in production**: For high-traffic deployments, configure sampling in the collector to reduce data volume.

2. **Monitor collector health**: Set up alerts for collector errors and queue depth.

3. **Secure the endpoint**: In production, use TLS and authentication for the OTLP endpoint.

4. **Set appropriate retention**: 90 days is the default; adjust based on your compliance and debugging needs.

5. **Index custom attributes**: If you query specific attributes frequently, add ClickHouse indexes.

## Further Reading

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [OpenTelemetry Collector Configuration](https://opentelemetry.io/docs/collector/configuration/)
- [ClickHouse Documentation](https://clickhouse.com/docs)
