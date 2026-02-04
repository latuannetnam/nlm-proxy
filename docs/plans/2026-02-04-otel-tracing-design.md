# OpenTelemetry Tracing for Smart Router

**Date:** 2026-02-04
**Status:** Proposed

## Overview

Add request/response tracing to the OpenAI proxy smart router using OpenTelemetry with ClickHouse storage for analytics.

**Goals:**
- Track which requests map to which NotebookLM responses
- Capture notebook selection decisions and reasoning
- Enable analytics queries on routing patterns
- Store full query and response content

**Non-Goals:**
- Real-time alerting (can be added later)
- Metrics collection (traces only for now)

## Decision Summary

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Standard | OpenTelemetry | CNCF standard, vendor-neutral, Python SDK mature |
| Storage | ClickHouse | Analytics-first, SQL queries, handles volume well |
| Collector | OTel Collector | Standard component, ClickHouse exporter built-in |
| Deployment | Docker Compose | Self-contained, reproducible, easy local dev |
| Instrumentation | Decorator style | Clean, readable, visible at function signature |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           nlm-proxy                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  │
│  │   OpenAI    │───▶│   Smart     │───▶│  NotebookLM / External LLM  │  │
│  │   Server    │    │   Router    │    │                             │  │
│  └──────┬──────┘    └──────┬──────┘    └──────────────┬──────────────┘  │
│         │                  │                          │                  │
│         └──────────────────┼──────────────────────────┘                  │
│                            ▼                                             │
│                   ┌─────────────────┐                                    │
│                   │ OTel SDK        │                                    │
│                   │ (Tracing)       │                                    │
│                   └────────┬────────┘                                    │
└────────────────────────────┼────────────────────────────────────────────┘
                             │ OTLP (gRPC :4317)
                             ▼
                   ┌─────────────────┐
                   │  OTel Collector │
                   │  (Docker)       │
                   └────────┬────────┘
                            │ ClickHouse Exporter
                            ▼
                   ┌─────────────────┐
                   │   ClickHouse    │
                   │   (Docker)      │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Grafana (opt.)  │
                   │ for dashboards  │
                   └─────────────────┘
```

## Span Structure

```
[smart_router.request]                    # Parent span
├── trace_id: "abc123..."
├── request_id: "req_456"
├── user_query: "Explain transformers"
├── model: "smart-router"
│
├── [smart_router.classify]               # Classification phase
│   ├── duration_ms: 150
│   ├── result: "NOTEBOOKLM"
│   └── llm_model: "claude-3-haiku"
│
├── [smart_router.select_notebook]        # Notebook selection
│   ├── duration_ms: 200
│   ├── notebook_id: "uuid-..."
│   ├── notebook_title: "ML Research"
│   ├── routing_reasoning: "Selected based on ML topics"
│   └── candidates_count: 5
│
└── [smart_router.execute]                # Query execution
    ├── duration_ms: 800
    ├── request_type: "NOTEBOOKLM"
    ├── notebook_id: "uuid-..."
    ├── response_content: "Transformers are..."
    ├── response_length: 1250
    └── success: true
```

## ClickHouse Schema

```sql
CREATE TABLE routing_traces (
    trace_id String,
    span_id String,
    parent_span_id String,
    span_name String,
    timestamp DateTime64(3),
    duration_ms UInt32,
    request_id String,
    user_query String,
    model String,
    request_type LowCardinality(String),
    notebook_id Nullable(String),
    notebook_title Nullable(String),
    routing_reasoning Nullable(String),
    candidates_count Nullable(UInt8),
    response_content String,
    response_length UInt32,
    success Bool,
    error_message Nullable(String),
    llm_model Nullable(String),

    INDEX idx_notebook_id notebook_id TYPE bloom_filter GRANULARITY 4,
    INDEX idx_request_type request_type TYPE set(2) GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, trace_id, span_name)
TTL timestamp + INTERVAL 90 DAY DELETE;
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NLM_PROXY_OTEL_ENABLED` | `false` | Enable tracing |
| `NLM_PROXY_OTEL_ENDPOINT` | `http://localhost:4317` | OTLP endpoint |
| `NLM_PROXY_OTEL_SERVICE_NAME` | `nlm-proxy` | Service name in traces |

## Example Analytics Queries

```sql
-- Notebook popularity
SELECT notebook_title, COUNT(*) as requests
FROM nlm_traces.routing_traces
WHERE span_name = 'smart_router.execute'
  AND timestamp > now() - INTERVAL 7 DAY
GROUP BY notebook_title
ORDER BY requests DESC;

-- Request type distribution
SELECT request_type,
       COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
FROM nlm_traces.routing_traces
WHERE span_name = 'smart_router.execute'
GROUP BY request_type;

-- Average latency by phase
SELECT span_name, AVG(duration_ms) as avg_ms
FROM nlm_traces.routing_traces
GROUP BY span_name;

-- Requests over time
SELECT toStartOfHour(timestamp) as hour,
       request_type,
       COUNT(*) as count
FROM nlm_traces.routing_traces
WHERE span_name = 'smart_router.execute'
GROUP BY hour, request_type
ORDER BY hour;
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/nlm_proxy/core/tracing.py` | Create |
| `src/nlm_proxy/core/config.py` | Modify |
| `src/nlm_proxy/openai/router.py` | Modify |
| `src/nlm_proxy/openai/server.py` | Modify |
| `pyproject.toml` | Modify |
| `docker-compose.otel.yml` | Create |
| `docker/clickhouse/init.sql` | Create |
| `docker/otel/config.yaml` | Create |
| `.env.example` | Modify |
| `tests/test_tracing.py` | Create |

## Alternatives Considered

1. **Jaeger** - Simpler, but lacks SQL analytics capability
2. **Signoz** - Good but heavier (6 containers vs 2)
3. **Grafana Tempo** - TraceQL less powerful than ClickHouse SQL
4. **Structured logging only** - No trace correlation or visualization
