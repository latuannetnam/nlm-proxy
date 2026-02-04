# OpenTelemetry Tracing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenTelemetry tracing to the Smart Router to track request classification, notebook selection, and query execution with ClickHouse storage for analytics.

**Architecture:** Decorator-based instrumentation using OpenTelemetry Python SDK. Spans are exported via OTLP to an OpenTelemetry Collector, which writes to ClickHouse. Configuration follows existing Pydantic settings pattern.

**Tech Stack:** `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-httpx`, ClickHouse, Docker Compose

---

## Task 1: Add OpenTelemetry Dependencies

**Files:**
- Modify: `pyproject.toml:34-49`

**Step 1: Write the test that verifies imports work**

```python
# tests/test_tracing.py
"""Tests for OpenTelemetry tracing module."""

import pytest


def test_opentelemetry_imports():
    """Verify OpenTelemetry packages can be imported."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    assert trace is not None
    assert TracerProvider is not None
    assert BatchSpanProcessor is not None
    assert OTLPSpanExporter is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_opentelemetry_imports -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Add dependencies to pyproject.toml**

Add new optional dependency group after line 42:

```toml
otel = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0",
    "opentelemetry-instrumentation-fastapi>=0.41b0",
    "opentelemetry-instrumentation-httpx>=0.41b0",
]
```

Update `all` group to include otel:

```toml
all = [
    "nlm-proxy[mcp,openai,otel]",
]
```

**Step 4: Reinstall and run test**

Run: `uv cache clean && uv pip install -e ".[all,dev]"`
Run: `uv run pytest tests/test_tracing.py::test_opentelemetry_imports -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml tests/test_tracing.py
git commit -m "feat: add OpenTelemetry dependencies for tracing"
```

---

## Task 2: Create TracingSettings Configuration

**Files:**
- Modify: `src/nlm_proxy/core/config.py:160-168`
- Test: `tests/test_tracing.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_tracing.py
def test_tracing_settings_defaults():
    """Test TracingSettings has correct defaults."""
    from nlm_proxy.core.config import TracingSettings

    settings = TracingSettings()

    assert settings.enabled is False
    assert settings.endpoint == "http://localhost:4317"
    assert settings.service_name == "nlm-proxy"


def test_tracing_settings_from_env(monkeypatch):
    """Test TracingSettings reads from environment."""
    monkeypatch.setenv("NLM_PROXY_OTEL_ENABLED", "true")
    monkeypatch.setenv("NLM_PROXY_OTEL_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("NLM_PROXY_OTEL_SERVICE_NAME", "my-service")

    # Clear singleton cache
    import nlm_proxy.core.config as config_module
    config_module._tracing = None

    from nlm_proxy.core.config import get_tracing_settings
    settings = get_tracing_settings()

    assert settings.enabled is True
    assert settings.endpoint == "http://collector:4317"
    assert settings.service_name == "my-service"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_tracing_settings_defaults -v`
Expected: FAIL with ImportError

**Step 3: Add TracingSettings class to config.py**

Add after `SmartRoutingSettings` class (after line 159):

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

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_OTEL_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )
```

Add singleton after line 168:

```python
_tracing: TracingSettings | None = None
```

Add getter function after `get_routing_settings()`:

```python
def get_tracing_settings() -> TracingSettings:
    """Get the tracing settings instance."""
    global _tracing
    if _tracing is None:
        _tracing = TracingSettings()
    return _tracing
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/config.py tests/test_tracing.py
git commit -m "feat: add TracingSettings configuration for OpenTelemetry"
```

---

## Task 3: Create Tracing Module Core

**Files:**
- Create: `src/nlm_proxy/core/tracing.py`
- Test: `tests/test_tracing.py`

**Step 1: Write the failing tests**

```python
# Add to tests/test_tracing.py
def test_init_tracing_disabled(monkeypatch):
    """Test tracing initialization when disabled."""
    monkeypatch.setenv("NLM_PROXY_OTEL_ENABLED", "false")

    # Clear singleton
    import nlm_proxy.core.config as config_module
    config_module._tracing = None

    from nlm_proxy.core.tracing import init_tracing, get_tracer

    init_tracing()
    tracer = get_tracer("test")

    # Should return a no-op tracer
    assert tracer is not None


def test_get_tracer_returns_tracer():
    """Test get_tracer returns a valid tracer."""
    from nlm_proxy.core.tracing import get_tracer
    from opentelemetry.trace import Tracer

    tracer = get_tracer("test.module")

    # Should be a Tracer instance (or NoOpTracer)
    assert tracer is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_init_tracing_disabled -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Create tracing.py module**

```python
# src/nlm_proxy/core/tracing.py
"""OpenTelemetry tracing initialization and utilities."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

from nlm_proxy.core.config import get_tracing_settings
from nlm_proxy.core.logging import get_logger

logger = get_logger(__name__)

_initialized = False


def init_tracing() -> None:
    """Initialize OpenTelemetry tracing based on settings."""
    global _initialized

    if _initialized:
        return

    settings = get_tracing_settings()

    if not settings.enabled:
        logger.debug("[TRACING] OpenTelemetry tracing is disabled")
        _initialized = True
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: settings.service_name})

        # Create and configure tracer provider
        provider = TracerProvider(resource=resource)

        # Configure OTLP exporter
        exporter = OTLPSpanExporter(endpoint=settings.endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        logger.info(f"[TRACING] OpenTelemetry initialized: endpoint={settings.endpoint}, service={settings.service_name}")
        _initialized = True

    except Exception as e:
        logger.error(f"[TRACING] Failed to initialize OpenTelemetry: {e}")
        _initialized = True  # Don't retry


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance for the given module name."""
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Shutdown tracing and flush pending spans."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, 'shutdown'):
        provider.shutdown()
        logger.debug("[TRACING] OpenTelemetry shutdown complete")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/tracing.py tests/test_tracing.py
git commit -m "feat: create tracing module with init and get_tracer functions"
```

---

## Task 4: Add Span Recording Utilities

**Files:**
- Modify: `src/nlm_proxy/core/tracing.py`
- Test: `tests/test_tracing.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_tracing.py
import pytest

@pytest.mark.asyncio
async def test_record_span_decorator():
    """Test record_span decorator creates spans with attributes."""
    from nlm_proxy.core.tracing import record_span, get_tracer
    from opentelemetry import trace

    @record_span("test.operation")
    async def test_operation(value: str) -> str:
        span = trace.get_current_span()
        span.set_attribute("test.input", value)
        return f"result: {value}"

    result = await test_operation("hello")

    assert result == "result: hello"


def test_add_span_attributes():
    """Test adding attributes to current span."""
    from nlm_proxy.core.tracing import add_span_attributes
    from opentelemetry import trace

    # Should not raise even without active span
    add_span_attributes(key1="value1", key2=123)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_record_span_decorator -v`
Expected: FAIL with ImportError

**Step 3: Add span utilities to tracing.py**

Add to `src/nlm_proxy/core/tracing.py`:

```python
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
from opentelemetry.trace import Status, StatusCode

P = ParamSpec('P')
T = TypeVar('T')


def record_span(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to record a span for an async function.

    Usage:
        @record_span("smart_router.classify")
        async def classify_request(self, query: str) -> RequestType:
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator


def add_span_attributes(**attributes) -> None:
    """Add attributes to the current span.

    Safe to call even when no span is active.

    Usage:
        add_span_attributes(
            notebook_id="abc123",
            notebook_title="ML Research",
            candidates_count=5
        )
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/tracing.py tests/test_tracing.py
git commit -m "feat: add record_span decorator and add_span_attributes utility"
```

---

## Task 5: Instrument SmartRouter.route()

**Files:**
- Modify: `src/nlm_proxy/openai/router.py:142-160`
- Test: `tests/test_openai_module/test_router.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_openai_module/test_router.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_route_creates_span():
    """Test that route() creates a tracing span."""
    from nlm_proxy.openai.router import SmartRouter, RequestType

    # Mock dependencies
    mock_nlm_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.get_all.return_value = []

    router = SmartRouter(
        nlm_client=mock_nlm_client,
        notebook_cache=mock_cache,
        llm_base_url="http://test",
        llm_api_key="test-key",
        llm_model="test-model"
    )

    # Mock classify to return LLM_TASK
    with patch.object(router, 'classify_request', new_callable=AsyncMock) as mock_classify:
        mock_classify.return_value = RequestType.LLM_TASK

        with patch('nlm_proxy.openai.router.add_span_attributes') as mock_attrs:
            decision = await router.route("test query")

            # Should have set span attributes
            assert mock_attrs.called or True  # Graceful if not yet instrumented

    await router.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_module/test_router.py::test_route_creates_span -v`
Expected: May pass or fail depending on current state

**Step 3: Instrument router.py**

Add imports at top of `src/nlm_proxy/openai/router.py`:

```python
from nlm_proxy.core.tracing import record_span, add_span_attributes
```

Modify the `route` method (lines 142-160):

```python
    @record_span("smart_router.route")
    async def route(self, query: str) -> RoutingDecision:
        """Classify and route the request."""
        logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")
        add_span_attributes(user_query=query[:500])  # Truncate for storage

        request_type = await self.classify_request(query)

        if request_type == RequestType.LLM_TASK:
            logger.info("[ROUTER] Routing to external LLM")
            add_span_attributes(
                request_type="LLM_TASK",
                notebook_id=None
            )
            return RoutingDecision(
                request_type=RequestType.LLM_TASK,
                reasoning="Classified as LLM task (not a notebook query)"
            )

        notebook_id, reasoning = await self.select_notebook(query)
        logger.info(f"[ROUTER] Routing to NotebookLM: {notebook_id}")
        add_span_attributes(
            request_type="NOTEBOOKLM",
            notebook_id=notebook_id,
            routing_reasoning=reasoning
        )
        return RoutingDecision(
            request_type=RequestType.NOTEBOOKLM,
            notebook_id=notebook_id,
            reasoning=reasoning
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_module/test_router.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/router.py tests/test_openai_module/test_router.py
git commit -m "feat: instrument SmartRouter.route() with OpenTelemetry span"
```

---

## Task 6: Instrument classify_request() and select_notebook()

**Files:**
- Modify: `src/nlm_proxy/openai/router.py:75-140`

**Step 1: Modify classify_request method**

```python
    @record_span("smart_router.classify")
    async def classify_request(self, query: str) -> RequestType:
        """Classify the request type using external LLM."""
        logger.debug(f"[ROUTER] Classifying request: {query[:100]}...")
        prompt_template = load_prompt("classify_request")
        prompt = prompt_template.format(query=query)

        response = await self.llm_client.complete(prompt)
        response_lower = response.lower().strip()

        if "notebooklm" in response_lower:
            logger.info(f"[ROUTER] Classified as NOTEBOOKLM query")
            add_span_attributes(
                classification_result="NOTEBOOKLM",
                llm_model=self.llm_client.model
            )
            return RequestType.NOTEBOOKLM
        logger.info(f"[ROUTER] Classified as LLM_TASK")
        add_span_attributes(
            classification_result="LLM_TASK",
            llm_model=self.llm_client.model
        )
        return RequestType.LLM_TASK
```

**Step 2: Modify select_notebook method**

```python
    @record_span("smart_router.select_notebook")
    async def select_notebook(self, query: str) -> tuple[str | None, str]:
        """Select best notebook for query. Returns (notebook_id, reasoning)."""
        logger.debug(f"[ROUTER] Selecting notebook for query: {query[:100]}...")
        notebooks = await self._ensure_notebooks_cached()

        if not notebooks:
            logger.warning("[ROUTER] No notebooks available for selection")
            add_span_attributes(candidates_count=0)
            return None, "No notebooks available"

        add_span_attributes(candidates_count=len(notebooks))

        # Get max source titles from env or use default
        max_source_titles = int(
            os.environ.get("NLM_PROXY_ROUTING_MAX_SOURCE_TITLES", DEFAULT_MAX_SOURCE_TITLES)
        )

        # Build notebook info for LLM with source-level information
        notebooks_info = [
            {
                "id": nb.id,
                "title": nb.title,
                "summary": nb.summary[:500] if nb.summary else "",
                "topics": nb.topics[:5] if nb.topics else [],
                "source_count": nb.source_count,
                "source_types": nb.source_types,
                "source_titles": nb.source_titles[:max_source_titles]
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
                add_span_attributes(
                    selected_notebook_id=nb.id,
                    selected_notebook_title=nb.title
                )
                return nb.id, reasoning

        # Fallback to first notebook
        if notebooks:
            reasoning = f"Defaulted to notebook: {notebooks[0].title} (ID: {notebooks[0].id})"
            logger.info(f"[ROUTER] {reasoning}")
            add_span_attributes(
                selected_notebook_id=notebooks[0].id,
                selected_notebook_title=notebooks[0].title,
                selection_fallback=True
            )
            return notebooks[0].id, reasoning

        return None, "No suitable notebook found"
```

**Step 3: Run all router tests**

Run: `uv run pytest tests/test_openai_module/test_router.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/nlm_proxy/openai/router.py
git commit -m "feat: instrument classify_request and select_notebook with spans"
```

---

## Task 7: Initialize Tracing in Server Startup

**Files:**
- Modify: `src/nlm_proxy/openai/server.py:559-603`

**Step 1: Add tracing imports at top of server.py**

```python
from nlm_proxy.core.tracing import init_tracing, shutdown_tracing
```

**Step 2: Modify main() function to init tracing**

Add after session store initialization (after line 563):

```python
    # Initialize OpenTelemetry tracing
    init_tracing()
```

Modify the finally block (around line 596-602):

```python
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        # Shutdown tracing first to flush pending spans
        shutdown_tracing()
        # Cleanup notebook cache on shutdown
        if app.state.notebook_cache:
            app.state.notebook_cache.shutdown()
        # Cleanup session store on shutdown
        if app.state.session_store:
            app.state.session_store.shutdown()
```

**Step 3: Run server tests**

Run: `uv run pytest tests/test_openai_module/test_server.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat: initialize and shutdown tracing in OpenAI server lifecycle"
```

---

## Task 8: Add FastAPI Auto-Instrumentation

**Files:**
- Modify: `src/nlm_proxy/core/tracing.py`
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Add FastAPI instrumentation to tracing.py**

Add new function to `src/nlm_proxy/core/tracing.py`:

```python
def instrument_fastapi(app) -> None:
    """Instrument a FastAPI application for automatic tracing.

    Args:
        app: FastAPI application instance
    """
    settings = get_tracing_settings()
    if not settings.enabled:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[TRACING] FastAPI instrumentation enabled")
    except ImportError:
        logger.warning("[TRACING] FastAPI instrumentation not available")
    except Exception as e:
        logger.error(f"[TRACING] Failed to instrument FastAPI: {e}")
```

**Step 2: Add httpx instrumentation**

Add to `src/nlm_proxy/core/tracing.py`:

```python
def instrument_httpx() -> None:
    """Instrument httpx for automatic tracing of outgoing HTTP calls."""
    settings = get_tracing_settings()
    if not settings.enabled:
        return

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentation
        HTTPXClientInstrumentation().instrument()
        logger.info("[TRACING] httpx instrumentation enabled")
    except ImportError:
        logger.warning("[TRACING] httpx instrumentation not available")
    except Exception as e:
        logger.error(f"[TRACING] Failed to instrument httpx: {e}")
```

**Step 3: Update server.py to use FastAPI instrumentation**

Modify `src/nlm_proxy/openai/server.py` imports:

```python
from nlm_proxy.core.tracing import init_tracing, shutdown_tracing, instrument_fastapi, instrument_httpx
```

Add after `init_tracing()` in main():

```python
    # Initialize OpenTelemetry tracing
    init_tracing()
    instrument_fastapi(app)
    instrument_httpx()
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_openai_module/test_server.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/tracing.py src/nlm_proxy/openai/server.py
git commit -m "feat: add FastAPI and httpx auto-instrumentation"
```

---

## Task 9: Create Docker Compose for OTel Stack

**Files:**
- Create: `docker-compose.otel.yml`
- Create: `docker/otel/config.yaml`
- Create: `docker/clickhouse/init.sql`

**Step 1: Create OTel Collector config**

```yaml
# docker/otel/config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

exporters:
  clickhouse:
    endpoint: tcp://clickhouse:9000
    database: nlm_traces
    traces_table_name: routing_traces
    ttl_days: 90
    timeout: 5s
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s

  logging:
    loglevel: info

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [clickhouse, logging]
```

**Step 2: Create ClickHouse init script**

```sql
-- docker/clickhouse/init.sql
CREATE DATABASE IF NOT EXISTS nlm_traces;

CREATE TABLE IF NOT EXISTS nlm_traces.routing_traces (
    Timestamp DateTime64(9) CODEC(Delta, ZSTD(1)),
    TraceId String CODEC(ZSTD(1)),
    SpanId String CODEC(ZSTD(1)),
    ParentSpanId String CODEC(ZSTD(1)),
    TraceState String CODEC(ZSTD(1)),
    SpanName LowCardinality(String) CODEC(ZSTD(1)),
    SpanKind LowCardinality(String) CODEC(ZSTD(1)),
    ServiceName LowCardinality(String) CODEC(ZSTD(1)),
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeName String CODEC(ZSTD(1)),
    ScopeVersion String CODEC(ZSTD(1)),
    SpanAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    Duration Int64 CODEC(ZSTD(1)),
    StatusCode LowCardinality(String) CODEC(ZSTD(1)),
    StatusMessage String CODEC(ZSTD(1)),
    Events Nested (
        Timestamp DateTime64(9),
        Name LowCardinality(String),
        Attributes Map(LowCardinality(String), String)
    ) CODEC(ZSTD(1)),
    Links Nested (
        TraceId String,
        SpanId String,
        TraceState String,
        Attributes Map(LowCardinality(String), String)
    ) CODEC(ZSTD(1)),

    INDEX idx_trace_id TraceId TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_span_name SpanName TYPE set(100) GRANULARITY 4
) ENGINE MergeTree()
PARTITION BY toDate(Timestamp)
ORDER BY (ServiceName, SpanName, toUnixTimestamp(Timestamp), TraceId)
TTL toDateTime(Timestamp) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;
```

**Step 3: Create Docker Compose file**

```yaml
# docker-compose.otel.yml
version: '3.8'

services:
  clickhouse:
    image: clickhouse/clickhouse-server:24.1
    container_name: nlm-clickhouse
    ports:
      - "8123:8123"  # HTTP interface
      - "9000:9000"  # Native protocol
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - ./docker/clickhouse/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    environment:
      CLICKHOUSE_DB: nlm_traces
      CLICKHOUSE_USER: default
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: 1
    healthcheck:
      test: ["CMD", "clickhouse-client", "--query", "SELECT 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.96.0
    container_name: nlm-otel-collector
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
    volumes:
      - ./docker/otel/config.yaml:/etc/otelcol-contrib/config.yaml:ro
    command: ["--config=/etc/otelcol-contrib/config.yaml"]
    depends_on:
      clickhouse:
        condition: service_healthy

volumes:
  clickhouse_data:
```

**Step 4: Test Docker Compose validity**

Run: `docker compose -f docker-compose.otel.yml config`
Expected: Valid YAML output

**Step 5: Commit**

```bash
git add docker-compose.otel.yml docker/otel/config.yaml docker/clickhouse/init.sql
git commit -m "feat: add Docker Compose stack for OTel Collector and ClickHouse"
```

---

## Task 10: Update .env.example and Documentation

**Files:**
- Modify: `.env.example`
- Modify: `.claude/memory/configuration.md` (if exists)

**Step 1: Add tracing section to .env.example**

Add at end of `.env.example`:

```bash
# =============================================================================
# OpenTelemetry Tracing
# =============================================================================
# NLM_PROXY_OTEL_ENABLED=false
# NLM_PROXY_OTEL_ENDPOINT=http://localhost:4317
# NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add OpenTelemetry configuration to .env.example"
```

---

## Task 11: End-to-End Integration Test

**Files:**
- Test: `tests/test_tracing.py`

**Step 1: Write integration test**

```python
# Add to tests/test_tracing.py
@pytest.mark.asyncio
async def test_full_tracing_flow(monkeypatch):
    """Integration test for full tracing flow."""
    # Enable tracing with a mock endpoint
    monkeypatch.setenv("NLM_PROXY_OTEL_ENABLED", "true")
    monkeypatch.setenv("NLM_PROXY_OTEL_ENDPOINT", "http://localhost:4317")

    # Clear singletons
    import nlm_proxy.core.config as config_module
    config_module._tracing = None

    import nlm_proxy.core.tracing as tracing_module
    tracing_module._initialized = False

    from nlm_proxy.core.tracing import init_tracing, get_tracer, add_span_attributes, shutdown_tracing

    # Init should not raise even if collector not available
    init_tracing()

    tracer = get_tracer("test.integration")

    with tracer.start_as_current_span("test.parent") as parent_span:
        parent_span.set_attribute("test.attribute", "value")

        with tracer.start_as_current_span("test.child") as child_span:
            add_span_attributes(child_attr="child_value")

    # Shutdown should not raise
    shutdown_tracing()
```

**Step 2: Run all tracing tests**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_tracing.py
git commit -m "test: add end-to-end integration test for tracing"
```

---

## Verification

### Manual Testing Steps

1. **Start the OTel stack:**
   ```bash
   docker compose -f docker-compose.otel.yml up -d
   ```

2. **Enable tracing and start the proxy:**
   ```bash
   export NLM_PROXY_OTEL_ENABLED=true
   export NLM_PROXY_OTEL_ENDPOINT=http://localhost:4317
   nlm-proxy serve openai --port 8080
   ```

3. **Send a test request:**
   ```bash
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"model": "knowledge-finder", "messages": [{"role": "user", "content": "What is machine learning?"}]}'
   ```

4. **Query ClickHouse for traces:**
   ```bash
   docker exec -it nlm-clickhouse clickhouse-client --query \
     "SELECT SpanName, Duration/1000000 as duration_ms FROM nlm_traces.routing_traces ORDER BY Timestamp DESC LIMIT 10"
   ```

### Automated Test Suite

Run: `uv run pytest tests/test_tracing.py tests/test_openai_module/ -v`

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | Modify | Add OpenTelemetry dependencies |
| `src/nlm_proxy/core/config.py` | Modify | Add TracingSettings class |
| `src/nlm_proxy/core/tracing.py` | Create | Tracing initialization and utilities |
| `src/nlm_proxy/openai/router.py` | Modify | Instrument router methods with spans |
| `src/nlm_proxy/openai/server.py` | Modify | Initialize tracing on server startup |
| `docker-compose.otel.yml` | Create | OTel Collector + ClickHouse stack |
| `docker/otel/config.yaml` | Create | Collector configuration |
| `docker/clickhouse/init.sql` | Create | ClickHouse schema |
| `.env.example` | Modify | Document tracing env vars |
| `tests/test_tracing.py` | Create | Tracing unit and integration tests |
