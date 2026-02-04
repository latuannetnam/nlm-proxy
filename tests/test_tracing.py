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


def test_tracing_settings_defaults(monkeypatch, tmp_path):
    """Test TracingSettings has correct defaults."""
    # Clear any environment variables that might override defaults
    monkeypatch.delenv("NLM_PROXY_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("NLM_PROXY_OTEL_ENDPOINT", raising=False)
    monkeypatch.delenv("NLM_PROXY_OTEL_SERVICE_NAME", raising=False)

    # Change to temp directory to avoid reading .env files
    monkeypatch.chdir(tmp_path)

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

    tracer = get_tracer("test.module")

    # Should be a Tracer instance (or NoOpTracer)
    assert tracer is not None


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

    # Should not raise even without active span
    add_span_attributes(key1="value1", key2=123)


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


def test_tracing_settings_request_max_length_default(monkeypatch, tmp_path):
    """Test TracingSettings has correct default for request_max_length."""
    # Clear env vars and change to temp dir to avoid .env files
    monkeypatch.delenv("NLM_PROXY_OTEL_REQUEST_MAX_LENGTH", raising=False)
    monkeypatch.chdir(tmp_path)

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


def test_tracing_settings_response_max_length_default(monkeypatch, tmp_path):
    """Test TracingSettings has correct default for response_max_length."""
    # Clear env vars and change to temp dir to avoid .env files
    monkeypatch.delenv("NLM_PROXY_OTEL_RESPONSE_MAX_LENGTH", raising=False)
    monkeypatch.chdir(tmp_path)

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
