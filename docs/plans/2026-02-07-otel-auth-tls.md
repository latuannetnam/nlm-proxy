# OTel Authentication & TLS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add authentication (Bearer token) and TLS support to OpenTelemetry tracing with protocol selection (gRPC or HTTP).

**Architecture:** Extend `TracingSettings` with new config fields, refactor exporter creation into protocol-specific factory functions. HTTP exporter enables "TLS + skip verify" which gRPC Python doesn't support.

**Tech Stack:** OpenTelemetry Python SDK, gRPC, requests (via HTTP exporter)

---

## Task 1: Add Configuration Fields

**Files:**
- Modify: `src/nlm_proxy/core/config.py:178-212`
- Test: `tests/core/test_config_tracing.py` (create)

**Step 1: Write the failing test**

Create `tests/core/test_config_tracing.py`:

```python
"""Tests for TracingSettings configuration."""

import os
import pytest
from nlm_proxy.core.config import TracingSettings


class TestTracingSettings:
    """Test TracingSettings fields and defaults."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = TracingSettings()

        assert settings.enabled is False
        assert settings.endpoint == "localhost:4317"
        assert settings.service_name == "nlm-proxy"
        assert settings.protocol == "grpc"
        assert settings.api_key is None
        assert settings.ca_cert_path is None
        assert settings.verify_cert is True
        assert settings.insecure is True

    def test_protocol_validation(self):
        """Test protocol must be grpc or http."""
        settings = TracingSettings(protocol="http")
        assert settings.protocol == "http"

        settings = TracingSettings(protocol="grpc")
        assert settings.protocol == "grpc"

    def test_env_var_loading(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("NLM_PROXY_OTEL_PROTOCOL", "http")
        monkeypatch.setenv("NLM_PROXY_OTEL_API_KEY", "test-token")
        monkeypatch.setenv("NLM_PROXY_OTEL_CA_CERT_PATH", "/path/to/ca.pem")
        monkeypatch.setenv("NLM_PROXY_OTEL_VERIFY_CERT", "false")
        monkeypatch.setenv("NLM_PROXY_OTEL_INSECURE", "false")

        settings = TracingSettings()

        assert settings.protocol == "http"
        assert settings.api_key == "test-token"
        assert settings.ca_cert_path == "/path/to/ca.pem"
        assert settings.verify_cert is False
        assert settings.insecure is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_config_tracing.py -v`
Expected: FAIL with "AttributeError: 'TracingSettings' object has no attribute 'protocol'"

**Step 3: Add new fields to TracingSettings**

Modify `src/nlm_proxy/core/config.py` lines 178-212, update `TracingSettings`:

```python
class TracingSettings(BaseSettings):
    """OpenTelemetry tracing configuration."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    endpoint: str = Field(
        default="localhost:4317",
        description="OTLP collector endpoint (host:port)"
    )
    service_name: str = Field(
        default="nlm-proxy",
        description="Service name in traces"
    )
    protocol: Literal["grpc", "http"] = Field(
        default="grpc",
        description="Exporter protocol: grpc or http"
    )
    api_key: str | None = Field(
        default=None,
        description="Bearer token for collector authentication"
    )
    ca_cert_path: str | None = Field(
        default=None,
        description="Path to CA certificate for TLS verification"
    )
    verify_cert: bool = Field(
        default=True,
        description="Verify server certificate (HTTP only, gRPC always verifies)"
    )
    insecure: bool = Field(
        default=True,
        description="Use plain text (no TLS). Set to false for TLS."
    )
    export_timeout: int = Field(
        default=2,
        description="Export timeout in seconds (default: 2s for fast failure)"
    )
    max_queue_size: int = Field(
        default=2048,
        description="Max span queue size (default: 2048, drops oldest when full)"
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

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_config_tracing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/core/test_config_tracing.py src/nlm_proxy/core/config.py
git commit -m "feat(tracing): add protocol, auth, and TLS config fields"
```

---

## Task 2: Add HTTP Exporter Dependency

**Files:**
- Modify: `pyproject.toml:43-49`

**Step 1: Update otel dependencies**

Modify `pyproject.toml` otel section:

```toml
otel = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp-proto-grpc>=1.20.0",
    "opentelemetry-exporter-otlp-proto-http>=1.20.0",
    "opentelemetry-instrumentation-fastapi>=0.41b0",
    "opentelemetry-instrumentation-httpx>=0.41b0",
]
```

**Step 2: Reinstall dependencies**

Run: `uv cache clean && uv sync --all-extras`
Expected: Dependencies installed successfully

**Step 3: Verify imports work**

Run: `uv run python -c "from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter; print('HTTP exporter OK')"`
Expected: "HTTP exporter OK"

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(deps): add OTLP HTTP exporter dependency"
```

---

## Task 3: Implement Exporter Factory Functions

**Files:**
- Modify: `src/nlm_proxy/core/tracing.py:37-51`
- Test: `tests/core/test_tracing_exporter.py` (create)

**Step 1: Write the failing test**

Create `tests/core/test_tracing_exporter.py`:

```python
"""Tests for OTel exporter creation."""

import pytest
from unittest.mock import patch, MagicMock
from nlm_proxy.core.config import TracingSettings


class TestCreateExporter:
    """Test exporter factory functions."""

    def test_grpc_exporter_insecure(self):
        """Test gRPC exporter with insecure=True."""
        from nlm_proxy.core.tracing import _create_exporter

        settings = TracingSettings(
            enabled=True,
            endpoint="localhost:4317",
            protocol="grpc",
            insecure=True,
        )

        with patch("nlm_proxy.core.tracing._create_grpc_exporter") as mock:
            mock.return_value = MagicMock()
            exporter = _create_exporter(settings)
            mock.assert_called_once()

    def test_http_exporter_with_auth(self):
        """Test HTTP exporter with Bearer token."""
        from nlm_proxy.core.tracing import _create_exporter

        settings = TracingSettings(
            enabled=True,
            endpoint="localhost:4318",
            protocol="http",
            api_key="test-token",
            insecure=False,
            verify_cert=False,
        )

        with patch("nlm_proxy.core.tracing._create_http_exporter") as mock:
            mock.return_value = MagicMock()
            exporter = _create_exporter(settings)
            mock.assert_called_once()
            # Verify headers passed
            call_args = mock.call_args
            headers = call_args[0][1]  # Second positional arg
            assert headers == {"authorization": "Bearer test-token"}

    def test_http_exporter_url_construction_insecure(self):
        """Test HTTP URL is constructed with http:// when insecure."""
        from nlm_proxy.core.tracing import _build_http_url

        url = _build_http_url("localhost:4318", insecure=True)
        assert url == "http://localhost:4318/v1/traces"

    def test_http_exporter_url_construction_secure(self):
        """Test HTTP URL is constructed with https:// when secure."""
        from nlm_proxy.core.tracing import _build_http_url

        url = _build_http_url("collector.example.com:4318", insecure=False)
        assert url == "https://collector.example.com:4318/v1/traces"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_tracing_exporter.py -v`
Expected: FAIL with "cannot import name '_create_exporter'"

**Step 3: Implement exporter factory functions**

Modify `src/nlm_proxy/core/tracing.py`, add after line 20 (before `init_tracing`):

```python
from pathlib import Path
from opentelemetry.sdk.trace.export import SpanExporter


def _build_http_url(endpoint: str, insecure: bool) -> str:
    """Build full HTTP URL from host:port endpoint."""
    scheme = "http" if insecure else "https"
    return f"{scheme}://{endpoint}/v1/traces"


def _create_exporter(settings) -> SpanExporter:
    """Create appropriate exporter based on protocol setting."""
    headers = None
    if settings.api_key:
        headers = {"authorization": f"Bearer {settings.api_key}"}

    if settings.protocol == "http":
        return _create_http_exporter(settings, headers)
    else:
        return _create_grpc_exporter(settings, headers)


def _create_http_exporter(settings, headers: dict | None) -> SpanExporter:
    """Create HTTP exporter with TLS options."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter
    )

    # Build full URL
    endpoint_url = _build_http_url(settings.endpoint, settings.insecure)

    # Determine certificate_verification value
    if settings.insecure:
        # Plain HTTP, no cert verification needed
        cert_verify = True  # Ignored for http://
    elif not settings.verify_cert:
        cert_verify = False
    elif settings.ca_cert_path:
        cert_verify = settings.ca_cert_path
    else:
        cert_verify = True  # System CA

    logger.debug(
        f"[TRACING] Creating HTTP exporter: endpoint={endpoint_url}, "
        f"verify={cert_verify}, auth={'enabled' if headers else 'disabled'}"
    )

    return HTTPSpanExporter(
        endpoint=endpoint_url,
        headers=headers,
        certificate_verification=cert_verify,
        timeout=settings.export_timeout,
    )


def _create_grpc_exporter(settings, headers: dict | None) -> SpanExporter:
    """Create gRPC exporter with TLS options."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter
    )

    credentials = None

    if not settings.insecure:
        from grpc import ssl_channel_credentials

        # Warn if verify_cert=False (not supported in gRPC)
        if not settings.verify_cert:
            logger.warning(
                "[TRACING] verify_cert=False not supported with gRPC protocol, "
                "certificates will be validated. Use protocol=http for skip-verify."
            )

        ca_cert = None
        if settings.ca_cert_path:
            cert_path = Path(settings.ca_cert_path)
            if not cert_path.exists():
                raise FileNotFoundError(
                    f"[TRACING] CA certificate not found: {settings.ca_cert_path}"
                )
            with open(cert_path, "rb") as f:
                ca_cert = f.read()

        credentials = ssl_channel_credentials(root_certificates=ca_cert)

    # gRPC headers format: list of tuples
    grpc_headers = [(k, v) for k, v in headers.items()] if headers else None

    logger.debug(
        f"[TRACING] Creating gRPC exporter: endpoint={settings.endpoint}, "
        f"insecure={settings.insecure}, auth={'enabled' if headers else 'disabled'}"
    )

    return GRPCSpanExporter(
        endpoint=settings.endpoint,
        insecure=settings.insecure,
        credentials=credentials,
        headers=grpc_headers,
        timeout=settings.export_timeout,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_tracing_exporter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/tracing.py tests/core/test_tracing_exporter.py
git commit -m "feat(tracing): add exporter factory functions for gRPC and HTTP"
```

---

## Task 4: Update init_tracing to Use Factory

**Files:**
- Modify: `src/nlm_proxy/core/tracing.py:37-51`
- Test: `tests/core/test_tracing_init.py` (create)

**Step 1: Write the failing test**

Create `tests/core/test_tracing_init.py`:

```python
"""Tests for init_tracing with new config options."""

import pytest
from unittest.mock import patch, MagicMock


class TestInitTracing:
    """Test init_tracing with various configurations."""

    def setup_method(self):
        """Reset tracing state before each test."""
        import nlm_proxy.core.tracing as tracing_module
        tracing_module._initialized = False

    def test_init_with_http_protocol(self):
        """Test initialization with HTTP protocol."""
        from nlm_proxy.core.config import TracingSettings

        settings = TracingSettings(
            enabled=True,
            endpoint="localhost:4318",
            protocol="http",
            insecure=True,
        )

        with patch("nlm_proxy.core.tracing.get_tracing_settings", return_value=settings):
            with patch("nlm_proxy.core.tracing._create_exporter") as mock_create:
                mock_create.return_value = MagicMock()

                from nlm_proxy.core.tracing import init_tracing
                init_tracing()

                mock_create.assert_called_once_with(settings)

    def test_init_logs_tls_mode(self, caplog):
        """Test that TLS mode is logged on init."""
        from nlm_proxy.core.config import TracingSettings

        settings = TracingSettings(
            enabled=True,
            endpoint="localhost:4318",
            protocol="http",
            insecure=False,
            verify_cert=False,
        )

        with patch("nlm_proxy.core.tracing.get_tracing_settings", return_value=settings):
            with patch("nlm_proxy.core.tracing._create_exporter") as mock_create:
                mock_create.return_value = MagicMock()

                from nlm_proxy.core.tracing import init_tracing
                import logging
                with caplog.at_level(logging.DEBUG):
                    init_tracing()

        # Verify logging happened (specific message checked in debug logs)
        assert "TRACING" in caplog.text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_tracing_init.py -v`
Expected: FAIL (init_tracing still uses old exporter creation)

**Step 3: Update init_tracing to use _create_exporter**

Modify `src/nlm_proxy/core/tracing.py`, replace lines 37-51 in `init_tracing()`:

```python
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
        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: settings.service_name})

        # Create and configure tracer provider
        provider = TracerProvider(resource=resource)

        # Create exporter using factory (handles protocol, TLS, auth)
        exporter = _create_exporter(settings)

        # Configure processor to drop spans instead of blocking when queue is full
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=settings.max_queue_size,
            schedule_delay_millis=5000,  # Export every 5s (default)
            max_export_batch_size=512,   # Batch size (default)
            export_timeout_millis=settings.export_timeout * 1000  # Convert to ms
        )

        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Log configuration summary
        tls_status = "TLS" if not settings.insecure else "plain"
        auth_status = "auth=enabled" if settings.api_key else "auth=disabled"
        logger.info(
            f"[TRACING] OpenTelemetry initialized: protocol={settings.protocol}, "
            f"endpoint={settings.endpoint}, {tls_status}, {auth_status}, "
            f"service={settings.service_name}"
        )
        _initialized = True

    except Exception as e:
        logger.error(f"[TRACING] Failed to initialize OpenTelemetry: {e}")
        logger.warning("[TRACING] Tracing disabled - server will continue without observability")
        _initialized = True  # Don't retry, server continues normally
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_tracing_init.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/tracing.py tests/core/test_tracing_init.py
git commit -m "feat(tracing): use exporter factory in init_tracing"
```

---

## Task 5: Add Validation Logging

**Files:**
- Modify: `src/nlm_proxy/core/tracing.py`
- Test: `tests/core/test_tracing_validation.py` (create)

**Step 1: Write the failing test**

Create `tests/core/test_tracing_validation.py`:

```python
"""Tests for tracing configuration validation."""

import pytest
import tempfile
from pathlib import Path


class TestTracingValidation:
    """Test configuration validation and warnings."""

    def test_missing_ca_cert_raises_error(self):
        """Test that missing CA cert file raises FileNotFoundError."""
        from nlm_proxy.core.config import TracingSettings
        from nlm_proxy.core.tracing import _create_grpc_exporter

        settings = TracingSettings(
            enabled=True,
            protocol="grpc",
            insecure=False,
            ca_cert_path="/nonexistent/path/ca.pem",
        )

        with pytest.raises(FileNotFoundError, match="CA certificate not found"):
            _create_grpc_exporter(settings, None)

    def test_valid_ca_cert_loads_successfully(self):
        """Test that valid CA cert file is loaded."""
        from nlm_proxy.core.config import TracingSettings
        from nlm_proxy.core.tracing import _create_grpc_exporter
        from unittest.mock import patch, MagicMock

        # Create a temp cert file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pem", delete=False) as f:
            f.write(b"-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----")
            cert_path = f.name

        try:
            settings = TracingSettings(
                enabled=True,
                protocol="grpc",
                insecure=False,
                ca_cert_path=cert_path,
            )

            with patch("nlm_proxy.core.tracing.ssl_channel_credentials") as mock_ssl:
                with patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"):
                    mock_ssl.return_value = MagicMock()
                    _create_grpc_exporter(settings, None)

                    # Verify cert was read and passed
                    mock_ssl.assert_called_once()
                    call_args = mock_ssl.call_args
                    assert call_args[1]["root_certificates"] is not None
        finally:
            Path(cert_path).unlink()

    def test_grpc_verify_cert_false_logs_warning(self, caplog):
        """Test warning logged when verify_cert=False with gRPC."""
        from nlm_proxy.core.config import TracingSettings
        from nlm_proxy.core.tracing import _create_grpc_exporter
        from unittest.mock import patch, MagicMock
        import logging

        settings = TracingSettings(
            enabled=True,
            protocol="grpc",
            insecure=False,
            verify_cert=False,  # Not supported in gRPC
        )

        with patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"):
            with caplog.at_level(logging.WARNING):
                _create_grpc_exporter(settings, None)

        assert "verify_cert=False not supported with gRPC" in caplog.text
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/core/test_tracing_validation.py -v`
Expected: PASS (validation already implemented in Task 3)

**Step 3: Commit**

```bash
git add tests/core/test_tracing_validation.py
git commit -m "test(tracing): add validation tests for CA cert and warnings"
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `.env.example`
- Modify: `.claude/memory/configuration.md`

**Step 1: Update .env.example**

Add to `.env.example`:

```bash
# =============================================================================
# OpenTelemetry Tracing
# =============================================================================
# NLM_PROXY_OTEL_ENABLED=false
# NLM_PROXY_OTEL_ENDPOINT=localhost:4317
# NLM_PROXY_OTEL_SERVICE_NAME=nlm-proxy

# Protocol: grpc or http
# - grpc: Default, lower overhead, no skip-verify support
# - http: Supports skip-verify for self-signed certs
# NLM_PROXY_OTEL_PROTOCOL=grpc

# TLS Configuration
# NLM_PROXY_OTEL_INSECURE=true          # true=plain text, false=TLS enabled
# NLM_PROXY_OTEL_VERIFY_CERT=true       # false to skip cert validation (HTTP only)
# NLM_PROXY_OTEL_CA_CERT_PATH=          # Path to private CA certificate

# Authentication
# NLM_PROXY_OTEL_API_KEY=               # Bearer token for collector auth
```

**Step 2: Update memory documentation**

Add to `.claude/memory/configuration.md` under tracing section:

```markdown
## Tracing Authentication & TLS

| Variable | Default | Description |
|----------|---------|-------------|
| `NLM_PROXY_OTEL_PROTOCOL` | `grpc` | Exporter protocol: `grpc` or `http` |
| `NLM_PROXY_OTEL_API_KEY` | (none) | Bearer token for collector authentication |
| `NLM_PROXY_OTEL_INSECURE` | `true` | `true`=plain text, `false`=TLS enabled |
| `NLM_PROXY_OTEL_VERIFY_CERT` | `true` | Skip cert validation (HTTP only) |
| `NLM_PROXY_OTEL_CA_CERT_PATH` | (none) | Path to private CA certificate |

### Common Configurations

**Local development (plain text):**
```bash
NLM_PROXY_OTEL_PROTOCOL=grpc
NLM_PROXY_OTEL_INSECURE=true
```

**Production with private CA:**
```bash
NLM_PROXY_OTEL_PROTOCOL=http
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_CA_CERT_PATH=/etc/ssl/otel-ca.pem
NLM_PROXY_OTEL_API_KEY=your-bearer-token
```

**Development with self-signed cert (skip verify):**
```bash
NLM_PROXY_OTEL_PROTOCOL=http
NLM_PROXY_OTEL_INSECURE=false
NLM_PROXY_OTEL_VERIFY_CERT=false
```

**Note:** gRPC protocol does not support `verify_cert=false`. Use HTTP protocol for skip-verify scenarios.
```

**Step 3: Commit**

```bash
git add .env.example .claude/memory/configuration.md
git commit -m "docs: add OTel auth and TLS configuration documentation"
```

---

## Task 7: Integration Test

**Files:**
- Test: `tests/core/test_tracing_integration.py` (create)

**Step 1: Write integration test**

Create `tests/core/test_tracing_integration.py`:

```python
"""Integration tests for tracing with various configurations."""

import pytest
from unittest.mock import patch, MagicMock


class TestTracingIntegration:
    """End-to-end tests for tracing initialization."""

    def setup_method(self):
        """Reset tracing state."""
        import nlm_proxy.core.tracing as tracing_module
        tracing_module._initialized = False

    def test_backward_compatibility_grpc_insecure(self):
        """Test default config (gRPC + insecure) still works."""
        from nlm_proxy.core.config import TracingSettings
        from nlm_proxy.core.tracing import init_tracing

        settings = TracingSettings(
            enabled=True,
            endpoint="localhost:4317",
            # All other fields use defaults
        )

        with patch("nlm_proxy.core.tracing.get_tracing_settings", return_value=settings):
            with patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter") as mock:
                mock.return_value = MagicMock()
                init_tracing()

                # Verify gRPC exporter created with insecure=True
                mock.assert_called_once()
                call_kwargs = mock.call_args[1]
                assert call_kwargs["insecure"] is True

    def test_http_with_bearer_auth(self):
        """Test HTTP exporter with Bearer token auth."""
        from nlm_proxy.core.config import TracingSettings
        from nlm_proxy.core.tracing import init_tracing

        settings = TracingSettings(
            enabled=True,
            endpoint="localhost:4318",
            protocol="http",
            api_key="my-secret-token",
            insecure=True,
        )

        with patch("nlm_proxy.core.tracing.get_tracing_settings", return_value=settings):
            with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as mock:
                mock.return_value = MagicMock()
                init_tracing()

                mock.assert_called_once()
                call_kwargs = mock.call_args[1]
                assert call_kwargs["headers"] == {"authorization": "Bearer my-secret-token"}
                assert call_kwargs["endpoint"] == "http://localhost:4318/v1/traces"

    def test_http_tls_skip_verify(self):
        """Test HTTP with TLS but skip certificate verification."""
        from nlm_proxy.core.config import TracingSettings
        from nlm_proxy.core.tracing import init_tracing

        settings = TracingSettings(
            enabled=True,
            endpoint="collector:4318",
            protocol="http",
            insecure=False,
            verify_cert=False,
        )

        with patch("nlm_proxy.core.tracing.get_tracing_settings", return_value=settings):
            with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as mock:
                mock.return_value = MagicMock()
                init_tracing()

                mock.assert_called_once()
                call_kwargs = mock.call_args[1]
                assert call_kwargs["certificate_verification"] is False
                assert call_kwargs["endpoint"] == "https://collector:4318/v1/traces"
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/core/test_tracing_integration.py -v`
Expected: PASS

**Step 3: Run all tracing tests**

Run: `uv run pytest tests/core/test_tracing*.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/core/test_tracing_integration.py
git commit -m "test(tracing): add integration tests for auth and TLS configs"
```

---

## Verification Checklist

After completing all tasks, verify:

1. **Backward compatibility:** Default config works (gRPC + insecure)
   ```bash
   NLM_PROXY_OTEL_ENABLED=true nlm-proxy serve openai --port 8080
   # Should connect to localhost:4317 with plain gRPC
   ```

2. **HTTP + skip verify:**
   ```bash
   NLM_PROXY_OTEL_ENABLED=true \
   NLM_PROXY_OTEL_PROTOCOL=http \
   NLM_PROXY_OTEL_INSECURE=false \
   NLM_PROXY_OTEL_VERIFY_CERT=false \
   nlm-proxy serve openai --port 8080
   ```

3. **HTTP + private CA + auth:**
   ```bash
   NLM_PROXY_OTEL_ENABLED=true \
   NLM_PROXY_OTEL_PROTOCOL=http \
   NLM_PROXY_OTEL_INSECURE=false \
   NLM_PROXY_OTEL_CA_CERT_PATH=/path/to/ca.pem \
   NLM_PROXY_OTEL_API_KEY=my-token \
   nlm-proxy serve openai --port 8080
   ```

4. **All tests pass:**
   ```bash
   uv run pytest tests/core/test_tracing*.py tests/core/test_config_tracing.py -v
   ```
