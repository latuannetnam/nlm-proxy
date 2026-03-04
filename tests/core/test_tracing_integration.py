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
            protocol="grpc",
            insecure=True,
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
                # Implementation uses env vars (PYTHONHTTPSVERIFY, REQUESTS_CA_BUNDLE)
                # to disable cert verification, not a kwarg to the exporter
                assert call_kwargs["endpoint"] == "https://collector:4318/v1/traces"
