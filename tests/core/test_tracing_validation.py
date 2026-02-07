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

            with patch("grpc.ssl_channel_credentials") as mock_ssl:
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
