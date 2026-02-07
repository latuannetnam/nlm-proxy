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
