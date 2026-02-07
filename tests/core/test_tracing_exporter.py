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
