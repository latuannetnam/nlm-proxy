"""Tests for TracingSettings configuration."""

import os
import pytest
from unittest.mock import patch
from nlm_proxy.core.config import TracingSettings


class TestTracingSettings:
    """Test TracingSettings fields and defaults."""

    def test_default_values(self, monkeypatch, tmp_path):
        """Test default configuration values."""
        # Clear any environment variables that might affect defaults
        for key in ["ENABLED", "ENDPOINT", "SERVICE_NAME", "PROTOCOL", "API_KEY",
                    "CA_CERT_PATH", "VERIFY_CERT", "INSECURE", "EXPORT_TIMEOUT",
                    "MAX_QUEUE_SIZE", "REQUEST_MAX_LENGTH", "RESPONSE_MAX_LENGTH"]:
            monkeypatch.delenv(f"NLM_PROXY_OTEL_{key}", raising=False)

        # Change to temp directory with no .env file to prevent loading user's config
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))  # Prevent loading from ~/.nlm-proxy/.env

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
