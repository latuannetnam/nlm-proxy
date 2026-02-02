"""Tests for configuration system.

Tests verify:
1. Default values work correctly
2. Environment variables override defaults
3. .env file loading works
4. Settings precedence is correct (env > .env > defaults)
5. All settings classes load properly
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch


class TestSharedSettings:
    """Test SharedSettings class."""

    def test_default_values(self):
        """Default values should be applied when no env vars set."""
        # Clear any existing env vars
        env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_")}
        with patch.dict(os.environ, env, clear=True):
            # Force reload by clearing singleton
            import nlm_proxy.core.config as config
            config._shared = None

            settings = config.SharedSettings()

            assert settings.debug is False
            assert settings.auth_dir == Path.home() / ".nlm-proxy"

    def test_env_override_debug(self):
        """NLM_PROXY_DEBUG should override default."""
        with patch.dict(os.environ, {"NLM_PROXY_DEBUG": "true"}, clear=False):
            from nlm_proxy.core.config import SharedSettings
            settings = SharedSettings()
            assert settings.debug is True

    def test_env_override_auth_dir(self):
        """NLM_PROXY_AUTH_DIR should override default."""
        with patch.dict(os.environ, {"NLM_PROXY_AUTH_DIR": "/custom/path"}, clear=False):
            from nlm_proxy.core.config import SharedSettings
            settings = SharedSettings()
            assert settings.auth_dir == Path("/custom/path")


class TestMCPSettings:
    """Test MCPSettings class."""

    def test_default_values(self):
        """Default values for MCP settings."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_MCP_")}
        with patch.dict(os.environ, env, clear=True):
            from nlm_proxy.core.config import MCPSettings
            settings = MCPSettings()

            assert settings.port == 8000
            assert settings.transport == "stdio"

    def test_env_override_port(self):
        """NLM_PROXY_MCP_PORT should override default."""
        with patch.dict(os.environ, {"NLM_PROXY_MCP_PORT": "9000"}, clear=False):
            from nlm_proxy.core.config import MCPSettings
            settings = MCPSettings()
            assert settings.port == 9000

    def test_env_override_transport(self):
        """NLM_PROXY_MCP_TRANSPORT should override default."""
        with patch.dict(os.environ, {"NLM_PROXY_MCP_TRANSPORT": "http"}, clear=False):
            from nlm_proxy.core.config import MCPSettings
            settings = MCPSettings()
            assert settings.transport == "http"


class TestOpenAISettings:
    """Test OpenAISettings class."""

    def test_default_values(self):
        """Default values for OpenAI settings (with required api_key)."""
        # Test that env vars are picked up correctly
        env_vars = {
            "NLM_PROXY_OPENAI_API_KEY": "test-key",
            "NLM_PROXY_OPENAI_HOST": "0.0.0.0",
            "NLM_PROXY_OPENAI_PORT": "8080",
            "NLM_PROXY_OPENAI_SESSION_TTL": "86400",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            from nlm_proxy.core.config import OpenAISettings
            settings = OpenAISettings()

            assert settings.host == "0.0.0.0"
            assert settings.port == 8080
            assert settings.session_ttl == 86400
            assert settings.api_key == "test-key"

    def test_env_override_all(self):
        """All OpenAI env vars should work."""
        env_vars = {
            "NLM_PROXY_OPENAI_HOST": "127.0.0.1",
            "NLM_PROXY_OPENAI_PORT": "3000",
            "NLM_PROXY_OPENAI_SESSION_TTL": "3600",
            "NLM_PROXY_OPENAI_API_KEY": "my-secret-key",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            from nlm_proxy.core.config import OpenAISettings
            settings = OpenAISettings()

            assert settings.host == "127.0.0.1"
            assert settings.port == 3000
            assert settings.session_ttl == 3600
            assert settings.api_key == "my-secret-key"

    def test_api_key_required(self):
        """OpenAISettings should require api_key."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_OPENAI_")}
        with patch.dict(os.environ, env, clear=True):
            from pydantic import ValidationError
            from nlm_proxy.core.config import OpenAISettings
            with pytest.raises(ValidationError) as exc_info:
                OpenAISettings()
            assert "api_key" in str(exc_info.value)

    def test_api_key_from_env(self):
        """NLM_PROXY_OPENAI_API_KEY should set api_key."""
        with patch.dict(os.environ, {"NLM_PROXY_OPENAI_API_KEY": "test-key-123"}, clear=False):
            from nlm_proxy.core.config import OpenAISettings
            settings = OpenAISettings()
            assert settings.api_key == "test-key-123"


class TestAuthSettings:
    """Test AuthSettings class."""

    def test_default_values(self):
        """Default values for Auth settings."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_AUTH_")}
        with patch.dict(os.environ, env, clear=True):
            from nlm_proxy.core.config import AuthSettings
            settings = AuthSettings()

            assert settings.chrome_port == 9222
            assert settings.auto_launch is True

    def test_env_override_chrome_port(self):
        """NLM_PROXY_AUTH_CHROME_PORT should override default."""
        with patch.dict(os.environ, {"NLM_PROXY_AUTH_CHROME_PORT": "9333"}, clear=False):
            from nlm_proxy.core.config import AuthSettings
            settings = AuthSettings()
            assert settings.chrome_port == 9333

    def test_env_override_auto_launch(self):
        """NLM_PROXY_AUTH_AUTO_LAUNCH should override default."""
        with patch.dict(os.environ, {"NLM_PROXY_AUTH_AUTO_LAUNCH": "false"}, clear=False):
            from nlm_proxy.core.config import AuthSettings
            settings = AuthSettings()
            assert settings.auto_launch is False


class TestLoggingSettings:
    """Test LoggingSettings class (existing, unchanged)."""

    def test_default_values(self):
        """Default values for Logging settings."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_LOG_")}
        with patch.dict(os.environ, env, clear=True):
            from nlm_proxy.core.config import LoggingSettings
            settings = LoggingSettings()

            assert settings.level == "INFO"
            assert settings.max_size == 10485760
            assert settings.backup_count == 5

    def test_env_override_level(self):
        """NLM_PROXY_LOG_LEVEL should override default."""
        with patch.dict(os.environ, {"NLM_PROXY_LOG_LEVEL": "DEBUG"}, clear=False):
            from nlm_proxy.core.config import LoggingSettings
            settings = LoggingSettings()
            assert settings.level == "DEBUG"


class TestEnvFileLoading:
    """Test .env file loading."""

    def test_env_file_loading(self, tmp_path):
        """Settings should load from .env file."""
        # Create a temp .env file
        env_file = tmp_path / ".env"
        env_file.write_text("NLM_PROXY_MCP_PORT=7777\n")

        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Clear env and reload
            env = {k: v for k, v in os.environ.items() if not k.startswith("NLM_PROXY_MCP_")}
            with patch.dict(os.environ, env, clear=True):
                from nlm_proxy.core.config import MCPSettings
                settings = MCPSettings()
                assert settings.port == 7777
        finally:
            os.chdir(original_cwd)

    def test_env_var_beats_env_file(self, tmp_path):
        """Environment variable should override .env file."""
        # Create a temp .env file
        env_file = tmp_path / ".env"
        env_file.write_text("NLM_PROXY_MCP_PORT=7777\n")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Set env var that should win
            with patch.dict(os.environ, {"NLM_PROXY_MCP_PORT": "8888"}, clear=False):
                from nlm_proxy.core.config import MCPSettings
                settings = MCPSettings()
                assert settings.port == 8888  # Env var wins
        finally:
            os.chdir(original_cwd)


class TestSingletonAccessors:
    """Test singleton accessor functions."""

    def test_get_shared_settings_returns_same_instance(self):
        """get_shared_settings should return singleton."""
        import nlm_proxy.core.config as config
        config._shared = None  # Reset

        s1 = config.get_shared_settings()
        s2 = config.get_shared_settings()
        assert s1 is s2

    def test_get_mcp_settings_returns_same_instance(self):
        """get_mcp_settings should return singleton."""
        import nlm_proxy.core.config as config
        config._mcp = None  # Reset

        s1 = config.get_mcp_settings()
        s2 = config.get_mcp_settings()
        assert s1 is s2

    def test_get_openai_settings_returns_same_instance(self):
        """get_openai_settings should return singleton."""
        import nlm_proxy.core.config as config
        config._openai = None  # Reset

        s1 = config.get_openai_settings()
        s2 = config.get_openai_settings()
        assert s1 is s2

    def test_get_auth_settings_returns_same_instance(self):
        """get_auth_settings should return singleton."""
        import nlm_proxy.core.config as config
        config._auth = None  # Reset

        s1 = config.get_auth_settings()
        s2 = config.get_auth_settings()
        assert s1 is s2


class TestSmartRoutingSettings:
    """Test SmartRoutingSettings class."""

    def test_smart_routing_settings_defaults(self):
        """Test SmartRoutingSettings has correct defaults."""
        from nlm_proxy.core.config import SmartRoutingSettings

        settings = SmartRoutingSettings(llm_api_key="test-key")

        assert settings.llm_base_url == "https://api.openai.com/v1"
        assert settings.llm_api_key == "test-key"
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.router_model_name == "knowledge-finder"
        assert settings.allowed_notebooks == []
        assert settings.summary_cache_ttl == 3600

    def test_smart_routing_settings_from_env(self, monkeypatch):
        """Test SmartRoutingSettings loads from environment."""
        monkeypatch.setenv("NLM_PROXY_ROUTING_LLM_BASE_URL", "https://custom.api/v1")
        monkeypatch.setenv("NLM_PROXY_ROUTING_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("NLM_PROXY_ROUTING_LLM_MODEL", "gpt-4o")

        from nlm_proxy.core.config import SmartRoutingSettings
        settings = SmartRoutingSettings()

        assert settings.llm_base_url == "https://custom.api/v1"
        assert settings.llm_api_key == "sk-test"
        assert settings.llm_model == "gpt-4o"

    def test_get_routing_settings_returns_same_instance(self):
        """get_routing_settings should return singleton."""
        import nlm_proxy.core.config as config
        config._routing = None  # Reset

        s1 = config.get_routing_settings()
        s2 = config.get_routing_settings()
        assert s1 is s2
