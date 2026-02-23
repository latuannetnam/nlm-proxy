"""Tests for AuthRefreshService and related auth refresh functionality."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class TestAuthRefreshService:
    """Test AuthRefreshService lifecycle and refresh logic."""

    def test_import_path_fixed(self):
        """Verify the headless auth import path is correct (was broken)."""
        from nlm_proxy.core.auth_cli import run_headless_auth  # noqa: F401
        # If this import succeeds, the corrected import path works

    def test_service_starts_and_stops(self):
        """AuthRefreshService should start daemon threads and stop cleanly."""
        from nlm_proxy.core.auth_refresh import AuthRefreshService

        # Use long intervals so loops don't run during the test
        service = AuthRefreshService(
            csrf_refresh_interval=9999,
            cookie_refresh_interval=9999,
        )
        assert not service.is_running

        service.start()
        time.sleep(0.1)  # Allow threads to spin up
        assert service.is_running

        service.stop()
        assert not service.is_running

    def test_service_start_idempotent(self):
        """Calling start() twice should not create extra threads."""
        from nlm_proxy.core.auth_refresh import AuthRefreshService

        service = AuthRefreshService(csrf_refresh_interval=9999, cookie_refresh_interval=9999)
        service.start()
        time.sleep(0.05)

        original_csrf_thread = service._csrf_thread

        service.start()  # Second call -- should be a no-op
        time.sleep(0.05)

        # Thread should not have been replaced
        assert service._csrf_thread is original_csrf_thread

        service.stop()

    def test_csrf_refresh_no_cached_tokens(self):
        """_do_csrf_refresh should return False when no tokens are cached."""
        from nlm_proxy.core.auth_refresh import AuthRefreshService

        service = AuthRefreshService()

        with patch("nlm_proxy.core.auth.load_cached_tokens", return_value=None):
            result = service._do_csrf_refresh()

        assert result is False

    def test_csrf_refresh_redirects_to_login(self):
        """_do_csrf_refresh should return False when cookies are expired (login redirect)."""
        from nlm_proxy.core.auth_refresh import AuthRefreshService
        from nlm_proxy.core.auth import AuthTokens

        service = AuthRefreshService()
        fake_tokens = AuthTokens(cookies={"SID": "x"})

        mock_response = MagicMock()
        mock_response.url = "https://accounts.google.com/ServiceLogin"
        mock_response.status_code = 200
        mock_response.text = "login page"

        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_response
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)

        with patch("nlm_proxy.core.auth.load_cached_tokens", return_value=fake_tokens), \
             patch("nlm_proxy.core.auth_cli.check_if_logged_in_by_url", return_value=False), \
             patch("httpx.Client", return_value=mock_http_client):
            result = service._do_csrf_refresh()

        assert result is False

    def test_csrf_refresh_success(self):
        """_do_csrf_refresh should return True and save updated tokens on success."""
        from nlm_proxy.core.auth_refresh import AuthRefreshService
        from nlm_proxy.core.auth import AuthTokens

        service = AuthRefreshService()
        fake_tokens = AuthTokens(cookies={"SID": "x"}, csrf_token="old_csrf")

        mock_response = MagicMock()
        mock_response.url = "https://notebooklm.google.com/"
        mock_response.status_code = 200
        mock_response.text = '<script>{"SNlM0e":"new_csrf","FdrFJe":"123456"}</script>'

        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_response
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)

        with patch("nlm_proxy.core.auth.load_cached_tokens", return_value=fake_tokens), \
             patch("nlm_proxy.core.auth.save_tokens_to_cache") as mock_save, \
             patch("nlm_proxy.core.auth.extract_csrf_from_page_source", return_value="new_csrf"), \
             patch("nlm_proxy.core.auth_cli.extract_session_id_from_html", return_value="123456"), \
             patch("nlm_proxy.core.auth_cli.check_if_logged_in_by_url", return_value=True), \
             patch("httpx.Client", return_value=mock_http_client):
            result = service._do_csrf_refresh()

        assert result is True
        mock_save.assert_called_once()
        assert fake_tokens.csrf_token == "new_csrf"

    def test_cookie_refresh_no_chrome_profile(self):
        """_do_cookie_refresh should return False when no Chrome profile exists."""
        from nlm_proxy.core.auth_refresh import AuthRefreshService

        service = AuthRefreshService()

        with patch("nlm_proxy.core.auth_cli.has_chrome_profile", return_value=False):
            result = service._do_cookie_refresh()

        assert result is False

    def test_cookie_refresh_headless_auth_succeeds(self):
        """_do_cookie_refresh should return True when headless auth succeeds."""
        from nlm_proxy.core.auth_refresh import AuthRefreshService
        from nlm_proxy.core.auth import AuthTokens

        service = AuthRefreshService()
        fresh_tokens = AuthTokens(
            cookies={"SID": "fresh"},
            csrf_token="fresh_csrf",
            session_id="fresh_sid",
        )

        with patch("nlm_proxy.core.auth_cli.has_chrome_profile", return_value=True), \
             patch("nlm_proxy.core.auth_cli.run_headless_auth", return_value=fresh_tokens):
            result = service._do_cookie_refresh()

        assert result is True

    def test_cookie_refresh_headless_auth_fails(self):
        """_do_cookie_refresh should return False when headless auth returns None."""
        from nlm_proxy.core.auth_refresh import AuthRefreshService

        service = AuthRefreshService()

        with patch("nlm_proxy.core.auth_cli.has_chrome_profile", return_value=True), \
             patch("nlm_proxy.core.auth_cli.run_headless_auth", return_value=None):
            result = service._do_cookie_refresh()

        assert result is False



class TestStandaloneRefreshHelpers:
    """Test the standalone helpers used by the CLI auth refresh command."""

    def test_refresh_csrf_once(self):
        """refresh_csrf_once() should delegate to AuthRefreshService._do_csrf_refresh."""
        from nlm_proxy.core.auth_refresh import refresh_csrf_once

        with patch("nlm_proxy.core.auth_refresh.AuthRefreshService._do_csrf_refresh",
                   return_value=True) as mock_method:
            result = refresh_csrf_once()

        assert result is True
        mock_method.assert_called_once()

    def test_refresh_cookies_once(self):
        """refresh_cookies_once() should delegate to AuthRefreshService._do_cookie_refresh."""
        from nlm_proxy.core.auth_refresh import refresh_cookies_once

        with patch("nlm_proxy.core.auth_refresh.AuthRefreshService._do_cookie_refresh",
                   return_value=False) as mock_method:
            result = refresh_cookies_once()

        assert result is False
        mock_method.assert_called_once()


class TestConfigNewFields:
    """Test that new AuthSettings fields load correctly."""

    def test_auth_settings_new_fields_have_defaults(self):
        """New AuthSettings fields should have sensible defaults."""
        from nlm_proxy.core.config import AuthSettings

        # Reset singleton so test gets a fresh instance
        import nlm_proxy.core.config as cfg_module
        original = cfg_module._auth
        cfg_module._auth = None

        try:
            settings = AuthSettings()
            assert settings.auto_refresh_enabled is True
            assert settings.csrf_refresh_interval == 1800    # 30 min
            assert settings.cookie_refresh_interval == 21600  # 6 h
            assert settings.headless_port == 9223
        finally:
            cfg_module._auth = original

    def test_auth_settings_env_override(self, monkeypatch):
        """NLM_PROXY_AUTH_* env vars should override AuthSettings defaults."""
        from nlm_proxy.core.config import AuthSettings

        monkeypatch.setenv("NLM_PROXY_AUTH_AUTO_REFRESH_ENABLED", "false")
        monkeypatch.setenv("NLM_PROXY_AUTH_CSRF_REFRESH_INTERVAL", "300")
        monkeypatch.setenv("NLM_PROXY_AUTH_COOKIE_REFRESH_INTERVAL", "7200")
        monkeypatch.setenv("NLM_PROXY_AUTH_HEADLESS_PORT", "9224")

        settings = AuthSettings()
        assert settings.auto_refresh_enabled is False
        assert settings.csrf_refresh_interval == 300
        assert settings.cookie_refresh_interval == 7200
        assert settings.headless_port == 9224
