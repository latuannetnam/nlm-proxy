"""Background authentication token refresh service for NLM Proxy.

Proactively refreshes CSRF tokens (via HTTP page fetch) and Google cookies
(via headless Chrome) to prevent authentication failures during long-running
proxy sessions.
"""

import asyncio
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("nlm_proxy.auth_refresh")


class AuthRefreshService:
    """Background service that proactively refreshes NotebookLM auth tokens.

    Two refresh loops run as daemon threads:
    - CSRF/Session refresh: re-fetches the NotebookLM homepage to extract fresh
      CSRF token (SNlM0e) and session ID (FdrFJe). Fast (~1-2s), requires
      valid cookies. Runs every `csrf_refresh_interval` seconds (default: 30 min).
    - Cookie refresh: runs headless Chrome with the saved Chrome profile to
      extract fresh Google cookies. Slower (~10s), requires a saved Chrome login.
      Runs every `cookie_refresh_interval` seconds (default: 6 hours).

    Both loops write the refreshed tokens to ~/.nlm-proxy/auth.json so that
    running proxy servers pick them up on the next request (via the existing
    Layer 2 disk-reload mechanism in NotebookLMClient._call_rpc).
    """

    def __init__(
        self,
        csrf_refresh_interval: int = 1800,    # 30 minutes
        cookie_refresh_interval: int = 21600,  # 6 hours
        headless_port: int = 9223,
    ):
        """Initialize the refresh service.

        Args:
            csrf_refresh_interval: Seconds between CSRF/session token refreshes.
            cookie_refresh_interval: Seconds between full cookie refreshes via headless Chrome.
            headless_port: Chrome DevTools port for headless auth (should differ from
                           the interactive auth port to avoid conflicts).
        """
        self.csrf_refresh_interval = csrf_refresh_interval
        self.cookie_refresh_interval = cookie_refresh_interval
        self.headless_port = headless_port

        self._stop_event = threading.Event()
        self._csrf_thread: Optional[threading.Thread] = None
        self._cookie_thread: Optional[threading.Thread] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the background refresh threads."""
        if self._csrf_thread and self._csrf_thread.is_alive():
            logger.warning("[AUTH_REFRESH] Service already running; ignoring start()")
            return

        self._stop_event.clear()

        self._csrf_thread = threading.Thread(
            target=self._csrf_refresh_loop,
            name="auth-csrf-refresh",
            daemon=True,
        )
        self._csrf_thread.start()

        self._cookie_thread = threading.Thread(
            target=self._cookie_refresh_loop,
            name="auth-cookie-refresh",
            daemon=True,
        )
        self._cookie_thread.start()

        logger.info(
            f"[AUTH_REFRESH] Service started — CSRF every {self.csrf_refresh_interval}s, "
            f"cookies every {self.cookie_refresh_interval}s"
        )

    def stop(self) -> None:
        """Signal the refresh threads to stop and wait for them to exit."""
        self._stop_event.set()

        if self._csrf_thread:
            self._csrf_thread.join(timeout=5)
            self._csrf_thread = None

        if self._cookie_thread:
            self._cookie_thread.join(timeout=5)
            self._cookie_thread = None

        logger.info("[AUTH_REFRESH] Service stopped")

    @property
    def is_running(self) -> bool:
        """Return True if the service threads are alive."""
        return (
            (self._csrf_thread is not None and self._csrf_thread.is_alive())
            or (self._cookie_thread is not None and self._cookie_thread.is_alive())
        )

    # =========================================================================
    # CSRF Refresh Loop
    # =========================================================================

    def _csrf_refresh_loop(self) -> None:
        """Run the CSRF/session token refresh on a fixed interval.

        Waits for `csrf_refresh_interval` seconds before the first refresh,
        then repeats until the stop event is set.
        """
        logger.debug(
            f"[AUTH_REFRESH] CSRF thread started (interval={self.csrf_refresh_interval}s)"
        )
        while not self._stop_event.wait(timeout=self.csrf_refresh_interval):
            try:
                self._do_csrf_refresh()
            except Exception as e:
                logger.warning(f"[AUTH_REFRESH] CSRF refresh error (will retry): {e}")

        logger.debug("[AUTH_REFRESH] CSRF thread exiting")

    def _do_csrf_refresh(self) -> bool:
        """Fetch the NotebookLM homepage and update the cached CSRF/session tokens.

        Returns:
            True on success, False if no cookies are cached or refresh failed.
        """
        from nlm_proxy.core.auth import load_cached_tokens, save_tokens_to_cache, extract_csrf_from_page_source
        from nlm_proxy.core.auth_cli import (
            extract_session_id_from_html,
            check_if_logged_in_by_url,
        )
        import httpx

        cached = load_cached_tokens()
        if not cached or not cached.cookies:
            logger.debug("[AUTH_REFRESH] No cached tokens — skipping CSRF refresh")
            return False

        cookie_header = "; ".join(f"{k}={v}" for k, v in cached.cookies.items())

        # Browser-like headers required for page fetch (same as NotebookLMClient)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cookie": cookie_header,
        }

        try:
            with httpx.Client(headers=headers, follow_redirects=True, timeout=15.0) as client:
                response = client.get("https://notebooklm.google.com/")

            # Check for redirect to login page (cookies expired)
            if not check_if_logged_in_by_url(str(response.url)):
                logger.warning(
                    "[AUTH_REFRESH] Cookies expired — redirected to Google login. "
                    "Run 'nlm-proxy auth refresh --full' or 'nlm-proxy auth extract' to re-authenticate."
                )
                return False

            if response.status_code != 200:
                logger.warning(
                    f"[AUTH_REFRESH] CSRF refresh failed: HTTP {response.status_code}"
                )
                return False

            html = response.text
            csrf_token = extract_csrf_from_page_source(html)
            session_id = extract_session_id_from_html(html)

            if not csrf_token:
                logger.warning(
                    "[AUTH_REFRESH] Could not extract CSRF token from page — "
                    "NotebookLM page structure may have changed"
                )
                return False

            # Update the cached tokens in-place
            cached.csrf_token = csrf_token
            if session_id:
                cached.session_id = session_id

            save_tokens_to_cache(cached, silent=True)
            logger.info("[AUTH_REFRESH] CSRF token refreshed successfully")
            return True

        except Exception as e:
            logger.warning(f"[AUTH_REFRESH] CSRF refresh HTTP error: {e}")
            return False

    # =========================================================================
    # Cookie Refresh Loop
    # =========================================================================

    def _cookie_refresh_loop(self) -> None:
        """Run the full cookie refresh (via headless Chrome) on a fixed interval.

        Waits for `cookie_refresh_interval` seconds before the first refresh,
        then repeats until the stop event is set.
        """
        logger.debug(
            f"[AUTH_REFRESH] Cookie thread started (interval={self.cookie_refresh_interval}s)"
        )
        while not self._stop_event.wait(timeout=self.cookie_refresh_interval):
            try:
                self._do_cookie_refresh()
            except Exception as e:
                logger.warning(f"[AUTH_REFRESH] Cookie refresh error (will retry): {e}")

        logger.debug("[AUTH_REFRESH] Cookie thread exiting")

    def _do_cookie_refresh(self) -> bool:
        """Launch headless Chrome and extract fresh Google cookies.

        Only works if the user has previously run 'nlm-proxy auth extract'
        and the Chrome profile at ~/.notebooklm-mcp/chrome-profile has a
        saved Google login session.

        Returns:
            True on success, False if headless auth is unavailable or failed.
        """
        from nlm_proxy.core.auth_cli import run_headless_auth, has_chrome_profile

        if not has_chrome_profile():
            logger.debug(
                "[AUTH_REFRESH] No Chrome profile found — skipping cookie refresh. "
                "Run 'nlm-proxy auth extract' once to enable automatic cookie refresh."
            )
            return False

        logger.info("[AUTH_REFRESH] Running headless Chrome cookie refresh...")
        tokens = run_headless_auth(port=self.headless_port)
        if tokens:
            logger.info("[AUTH_REFRESH] Cookies refreshed successfully via headless Chrome")
            return True
        else:
            logger.warning(
                "[AUTH_REFRESH] Headless Chrome cookie refresh failed — "
                "Chrome profile may have an expired Google session. "
                "Run 'nlm-proxy auth extract' to re-authenticate."
            )
            return False


# =========================================================================
# Standalone refresh helpers (used by the CLI `auth refresh` command)
# =========================================================================

def refresh_csrf_once() -> bool:
    """Perform a single CSRF/session token refresh. Used by `nlm-proxy auth refresh`.

    Returns:
        True on success, False otherwise.
    """
    service = AuthRefreshService()
    return service._do_csrf_refresh()


def refresh_cookies_once(headless_port: int = 9223) -> bool:
    """Perform a single full cookie refresh via headless Chrome.
    Used by `nlm-proxy auth refresh --full`.

    Returns:
        True on success, False otherwise.
    """
    service = AuthRefreshService(headless_port=headless_port)
    return service._do_cookie_refresh()
