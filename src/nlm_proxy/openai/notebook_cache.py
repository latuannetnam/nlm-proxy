"""Notebook summary cache for smart routing with proactive background refresh."""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nlm_proxy.core.logging import get_logger

if TYPE_CHECKING:
    from nlm_proxy.core import NotebookLMClient

logger = get_logger(__name__)


@dataclass
class NotebookInfo:
    """Cached notebook information."""
    id: str
    title: str
    summary: str
    topics: list[str]
    cached_at: float


class NotebookCache:
    """Thread-safe cache for notebook summaries with proactive background refresh.

    This cache pre-fetches all notebook summaries at initialization and
    refreshes them in the background before the TTL expires, ensuring
    the cache is always warm for incoming requests.
    """

    def __init__(
        self,
        nlm_client: "NotebookLMClient",
        ttl_seconds: int = 3600,
        allowed_notebooks: list[str] | None = None
    ):
        """Initialize the cache with proactive refresh.

        Args:
            nlm_client: NotebookLM client for fetching notebook data
            ttl_seconds: Cache TTL in seconds (default: 1 hour)
            allowed_notebooks: Optional list of notebook IDs to cache (default: all)
        """
        self._nlm_client = nlm_client
        self._ttl_seconds = ttl_seconds
        self._allowed_notebooks = allowed_notebooks or []
        self._cache: dict[str, NotebookInfo] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._refresh_thread: threading.Thread | None = None

        # Pre-fetch on init (blocking) - ensures cache is warm before server accepts requests
        self._initial_fetch()

        # Start background refresh thread
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
            name="notebook-cache-refresh"
        )
        self._refresh_thread.start()
        logger.debug("[CACHE] Background refresh thread started")

    def _initial_fetch(self) -> None:
        """Blocking fetch at startup to warm the cache."""
        logger.info("[CACHE] Performing initial notebook fetch...")
        try:
            asyncio.run(self._fetch_all_summaries())
            logger.info(f"[CACHE] Initial fetch complete: {len(self._cache)} notebooks cached")
        except Exception as e:
            logger.error(f"[CACHE] Initial fetch failed: {e}")

    def _refresh_loop(self) -> None:
        """Background thread that refreshes cache before TTL expires."""
        # Refresh at 80% of TTL to ensure cache never expires
        refresh_interval = self._ttl_seconds * 0.8
        logger.debug(f"[CACHE] Refresh interval set to {refresh_interval:.0f}s (80% of {self._ttl_seconds}s TTL)")

        while not self._shutdown.wait(timeout=refresh_interval):
            try:
                logger.debug("[CACHE] Background refresh starting...")
                asyncio.run(self._fetch_all_summaries())
                logger.info(f"[CACHE] Background refresh complete: {len(self._cache)} notebooks")
            except Exception as e:
                logger.error(f"[CACHE] Background refresh failed: {e}")

    async def _fetch_all_summaries(self) -> None:
        """Fetch all notebook summaries from NotebookLM."""
        notebooks = await self._nlm_client.list_notebooks()
        logger.debug(f"[CACHE] Found {len(notebooks)} notebooks in NotebookLM")

        # Filter to allowed notebooks if configured
        if self._allowed_notebooks:
            notebooks = [nb for nb in notebooks if nb.id in self._allowed_notebooks]
            logger.debug(f"[CACHE] Filtered to {len(notebooks)} allowed notebooks")

        # Fetch summary for each notebook
        for nb in notebooks:
            try:
                logger.debug(f"[CACHE] Fetching summary for: {nb.title} ({nb.id})")
                summary_data = await self._nlm_client.get_notebook_summary(nb.id)
                self.set(
                    notebook_id=nb.id,
                    title=nb.title,
                    summary=summary_data.get("summary", ""),
                    topics=summary_data.get("suggested_topics", [])
                )
            except Exception as e:
                logger.warning(f"[CACHE] Failed to get summary for {nb.id}: {e}")
                # Cache with just the title so routing can still work
                self.set(notebook_id=nb.id, title=nb.title, summary="", topics=[])

    def get(self, notebook_id: str) -> NotebookInfo | None:
        """Get cached notebook info if not expired."""
        with self._lock:
            info = self._cache.get(notebook_id)
            if info is None:
                return None
            if time.time() - info.cached_at > self._ttl_seconds:
                del self._cache[notebook_id]
                return None
            return info

    def set(self, notebook_id: str, title: str, summary: str, topics: list[str]) -> None:
        """Cache notebook info."""
        with self._lock:
            self._cache[notebook_id] = NotebookInfo(
                id=notebook_id,
                title=title,
                summary=summary,
                topics=topics,
                cached_at=time.time()
            )

    def get_all(self) -> list[NotebookInfo]:
        """Get all non-expired cached notebooks."""
        with self._lock:
            current_time = time.time()
            valid = []
            expired = []
            for nb_id, info in self._cache.items():
                if current_time - info.cached_at > self._ttl_seconds:
                    expired.append(nb_id)
                else:
                    valid.append(info)
            for nb_id in expired:
                del self._cache[nb_id]
            return valid

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def shutdown(self) -> None:
        """Stop the refresh thread gracefully."""
        logger.debug("[CACHE] Shutting down refresh thread...")
        self._shutdown.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)
            logger.debug("[CACHE] Refresh thread stopped")
