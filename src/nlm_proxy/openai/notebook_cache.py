"""Notebook summary cache for smart routing with proactive background refresh."""

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nlm_proxy.core.logging import get_logger

if TYPE_CHECKING:
    from nlm_proxy.core import NotebookLMClient

logger = get_logger(__name__)

# Configuration defaults
DEFAULT_SOURCE_FETCH_CONCURRENCY = 10
DEFAULT_MAX_SOURCE_TITLES = 15
MAX_SOURCE_TITLE_LENGTH = 100


def _extract_first_sentence(text: str, max_chars: int = 80) -> str:
    """Extract first sentence or truncate to max_chars.

    Handles:
    - Empty/None text
    - Markdown cleanup (removes leading ** or ## markers)
    - Sentence extraction (splits on . ! ?)
    - Word-boundary truncation with ellipsis
    """
    if not text:
        return ""

    # Clean up markdown formatting at the start
    cleaned = text.strip()
    while cleaned.startswith(("**", "##", "# ", "- ")):
        if cleaned.startswith("**"):
            cleaned = cleaned[2:]
        elif cleaned.startswith("##"):
            cleaned = cleaned[2:]
        elif cleaned.startswith("# "):
            cleaned = cleaned[2:]
        elif cleaned.startswith("- "):
            cleaned = cleaned[2:]
        cleaned = cleaned.strip()

    # Try to extract first sentence
    for end_char in ".!?":
        idx = cleaned.find(end_char)
        if 0 < idx < max_chars:
            return cleaned[:idx + 1].strip()

    # No sentence boundary found within limit - truncate at word boundary
    if len(cleaned) <= max_chars:
        return cleaned

    # Find last space before max_chars
    truncated = cleaned[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space] + "..."

    return truncated + "..."


@dataclass
class SourceInfo:
    """Cached source information for a notebook."""
    id: str
    title: str
    source_type: str  # "pdf", "url", "text", "gdoc", etc.
    summary: str = ""  # Stored but not passed to selection LLM
    keywords: list[str] = field(default_factory=list)


@dataclass
class NotebookInfo:
    """Cached notebook information."""
    id: str
    title: str
    summary: str
    topics: list[str]
    cached_at: float
    sources: list[SourceInfo] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        """Total number of sources in the notebook."""
        return len(self.sources)

    @property
    def source_types(self) -> dict[str, int]:
        """Count of sources by type."""
        counts: dict[str, int] = {}
        for src in self.sources:
            counts[src.source_type] = counts.get(src.source_type, 0) + 1
        return counts

    @property
    def source_titles(self) -> list[str]:
        """List of source titles (truncated to max length)."""
        return [
            src.title[:MAX_SOURCE_TITLE_LENGTH] if len(src.title) > MAX_SOURCE_TITLE_LENGTH
            else src.title
            for src in self.sources
        ]

    def get_source_descriptions(
        self,
        max_sources: int = 10,
        max_keywords: int = 5,
        summary_max_chars: int = 80
    ) -> list[dict]:
        """Get source info with keywords and truncated summaries.

        Args:
            max_sources: Max sources to include full descriptions for
            max_keywords: Max keywords per source
            summary_max_chars: Max chars for summary (first sentence or truncated)

        Returns:
            List of dicts with title, keywords, and summary.
            Sources beyond max_sources get title only.
        """
        result = []
        for i, src in enumerate(self.sources):
            title = src.title[:MAX_SOURCE_TITLE_LENGTH] if len(src.title) > MAX_SOURCE_TITLE_LENGTH else src.title

            if i < max_sources:
                # Full description for first N sources
                entry: dict = {"title": title}
                if src.keywords:
                    entry["keywords"] = src.keywords[:max_keywords]
                if src.summary:
                    entry["summary"] = _extract_first_sentence(src.summary, summary_max_chars)
                result.append(entry)
            else:
                # Title only for remaining sources
                result.append({"title": title})

        return result


class NotebookCache:
    """Thread-safe cache for notebook summaries with proactive background refresh.

    This cache pre-fetches all notebook summaries and source information at
    initialization and refreshes them in the background before the TTL expires,
    ensuring the cache is always warm for incoming requests.
    """

    def __init__(
        self,
        nlm_client: "NotebookLMClient",
        ttl_seconds: int = 3600,
        allowed_notebooks: list[str] | None = None,
        source_fetch_concurrency: int | None = None,
    ):
        """Initialize the cache with proactive refresh.

        Args:
            nlm_client: NotebookLM client for fetching notebook data
            ttl_seconds: Cache TTL in seconds (default: 1 hour)
            allowed_notebooks: Optional list of notebook IDs to cache (default: all)
            source_fetch_concurrency: Max concurrent source fetches (default: from env or 10)
        """
        self._nlm_client = nlm_client
        self._ttl_seconds = ttl_seconds
        self._allowed_notebooks = allowed_notebooks or []
        self._cache: dict[str, NotebookInfo] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._refresh_thread: threading.Thread | None = None

        # Configure source fetch concurrency from env or parameter
        self._source_fetch_concurrency = source_fetch_concurrency or int(
            os.environ.get("NLM_PROXY_ROUTING_SOURCE_FETCH_CONCURRENCY", DEFAULT_SOURCE_FETCH_CONCURRENCY)
        )

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
            # Use new_event_loop() instead of asyncio.run() to avoid conflicts
            # with the background thread's event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._fetch_all_summaries())
                logger.info(f"[CACHE] Initial fetch complete: {len(self._cache)} notebooks cached")
                # Close the client to release async resources (locks, httpx client)
                # bound to this event loop. They will be recreated fresh in the
                # background thread's event loop on next use.
                loop.run_until_complete(self._nlm_client.close())
            finally:
                # Clear the global event loop reference BEFORE closing
                # This prevents the background thread from seeing a closed loop
                asyncio.set_event_loop(None)
                loop.close()
        except Exception as e:
            logger.error(f"[CACHE] Initial fetch failed: {e}")

    def _refresh_loop(self) -> None:
        """Background thread that refreshes cache before TTL expires."""
        # Refresh at 80% of TTL to ensure cache never expires
        refresh_interval = self._ttl_seconds * 0.8
        logger.debug(f"[CACHE] Refresh interval set to {refresh_interval:.0f}s (80% of {self._ttl_seconds}s TTL)")

        # Create a persistent event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while not self._shutdown.wait(timeout=refresh_interval):
                try:
                    logger.debug("[CACHE] Background refresh starting...")
                    loop.run_until_complete(self._fetch_all_summaries())
                    logger.info(f"[CACHE] Background refresh complete: {len(self._cache)} notebooks")
                except Exception as e:
                    logger.error(f"[CACHE] Background refresh failed: {e}")
        finally:
            # Clean up the event loop when thread shuts down
            loop.close()

    async def _fetch_all_summaries(self) -> None:
        """Fetch all notebook summaries and source information from NotebookLM."""
        notebooks = await self._nlm_client.list_notebooks()
        logger.debug(f"[CACHE] Found {len(notebooks)} notebooks in NotebookLM")

        # Filter to allowed notebooks if configured
        if self._allowed_notebooks:
            notebooks = [nb for nb in notebooks if nb.id in self._allowed_notebooks]
            logger.debug(f"[CACHE] Filtered to {len(notebooks)} allowed notebooks")

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._source_fetch_concurrency)

        async def fetch_notebook_with_sources(nb):
            """Fetch notebook summary and all source summaries."""
            try:
                logger.debug(f"[CACHE] Fetching summary for: {nb.title} ({nb.id})")

                # Fetch notebook summary and sources list in parallel
                summary_task = self._nlm_client.get_notebook_summary(nb.id)
                sources_task = self._nlm_client.get_notebook_sources_with_types(nb.id)

                summary_data, sources_list = await asyncio.gather(
                    summary_task, sources_task, return_exceptions=True
                )

                # Handle potential exceptions
                if isinstance(summary_data, Exception):
                    logger.warning(f"[CACHE] Failed to get summary for {nb.id}: {summary_data}")
                    summary_data = {"summary": "", "suggested_topics": []}

                if isinstance(sources_list, Exception):
                    logger.warning(f"[CACHE] Failed to get sources for {nb.id}: {sources_list}")
                    sources_list = []

                # Fetch source summaries in parallel with semaphore
                sources = await self._fetch_source_summaries(sources_list, semaphore)

                self.set(
                    notebook_id=nb.id,
                    title=nb.title,
                    summary=summary_data.get("summary", ""),
                    topics=summary_data.get("suggested_topics", []),
                    sources=sources
                )
                logger.debug(f"[CACHE] Cached {nb.title}: {len(sources)} sources")

            except Exception as e:
                logger.warning(f"[CACHE] Failed to fetch notebook {nb.id}: {e}")
                # Cache with just the title so routing can still work
                self.set(notebook_id=nb.id, title=nb.title, summary="", topics=[], sources=[])

        # Fetch all notebooks in parallel
        await asyncio.gather(*[fetch_notebook_with_sources(nb) for nb in notebooks])

    async def _fetch_source_summaries(
        self,
        sources_list: list[dict],
        semaphore: asyncio.Semaphore
    ) -> list[SourceInfo]:
        """Fetch summaries for all sources with concurrency control."""

        async def fetch_single_source(src: dict) -> SourceInfo:
            """Fetch summary for a single source."""
            source_id = src.get("id", "")
            title = src.get("title", "Untitled")
            source_type = src.get("source_type_name", "unknown")

            # Truncate long titles
            if len(title) > MAX_SOURCE_TITLE_LENGTH:
                title = title[:MAX_SOURCE_TITLE_LENGTH]

            async with semaphore:
                try:
                    guide_data = await self._nlm_client.get_source_guide(source_id)
                    return SourceInfo(
                        id=source_id,
                        title=title,
                        source_type=source_type,
                        summary=guide_data.get("summary", ""),
                        keywords=guide_data.get("keywords", [])
                    )
                except Exception as e:
                    logger.warning(f"[CACHE] Failed to get source guide for {source_id}: {e}")
                    # Return source with just basic info
                    return SourceInfo(
                        id=source_id,
                        title=title,
                        source_type=source_type,
                        summary="",
                        keywords=[]
                    )

        # Fetch all source summaries in parallel
        results = await asyncio.gather(*[fetch_single_source(src) for src in sources_list])
        return list(results)

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

    def set(
        self,
        notebook_id: str,
        title: str,
        summary: str,
        topics: list[str],
        sources: list[SourceInfo] | None = None
    ) -> None:
        """Cache notebook info with optional source information."""
        with self._lock:
            self._cache[notebook_id] = NotebookInfo(
                id=notebook_id,
                title=title,
                summary=summary,
                topics=topics,
                cached_at=time.time(),
                sources=sources or []
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
