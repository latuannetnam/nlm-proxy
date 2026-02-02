"""Notebook summary cache for smart routing."""

import threading
import time
from dataclasses import dataclass


@dataclass
class NotebookInfo:
    """Cached notebook information."""
    id: str
    title: str
    summary: str
    topics: list[str]
    cached_at: float


class NotebookCache:
    """Thread-safe cache for notebook summaries with TTL expiration."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, NotebookInfo] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, notebook_id: str) -> NotebookInfo | None:
        """Get cached notebook info if not expired."""
        with self._lock:
            info = self._cache.get(notebook_id)
            if info is None:
                return None
            if time.time() - info.cached_at > self._ttl:
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
                if current_time - info.cached_at > self._ttl:
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
