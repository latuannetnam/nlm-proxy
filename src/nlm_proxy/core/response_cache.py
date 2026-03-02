"""Response cache with three-layer lookup.

Layer 1: Exact hash match (0ms) — always active
Layer 2: Embedding pre-filter via fastembed + NumPy (~10-30ms) — smart routing only
Layer 3: LLM semantic verification (~1-2s) — smart routing only

Cache is global across all users, keyed by (notebook_id, normalized_query).
Supports LRU eviction and TTL-based expiration.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A single cached response entry."""

    query: str  # Original query text (for LLM comparison)
    query_hash: str  # Normalized hash key for exact match
    notebook_id: str  # Which notebook answered this
    answer: str  # The full answer text
    thinking: str | None  # Thinking/reasoning text
    conversation_id: str  # The conversation_id from NLM
    embedding: list[float] | None  # Pre-normalized query embedding vector
    cached_at: float  # time.time() when cached
    hit_count: int = 0  # Number of cache hits


class ResponseCache:
    """Three-layer response cache with LRU + TTL.

    Layer 1: exact hash match (synchronous)
    Layer 2: embedding similarity pre-filter (requires fastembed)
    Layer 3: LLM semantic verification (async, requires llm_client)
    """

    def __init__(
        self,
        max_entries: int = 1000,
        ttl_seconds: int = 14400,
        semantic_enabled: bool = True,
        llm_client: object | None = None,
        embedding_model: str | None = None,
        similarity_threshold: float = 0.7,
        similarity_exact_threshold: float = 0.95,
        top_k: int = 10,
    ):
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._semantic_enabled = semantic_enabled
        self._llm_client = llm_client
        self._similarity_threshold = similarity_threshold
        self._similarity_exact_threshold = similarity_exact_threshold
        self._top_k = top_k

        # Layer 1: hash → CachedResponse
        self._cache_by_hash: dict[str, CachedResponse] = {}
        # Notebook partition: notebook_id → list of CachedResponse
        self._cache_by_notebook: dict[str, list[CachedResponse]] = {}
        # LRU ordering: list of hash keys, most recent at end
        self._lru_order: list[str] = []

        # Layer 2: NumPy matrices per notebook (lazy build)
        self._notebook_matrices: dict[str, object] = {}  # np.ndarray
        self._matrix_dirty: dict[str, bool] = {}

        # Embedding model (loaded lazily)
        self._embedding_model_name = embedding_model
        self._embedding_model_instance = None

        # Thread safety
        self._lock = threading.Lock()

    # ── Hashing ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(notebook_id: str, query: str) -> str:
        """Compute deterministic hash key from notebook_id + normalized query."""
        normalized = f"{notebook_id}:{query.strip().lower()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ── Layer 1: Exact Match ─────────────────────────────────────────────

    def store(
        self,
        notebook_id: str,
        query: str,
        answer: str,
        thinking: str | None,
        conversation_id: str,
        embedding: list[float] | None = None,
    ) -> None:
        """Store a response in the cache.

        Empty/error responses are silently ignored.
        If an entry with the same hash exists, it is updated (bypass_cache refresh).
        """
        if not answer or not answer.strip():
            return

        query_hash = self._compute_hash(notebook_id, query)

        with self._lock:
            # Check if entry already exists (update case)
            existing = self._cache_by_hash.get(query_hash)
            if existing:
                # Update existing entry
                existing.answer = answer
                existing.thinking = thinking
                existing.conversation_id = conversation_id
                existing.cached_at = time.time()
                existing.hit_count = 0
                if embedding is not None:
                    existing.embedding = embedding
                    self._matrix_dirty[notebook_id] = True
                # Move to end of LRU
                if query_hash in self._lru_order:
                    self._lru_order.remove(query_hash)
                self._lru_order.append(query_hash)
                return

            # Create new entry
            entry = CachedResponse(
                query=query,
                query_hash=query_hash,
                notebook_id=notebook_id,
                answer=answer,
                thinking=thinking,
                conversation_id=conversation_id,
                embedding=embedding,
                cached_at=time.time(),
            )

            # Evict LRU if at capacity
            while len(self._cache_by_hash) >= self._max_entries:
                self._evict_oldest()

            # Store in hash map
            self._cache_by_hash[query_hash] = entry

            # Store in notebook partition
            if notebook_id not in self._cache_by_notebook:
                self._cache_by_notebook[notebook_id] = []
            self._cache_by_notebook[notebook_id].append(entry)

            # Track LRU order
            self._lru_order.append(query_hash)

            # Mark matrix dirty if embedding provided
            if embedding is not None:
                self._matrix_dirty[notebook_id] = True

    def lookup(
        self,
        notebook_id: str,
        query: str,
        bypass_cache: bool = False,
    ) -> CachedResponse | None:
        """Layer 1 exact-match lookup (synchronous).

        Returns CachedResponse on hit, None on miss.
        """
        if bypass_cache:
            return None

        query_hash = self._compute_hash(notebook_id, query)

        with self._lock:
            entry = self._cache_by_hash.get(query_hash)
            if entry is None:
                return None

            # Check TTL
            if time.time() - entry.cached_at > self._ttl_seconds:
                self._remove_entry(entry)
                return None

            # Update hit count and LRU position
            entry.hit_count += 1
            if query_hash in self._lru_order:
                self._lru_order.remove(query_hash)
            self._lru_order.append(query_hash)

            return entry

    # ── Invalidation ─────────────────────────────────────────────────────

    def invalidate_notebook(self, notebook_id: str) -> None:
        """Remove all cached entries for a specific notebook."""
        with self._lock:
            entries = self._cache_by_notebook.pop(notebook_id, [])
            for entry in entries:
                self._cache_by_hash.pop(entry.query_hash, None)
                if entry.query_hash in self._lru_order:
                    self._lru_order.remove(entry.query_hash)

            # Clean up matrices
            self._notebook_matrices.pop(notebook_id, None)
            self._matrix_dirty.pop(notebook_id, None)

        if entries:
            logger.info(
                "[CACHE] Invalidated %d entries for notebook %s",
                len(entries),
                notebook_id,
            )

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._cache_by_hash.clear()
            self._cache_by_notebook.clear()
            self._lru_order.clear()
            self._notebook_matrices.clear()
            self._matrix_dirty.clear()

        logger.info("[CACHE] All entries cleared")

    # ── Internal helpers ─────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry. Must hold self._lock."""
        if not self._lru_order:
            return
        oldest_hash = self._lru_order.pop(0)
        entry = self._cache_by_hash.pop(oldest_hash, None)
        if entry:
            nb_entries = self._cache_by_notebook.get(entry.notebook_id, [])
            if entry in nb_entries:
                nb_entries.remove(entry)
            if not nb_entries:
                self._cache_by_notebook.pop(entry.notebook_id, None)
                self._notebook_matrices.pop(entry.notebook_id, None)
                self._matrix_dirty.pop(entry.notebook_id, None)
            else:
                self._matrix_dirty[entry.notebook_id] = True

    def _remove_entry(self, entry: CachedResponse) -> None:
        """Remove a specific entry. Must hold self._lock."""
        self._cache_by_hash.pop(entry.query_hash, None)
        if entry.query_hash in self._lru_order:
            self._lru_order.remove(entry.query_hash)

        nb_entries = self._cache_by_notebook.get(entry.notebook_id, [])
        if entry in nb_entries:
            nb_entries.remove(entry)
        if not nb_entries:
            self._cache_by_notebook.pop(entry.notebook_id, None)
            self._notebook_matrices.pop(entry.notebook_id, None)
            self._matrix_dirty.pop(entry.notebook_id, None)
        else:
            self._matrix_dirty[entry.notebook_id] = True

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        """Total number of cached entries."""
        return len(self._cache_by_hash)

    @property
    def notebook_count(self) -> int:
        """Number of notebooks with cached entries."""
        return len(self._cache_by_notebook)
