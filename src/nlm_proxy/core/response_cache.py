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

    # ── Layer 2: Embedding Pre-filter ────────────────────────────────────

    @staticmethod
    def _normalize_embedding(vec: list[float]) -> list[float]:
        """L2-normalize embedding to unit vector for dot-product similarity."""
        import numpy as np
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr.tolist()

    def _rebuild_matrix(self, notebook_id: str) -> None:
        """Rebuild NumPy similarity matrix for a notebook. Must hold self._lock."""
        import numpy as np
        entries = self._cache_by_notebook.get(notebook_id, [])
        embeddings = [e.embedding for e in entries if e.embedding is not None]
        if embeddings:
            self._notebook_matrices[notebook_id] = np.array(
                embeddings, dtype=np.float32
            )
        else:
            self._notebook_matrices.pop(notebook_id, None)
        self._matrix_dirty[notebook_id] = False

    def _find_similar(
        self,
        query_emb: object,  # np.ndarray
        notebook_id: str,
        top_k: int | None = None,
    ) -> list[tuple[float, CachedResponse]]:
        """Find similar cached queries via NumPy vectorized cosine similarity.

        Returns list of (similarity_score, CachedResponse) sorted by score desc.
        Only entries above self._similarity_threshold are returned.
        If any entry is above self._similarity_exact_threshold, only that single
        entry is returned (early termination — skip LLM verification).
        """
        import numpy as np

        if top_k is None:
            top_k = self._top_k

        with self._lock:
            entries = self._cache_by_notebook.get(notebook_id, [])
            # Filter to entries that have embeddings
            entries_with_emb = [e for e in entries if e.embedding is not None]
            if not entries_with_emb:
                return []

            # Rebuild matrix if dirty or missing
            if self._matrix_dirty.get(notebook_id, True):
                self._rebuild_matrix(notebook_id)

            matrix = self._notebook_matrices.get(notebook_id)
            if matrix is None:
                return []

            # Ensure query embedding is a numpy array
            if not isinstance(query_emb, np.ndarray):
                query_emb = np.array(query_emb, dtype=np.float32)

            # Single matrix-vector multiply — all dot products at once
            # Pre-normalized embeddings means dot product = cosine similarity
            similarities = matrix @ query_emb  # (n,)

            # Early termination: near-perfect match → skip LLM verification
            max_sim = float(similarities.max())
            if max_sim >= self._similarity_exact_threshold:
                best_idx = int(similarities.argmax())
                return [(max_sim, entries_with_emb[best_idx])]

            # Filter by threshold and get top-K
            mask = similarities >= self._similarity_threshold
            if not mask.any():
                return []

            filtered_idx = np.where(mask)[0]
            sorted_idx = filtered_idx[
                np.argsort(similarities[filtered_idx])[::-1]
            ][:top_k]
            return [
                (float(similarities[i]), entries_with_emb[i]) for i in sorted_idx
            ]

    def _compute_embedding(self, query: str) -> object | None:
        """Compute query embedding using fastembed model.

        Returns numpy array or None if embedding model is not available.
        """
        if self._embedding_model_instance is None:
            if not self._embedding_model_name:
                return None
            try:
                from fastembed import TextEmbedding
                self._embedding_model_instance = TextEmbedding(
                    self._embedding_model_name
                )
                logger.info(
                    "[CACHE] Embedding model loaded: %s",
                    self._embedding_model_name,
                )
            except ImportError:
                logger.info(
                    "[CACHE] fastembed not installed, semantic matching disabled"
                )
                self._semantic_enabled = False
                return None
            except Exception:
                logger.exception("[CACHE] Failed to load embedding model")
                self._semantic_enabled = False
                return None

        import numpy as np
        try:
            embeddings = list(
                self._embedding_model_instance.embed([query])
            )
            if embeddings:
                emb = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                return emb
        except Exception:
            logger.exception("[CACHE] Failed to compute embedding")
        return None

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
