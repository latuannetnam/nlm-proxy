"""Response cache with three-layer lookup.

Layer 1: Exact hash match (0ms) — always active
Layer 2: Embedding pre-filter via LangChain HuggingFace + NumPy (~10-30ms) — smart routing only
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
    Layer 2: embedding similarity pre-filter (requires langchain-huggingface)
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
        # Global index: query-only hash → CachedResponse (no notebook_id)
        self._global_hash_index: dict[str, CachedResponse] = {}
        # Notebook partition: notebook_id → list of CachedResponse
        self._cache_by_notebook: dict[str, list[CachedResponse]] = {}
        # LRU ordering: list of hash keys, most recent at end
        self._lru_order: list[str] = []

        # Alias tracking (aliases not counted in LRU capacity)
        self._alias_hashes: set[str] = set()         # notebook-scoped alias hashes
        self._alias_global_hashes: set[str] = set()   # global alias hashes

        # Layer 2: NumPy matrices per notebook (lazy build)
        self._notebook_matrices: dict[str, object] = {}  # np.ndarray
        self._matrix_dirty: dict[str, bool] = {}

        # Embedding model
        self._embedding_model_name = embedding_model
        self._embedding_model_instance = None
        if self._semantic_enabled and self._embedding_model_name:
            self._load_embedding_model()

        # Warn if low threshold without LLM verification
        if self._similarity_threshold < 0.7 and self._llm_client is None:
            logger.warning(
                "[CACHE] L2 threshold (%.2f) is below 0.7 but no LLM client "
                "configured for L3 verification. Semantic matching may produce "
                "false positives. Consider raising threshold to 0.7+ or "
                "configuring an LLM client.",
                self._similarity_threshold,
            )

        # Thread safety
        self._lock = threading.Lock()

        # Stats counters
        self._stats = {
            "pre_routing_l1_hits": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "misses": 0,
            "bypasses": 0,
        }

    # ── Hashing ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(notebook_id: str, query: str) -> str:
        """Compute deterministic hash key from notebook_id + normalized query."""
        normalized = f"{notebook_id}:{query.strip().lower()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_global_hash(query: str) -> str:
        """Compute hash on query only (no notebook_id) for pre-routing lookup."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ── Pre-routing Global L1 ────────────────────────────────────────────

    def lookup_global(self, query: str) -> tuple[CachedResponse | None, str | None]:
        """Pre-routing L1 lookup: find cached entry by query only (no notebook_id).

        Returns (CachedResponse, hit_type) on hit, (None, None) on miss.
        Caller must validate notebook ACL.
        """
        global_hash = self._compute_global_hash(query)
        with self._lock:
            entry = self._global_hash_index.get(global_hash)
            if entry is None:
                return None, None
            # Check TTL
            age = time.time() - entry.cached_at
            if age > self._ttl_seconds:
                logger.debug("[CACHE] Global L1 EXPIRED for '%s'", query[:80])
                return None, None
            entry.hit_count += 1
            logger.info(
                "[CACHE] Global L1 HIT for '%s' (notebook=%s, hits=%d, age=%.0fs)",
                query[:80], entry.notebook_id[:12], entry.hit_count, age,
            )
            self._stats["pre_routing_l1_hits"] += 1
            return entry, "exact"

    # ── Alias Creation ────────────────────────────────────────────────────

    def create_alias(
        self, notebook_id: str, new_query: str, target_entry: CachedResponse
    ) -> None:
        """Create L1 alias: new query → existing cached entry.

        Aliases are NOT counted in LRU capacity.
        """
        alias_hash = self._compute_hash(notebook_id, new_query)
        global_hash = self._compute_global_hash(new_query)

        with self._lock:
            self._cache_by_hash[alias_hash] = target_entry
            self._global_hash_index[global_hash] = target_entry
            self._alias_hashes.add(alias_hash)
            self._alias_global_hashes.add(global_hash)

        logger.info(
            "[CACHE] Alias created: '%s' → '%s' (notebook=%s)",
            new_query[:60], target_entry.query[:60], notebook_id[:12],
        )

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
                # Update global index
                global_hash = self._compute_global_hash(query)
                self._global_hash_index[global_hash] = existing
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

            # Store in global hash index (query-only, for pre-routing lookup)
            global_hash = self._compute_global_hash(query)
            self._global_hash_index[global_hash] = entry

            # Store in notebook partition
            if notebook_id not in self._cache_by_notebook:
                self._cache_by_notebook[notebook_id] = []
            self._cache_by_notebook[notebook_id].append(entry)

            # Track LRU order
            self._lru_order.append(query_hash)

            # Mark matrix dirty if embedding provided
            if embedding is not None:
                self._matrix_dirty[notebook_id] = True

            logger.info(
                "[CACHE] STORED '%s' (notebook=%s, answer_len=%d, total_entries=%d)",
                query[:80], notebook_id[:12], len(answer),
                len(self._cache_by_hash),
            )

    def lookup(
        self,
        notebook_id: str,
        query: str,
        bypass_cache: bool = False,
    ) -> tuple[CachedResponse | None, str | None]:
        """Layer 1 exact-match lookup (synchronous).

        Returns (CachedResponse, "exact") on hit, (None, None) on miss.
        """
        if bypass_cache:
            logger.debug("[CACHE] L1 BYPASS for '%s'", query[:80])
            self._stats["bypasses"] += 1
            return None, None

        query_hash = self._compute_hash(notebook_id, query)

        with self._lock:
            entry = self._cache_by_hash.get(query_hash)
            if entry is None:
                logger.debug(
                    "[CACHE] L1 MISS for '%s' (hash=%s, notebook=%s)",
                    query[:80], query_hash[:12], notebook_id[:12],
                )
                return None, None

            # Check TTL
            age = time.time() - entry.cached_at
            if age > self._ttl_seconds:
                logger.info(
                    "[CACHE] L1 EXPIRED for '%s' (age=%.0fs, ttl=%ds)",
                    query[:80], age, self._ttl_seconds,
                )
                self._remove_entry(entry)
                return None, None

            # Update hit count and LRU position
            entry.hit_count += 1
            if query_hash in self._lru_order:
                self._lru_order.remove(query_hash)
            self._lru_order.append(query_hash)

            logger.info(
                "[CACHE] L1 HIT for '%s' (hits=%d, age=%.0fs, answer_len=%d)",
                query[:80], entry.hit_count, age, len(entry.answer),
            )
            self._stats["l1_hits"] += 1
            return entry, "exact"

    async def lookup_async(
        self,
        notebook_id: str,
        query: str,
        bypass_cache: bool = False,
    ) -> tuple[CachedResponse | None, str | None]:
        """Full three-layer async lookup: L1 → L2 → L3.

        Layer 1: Exact hash match (always)
        Layer 2: Embedding pre-filter (if semantic_enabled)
        Layer 3: LLM verification (if candidates found and not near-exact)

        Returns (CachedResponse, hit_type) on hit, (None, None) on miss.
        """
        if bypass_cache:
            logger.debug("[CACHE] BYPASS (async) for '%s'", query[:80])
            self._stats["bypasses"] += 1
            return None, None

        # Layer 1: exact hash match
        result, hit_type = self.lookup(notebook_id, query)
        if result is not None:
            return result, hit_type

        # Layers 2-3 only if semantic matching is enabled
        if not self._semantic_enabled:
            logger.debug("[CACHE] Semantic matching disabled, skipping L2/L3")
            self._stats["misses"] += 1
            return None, None

        # Layer 2: compute embedding and find similar
        logger.debug("[CACHE] L2 computing embedding for '%s'", query[:80])
        query_emb = self._compute_embedding(query)
        if query_emb is None:
            logger.debug("[CACHE] L2 embedding failed, skipping")
            self._stats["misses"] += 1
            return None, None

        candidates = self._find_similar(query_emb, notebook_id)
        if not candidates:
            logger.debug("[CACHE] L2 MISS — no similar entries for '%s'", query[:80])
            self._stats["misses"] += 1
            return None, None

        # Check if early termination (similarity >= exact threshold)
        if len(candidates) == 1 and candidates[0][0] >= self._similarity_exact_threshold:
            entry = candidates[0][1]
            with self._lock:
                entry.hit_count += 1
            logger.info(
                "[CACHE] L2 HIT (sim=%.4f, skip-LLM) for '%s' → '%s'",
                candidates[0][0], query[:60], entry.query[:60],
            )
            self._stats["l2_hits"] += 1
            self.create_alias(notebook_id, query, entry)
            return entry, "semantic"

        # Layer 3: LLM verification
        logger.info(
            "[CACHE] L3 verifying %d candidates for '%s'",
            len(candidates), query[:80],
        )
        matched = await self._verify_semantic_match(
            query, [c[1] for c in candidates]
        )
        if matched is not None:
            with self._lock:
                matched.hit_count += 1
            logger.info(
                "[CACHE] L3 HIT (LLM-verified) for '%s' → '%s'",
                query[:60], matched.query[:60],
            )
            self._stats["l3_hits"] += 1
            self.create_alias(notebook_id, query, matched)
            return matched, "semantic"

        logger.info("[CACHE] L3 MISS — LLM found no match for '%s'", query[:80])
        self._stats["l3_misses"] += 1
        self._stats["misses"] += 1
        return None, None

    # ── Invalidation ─────────────────────────────────────────────────────

    def invalidate_notebook(self, notebook_id: str) -> None:
        """Remove all cached entries for a specific notebook."""
        with self._lock:
            entries = self._cache_by_notebook.pop(notebook_id, [])
            for entry in entries:
                self._cache_by_hash.pop(entry.query_hash, None)
                if entry.query_hash in self._lru_order:
                    self._lru_order.remove(entry.query_hash)
                # Clean global index
                global_hash = self._compute_global_hash(entry.query)
                self._global_hash_index.pop(global_hash, None)

            # Clean aliases pointing to entries in this notebook
            for alias_hash in list(self._alias_hashes):
                if alias_hash in self._cache_by_hash:
                    entry = self._cache_by_hash[alias_hash]
                    if entry.notebook_id == notebook_id:
                        self._cache_by_hash.pop(alias_hash, None)
                        self._alias_hashes.discard(alias_hash)
            for gh in list(self._alias_global_hashes):
                if gh in self._global_hash_index:
                    entry = self._global_hash_index[gh]
                    if entry.notebook_id == notebook_id:
                        self._global_hash_index.pop(gh, None)
                        self._alias_global_hashes.discard(gh)

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
            self._global_hash_index.clear()
            self._cache_by_notebook.clear()
            self._lru_order.clear()
            self._notebook_matrices.clear()
            self._matrix_dirty.clear()
            self._alias_hashes.clear()
            self._alias_global_hashes.clear()
            for key in self._stats:
                self._stats[key] = 0

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
            logger.debug(
                "[CACHE] Rebuilt embedding matrix for notebook %s (shape=%s)",
                notebook_id[:12], self._notebook_matrices[notebook_id].shape,
            )
        else:
            self._notebook_matrices.pop(notebook_id, None)
            logger.debug(
                "[CACHE] Removed empty embedding matrix for notebook %s",
                notebook_id[:12],
            )
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
                logger.debug(
                    "[CACHE] L2 no entries with embeddings for notebook %s",
                    notebook_id[:12],
                )
                return []

            # Rebuild matrix if dirty or missing
            if self._matrix_dirty.get(notebook_id, True):
                self._rebuild_matrix(notebook_id)

            matrix = self._notebook_matrices.get(notebook_id)
            if matrix is None:
                logger.debug(
                    "[CACHE] L2 no embedding matrix found for notebook %s",
                    notebook_id[:12],
                )
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
                logger.info(
                    "[CACHE] L2 near-exact match (sim=%.4f, threshold=%.2f) "
                    "for query_emb.shape=%s → '%s'",
                    max_sim, self._similarity_exact_threshold,
                    query_emb.shape, entries_with_emb[best_idx].query[:60],
                )
                return [(max_sim, entries_with_emb[best_idx])]

            # Filter by threshold and get top-K
            mask = similarities >= self._similarity_threshold
            if not mask.any():
                logger.debug(
                    "[CACHE] L2 no candidates above threshold=%.2f "
                    "(max_sim=%.4f, %d entries checked)",
                    self._similarity_threshold, max_sim, len(entries_with_emb),
                )
                return []

            filtered_idx = np.where(mask)[0]
            sorted_idx = filtered_idx[
                np.argsort(similarities[filtered_idx])[::-1]
            ][:top_k]
            result = [
                (float(similarities[i]), entries_with_emb[i]) for i in sorted_idx
            ]
            logger.info(
                "[CACHE] L2 found %d candidates (best=%.4f, threshold=%.2f, "
                "%d entries checked): %s",
                len(result), result[0][0], self._similarity_threshold,
                len(entries_with_emb),
                [(f"{sim:.3f}", e.query[:40]) for sim, e in result],
            )
            return result

    def _load_embedding_model(self) -> None:
        """Load the LangChain HuggingFace embedding model."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            import numpy as np
            self._embedding_model_instance = HuggingFaceEmbeddings(
                model_name=self._embedding_model_name
            )
            self._np = np
            logger.info(
                "[CACHE] Embedding model loaded: %s",
                self._embedding_model_name,
            )
        except Exception:
            logger.exception("[CACHE] Failed to load embedding model")
            self._semantic_enabled = False

    def _compute_embedding(self, query: str) -> object | None:
        """Compute query embedding using LangChain HuggingFace model.

        Returns numpy array or None if embedding model is not available.
        """
        if self._embedding_model_instance is None:
            return None

        try:
            embedding = self._embedding_model_instance.embed_query(query)
            vec = self._np.array(embedding)
            norm = self._np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception:
            logger.exception("[CACHE] Failed to compute embedding")
        return None

    # ── Layer 3: LLM Semantic Verification ───────────────────────────────

    @staticmethod
    def _build_verification_prompt(
        new_query: str, cached_queries: list[str]
    ) -> str:
        """Build the LLM verification prompt.

        The prompt asks the LLM to determine if the new query is semantically
        equivalent to any of the cached queries (i.e., would produce the same
        answer from the same knowledge base).
        """
        numbered = "\n".join(
            f'{i + 1}. "{q}"' for i, q in enumerate(cached_queries)
        )
        return (
            "You are a cache lookup assistant. Determine if the new question "
            "is asking essentially the same thing as any previously cached "
            "question. Two questions match if they would produce the same "
            "answer from the same knowledge base.\n\n"
            "Rules:\n"
            "- Match: same intent, just different wording\n"
            '  "What are the key points?" ≈ "Summarize the main takeaways"\n'
            '  "Tóm tắt điểm chính" ≈ "Nêu các ý chính"\n'
            "- No match: related topic but different scope or different info "
            "requested\n"
            '  "What happened in Q1?" ≠ "What happened in Q2?"\n'
            '  "List the team members" ≠ "Who is the project lead?"\n\n'
            f'New question: "{new_query}"\n\n'
            f"Cached questions:\n{numbered}\n\n"
            "Reply with ONLY the number of the matching question, or -1 if "
            "no match."
        )

    def _parse_semantic_match(
        self, response: str, num_candidates: int
    ) -> int | None:
        """Parse LLM response to matched index (0-based) or None.

        Handles: exact number, -1, "none", "no match", number embedded in text.
        """
        text = response.strip()
        if not text:
            return None

        # Explicit no-match responses
        if text == "-1" or text.lower() in ("none", "no match"):
            return None

        # Try direct integer parse
        try:
            index = int(text) - 1  # 1-based → 0-based
            if 0 <= index < num_candidates:
                return index
            return None
        except ValueError:
            pass

        # Try to extract a number from mixed text
        match = re.search(r"\b(\d+)\b", text)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < num_candidates:
                return index

        return None

    async def _verify_semantic_match(
        self,
        query: str,
        candidates: list[CachedResponse],
    ) -> CachedResponse | None:
        """Ask LLM to verify if query matches any cached candidate.

        Returns the matched CachedResponse, or None on no match / error.
        Retries once if LLM returns an empty response.
        """
        if not self._llm_client or not candidates:
            return None

        cached_queries = [c.query for c in candidates]
        prompt = self._build_verification_prompt(query, cached_queries)

        max_attempts = 2  # retry once on empty response
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._llm_client.complete(
                    prompt, max_tokens=10
                )
                logger.debug(
                    "[CACHE] L3 raw LLM response (attempt %d): %r",
                    attempt, response,
                )

                # Retry on empty/whitespace response
                if not response or not response.strip():
                    if attempt < max_attempts:
                        logger.warning(
                            "[CACHE] L3 LLM returned empty response "
                            "(attempt %d/%d), retrying for '%s'",
                            attempt, max_attempts, query[:80],
                        )
                        continue
                    logger.warning(
                        "[CACHE] L3 LLM returned empty response after "
                        "%d attempts for '%s'",
                        max_attempts, query[:80],
                    )
                    return None

                matched_idx = self._parse_semantic_match(
                    response, len(candidates)
                )
                if matched_idx is not None:
                    logger.info(
                        "[CACHE] LLM verified semantic match: "
                        '"%s" ≈ "%s"',
                        query,
                        candidates[matched_idx].query,
                    )
                    return candidates[matched_idx]
                logger.debug(
                    "[CACHE] LLM: no semantic match for '%s' "
                    "(response=%r)", query[:80], response,
                )
                return None
            except TimeoutError:
                logger.warning(
                    "[CACHE] LLM verification timed out for '%s'", query
                )
            except Exception:
                logger.exception(
                    "[CACHE] LLM verification failed for '%s'", query
                )

        return None

    # ── Internal helpers ─────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry. Must hold self._lock."""
        if not self._lru_order:
            return
        oldest_hash = self._lru_order.pop(0)
        entry = self._cache_by_hash.pop(oldest_hash, None)
        if entry:
            # Clean global index
            global_hash = self._compute_global_hash(entry.query)
            self._global_hash_index.pop(global_hash, None)
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
        # Clean global index
        global_hash = self._compute_global_hash(entry.query)
        self._global_hash_index.pop(global_hash, None)
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

    def get_stats(self) -> dict:
        """Return cache statistics including hit/miss counters.

        'entries' counts only primary entries (aliases excluded).
        """
        with self._lock:
            alias_count = len(self._alias_hashes)
            total_hits = (
                self._stats["pre_routing_l1_hits"]
                + self._stats["l1_hits"]
                + self._stats["l2_hits"]
                + self._stats["l3_hits"]
            )
            total_lookups = total_hits + self._stats["misses"]
            hit_rate = (
                (total_hits / total_lookups * 100) if total_lookups > 0 else 0.0
            )
            return {
                "entries": len(self._cache_by_hash) - alias_count,
                "aliases": alias_count,
                "total_hits": total_hits,
                "total_misses": self._stats["misses"],
                "total_bypasses": self._stats["bypasses"],
                "hit_rate": round(hit_rate, 1),
                "pre_routing_l1_hits": self._stats["pre_routing_l1_hits"],
                "l1_hits": self._stats["l1_hits"],
                "l2_hits": self._stats["l2_hits"],
                "l3_hits": self._stats["l3_hits"],
                "l3_misses": self._stats["l3_misses"],
                "entry_count": len(self._cache_by_hash) - alias_count,
                "notebook_count": len(self._cache_by_notebook),
            }
