"""Tests for response cache — Layer 2 (embedding pre-filter via NumPy)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestEmbeddingPreFilter:
    """Test Layer 2: embedding similarity with NumPy."""

    def _make_cache_with_embeddings(self):
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(
            max_entries=100,
            ttl_seconds=3600,
            semantic_enabled=True,
            embedding_model=None,  # We'll mock embeddings
        )
        return cache

    def test_find_similar_returns_candidates(self):
        """Pre-filter should return similar candidates sorted by similarity."""
        cache = self._make_cache_with_embeddings()

        # Store entries with known embeddings (pre-normalized unit vectors)
        emb_a = np.array([1.0, 0.0, 0.0])
        emb_a /= np.linalg.norm(emb_a)
        emb_b = np.array([0.9, 0.1, 0.0])
        emb_b /= np.linalg.norm(emb_b)
        emb_c = np.array([0.0, 1.0, 0.0])  # orthogonal — low similarity
        emb_c /= np.linalg.norm(emb_c)

        cache.store("nb-1", "query A", "answer A", None, "conv-a", embedding=emb_a.tolist())
        cache.store("nb-1", "query B", "answer B", None, "conv-b", embedding=emb_b.tolist())
        cache.store("nb-1", "query C", "answer C", None, "conv-c", embedding=emb_c.tolist())

        # Query similar to A and B
        query_emb = np.array([0.95, 0.05, 0.0])
        query_emb /= np.linalg.norm(query_emb)

        candidates = cache._find_similar(query_emb, "nb-1")
        assert len(candidates) >= 1
        # Best candidate should be most similar (query A)
        assert candidates[0][1].query == "query A"

    def test_find_similar_filters_by_threshold(self):
        """Candidates below similarity threshold should be excluded."""
        cache = self._make_cache_with_embeddings()

        emb_a = np.array([1.0, 0.0, 0.0])
        emb_a /= np.linalg.norm(emb_a)

        cache.store("nb-1", "query A", "answer A", None, "conv-a", embedding=emb_a.tolist())

        # Orthogonal query — similarity near 0
        query_emb = np.array([0.0, 1.0, 0.0])
        query_emb /= np.linalg.norm(query_emb)

        candidates = cache._find_similar(query_emb, "nb-1")
        assert len(candidates) == 0

    def test_find_similar_partition_by_notebook(self):
        """Only entries from the queried notebook should be searched."""
        cache = self._make_cache_with_embeddings()

        emb = np.array([1.0, 0.0, 0.0])
        emb /= np.linalg.norm(emb)

        cache.store("nb-1", "query A", "answer A", None, "conv-a", embedding=emb.tolist())
        cache.store("nb-2", "query B", "answer B", None, "conv-b", embedding=emb.tolist())

        query_emb = np.array([1.0, 0.0, 0.0])
        query_emb /= np.linalg.norm(query_emb)

        candidates = cache._find_similar(query_emb, "nb-1")
        assert all(c[1].notebook_id == "nb-1" for c in candidates)

    def test_l2_hit_above_threshold(self):
        """Similarity >= 0.93 should return single result."""
        cache = self._make_cache_with_embeddings()

        emb = np.array([1.0, 0.0, 0.0])
        emb /= np.linalg.norm(emb)

        cache.store("nb-1", "exact match", "answer", None, "conv-a", embedding=emb.tolist())

        # Near-identical query
        query_emb = np.array([0.999, 0.001, 0.0])
        query_emb /= np.linalg.norm(query_emb)

        candidates = cache._find_similar(query_emb, "nb-1")
        assert len(candidates) == 1
        assert candidates[0][0] >= 0.93

    def test_matrix_rebuild_on_new_entry(self):
        """Matrix should be rebuilt after new entries are added."""
        cache = self._make_cache_with_embeddings()

        emb = np.array([1.0, 0.0, 0.0])
        emb /= np.linalg.norm(emb)

        cache.store("nb-1", "q1", "a1", None, "c1", embedding=emb.tolist())

        # First lookup triggers matrix build
        cache._find_similar(emb, "nb-1")
        assert not cache._matrix_dirty.get("nb-1", True)

        # Adding new entry should mark dirty
        emb2 = np.array([0.0, 1.0, 0.0])
        emb2 /= np.linalg.norm(emb2)
        cache.store("nb-1", "q2", "a2", None, "c2", embedding=emb2.tolist())
        assert cache._matrix_dirty.get("nb-1", False)

    def test_invalidate_notebook_clears_matrix(self):
        """Invalidating notebook should clear its matrix."""
        cache = self._make_cache_with_embeddings()

        emb = np.array([1.0, 0.0, 0.0])
        emb /= np.linalg.norm(emb)

        cache.store("nb-1", "q1", "a1", None, "c1", embedding=emb.tolist())
        cache._find_similar(emb, "nb-1")  # build matrix

        cache.invalidate_notebook("nb-1")
        assert "nb-1" not in cache._notebook_matrices


class TestLookupAsync:
    """Test full two-layer async lookup (L1 → L2)."""

    def _make_cache_with_mock_embedding(self):
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(
            max_entries=100,
            ttl_seconds=3600,
            semantic_enabled=True,
            embedding_model=None,  # We'll set _np manually
            similarity_threshold=0.93,
        )
        cache._np = np
        return cache

    @pytest.mark.asyncio
    async def test_lookup_async_l2_hit(self):
        """lookup_async should return L2 hit when similarity > threshold."""
        cache = self._make_cache_with_mock_embedding()

        emb = np.array([1.0, 0.0, 0.0])
        emb /= np.linalg.norm(emb)

        cache.store("nb-1", "cached query", "cached answer", None, "conv-a",
                    embedding=emb.tolist())

        # Mock _compute_embedding to return near-identical vector
        query_emb = np.array([0.999, 0.001, 0.0])
        query_emb /= np.linalg.norm(query_emb)
        cache._compute_embedding = MagicMock(return_value=query_emb)

        result, hit_type = await cache.lookup_async("nb-1", "similar query")
        assert result is not None
        assert result.answer == "cached answer"
        assert hit_type == "semantic"
        assert cache._stats["l2_hits"] == 1

    @pytest.mark.asyncio
    async def test_lookup_async_l2_miss(self):
        """lookup_async should return miss when similarity < threshold."""
        cache = self._make_cache_with_mock_embedding()

        emb = np.array([1.0, 0.0, 0.0])
        emb /= np.linalg.norm(emb)

        cache.store("nb-1", "cached query", "cached answer", None, "conv-a",
                    embedding=emb.tolist())

        # Mock _compute_embedding to return orthogonal vector
        query_emb = np.array([0.0, 1.0, 0.0])
        query_emb /= np.linalg.norm(query_emb)
        cache._compute_embedding = MagicMock(return_value=query_emb)

        result, hit_type = await cache.lookup_async("nb-1", "different query")
        assert result is None
        assert hit_type is None
        assert cache._stats["misses"] == 1

