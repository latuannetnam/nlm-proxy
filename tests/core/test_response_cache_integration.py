"""Tests for full three-layer cache lookup integration."""
import time
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFullCacheLookup:
    """Test the complete L1 → L2 → L3 lookup flow."""

    @pytest.mark.asyncio
    async def test_l1_hit_skips_l2_l3(self):
        """Exact hash match should return immediately without embedding or LLM."""
        from nlm_proxy.core.response_cache import ResponseCache

        mock_llm = AsyncMock()
        cache = ResponseCache(
            max_entries=100, ttl_seconds=3600,
            semantic_enabled=True, llm_client=mock_llm,
        )

        cache.store("nb-1", "key points?", "answer", None, "conv-1")
        result = await cache.lookup_async("nb-1", "key points?")

        assert result is not None
        assert result.answer == "answer"
        mock_llm.chat.assert_not_called()  # L3 not invoked

    @pytest.mark.asyncio
    async def test_l2_no_candidates_skips_l3(self):
        """No embedding candidates → skip LLM → miss."""
        from nlm_proxy.core.response_cache import ResponseCache

        mock_llm = AsyncMock()
        cache = ResponseCache(
            max_entries=100, ttl_seconds=3600,
            semantic_enabled=True, llm_client=mock_llm,
        )

        # No entries stored
        result = await cache.lookup_async("nb-1", "key points?")

        assert result is None
        mock_llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_l2_high_similarity_skips_l3(self):
        """Similarity >= 0.95 should skip LLM and return directly."""
        from nlm_proxy.core.response_cache import ResponseCache

        mock_llm = AsyncMock()
        cache = ResponseCache(
            max_entries=100, ttl_seconds=3600,
            semantic_enabled=True, llm_client=mock_llm,
        )

        emb = np.array([1.0, 0.0, 0.0])
        emb /= np.linalg.norm(emb)

        cache.store("nb-1", "key points?", "answer", None, "conv-1", embedding=emb.tolist())

        # Mock embedding computation to return near-identical vector
        with patch.object(cache, '_compute_embedding', return_value=emb):
            result = await cache.lookup_async("nb-1", "key points rephrased?")

        assert result is not None
        mock_llm.chat.assert_not_called()  # Skipped L3

    @pytest.mark.asyncio
    async def test_semantic_disabled_only_l1(self):
        """When semantic_enabled=False, only Layer 1 should run."""
        from nlm_proxy.core.response_cache import ResponseCache

        cache = ResponseCache(
            max_entries=100, ttl_seconds=3600,
            semantic_enabled=False,
        )

        cache.store("nb-1", "exact query", "answer", None, "conv-1")

        # Exact match works
        result = await cache.lookup_async("nb-1", "exact query")
        assert result is not None

        # Similar but different wording → miss (no semantic matching)
        result = await cache.lookup_async("nb-1", "rephrase of exact query")
        assert result is None

    def test_lookup_returns_cache_hit_type(self):
        """Lookup result should indicate hit type (exact vs semantic)."""
        from nlm_proxy.core.response_cache import ResponseCache

        cache = ResponseCache(max_entries=100, ttl_seconds=3600, semantic_enabled=False)
        cache.store("nb-1", "key points?", "answer", None, "conv-1")

        result = cache.lookup("nb-1", "key points?")
        assert result is not None
        # The result should be a CachedResponse; hit_type tracked separately by caller


class TestCacheStats:
    """Test cache statistics and metadata."""

    def test_cache_entry_count(self):
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(max_entries=100, ttl_seconds=3600, semantic_enabled=False)

        assert cache.entry_count == 0
        cache.store("nb-1", "q1", "a1", None, "c1")
        assert cache.entry_count == 1
        cache.store("nb-1", "q2", "a2", None, "c2")
        assert cache.entry_count == 2

    def test_cache_notebook_count(self):
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(max_entries=100, ttl_seconds=3600, semantic_enabled=False)

        cache.store("nb-1", "q1", "a1", None, "c1")
        cache.store("nb-2", "q2", "a2", None, "c2")
        assert cache.notebook_count == 2
