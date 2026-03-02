"""Tests for response cache — Layer 1 (exact hash match)."""
import time
import pytest


class TestCachedResponse:
    """Test CachedResponse dataclass."""

    def test_create_cached_response(self):
        from nlm_proxy.core.response_cache import CachedResponse
        entry = CachedResponse(
            query="What are the key points?",
            query_hash="abc123",
            notebook_id="nb-1",
            answer="The key points are...",
            thinking="Let me think...",
            conversation_id="conv-1",
            embedding=None,
            cached_at=time.time(),
        )
        assert entry.hit_count == 0
        assert entry.answer == "The key points are..."


class TestResponseCacheLayer1:
    """Test Layer 1: exact hash match."""

    def _make_cache(self, **kwargs):
        from nlm_proxy.core.response_cache import ResponseCache
        defaults = dict(
            max_entries=100,
            ttl_seconds=3600,
            semantic_enabled=False,  # Layer 1 only
        )
        defaults.update(kwargs)
        return ResponseCache(**defaults)

    def test_store_and_exact_lookup(self):
        """Exact same query should hit cache."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="What are the key points?",
            answer="The key points are...",
            thinking=None,
            conversation_id="conv-1",
        )
        result = cache.lookup(notebook_id="nb-1", query="What are the key points?")
        assert result is not None
        assert result.answer == "The key points are..."
        assert result.hit_count == 1

    def test_exact_match_case_insensitive(self):
        """Query matching should be case-insensitive."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="What are the key points?",
            answer="Answer",
            thinking=None,
            conversation_id="conv-1",
        )
        result = cache.lookup(notebook_id="nb-1", query="WHAT ARE THE KEY POINTS?")
        assert result is not None

    def test_exact_match_strips_whitespace(self):
        """Query matching should strip whitespace."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="What are the key points?",
            answer="Answer",
            thinking=None,
            conversation_id="conv-1",
        )
        result = cache.lookup(notebook_id="nb-1", query="  What are the key points?  ")
        assert result is not None

    def test_different_notebook_no_match(self):
        """Same query in different notebook should NOT match."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="Answer for nb-1",
            thinking=None,
            conversation_id="conv-1",
        )
        result = cache.lookup(notebook_id="nb-2", query="key points?")
        assert result is None

    def test_different_query_no_match(self):
        """Different query should NOT match."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="What are the key points?",
            answer="Answer",
            thinking=None,
            conversation_id="conv-1",
        )
        result = cache.lookup(notebook_id="nb-1", query="Who are the team members?")
        assert result is None

    def test_ttl_expiration(self):
        """Entries should expire after TTL."""
        cache = self._make_cache(ttl_seconds=0.1)
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="Answer",
            thinking=None,
            conversation_id="conv-1",
        )
        assert cache.lookup(notebook_id="nb-1", query="key points?") is not None
        time.sleep(0.15)
        assert cache.lookup(notebook_id="nb-1", query="key points?") is None

    def test_lru_eviction(self):
        """LRU should evict oldest entries when max_entries exceeded."""
        cache = self._make_cache(max_entries=3)
        for i in range(4):
            cache.store(
                notebook_id="nb-1",
                query=f"query {i}",
                answer=f"answer {i}",
                thinking=None,
                conversation_id=f"conv-{i}",
            )
        # First entry should be evicted
        assert cache.lookup(notebook_id="nb-1", query="query 0") is None
        # Last entry should exist
        assert cache.lookup(notebook_id="nb-1", query="query 3") is not None

    def test_hit_count_increments(self):
        """hit_count should increment on each cache hit."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="Answer",
            thinking=None,
            conversation_id="conv-1",
        )
        cache.lookup(notebook_id="nb-1", query="key points?")
        cache.lookup(notebook_id="nb-1", query="key points?")
        result = cache.lookup(notebook_id="nb-1", query="key points?")
        assert result.hit_count == 3

    def test_bypass_cache_flag(self):
        """bypass_cache=True should skip lookup."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="Answer",
            thinking=None,
            conversation_id="conv-1",
        )
        result = cache.lookup(
            notebook_id="nb-1",
            query="key points?",
            bypass_cache=True,
        )
        assert result is None

    def test_store_with_bypass_updates_existing(self):
        """bypass_cache store should update existing entry."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="Old answer",
            thinking=None,
            conversation_id="conv-1",
        )
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="Fresh answer",
            thinking=None,
            conversation_id="conv-2",
        )
        result = cache.lookup(notebook_id="nb-1", query="key points?")
        assert result.answer == "Fresh answer"

    def test_empty_or_error_response_not_cached(self):
        """Empty or error responses should not be cached."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="",
            thinking=None,
            conversation_id="conv-1",
        )
        result = cache.lookup(notebook_id="nb-1", query="key points?")
        assert result is None

    def test_invalidate_notebook(self):
        """invalidate_notebook should clear all entries for that notebook."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1", query="q1", answer="a1",
            thinking=None, conversation_id="c1",
        )
        cache.store(
            notebook_id="nb-1", query="q2", answer="a2",
            thinking=None, conversation_id="c2",
        )
        cache.store(
            notebook_id="nb-2", query="q3", answer="a3",
            thinking=None, conversation_id="c3",
        )

        cache.invalidate_notebook("nb-1")

        assert cache.lookup(notebook_id="nb-1", query="q1") is None
        assert cache.lookup(notebook_id="nb-1", query="q2") is None
        assert cache.lookup(notebook_id="nb-2", query="q3") is not None

    def test_clear_all(self):
        """clear() should remove all entries."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1", query="q1", answer="a1",
            thinking=None, conversation_id="c1",
        )
        cache.store(
            notebook_id="nb-2", query="q2", answer="a2",
            thinking=None, conversation_id="c2",
        )

        cache.clear()

        assert cache.lookup(notebook_id="nb-1", query="q1") is None
        assert cache.lookup(notebook_id="nb-2", query="q2") is None

    def test_global_cache_across_users(self):
        """Cache is global — same notebook_id + query returns same entry."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="Shared answer",
            thinking=None,
            conversation_id="conv-1",
        )
        # "User B" queries same thing — should get cached answer
        result = cache.lookup(notebook_id="nb-1", query="key points?")
        assert result is not None
        assert result.answer == "Shared answer"

    def test_thinking_text_stored(self):
        """Thinking text should be stored and returned on cache hit."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb-1",
            query="key points?",
            answer="Answer",
            thinking="Let me analyze the document...",
            conversation_id="conv-1",
        )
        result = cache.lookup(notebook_id="nb-1", query="key points?")
        assert result.thinking == "Let me analyze the document..."
