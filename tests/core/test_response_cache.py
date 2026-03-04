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
        result, hit_type = cache.lookup(notebook_id="nb-1", query="What are the key points?")
        assert result is not None
        assert result.answer == "The key points are..."
        assert result.hit_count == 1
        assert hit_type == "exact"

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
        result, hit_type = cache.lookup(notebook_id="nb-1", query="WHAT ARE THE KEY POINTS?")
        assert result is not None
        assert hit_type == "exact"

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
        result, hit_type = cache.lookup(notebook_id="nb-1", query="  What are the key points?  ")
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
        result, hit_type = cache.lookup(notebook_id="nb-2", query="key points?")
        assert result is None
        assert hit_type is None

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
        result, hit_type = cache.lookup(notebook_id="nb-1", query="Who are the team members?")
        assert result is None
        assert hit_type is None

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
        result, hit_type = cache.lookup(notebook_id="nb-1", query="key points?")
        assert result is not None
        time.sleep(0.15)
        result, hit_type = cache.lookup(notebook_id="nb-1", query="key points?")
        assert result is None

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
        result, _ = cache.lookup(notebook_id="nb-1", query="query 0")
        assert result is None
        # Last entry should exist
        result, _ = cache.lookup(notebook_id="nb-1", query="query 3")
        assert result is not None

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
        result, _ = cache.lookup(notebook_id="nb-1", query="key points?")
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
        result, hit_type = cache.lookup(
            notebook_id="nb-1",
            query="key points?",
            bypass_cache=True,
        )
        assert result is None
        assert hit_type is None

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
        result, _ = cache.lookup(notebook_id="nb-1", query="key points?")
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
        result, _ = cache.lookup(notebook_id="nb-1", query="key points?")
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

        assert cache.lookup(notebook_id="nb-1", query="q1")[0] is None
        assert cache.lookup(notebook_id="nb-1", query="q2")[0] is None
        assert cache.lookup(notebook_id="nb-2", query="q3")[0] is not None

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

        assert cache.lookup(notebook_id="nb-1", query="q1")[0] is None
        assert cache.lookup(notebook_id="nb-2", query="q2")[0] is None

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
        result, _ = cache.lookup(notebook_id="nb-1", query="key points?")
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
        result, _ = cache.lookup(notebook_id="nb-1", query="key points?")
        assert result.thinking == "Let me analyze the document..."


class TestGlobalLookup:
    """Test pre-routing global L1 lookup (notebook-agnostic)."""

    def _make_cache(self, **kwargs):
        from nlm_proxy.core.response_cache import ResponseCache
        defaults = dict(max_entries=100, ttl_seconds=3600, semantic_enabled=False)
        defaults.update(kwargs)
        return ResponseCache(**defaults)

    def test_lookup_global_hit(self):
        """Global lookup finds entry without knowing notebook_id."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb1", query="test query", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        result, hit_type = cache.lookup_global("test query")
        assert result is not None
        assert result.answer == "answer"
        assert result.notebook_id == "nb1"
        assert hit_type == "exact"

    def test_lookup_global_miss(self):
        """Global lookup returns (None, None) when query not cached."""
        cache = self._make_cache()
        result, hit_type = cache.lookup_global("unknown query")
        assert result is None
        assert hit_type is None

    def test_lookup_global_case_insensitive(self):
        """Global lookup normalizes case."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb1", query="Hello World", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        result, hit_type = cache.lookup_global("hello world")
        assert result is not None

    def test_lookup_global_after_eviction(self):
        """Global index cleaned up on eviction."""
        cache = self._make_cache(max_entries=2)
        cache.store(notebook_id="nb1", query="q1", answer="a1", thinking=None, conversation_id="c1")
        cache.store(notebook_id="nb1", query="q2", answer="a2", thinking=None, conversation_id="c2")
        cache.store(notebook_id="nb1", query="q3", answer="a3", thinking=None, conversation_id="c3")
        # q1 should be evicted
        assert cache.lookup_global("q1")[0] is None
        assert cache.lookup_global("q3")[0] is not None

    def test_lookup_global_after_invalidation(self):
        """Global index cleaned up on notebook invalidation."""
        cache = self._make_cache()
        cache.store(notebook_id="nb1", query="q1", answer="a1", thinking=None, conversation_id="c1")
        cache.invalidate_notebook("nb1")
        assert cache.lookup_global("q1")[0] is None

    def test_lookup_global_after_clear(self):
        """Global index cleaned up on clear."""
        cache = self._make_cache()
        cache.store(notebook_id="nb1", query="q1", answer="a1", thinking=None, conversation_id="c1")
        cache.clear()
        assert cache.lookup_global("q1")[0] is None


class TestAliasCreation:
    """Test alias creation on semantic match."""

    def _make_cache(self, **kwargs):
        from nlm_proxy.core.response_cache import ResponseCache
        defaults = dict(max_entries=100, ttl_seconds=3600, semantic_enabled=False)
        defaults.update(kwargs)
        return ResponseCache(**defaults)

    def test_create_alias_enables_l1_hit(self):
        """After alias creation, new query gets L1 hit."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb1", query="original query", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        original_entry, _ = cache.lookup("nb1", "original query")

        # Create alias
        cache.create_alias("nb1", "rewritten query", original_entry)

        # Alias should hit L1
        result, hit_type = cache.lookup("nb1", "rewritten query")
        assert result is not None
        assert result.answer == "answer"
        assert hit_type == "exact"

    def test_alias_in_global_index(self):
        """Alias also available in global lookup."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb1", query="original", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        entry, _ = cache.lookup("nb1", "original")
        cache.create_alias("nb1", "alias query", entry)
        result, hit_type = cache.lookup_global("alias query")
        assert result is not None
        assert result.answer == "answer"
        assert hit_type == "exact"

    def test_alias_not_counted_in_lru(self):
        """Aliases don't consume LRU capacity."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb1", query="primary", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        entry, _ = cache.lookup("nb1", "primary")
        # Create 5 aliases
        for i in range(5):
            cache.create_alias("nb1", f"alias {i}", entry)
        stats = cache.get_stats()
        # Only 1 primary entry, aliases don't count
        assert stats["entries"] == 1
        assert stats["aliases"] == 5

    def test_alias_cleaned_on_invalidation(self):
        """Aliases cleaned up when notebook invalidated."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb1", query="primary", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        entry, _ = cache.lookup("nb1", "primary")
        cache.create_alias("nb1", "alias", entry)
        cache.invalidate_notebook("nb1")
        assert cache.lookup("nb1", "alias")[0] is None
        assert cache.lookup_global("alias")[0] is None

    def test_alias_cleaned_on_clear(self):
        """Aliases cleaned up on full clear."""
        cache = self._make_cache()
        cache.store(
            notebook_id="nb1", query="primary", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        entry, _ = cache.lookup("nb1", "primary")
        cache.create_alias("nb1", "alias", entry)
        cache.clear()
        assert cache.lookup("nb1", "alias")[0] is None


class TestLookupReturnsTuples:
    """Verify lookup methods return (CachedResponse, hit_type) tuples."""

    def test_lookup_returns_cache_hit_type(self):
        """Lookup result should return (CachedResponse, hit_type) tuple."""
        from nlm_proxy.core.response_cache import ResponseCache

        cache = ResponseCache(max_entries=100, ttl_seconds=3600, semantic_enabled=False)
        cache.store("nb-1", "key points?", "answer", None, "conv-1")

        result, hit_type = cache.lookup("nb-1", "key points?")
        assert result is not None
        assert hit_type == "exact"

        result, hit_type = cache.lookup("nb-1", "nonexistent")
        assert result is None
        assert hit_type is None

    def test_lookup_global_returns_tuple(self):
        """lookup_global should return (CachedResponse, hit_type) tuple."""
        from nlm_proxy.core.response_cache import ResponseCache

        cache = ResponseCache(max_entries=100, ttl_seconds=3600, semantic_enabled=False)
        cache.store("nb-1", "test", "answer", None, "conv-1")

        result, hit_type = cache.lookup_global("test")
        assert result is not None
        assert hit_type == "exact"

        result, hit_type = cache.lookup_global("missing")
        assert result is None
        assert hit_type is None
