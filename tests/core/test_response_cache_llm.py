"""Tests for response cache — Layer 3 (LLM semantic verification)."""
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestLLMVerification:
    """Test Layer 3: LLM semantic matching."""

    def test_parse_semantic_match_exact_number(self):
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(max_entries=10, ttl_seconds=3600, semantic_enabled=False)

        assert cache._parse_semantic_match("1", 3) == 0  # 1-based → 0-based
        assert cache._parse_semantic_match("2", 3) == 1
        assert cache._parse_semantic_match("3", 3) == 2

    def test_parse_semantic_match_negative_one(self):
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(max_entries=10, ttl_seconds=3600, semantic_enabled=False)

        assert cache._parse_semantic_match("-1", 3) is None
        assert cache._parse_semantic_match("none", 3) is None
        assert cache._parse_semantic_match("no match", 3) is None

    def test_parse_semantic_match_out_of_range(self):
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(max_entries=10, ttl_seconds=3600, semantic_enabled=False)

        assert cache._parse_semantic_match("5", 3) is None  # >3 candidates
        assert cache._parse_semantic_match("0", 3) is None  # 0 is invalid (1-based)

    def test_parse_semantic_match_with_explanation(self):
        """LLM might include explanation text around the number."""
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(max_entries=10, ttl_seconds=3600, semantic_enabled=False)

        assert cache._parse_semantic_match("The answer is 2", 3) == 1
        assert cache._parse_semantic_match("Question 1 matches", 3) == 0

    def test_parse_semantic_match_garbage(self):
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(max_entries=10, ttl_seconds=3600, semantic_enabled=False)

        assert cache._parse_semantic_match("I don't know", 3) is None
        assert cache._parse_semantic_match("", 3) is None

    def test_build_verification_prompt(self):
        """Prompt should contain new query and all cached candidates."""
        from nlm_proxy.core.response_cache import ResponseCache
        cache = ResponseCache(max_entries=10, ttl_seconds=3600, semantic_enabled=False)

        prompt = cache._build_verification_prompt(
            "What are the key points?",
            ["Summarize main takeaways", "List team members"]
        )
        assert "What are the key points?" in prompt
        assert "Summarize main takeaways" in prompt
        assert "List team members" in prompt
        assert "-1" in prompt  # Should mention -1 for no match

    @pytest.mark.asyncio
    async def test_verify_semantic_match_returns_matched_entry(self):
        """LLM confirms match → return the matched candidate."""
        from nlm_proxy.core.response_cache import ResponseCache, CachedResponse
        import time

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "1"  # LLM says candidate 1 matches

        cache = ResponseCache(
            max_entries=10, ttl_seconds=3600,
            semantic_enabled=True, llm_client=mock_llm,
        )

        candidates = [
            CachedResponse(
                query="Summarize the key points",
                query_hash="h1", notebook_id="nb-1",
                answer="The key points are...",
                thinking=None, conversation_id="conv-1",
                embedding=None, cached_at=time.time(),
            )
        ]

        result = await cache._verify_semantic_match("What are the key points?", candidates)
        assert result is not None
        assert result.answer == "The key points are..."

    @pytest.mark.asyncio
    async def test_verify_semantic_match_no_match(self):
        """LLM says no match → return None."""
        from nlm_proxy.core.response_cache import ResponseCache, CachedResponse
        import time

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "-1"

        cache = ResponseCache(
            max_entries=10, ttl_seconds=3600,
            semantic_enabled=True, llm_client=mock_llm,
        )

        candidates = [
            CachedResponse(
                query="Who is the CEO?",
                query_hash="h1", notebook_id="nb-1",
                answer="The CEO is...",
                thinking=None, conversation_id="conv-1",
                embedding=None, cached_at=time.time(),
            )
        ]

        result = await cache._verify_semantic_match("What is the budget?", candidates)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_semantic_match_llm_timeout(self):
        """LLM timeout should return None (treat as miss)."""
        from nlm_proxy.core.response_cache import ResponseCache, CachedResponse
        import time

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = TimeoutError("LLM timeout")

        cache = ResponseCache(
            max_entries=10, ttl_seconds=3600,
            semantic_enabled=True, llm_client=mock_llm,
        )

        candidates = [
            CachedResponse(
                query="key points", query_hash="h1", notebook_id="nb-1",
                answer="answer", thinking=None, conversation_id="conv-1",
                embedding=None, cached_at=time.time(),
            )
        ]

        result = await cache._verify_semantic_match("key points?", candidates)
        assert result is None  # Timeout → miss, not crash
