# Response Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a three-layer response cache (exact match → embedding pre-filter → LLM verification) that eliminates 40-50s latency for repeated/semantically similar queries across 300+ users.

**Architecture:** `ResponseCache` lives in `core/response_cache.py` with three lookup layers. Cache check happens at the server level in `server.py` before session store lookup and NLM query. `NotebookCache` provides source IDs and triggers auto-invalidation on source changes.

**Tech Stack:** Python 3.11+, fastembed (ONNX/multilingual), NumPy (vectorized similarity), pydantic-settings (config), pytest + AsyncMock (tests)

**Design Document:** [2026-03-02-response-cache-design.md](file:///d:/latuan/Programming/nlm-proxy/docs/plans/2026-03-02-response-cache-design.md)

---

## Task 1: CacheSettings Configuration

**Files:**
- Modify: `src/nlm_proxy/core/config.py`
- Modify: `tests/test_config.py`
- Modify: `.env.example`

**Step 1: Write the failing test**

Add `TestCacheSettings` to `tests/test_config.py`:

```python
class TestCacheSettings:
    """Test CacheSettings class."""

    def test_default_values(self):
        """Test CacheSettings default values."""
        from nlm_proxy.core.config import CacheSettings
        settings = CacheSettings()

        assert settings.response_cache_enabled is True
        assert settings.response_cache_ttl == 14400
        assert settings.response_cache_max_entries == 1000
        assert settings.semantic_match_enabled is True
        assert settings.embedding_model == "intfloat/multilingual-e5-small"
        assert settings.similarity_threshold == 0.7
        assert settings.similarity_exact_threshold == 0.95
        assert settings.semantic_match_top_k == 10

    def test_env_override(self, monkeypatch):
        """Test CacheSettings loads from environment."""
        monkeypatch.setenv("NLM_PROXY_CACHE_RESPONSE_CACHE_TTL", "7200")
        monkeypatch.setenv("NLM_PROXY_CACHE_RESPONSE_CACHE_MAX_ENTRIES", "500")
        monkeypatch.setenv("NLM_PROXY_CACHE_SIMILARITY_THRESHOLD", "0.8")

        from nlm_proxy.core.config import CacheSettings
        settings = CacheSettings()

        assert settings.response_cache_ttl == 7200
        assert settings.response_cache_max_entries == 500
        assert settings.similarity_threshold == 0.8

    def test_disable_cache(self, monkeypatch):
        """Test disabling cache via env."""
        monkeypatch.setenv("NLM_PROXY_CACHE_RESPONSE_CACHE_ENABLED", "false")

        from nlm_proxy.core.config import CacheSettings
        settings = CacheSettings()

        assert settings.response_cache_enabled is False

    def test_get_cache_settings_singleton(self):
        """get_cache_settings should return singleton."""
        import nlm_proxy.core.config as config
        config._cache = None

        s1 = config.get_cache_settings()
        s2 = config.get_cache_settings()
        assert s1 is s2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestCacheSettings -v`
Expected: FAIL with `ImportError: cannot import name 'CacheSettings'`

**Step 3: Write minimal implementation**

Add to `src/nlm_proxy/core/config.py`:

```python
class CacheSettings(BaseSettings):
    """Response cache configuration."""
    response_cache_enabled: bool = Field(default=True, description="Enable response caching")
    response_cache_ttl: int = Field(default=14400, description="Response cache TTL in seconds (4h)")
    response_cache_max_entries: int = Field(default=1000, description="Max cached responses (LRU)")
    semantic_match_enabled: bool = Field(default=True, description="Enable semantic matching")
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small",
        description="fastembed model for embeddings"
    )
    similarity_threshold: float = Field(default=0.7, description="Min cosine similarity")
    similarity_exact_threshold: float = Field(default=0.95, description="Skip LLM verification threshold")
    semantic_match_top_k: int = Field(default=10, description="Max candidates sent to LLM")
    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_CACHE_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

# Add singleton
_cache: CacheSettings | None = None

def get_cache_settings() -> CacheSettings:
    """Get the cache settings instance."""
    global _cache
    if _cache is None:
        _cache = CacheSettings()
    return _cache
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::TestCacheSettings -v`
Expected: PASS

**Step 5: Update `.env.example`**

Add cache settings section to `.env.example`.

**Step 6: Commit**

```bash
git add src/nlm_proxy/core/config.py tests/test_config.py .env.example
git commit -m "feat(cache): add CacheSettings configuration"
```

---

## Task 2: `bypass_cache` Field on `ChatCompletionRequest`

**Files:**
- Modify: `src/nlm_proxy/openai/types.py`
- Modify: `tests/test_openai_types.py`

**Step 1: Write the failing test**

Add to `tests/test_openai_types.py`:

```python
def test_bypass_cache_default_false():
    """bypass_cache should default to False."""
    from nlm_proxy.openai.types import ChatCompletionRequest
    req = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}]
    )
    assert req.bypass_cache is False

def test_bypass_cache_from_extra_body():
    """bypass_cache should be settable via extra_body."""
    from nlm_proxy.openai.types import ChatCompletionRequest
    req = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        bypass_cache=True
    )
    assert req.bypass_cache is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_types.py::test_bypass_cache_default_false -v`
Expected: FAIL with `ValidationError`

**Step 3: Write minimal implementation**

Add to `ChatCompletionRequest` in `types.py`:

```python
bypass_cache: bool = False  # Skip response cache, query NLM directly
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_types.py -v -k "bypass_cache"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/types.py tests/test_openai_types.py
git commit -m "feat(cache): add bypass_cache field to ChatCompletionRequest"
```

---

## Task 3: ResponseCache Core — Layer 1 (Exact Match)

**Files:**
- Create: `src/nlm_proxy/core/response_cache.py`
- Create: `tests/core/test_response_cache.py`

**Step 1: Write failing tests for core data structures and Layer 1**

Create `tests/core/test_response_cache.py`:

```python
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
        cache.store(notebook_id="nb-1", query="q1", answer="a1", thinking=None, conversation_id="c1")
        cache.store(notebook_id="nb-1", query="q2", answer="a2", thinking=None, conversation_id="c2")
        cache.store(notebook_id="nb-2", query="q3", answer="a3", thinking=None, conversation_id="c3")

        cache.invalidate_notebook("nb-1")

        assert cache.lookup(notebook_id="nb-1", query="q1") is None
        assert cache.lookup(notebook_id="nb-1", query="q2") is None
        assert cache.lookup(notebook_id="nb-2", query="q3") is not None

    def test_clear_all(self):
        """clear() should remove all entries."""
        cache = self._make_cache()
        cache.store(notebook_id="nb-1", query="q1", answer="a1", thinking=None, conversation_id="c1")
        cache.store(notebook_id="nb-2", query="q2", answer="a2", thinking=None, conversation_id="c2")

        cache.clear()

        assert cache.lookup(notebook_id="nb-1", query="q1") is None
        assert cache.lookup(notebook_id="nb-2", query="q2") is None

    def test_global_cache_across_users(self):
        """Cache is global — same notebook_id + query returns same entry regardless of caller."""
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_response_cache.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/nlm_proxy/core/response_cache.py` with `CachedResponse` dataclass and `ResponseCache` class implementing Layer 1 (exact hash match with LRU + TTL + thread safety). See design doc for full data structures.

Key methods:
- `__init__(max_entries, ttl_seconds, semantic_enabled, llm_client, embedding_model, ...)`
- `_compute_hash(notebook_id, query)` — `hash(notebook_id + query.strip().lower())`
- `store(notebook_id, query, answer, thinking, conversation_id)` — add to both `_cache_by_hash` and `_cache_by_notebook`
- `lookup(notebook_id, query, bypass_cache=False)` — Layer 1 exact match
- `invalidate_notebook(notebook_id)` — clear entries by notebook
- `clear()` — clear all

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_response_cache.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_response_cache.py
git commit -m "feat(cache): implement ResponseCache Layer 1 — exact hash match"
```

---

## Task 4: ResponseCache — Layer 2 (Embedding Pre-filter)

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py`
- Create: `tests/core/test_response_cache_semantic.py`

**Step 1: Write failing tests for Layer 2**

Create `tests/core/test_response_cache_semantic.py`:

```python
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
            llm_client=None,  # No Layer 3 for these tests
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

        candidates = cache._find_similar(query_emb, "nb-1", top_k=2)
        assert len(candidates) >= 1
        # First candidate should be most similar (query A)
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

    def test_early_termination_above_exact_threshold(self):
        """Similarity >= 0.95 should return single result (skip LLM)."""
        cache = self._make_cache_with_embeddings()

        emb = np.array([1.0, 0.0, 0.0])
        emb /= np.linalg.norm(emb)

        cache.store("nb-1", "exact match", "answer", None, "conv-a", embedding=emb.tolist())

        # Near-identical query
        query_emb = np.array([0.999, 0.001, 0.0])
        query_emb /= np.linalg.norm(query_emb)

        candidates = cache._find_similar(query_emb, "nb-1")
        assert len(candidates) == 1
        assert candidates[0][0] >= 0.95

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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_response_cache_semantic.py -v`
Expected: FAIL

**Step 3: Implement Layer 2**

Add to `ResponseCache`:
- `_normalize_embedding(vec)` — L2 normalize to unit vector
- `_rebuild_matrix(notebook_id)` — build NumPy matrix from notebook entries
- `_find_similar(query_emb, notebook_id, top_k)` — NumPy vectorized similarity with early termination
- Update `store()` to accept optional `embedding` and track `_matrix_dirty`
- Update `invalidate_notebook()` to clean up matrices

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_response_cache_semantic.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_response_cache_semantic.py
git commit -m "feat(cache): implement Layer 2 — NumPy embedding pre-filter"
```

---

## Task 5: ResponseCache — Layer 3 (LLM Verification)

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py`
- Create: `tests/core/test_response_cache_llm.py`

**Step 1: Write failing tests for Layer 3**

Create `tests/core/test_response_cache_llm.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_response_cache_llm.py -v`
Expected: FAIL

**Step 3: Implement Layer 3**

Add to `ResponseCache`:
- `_build_verification_prompt(new_query, cached_queries)` — build the strict matching prompt
- `_parse_semantic_match(response, num_candidates)` — parse LLM response
- `_verify_semantic_match(query, candidates)` — async call to LLM with timeout

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_response_cache_llm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_response_cache_llm.py
git commit -m "feat(cache): implement Layer 3 — LLM semantic verification"
```

---

## Task 6: ResponseCache — Full Lookup Integration

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py`
- Create: `tests/core/test_response_cache_integration.py`

**Step 1: Write failing tests for the full three-layer lookup**

Create `tests/core/test_response_cache_integration.py`:

```python
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
        result = cache.lookup("nb-1", "exact query")
        assert result is not None

        # Similar but different wording → miss (no semantic matching)
        result = cache.lookup("nb-1", "rephrase of exact query")
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_response_cache_integration.py -v`
Expected: FAIL

**Step 3: Implement full async lookup**

Add to `ResponseCache`:
- `lookup_async(notebook_id, query, bypass_cache)` — full three-layer async lookup (L1 → L2 → L3)
- `_compute_embedding(query)` — call fastembed model
- `entry_count` / `notebook_count` properties

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_response_cache_integration.py tests/core/test_response_cache.py tests/core/test_response_cache_semantic.py tests/core/test_response_cache_llm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_response_cache_integration.py
git commit -m "feat(cache): implement full three-layer lookup with async support"
```

---

## Task 7: NotebookCache — Source Change Detection

**Files:**
- Modify: `src/nlm_proxy/openai/notebook_cache.py`
- Modify: `tests/test_openai_module/test_notebook_cache.py`

**Step 1: Write failing tests**

Add to `tests/test_openai_module/test_notebook_cache.py`:

```python
def test_on_sources_changed_callback_fires():
    """Callback should fire when sources change."""
    from nlm_proxy.openai.notebook_cache import NotebookCache, SourceInfo
    from unittest.mock import AsyncMock, MagicMock

    callback = MagicMock()
    mock_client = MagicMock()
    mock_client.list_notebooks = AsyncMock(return_value=[])

    cache = NotebookCache(nlm_client=mock_client, ttl_seconds=3600, on_sources_changed=callback)

    # First set — no previous sources, no callback
    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-1", title="Doc 1", source_type="pdf"),
    ])
    callback.assert_not_called()

    # Same sources — no callback
    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-1", title="Doc 1", source_type="pdf"),
    ])
    callback.assert_not_called()

    # Different sources — callback fires
    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-1", title="Doc 1", source_type="pdf"),
        SourceInfo(id="src-2", title="Doc 2", source_type="url"),
    ])
    callback.assert_called_once_with("nb-1")


def test_on_sources_changed_no_callback():
    """Without callback, source changes should not crash."""
    from nlm_proxy.openai.notebook_cache import NotebookCache, SourceInfo
    from unittest.mock import AsyncMock, MagicMock

    mock_client = MagicMock()
    mock_client.list_notebooks = AsyncMock(return_value=[])

    cache = NotebookCache(nlm_client=mock_client, ttl_seconds=3600)  # No callback

    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-1", title="Doc 1", source_type="pdf"),
    ])
    # Change sources — should not crash
    cache.set("nb-1", "Test", "Summary", [], sources=[
        SourceInfo(id="src-2", title="Doc 2", source_type="url"),
    ])
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_module/test_notebook_cache.py -v -k "sources_changed"`
Expected: FAIL (no `on_sources_changed` parameter)

**Step 3: Implement source change detection**

Modify `NotebookCache.__init__` to accept `on_sources_changed: Callable[[str], None] | None = None`. Modify `set()` to compare old vs new source IDs and fire callback when different.

**Step 4: Run all notebook cache tests**

Run: `uv run pytest tests/test_openai_module/test_notebook_cache.py -v`
Expected: All PASS (including existing tests)

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/notebook_cache.py tests/test_openai_module/test_notebook_cache.py
git commit -m "feat(cache): add on_sources_changed callback to NotebookCache"
```

---

## Task 8: Source ID Provider — Read from NotebookCache

**Files:**
- Modify: `src/nlm_proxy/core/client.py`

**Step 1: Modify `query()` and `query_stream()` in `client.py`**

Add `notebook_cache` parameter to `NotebookLMClient.__init__` (optional, default None).

In both `query()` and `query_stream()`, before the `get_notebook()` fallback:

```python
if source_ids is None and self._notebook_cache:
    info = self._notebook_cache.get(notebook_id)
    if info and info.sources:
        source_ids = [s.id for s in info.sources]
        logger.debug(f"[CACHE] Source IDs from NotebookCache: {len(source_ids)} sources")
```

**Step 2: Run existing tests to ensure no regressions**

Run: `uv run pytest tests/ -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add src/nlm_proxy/core/client.py
git commit -m "feat(cache): read source IDs from NotebookCache in query/query_stream"
```

---

## Task 9: Server Integration — Wire Caches Together

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Wire ResponseCache + NotebookCache in server startup**

In the `lifespan` function of `server.py`:

1. Import `CacheSettings`, `ResponseCache`, `ExternalLLMClient`
2. Create `ExternalLLMClient` for Layer 3 (reusing routing LLM settings)
3. Create `ResponseCache` with settings from `CacheSettings`
4. Create `NotebookCache` with `on_sources_changed=response_cache.invalidate_notebook`
5. Store on `app.state.response_cache`
6. Pass `notebook_cache` to `NotebookLMClient`

**Step 2: Add cache check to `chat_completions` and `handle_smart_routing`**

Before session store lookup:

```python
# First-turn detection
is_first_turn = (request.conversation_id is None)
if chat_id and app.state.session_store:
    stored_conv_id = app.state.session_store.get(chat_id)
    if stored_conv_id:
        is_first_turn = False

# Cache check (first-turn only)
if is_first_turn and not request.bypass_cache and app.state.response_cache:
    cache_result = app.state.response_cache.lookup(notebook_id, query_text)
    # or lookup_async for smart routing
    if cache_result:
        # Return cached response with X-Cache-Status header
        ...
```

**Step 3: Add `X-Cache-Status` header and `system_fingerprint` encoding**

For both streaming and non-streaming responses.

**Step 4: Add thinking collection in streaming paths**

Collect thinking + answer during streaming, store in cache after stream completes.

**Step 5: Run existing server tests**

Run: `uv run pytest tests/test_openai_module/test_server.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(cache): integrate ResponseCache into server with cache check flow"
```

---

## Task 10: pyproject.toml — Add Cache Extras

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add cache extras**

```toml
[project.optional-dependencies]
cache = ["fastembed>=0.4"]
cache-gpu = ["fastembed-gpu>=0.4"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

Note: `numpy` is NOT listed explicitly — it's a transitive dependency of `fastembed` via `onnxruntime`.

**Step 2: Verify installation**

Run: `uv pip install -e ".[cache,dev]"`
Expected: fastembed and numpy installed

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(cache): add cache and cache-gpu extras to pyproject.toml"
```

---

## Task 11: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `GEMINI.md`
- Modify: `docs/smart-routing-architecture.md`

**Step 1: Update all docs per project rules**

Add response cache documentation to each file:
- `README.md`: Add cache feature description, configuration, usage
- `.env.example`: Add `NLM_PROXY_CACHE_*` environment variables
- `GEMINI.md`: Update project overview mentioning cache, add quick commands
- `docs/smart-routing-architecture.md`: Update architecture to show cache layer

**Step 2: Commit**

```bash
git add README.md .env.example GEMINI.md docs/smart-routing-architecture.md
git commit -m "docs: add response cache documentation"
```

---

## Verification Plan

### Automated Tests

Run: `uv run pytest tests/ -v`

Expected test count per file:
| Test file | Tests |
|---|---|
| `tests/core/test_response_cache.py` | ~16 (Layer 1 + data structures) |
| `tests/core/test_response_cache_semantic.py` | ~7 (Layer 2 embedding) |
| `tests/core/test_response_cache_llm.py` | ~9 (Layer 3 LLM + prompt) |
| `tests/core/test_response_cache_integration.py` | ~7 (full L1→L2→L3 flow) |
| `tests/test_openai_module/test_notebook_cache.py` | +2 (source change callbacks) |
| `tests/test_config.py` | +4 (CacheSettings) |
| `tests/test_openai_types.py` | +2 (bypass_cache field) |

**Total new tests: ~47**

### Manual Verification

1. **Start OpenAI proxy**: `nlm-proxy serve openai --port 8080`
2. **Send a query**: Observe `[CACHE] MISS` in logs, response takes 40-50s
3. **Send same query again**: Observe `[CACHE] HIT (exact)` in logs, response is instant
4. **Send `bypass_cache=True`**: Observe `[CACHE] BYPASS` in logs, response takes 40-50s
5. **Check `X-Cache-Status` header** in responses
