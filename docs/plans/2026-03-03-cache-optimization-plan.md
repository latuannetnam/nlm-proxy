# Cache Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize response cache for client query rewriting — reduce cache-hit latency from 3-5s to <100ms and improve hit rate for rewritten queries.

**Architecture:** Two-phase cache lookup (pre-routing global L1 + post-routing L2/L3 with alias creation). Switch embedding model to MiniLM for RAM savings. Lower L2 threshold to 0.5 with config validation.

**Tech Stack:** Python, fastembed (MiniLM-L12-v2), NumPy, pytest

**Design doc:** [2026-03-03-cache-optimization-design.md](file:///d:/latuan/Programming/nlm-proxy/docs/plans/2026-03-03-cache-optimization-design.md)

---

### Task 1: Embedding Model Test Suite

**Files:**
- Create: `tests/core/test_embedding_models.py`

**Step 1: Write the embedding model tests**

```python
"""Test embedding model performance for Vietnamese and multilingual queries."""

import pytest

# Skip all tests if fastembed is not installed
fastembed = pytest.importorskip("fastembed")

from fastembed import TextEmbedding
import numpy as np
import time


# Use MiniLM (the target model for optimization)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="module")
def model():
    """Load embedding model once for all tests."""
    return TextEmbedding(MODEL_NAME)


def cosine_sim(model: TextEmbedding, text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    embeddings = list(model.embed([text_a, text_b]))
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestSameIntent:
    """Category 1: Same intent queries should be similar."""

    def test_hr_policy_vietnamese(self, model):
        sim = cosine_sim(model, "Chính sách nhân sự của công ty là gì?", "Cho tôi biết về chính sách nhân sự")
        assert sim > 0.5, f"Same intent Vietnamese: sim={sim:.4f}, expected >0.5"

    def test_department_staff_vietnamese(self, model):
        sim = cosine_sim(model, "Phòng TSD có những ai", "Danh sách nhân sự phòng TSD")
        assert sim > 0.5, f"Same intent Vietnamese: sim={sim:.4f}, expected >0.5"

    def test_greeting_with_context(self, model):
        sim = cosine_sim(model, "Xin chào", "Xin chào, tôi muốn tìm hiểu thông tin")
        assert sim > 0.3, f"Greeting + context: sim={sim:.4f}, expected >0.3"


class TestDifferentIntent:
    """Category 2: Different intent queries should be dissimilar."""

    def test_hr_vs_weather(self, model):
        sim = cosine_sim(model, "Chính sách nhân sự", "Thời tiết hôm nay")
        assert sim < 0.3, f"Different intent: sim={sim:.4f}, expected <0.3"

    def test_staff_vs_revenue(self, model):
        sim = cosine_sim(model, "Phòng TSD có những ai", "Doanh thu quý 3")
        assert sim < 0.3, f"Different intent: sim={sim:.4f}, expected <0.3"


class TestCrossLingual:
    """Category 3: Cross-lingual Vietnamese ↔ English."""

    def test_greeting_cross_lingual(self, model):
        sim = cosine_sim(model, "Xin chào, tôi muốn biết thông tin", "Hello, I want to know information")
        assert sim > 0.7, f"Cross-lingual: sim={sim:.4f}, expected >0.7"


class TestRewriteVariants:
    """Category 4: Query rewrite with context enrichment."""

    def test_short_to_contextual(self, model):
        sim = cosine_sim(model, "Có những ai", "Phòng TSD có những ai")
        assert sim > 0.3, f"Rewrite variant: sim={sim:.4f}, expected >0.3"

    def test_salary_rewrite(self, model):
        sim = cosine_sim(model, "Lương bao nhiêu", "Mức lương trung bình của nhân viên NetNam")
        assert sim > 0.3, f"Rewrite variant: sim={sim:.4f}, expected >0.3"


class TestPerformance:
    """Embedding performance benchmarks."""

    def test_single_query_latency(self, model):
        text = "Chính sách nhân sự của công ty là gì?"
        start = time.perf_counter()
        list(model.embed([text]))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Single query latency: {elapsed_ms:.1f}ms, expected <200ms"

    def test_batch_latency(self, model):
        texts = [f"Query number {i} about company policy" for i in range(10)]
        start = time.perf_counter()
        list(model.embed(texts))
        elapsed_ms = (time.perf_counter() - start) * 1000
        per_query = elapsed_ms / len(texts)
        assert per_query < 100, f"Batch per-query: {per_query:.1f}ms, expected <100ms"
```

**Step 2: Run tests**

```bash
uv run pytest tests/core/test_embedding_models.py -v
```
Expected: All tests PASS. Note actual similarity scores for Category 4 — if any fail, adjust thresholds or reconsider model choice.

**Step 3: Commit**

```bash
git add tests/core/test_embedding_models.py
git commit -m "test: add embedding model test suite for Vietnamese and multilingual queries"
```

---

### Task 2: Global Hash Index + `lookup_global()`

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py`
- Test: `tests/core/test_response_cache.py`

**Step 1: Write failing tests**

Add to existing `tests/core/test_response_cache.py`:

```python
class TestGlobalLookup:
    """Test pre-routing global L1 lookup (notebook-agnostic)."""

    def test_lookup_global_hit(self, cache):
        """Global lookup finds entry without knowing notebook_id."""
        cache.store(
            notebook_id="nb1", query="test query", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        result = cache.lookup_global("test query")
        assert result is not None
        assert result.answer == "answer"
        assert result.notebook_id == "nb1"

    def test_lookup_global_miss(self, cache):
        """Global lookup returns None when query not cached."""
        result = cache.lookup_global("unknown query")
        assert result is None

    def test_lookup_global_case_insensitive(self, cache):
        """Global lookup normalizes case."""
        cache.store(
            notebook_id="nb1", query="Hello World", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        result = cache.lookup_global("hello world")
        assert result is not None

    def test_lookup_global_after_eviction(self, cache_small):
        """Global index cleaned up on eviction."""
        # cache_small has max_entries=2
        cache_small.store(notebook_id="nb1", query="q1", answer="a1", thinking=None, conversation_id="c1")
        cache_small.store(notebook_id="nb1", query="q2", answer="a2", thinking=None, conversation_id="c2")
        cache_small.store(notebook_id="nb1", query="q3", answer="a3", thinking=None, conversation_id="c3")
        # q1 should be evicted
        assert cache_small.lookup_global("q1") is None
        assert cache_small.lookup_global("q3") is not None

    def test_lookup_global_after_invalidation(self, cache):
        """Global index cleaned up on notebook invalidation."""
        cache.store(notebook_id="nb1", query="q1", answer="a1", thinking=None, conversation_id="c1")
        cache.invalidate_notebook("nb1")
        assert cache.lookup_global("q1") is None

    def test_lookup_global_after_clear(self, cache):
        """Global index cleaned up on clear."""
        cache.store(notebook_id="nb1", query="q1", answer="a1", thinking=None, conversation_id="c1")
        cache.clear()
        assert cache.lookup_global("q1") is None
```

**Step 2: Run tests — expect failures**

```bash
uv run pytest tests/core/test_response_cache.py::TestGlobalLookup -v
```
Expected: FAIL — `lookup_global` method doesn't exist yet.

**Step 3: Implement global hash index**

In `response_cache.py`, modify `__init__`, `store`, `_evict_oldest`, `_remove_entry`, `invalidate_notebook`, `clear`:

```python
# In __init__, add:
self._global_hash_index: dict[str, CachedResponse] = {}

# Add new method:
@staticmethod
def _compute_global_hash(query: str) -> str:
    """Compute hash on query only (no notebook_id) for pre-routing lookup."""
    normalized = query.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

# Add new method:
def lookup_global(self, query: str) -> CachedResponse | None:
    """Pre-routing L1 lookup: find cached entry by query only (no notebook_id).

    Returns CachedResponse on hit (caller must validate notebook ACL), None on miss.
    """
    global_hash = self._compute_global_hash(query)
    with self._lock:
        entry = self._global_hash_index.get(global_hash)
        if entry is None:
            return None
        # Check TTL
        age = time.time() - entry.cached_at
        if age > self._ttl_seconds:
            logger.debug("[CACHE] Global L1 EXPIRED for '%s'", query[:80])
            return None
        entry.hit_count += 1
        logger.info(
            "[CACHE] Global L1 HIT for '%s' (notebook=%s, hits=%d, age=%.0fs)",
            query[:80], entry.notebook_id[:12], entry.hit_count, age,
        )
        self._last_hit_type = "exact"
        self._stats["l1_hits"] += 1
        return entry

# In store(), after self._cache_by_hash[query_hash] = entry, add:
global_hash = self._compute_global_hash(query)
self._global_hash_index[global_hash] = entry

# In store() update path (existing entry), also update global index:
global_hash = self._compute_global_hash(query)
self._global_hash_index[global_hash] = existing

# In _remove_entry(), add cleanup:
global_hash = self._compute_global_hash(entry.query)
self._global_hash_index.pop(global_hash, None)

# In invalidate_notebook(), inside the for loop, add:
global_hash = self._compute_global_hash(entry.query)
self._global_hash_index.pop(global_hash, None)

# In clear(), add:
self._global_hash_index.clear()
```

**Step 4: Run tests — expect pass**

```bash
uv run pytest tests/core/test_response_cache.py::TestGlobalLookup -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_response_cache.py
git commit -m "feat(cache): add global hash index for pre-routing L1 lookup"
```

---

### Task 3: Alias Creation on L2/L3 Match

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py`
- Test: `tests/core/test_response_cache.py`

**Step 1: Write failing tests**

```python
class TestAliasCreation:
    """Test alias creation on semantic match."""

    def test_create_alias_enables_l1_hit(self, cache):
        """After alias creation, new query gets L1 hit."""
        cache.store(
            notebook_id="nb1", query="original query", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        original_entry = cache.lookup("nb1", "original query")

        # Create alias
        cache.create_alias("nb1", "rewritten query", original_entry)

        # Alias should hit L1
        result = cache.lookup("nb1", "rewritten query")
        assert result is not None
        assert result.answer == "answer"

    def test_alias_in_global_index(self, cache):
        """Alias also available in global lookup."""
        cache.store(
            notebook_id="nb1", query="original", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        entry = cache.lookup("nb1", "original")
        cache.create_alias("nb1", "alias query", entry)
        result = cache.lookup_global("alias query")
        assert result is not None
        assert result.answer == "answer"

    def test_alias_not_counted_in_lru(self, cache):
        """Aliases don't consume LRU capacity."""
        cache.store(
            notebook_id="nb1", query="primary", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        entry = cache.lookup("nb1", "primary")
        # Create 5 aliases
        for i in range(5):
            cache.create_alias("nb1", f"alias {i}", entry)
        stats = cache.get_stats()
        # Only 1 primary entry, aliases don't count
        assert stats["entries"] == 1

    def test_alias_cleaned_on_invalidation(self, cache):
        """Aliases cleaned up when notebook invalidated."""
        cache.store(
            notebook_id="nb1", query="primary", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        entry = cache.lookup("nb1", "primary")
        cache.create_alias("nb1", "alias", entry)
        cache.invalidate_notebook("nb1")
        assert cache.lookup("nb1", "alias") is None
        assert cache.lookup_global("alias") is None

    def test_alias_cleaned_on_clear(self, cache):
        """Aliases cleaned up on full clear."""
        cache.store(
            notebook_id="nb1", query="primary", answer="answer",
            thinking=None, conversation_id="conv1",
        )
        entry = cache.lookup("nb1", "primary")
        cache.create_alias("nb1", "alias", entry)
        cache.clear()
        assert cache.lookup("nb1", "alias") is None
```

**Step 2: Run tests — expect failures**

```bash
uv run pytest tests/core/test_response_cache.py::TestAliasCreation -v
```

**Step 3: Implement `create_alias()` and alias tracking**

```python
# In __init__, add:
self._alias_hashes: set[str] = set()         # notebook-scoped alias hashes
self._alias_global_hashes: set[str] = set()   # global alias hashes

# New method:
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

# Modify get_stats() — count only primary entries:
# Change: "entries": len(self._cache_by_hash)
# To:     "entries": len(self._cache_by_hash) - len(self._alias_hashes)

# Modify _evict_oldest() — skip aliases in LRU:
# When iterating _lru_order, skip hashes that are in _alias_hashes

# Modify invalidate_notebook() — also clean aliases:
# After removing entries, clean aliases pointing to those entries
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

# Modify clear() — also clean alias tracking:
self._alias_hashes.clear()
self._alias_global_hashes.clear()
```

**Step 4: Run tests — expect pass**

```bash
uv run pytest tests/core/test_response_cache.py::TestAliasCreation -v
```

**Step 5: Wire alias creation into `lookup_async()`**

In `lookup_async()`, after L2 HIT or L3 HIT, call `create_alias()`:

```python
# After L2 near-exact match (line ~289):
self.create_alias(notebook_id, query, entry)
return entry

# After L3 LLM-verified match (line ~308):
self.create_alias(notebook_id, query, matched)
return matched
```

**Step 6: Run all cache tests**

```bash
uv run pytest tests/core/test_response_cache.py -v
```

**Step 7: Commit**

```bash
git add src/nlm_proxy/core/response_cache.py tests/core/test_response_cache.py
git commit -m "feat(cache): alias creation on L2/L3 match for improved hit rate"
```

---

### Task 4: Pre-Routing L1 in Server + ACL Check

**Files:**
- Modify: `src/nlm_proxy/openai/server.py:360-420` (streaming smart-routing)
- Modify: `src/nlm_proxy/openai/server.py:430-470` (non-streaming smart-routing)

**Step 1: Add pre-routing L1 check in streaming path**

In `handle_smart_routing()`, add BEFORE `router.route()`:

```python
query = user_messages[-1].content

# Phase 1: Pre-routing global L1 check (instant, skips routing)
if not request.bypass_cache and app.state.response_cache:
    cache_result = app.state.response_cache.lookup_global(query)
    if cache_result:
        cached_notebook_id = cache_result.notebook_id
        # ACL check: is the cached notebook accessible?
        if request_allowed_notebooks is None or cached_notebook_id in request_allowed_notebooks:
            hit_type = app.state.response_cache._last_hit_type or "exact"
            logger.info(
                "[CACHE] Pre-routing L1 HIT: query='%s', notebook=%s (skipped routing)",
                query[:80], cached_notebook_id[:12],
            )
            # ... return cached response (streaming or non-streaming)
        else:
            logger.debug(
                "[CACHE] Pre-routing L1 HIT but notebook %s not in allowed list, falling through",
                cached_notebook_id[:12],
            )

# Phase 2: Route normally (only reached on Phase 1 miss)
decision = await router.route(query, allowed_notebooks=request_allowed_notebooks)
```

**Step 2: Add same pre-routing L1 check in non-streaming path**

Same pattern in the non-streaming `with tracer.start_as_current_span(...)` block.

**Step 3: Manual test**

```bash
# Start proxy
.\run_proxy.ps1

# Send same query twice from Open WebUI
# Check logs for:
#   1st query: "L1 MISS" → NLM → "STORED"
#   2nd query: "Pre-routing L1 HIT ... (skipped routing)"
```

**Step 4: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat(cache): pre-routing global L1 check with ACL validation"
```

---

### Task 5: Lower L2 Threshold + Config Validation

**Files:**
- Modify: `src/nlm_proxy/core/config.py:273`
- Modify: `src/nlm_proxy/core/response_cache.py` (init validation)
- Modify: `.env.example`

**Step 1: Change default threshold**

In `config.py`:
```python
similarity_threshold: float = Field(
    default=0.5, description="Min cosine similarity for L2 pre-filter"
)
```

**Step 2: Add startup validation in `ResponseCache.__init__`**

```python
# After setting thresholds:
if self._similarity_threshold < 0.7 and self._llm_client is None:
    logger.warning(
        "[CACHE] L2 threshold (%.2f) is below 0.7 but no LLM client configured for L3 verification. "
        "Semantic matching may produce false positives. Consider raising threshold to 0.7+ "
        "or configuring an LLM client.",
        self._similarity_threshold,
    )
```

**Step 3: Update `.env.example`**

```env
NLM_PROXY_CACHE_SIMILARITY_THRESHOLD=0.5
```

**Step 4: Run tests**

```bash
uv run pytest tests/core/ -v
```

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/config.py src/nlm_proxy/core/response_cache.py .env.example
git commit -m "feat(cache): lower L2 threshold to 0.5, add config validation warning"
```

---

### Task 6: Switch Embedding Model to MiniLM

**Files:**
- Modify: `src/nlm_proxy/core/config.py:269-271`
- Modify: `.env.example`
- Modify: `.agent/memory/response-cache.md`
- Modify: `.agent/memory/configuration.md`

**Step 1: Update default model**

In `config.py`:
```python
embedding_model: str = Field(
    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    description="fastembed model for L2 embedding pre-filter",
)
```

**Step 2: Update `.env.example`**

```env
NLM_PROXY_CACHE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Step 3: Update memory docs**

Update default model name in `.agent/memory/response-cache.md` and `.agent/memory/configuration.md`.

**Step 4: Run embedding tests to verify MiniLM works**

```bash
uv run pytest tests/core/test_embedding_models.py -v
```

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/config.py .env.example .agent/memory/response-cache.md .agent/memory/configuration.md
git commit -m "feat(cache): switch to MiniLM model (0.22GB, saves ~0.8GB RAM)"
```

---

### Task 7: Documentation Update

**Files:**
- Modify: `GEMINI.md`
- Modify: `.agent/memory/response-cache.md`
- Modify: `docs/smart-routing-architecture.md` (if cache flow diagram exists)

**Step 1: Update response-cache.md**

Add sections for:
- Pre-routing global L1 lookup
- Alias creation on semantic match
- Updated thresholds and model

**Step 2: Update GEMINI.md**

Update any cache-related entries if needed.

**Step 3: Commit**

```bash
git add -A
git commit -m "docs: update cache documentation for optimization changes"
```

---

### Task 8: Integration Test

**Step 1: Start proxy and test full flow**

```bash
.\run_proxy.ps1
```

**Step 2: Verify with Open WebUI**

1. Open new chat, send "Xin chào" → expect L1 MISS → NLM → STORED
2. Same chat, send "Xin chào" again → expect pre-routing L1 HIT (instant, <100ms)
3. Open new chat, send "Xin chào" → expect pre-routing L1 HIT (cross-chat)
4. Check cache-stats for correct entry count (aliases not counted)

**Step 3: Verify cache stats**

```bash
.\scripts\cache-stats.ps1
```

Expected: entries=N (not inflated by aliases), hits increasing, semantic stats visible.

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: integration test verification complete"
```
