# Stage 0: Fix `_last_hit_type` Thread Safety

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the thread-unsafe `_last_hit_type` instance field with a return-value tuple from all cache lookup methods.

**Architecture:** Change `lookup()`, `lookup_async()`, and `lookup_global()` to return `(CachedResponse | None, str | None)` tuples. Remove the mutable `_last_hit_type` field. Update all callers in `server.py`.

**Inputs:** None — this is a standalone bug fix with zero LangChain dependency.

**Outputs:** All cache lookups return `(result, hit_type)` tuples. All 6 `_last_hit_type` references in `server.py` are replaced.

---

## Task 0.1: Update cache lookup return type

**Files:**
- Modify: `src/nlm_proxy/core/response_cache.py` (`lookup()`, `lookup_async()`, `lookup_global()`)
- Modify: `tests/core/test_response_cache_integration.py`

**Step 1: Write failing test**

Add to `tests/core/test_response_cache_integration.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_response_cache_integration.py -k "test_lookup_returns_cache_hit_type" -v`
Expected: FAIL — `ValueError: too many values to unpack`

**Step 3: Update `lookup()`, `lookup_async()`, `lookup_global()` to return tuples**

In `response_cache.py`:
- `lookup()`: change every `return cached_response` to `return (cached_response, "exact")`, every `return None` to `return (None, None)`, and semantic matches to `return (cached_response, "semantic")`
- `lookup_async()`: same pattern
- `lookup_global()`: same pattern
- Remove `self._last_hit_type` field from `__init__()` and all assignments to it

**Step 4: Update existing tests that unpack lookup results**

Search `tests/` for all calls to `.lookup(`, `.lookup_async(`, `.lookup_global(` and update them to unpack tuples:
```python
# Before:
result = cache.lookup(...)
# After:
result, hit_type = cache.lookup(...)
```

**Step 5: Run ALL tests**

Run: `uv run pytest -v`
Expected: Some server tests may fail (callers of `_last_hit_type`)

**Step 6: Commit cache changes**

```bash
git add src/nlm_proxy/core/response_cache.py tests/
git commit -m "fix: return (result, hit_type) tuple from cache lookup — thread safety"
```

---

## Task 0.2: Update all `_last_hit_type` callers in server.py

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Find and replace all `_last_hit_type` references**

There are **6 locations** in `server.py`. Search for `_last_hit_type`:

```python
# Pattern to find:
hit_type = app.state.response_cache._last_hit_type or "exact"
# Or:
cache_result = app.state.response_cache.lookup_global(query)

# Replace EACH cache lookup + _last_hit_type pair with:
cache_result, hit_type = app.state.response_cache.lookup_global(query)
# (remove the separate _last_hit_type line)
```

Locations (~lines 371-376, 436-438, 504-509, 564-566):
1. Streaming pre-routing L1 check
2. Streaming post-routing cache check
3. Non-streaming pre-routing L1 check
4. Non-streaming post-routing cache check

Also update `chat_completions()` direct notebook path if it uses `_last_hit_type`.

**Step 2: Run ALL tests**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "fix: update server.py to use (result, hit_type) tuple from cache lookup"
```

---

## 🔒 Stage 0 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS — cache API is now thread-safe, all callers updated.
