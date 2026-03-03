# Response Cache Optimization for Query Rewriting

## Problem

Chatbot clients (e.g., Open WebUI) rewrite follow-up queries into standalone questions before sending them to the proxy. This has two effects on the cache:

1. **Latency on cache HITs (~3-5s wasted):** Cache check happens after routing (classify + select notebook), so even exact-match cache hits still pay routing overhead.
2. **Low hit rate for rewrites:** The same user intent rewrites differently each time (e.g., "Xin chào" → "Xin chào, bạn có thể giúp mình tra cứu..."). L1 misses (different hash), L2 misses (sim=0.30, threshold=0.70).

### Request Flow (Open WebUI)

Per user message, the client sends 3 requests:
1. `### Task: Rewrite this follow-up...` → LLM_TASK → external LLM rewrites (2-3s)
2. Rewritten standalone query → NOTEBOOKLM → **cache lives here** (3-5s routing + 40-50s NLM)
3. `### Task: Suggest follow-up questions...` → LLM_TASK → external LLM suggests (2-3s)

Cache only matters for request #2. Requests #1 and #3 are client behavior we don't control.

---

## Design

### Part 1: Pre-Routing L1 Check (Reduce Latency)

Add a **global hash index** for O(1) lookup before routing.

**Current flow (cache HIT = 3-5s):**
```
Query → Route: classify (2s) + select notebook (2s) → L1 check → HIT (0ms)
```

**Proposed flow (cache HIT = <100ms):**
```
Query → Global L1 (0ms) → HIT + ACL ok? Return instantly (skip routing)
                         → MISS or ACL fail? → Route (3-5s) → L2/L3 (post-routing)
```

#### Implementation

Add a secondary index in `ResponseCache`:
```python
# Global index: query-only hash → CachedResponse (no notebook_id in hash)
self._global_hash_index: dict[str, CachedResponse] = {}
```

- `_compute_global_hash(query)` — hash only on `query.strip().lower()` (no notebook_id)
- On `store()`: also insert into global index
- On `_remove_entry()` / `_evict_oldest()`: also remove from global index
- New method `lookup_global(query)` → returns `(CachedResponse, notebook_id)` or `None`

#### ACL Safety for Pre-Routing L1

> [!CAUTION]
> Original design keys cache on `(notebook_id, query)`. A global hash bypasses notebook scoping, which could leak answers across notebook boundaries if ACL is active.

**Fix:** `lookup_global()` returns the cached entry's `notebook_id`. The caller validates it against the user's allowed notebooks **before** returning:

```python
# In handle_smart_routing(), BEFORE router.route():
if not request.bypass_cache and app.state.response_cache:
    cache_result = app.state.response_cache.lookup_global(query)
    if cache_result:
        cached_notebook_id = cache_result.notebook_id
        # ACL check: is this notebook accessible to the current user?
        if request_allowed_notebooks is None or cached_notebook_id in request_allowed_notebooks:
            # Safe to return — user has access to this notebook
            return cached_response(cache_result)
        else:
            # ACL violation — skip pre-routing hit, fall through to routing
            logger.debug(
                "[CACHE] Pre-routing L1 HIT but notebook %s not in allowed list, skipping",
                cached_notebook_id[:12],
            )

# Phase 2: Fall through to routing
decision = await router.route(query, ...)
```

**Edge case — multiple notebooks, same query:** If two notebooks can answer the same query, the global index stores whichever was cached last. The notebook-scoped L1 (post-routing) handles the case where routing picks a different notebook.

---

### Part 2a: Alias Creation on L2/L3 Match (Improve Hit Rate)

When L2/L3 confirms a semantic match, **also store the new query as an L1 alias** pointing to the same cached response. This converts future L2/L3 matches into instant L1 hits.

#### Alias Implementation

```python
def create_alias(self, notebook_id: str, new_query: str, target_entry: CachedResponse):
    """Create L1 alias: new query hash → existing cached entry."""
    alias_hash = self._compute_hash(notebook_id, new_query)
    global_hash = self._compute_global_hash(new_query)

    with self._lock:
        # Store as alias (not a full entry — no LRU slot consumed)
        self._cache_by_hash[alias_hash] = target_entry
        self._global_hash_index[global_hash] = target_entry
        # Track alias for cleanup
        self._alias_hashes.add(alias_hash)
        self._alias_global_hashes.add(global_hash)

    logger.info(
        "[CACHE] Alias created: '%s' → '%s' (notebook=%s)",
        new_query[:60], target_entry.query[:60], notebook_id[:12],
    )
```

#### Alias Lifecycle

- **TTL:** Aliases point to the same `CachedResponse` object — when the original expires via TTL, alias lookups find an expired entry and return miss. ✅ No stale data.
- **Eviction:** Aliases are NOT counted in LRU capacity. Only primary entries (with unique answers) consume LRU slots. This prevents alias explosion from eating cache capacity.
- **Invalidation:** When `invalidate_notebook()` is called, all aliases pointing to entries in that notebook are also cleaned up.
- **Cleanup tracking:** `self._alias_hashes: set[str]` and `self._alias_global_hashes: set[str]` track which hashes are aliases vs primary entries, so LRU eviction skips aliases.

**Effect:**
1. "Xin chào" → MISS → NLM → stored (1 LRU slot)
2. "Xin chào, bạn có thể..." → L1 MISS → L2/L3 match → **alias created** (0 LRU slots)
3. "Xin chào, bạn có thể..." (again) → **L1 HIT** (instant, pre-routing)

---

### Part 2b: Lower L2 Threshold (0.7 → 0.5)

L2 is a pre-filter, not the final judge. Current 0.7 threshold rejects valid candidates (sim=0.30 for "Xin chào" vs rewritten). Lowering to 0.5 lets more candidates reach L3 LLM verification for accurate judgement.

> [!WARNING]
> **Config validation required:** If `similarity_threshold < 0.7` and `llm_client` is not configured, log a startup warning:
> ```
> [CACHE] WARNING: L2 threshold (0.5) is below 0.7 but no LLM client configured for L3 verification.
> Semantic matching may produce false positives. Consider raising threshold to 0.7+ or configuring an LLM client.
> ```
> This prevents accidental false cache hits when L3 is not available to verify L2 candidates.

---

### Part 2c: Switch Embedding Model to MiniLM (0.22GB)

Since L2 is just a pre-filter for L3, we don't need the most accurate model — we need one that **doesn't miss candidates**. Trade-offs:

| Model | Size | Vietnamese sim | Role |
|-------|------|---------------|------|
| mpnet-base-v2 (current) | 1.0 GB | 0.82 | Overkill for pre-filter |
| **MiniLM-L12-v2** | **0.22 GB** | **0.66** | Sufficient for pre-filter |

MiniLM gives sim=0.66 for Vietnamese paraphrases. With threshold lowered to 0.5, this passes L2 and reaches L3 for verification. Saves ~0.8GB RAM on the 4GB VPS.

---

## Part 3: Embedding Model Verification

### Test Plan

Create a test suite (`tests/core/test_embedding_models.py`) that validates embedding model performance for Vietnamese and multilingual queries.

#### Test Cases

**Category 1: Same Intent — should be similar (sim > threshold)**
| Query A | Query B | Expected |
|---------|---------|----------|
| "Chính sách nhân sự của công ty là gì?" | "Cho tôi biết về chính sách nhân sự" | sim > 0.5 |
| "Phòng TSD có những ai" | "Danh sách nhân sự phòng TSD" | sim > 0.5 |
| "Xin chào" | "Xin chào, tôi muốn tìm hiểu thông tin" | sim > 0.3 |

**Category 2: Different Intent — should be dissimilar (sim < 0.3)**
| Query A | Query B | Expected |
|---------|---------|----------|
| "Chính sách nhân sự" | "Thời tiết hôm nay" | sim < 0.3 |
| "Phòng TSD có những ai" | "Doanh thu quý 3" | sim < 0.3 |

**Category 3: Cross-lingual — Vietnamese ↔ English**
| Query A | Query B | Expected |
|---------|---------|----------|
| "Xin chào, tôi muốn biết thông tin" | "Hello, I want to know information" | sim > 0.7 |

**Category 4: Rewrite variants — query + context enrichment**
| Original | Rewritten (with context) | Expected |
|----------|--------------------------|----------|
| "Có những ai" | "Phòng TSD có những ai" | sim > 0.3 |
| "Lương bao nhiêu" | "Mức lương trung bình của nhân viên NetNam" | sim > 0.3 |

#### Performance Benchmarks
- Embedding latency: < 50ms per query (single query)
- Model load time: < 10s (cold start)
- Memory usage: < 500MB (model in RAM)

#### Test Script
```bash
uv run pytest tests/core/test_embedding_models.py -v
```

---

## Compatibility with Original Design

Cross-referenced against [response-cache-design.md](file:///d:/latuan/Programming/nlm-proxy/docs/plans/2026-03-02-response-cache-design.md):

| Original Design Principle | Optimization Compatibility | Mitigation |
|--------------------------|---------------------------|------------|
| Cache keyed on `(notebook_id, query)` | Global L1 uses `query` only | ACL check at pre-routing hit time |
| First-turn only (lookup + store) | All turns (already changed) | Client rewrites → standalone queries |
| LRU capacity = 1000 entries | Aliases could inflate count | Aliases tracked separately, not counted in LRU |
| L3 strict matching guarantee | Aliases lock in L3 judgement | Aliases inherit TTL → expire with original |
| Graceful degradation (no LLM) | Lower L2 threshold risk | Config validation warning at startup |

---

## Summary

| Change | Latency | Hit Rate | RAM | Complexity |
|--------|---------|----------|-----|------------|
| Pre-routing Global L1 + ACL check | **-3-5s on exact hits** | — | ~negligible | Low |
| Alias creation on L2/L3 (no LRU cost) | **-3-5s on future repeats** | **+high** | ~negligible | Low |
| Lower L2 threshold (0.7→0.5) + validation | — | **+moderate** | — | Trivial |
| Switch to MiniLM (1GB→0.22GB) | — | Slight decrease | **-0.8GB** | Trivial |
| Embedding model test suite | — | Validates quality | — | Low |

**Net effect:** Repeated queries return in <100ms (was 3-5s). Rewritten queries get cached on first L3 match, then instant on subsequent hits. VPS RAM freed by ~0.8GB.
