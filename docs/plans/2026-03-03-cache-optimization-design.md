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

## Design

### Part 1: Pre-Routing L1 Check (Reduce Latency)

Add a **global hash index** for O(1) lookup before routing.

**Current flow (cache HIT = 3-5s):**
```
Query → Route: classify (2s) + select notebook (2s) → L1 check → HIT (0ms)
```

**Proposed flow (cache HIT = <100ms):**
```
Query → Global L1 (0ms) → HIT? Return instantly (skip routing)
                         → MISS? → Route (3-5s) → L2/L3 (post-routing)
```

#### Implementation

Add a secondary index in `ResponseCache`:
```python
# Global index: normalized_query_hash → CachedResponse (notebook-agnostic)
self._global_hash_index: dict[str, CachedResponse] = {}
```

- `_compute_global_hash(query)` — hash only on normalized query text (no notebook_id)
- On `store()`: also insert into global index
- On `_remove_entry()` / `_evict_oldest()`: also remove from global index
- New method `lookup_global(query)` — L1 only, no notebook needed

In `handle_smart_routing()`, check global L1 **before** `router.route()`:
```python
# Phase 1: Pre-routing L1 check (instant)
if not request.bypass_cache and app.state.response_cache:
    cache_result = app.state.response_cache.lookup_global(query)
    if cache_result:
        # Return cached response immediately — skip routing entirely
        ...

# Phase 2: Only if Phase 1 missed — route normally
decision = await router.route(query, ...)

# Phase 2b: Post-routing L2/L3 check (notebook-scoped semantic matching)
if decision.request_type == RequestType.NOTEBOOKLM:
    cache_result = await app.state.response_cache.lookup_async(
        decision.notebook_id, query
    )
```

### Part 2a: Alias Creation on L2/L3 Match (Improve Hit Rate)

When L2/L3 confirms a semantic match, **also store the new query as an L1 alias** pointing to the same cached response. This converts future L2/L3 matches into instant L1 hits.

```python
# After L3 confirms match:
matched_entry = existing_cached_response
# Create alias: store new query hash → same entry
alias_hash = self._compute_hash(notebook_id, new_query)
self._cache_by_hash[alias_hash] = matched_entry
# Also add to global index
global_hash = self._compute_global_hash(new_query)
self._global_hash_index[global_hash] = matched_entry
```

**Effect:**
1. "Xin chào" → MISS → NLM → stored
2. "Xin chào, bạn có thể..." → L1 MISS → L2/L3 match → **alias created**
3. "Xin chào, bạn có thể..." (again) → **L1 HIT** (instant)

### Part 2b: Lower L2 Threshold (0.7 → 0.5)

L2 is a pre-filter, not the final judge. Current 0.7 threshold rejects valid candidates (sim=0.30 for "Xin chào" vs rewritten). Lowering to 0.5 lets more candidates reach L3 LLM verification for accurate judgement.

> [!IMPORTANT]
> This requires L3 (LLM client) to be configured. Without L3, lower threshold increases false positive risk.

### Part 2c: Switch Embedding Model to MiniLM (0.22GB)

Since L2 is just a pre-filter for L3, we don't need the most accurate model — we need one that **doesn't miss candidates**. Trade-offs:

| Model | Size | Vietnamese sim | Role |
|-------|------|---------------|------|
| mpnet-base-v2 (current) | 1.0 GB | 0.82 | Overkill for pre-filter |
| **MiniLM-L12-v2** | **0.22 GB** | **0.66** | Sufficient for pre-filter |

MiniLM gives sim=0.66 for Vietnamese paraphrases. With threshold lowered to 0.5, this passes L2 and reaches L3 for verification. Saves ~0.8GB RAM on the 4GB VPS.

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

## Summary

| Change | Latency | Hit Rate | RAM | Complexity |
|--------|---------|----------|-----|------------|
| Pre-routing Global L1 | **-3-5s on exact hits** | — | ~negligible | Low |
| Alias creation on L2/L3 | **-3-5s on future repeats** | **+high** | ~negligible | Low |
| Lower L2 threshold (0.7→0.5) | — | **+moderate** | — | Trivial |
| Switch to MiniLM (1GB→0.22GB) | — | Slight decrease | **-0.8GB** | Trivial |

**Net effect:** Repeated queries return in <100ms (was 3-5s). Rewritten queries get cached on first L3 match, then instant on subsequent hits. VPS RAM freed by ~0.8GB.
