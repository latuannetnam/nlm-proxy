# Response Cache

Two-layer response cache eliminating 40-50s latency for repeated queries. Global, in-memory, volatile (lost on restart).

## Two-Phase Lookup Architecture

### Phase 1: Pre-routing Global L1 (instant, skips routing)

Before smart routing runs, `lookup_global(query)` checks a query-only hash index:
- Hit → validates notebook ACL → returns cached response instantly (~0ms)
- Miss → falls through to normal routing + Phase 2

Uses `_global_hash_index` (query hash without notebook_id) and `pre_routing_l1_hits` counter separated from post-routing `l1_hits`.

### Phase 2: Post-routing Two-Layer Lookup

| Layer | Mechanism | Latency | When |
|-------|-----------|---------|------|
| L1 | Exact hash match | ~0ms | Always |
| L2 | Embedding cosine similarity (HuggingFace + NumPy) | ~10-30ms | If semantic enabled |

> **Design Note:** L3 (LLM verification) was removed because it consistently misjudged HIT/MISS decisions in production, decreasing cache precision while adding 1-2s latency per lookup. L2 embedding similarity with a tuned threshold (0.93) proved more reliable and faster.

**Matching:** L2 similarity ≥ 0.93 → cache HIT (creates alias for future L1 hits).

## Alias Creation

On L2/L3 semantic match, `create_alias()` stores the new query as an alias pointing to the existing cached entry:
- Aliases get L1 hits on subsequent lookups (both per-notebook and global)
- Aliases are NOT counted in LRU capacity or `entries` stat
- Aliases are cleaned up on `invalidate_notebook()` and `clear()`
- Tracked via `_alias_hashes` and `_alias_global_hashes` sets

## Configuration

All variables use `NLM_PROXY_CACHE_` prefix (class: `CacheSettings`):

```bash
NLM_PROXY_CACHE_RESPONSE_CACHE_ENABLED=true
NLM_PROXY_CACHE_RESPONSE_CACHE_TTL=14400          # 4 hours
NLM_PROXY_CACHE_RESPONSE_CACHE_MAX_ENTRIES=1000
NLM_PROXY_CACHE_SEMANTIC_MATCH_ENABLED=true
NLM_PROXY_CACHE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  # 0.22GB
NLM_PROXY_CACHE_SIMILARITY_THRESHOLD=0.93           # L2 semantic match threshold
```



## Installation

`langchain-huggingface` and `numpy` are core dependencies — semantic matching (L2/L3) is always available.

```bash
uv pip install -e "."              # Core install (includes HuggingFace embeddings + numpy)
uv pip install -e ".[all]"         # All extras including dev tools
```

## Management API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/cache/stats` | GET | Hit/miss counters, layer breakdown, hit rate |
| `/v1/cache` | DELETE | Clear all cache entries |
| `/v1/cache/{notebook_id}` | DELETE | Clear entries for one notebook |

All require API key authentication.

## Stats Metrics

The `GET /v1/cache/stats` endpoint returns:

- `entries`, `aliases` — primary entries vs alias count
- `total_hits`, `total_misses`, `total_bypasses` — aggregate counters
- `hit_rate` — percentage (0-100)
- `pre_routing_l1_hits`, `l1_hits`, `l2_hits` — per-layer breakdown
- `entry_count`, `notebook_count` — storage usage
- `max_entries`, `ttl_seconds`, `semantic_enabled` — config

## Cache Signaling

Clients detect cache hits via:
- `X-Cache-Status` header: `HIT_PRE_ROUTING_EXACT`, `HIT_EXACT`, `HIT_SEMANTIC`, `MISS`, `BYPASS`
- `system_fingerprint`: `cache_exact_conv_{id}` or `cache_semantic_conv_{id}`

## Monitor Scripts

```powershell
.\scripts\cache-stats.ps1              # One-shot
.\scripts\cache-stats.ps1 -Watch       # Auto-refresh every 5s
```

```bash
./scripts/cache-stats.sh               # One-shot
./scripts/cache-stats.sh --watch       # Auto-refresh every 5s
```

## Auto-Invalidation

When `NotebookCache` detects source changes in a notebook, it calls `ResponseCache.invalidate_notebook()` to clear stale cache entries.

## Key Files

- `core/response_cache.py` — Cache implementation (L1/L2, global lookup, aliases, stats)
- `core/config.py` — `CacheSettings` class
- `openai/server.py` — Cache integration (pre-routing L1, check, store, endpoints)
- `openai/notebook_cache.py` — Auto-invalidation callback
- `tests/core/test_response_cache.py` — Unit tests (L1, global lookup, aliases)
- `tests/core/test_response_cache_semantic.py` — L2 embedding tests + async lookup tests
- `tests/core/test_embedding_models.py` — Embedding model validation suite
