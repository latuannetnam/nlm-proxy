# Response Cache

Three-layer response cache eliminating 40-50s latency for repeated queries. Global, in-memory, volatile (lost on restart).

## Three-Layer Lookup

| Layer | Mechanism | Latency | When |
|-------|-----------|---------|------|
| L1 | Exact hash match | ~0ms | Always |
| L2 | Embedding cosine similarity (fastembed + NumPy) | ~10-30ms | If semantic enabled |
| L3 | LLM verification | ~1-2s | If L2 finds candidates below exact threshold |

**Early termination:** L2 similarity ≥ 0.95 skips L3 (returns directly).

## Configuration

All variables use `NLM_PROXY_CACHE_` prefix (class: `CacheSettings`):

```bash
NLM_PROXY_CACHE_RESPONSE_CACHE_ENABLED=true
NLM_PROXY_CACHE_RESPONSE_CACHE_TTL=14400          # 4 hours
NLM_PROXY_CACHE_RESPONSE_CACHE_MAX_ENTRIES=1000
NLM_PROXY_CACHE_SEMANTIC_MATCH_ENABLED=true
NLM_PROXY_CACHE_EMBEDDING_MODEL=intfloat/multilingual-e5-small
NLM_PROXY_CACHE_SIMILARITY_THRESHOLD=0.7
NLM_PROXY_CACHE_SIMILARITY_EXACT_THRESHOLD=0.95
NLM_PROXY_CACHE_SEMANTIC_MATCH_TOP_K=10
```

## Installation

`fastembed` and `numpy` are core dependencies — semantic matching (L2/L3) is always available.

```bash
uv pip install -e "."              # Core install (includes fastembed + numpy)
uv pip install -e ".[cache-gpu]"   # GPU-accelerated embeddings
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

- `total_hits`, `total_misses`, `total_bypasses` — aggregate counters
- `hit_rate` — percentage (0-100)
- `l1_hits`, `l2_hits`, `l3_hits`, `l3_misses` — per-layer breakdown
- `entry_count`, `notebook_count` — storage usage
- `max_entries`, `ttl_seconds`, `semantic_enabled` — config

## Cache Signaling

Clients detect cache hits via:
- `X-Cache-Status` header: `HIT_EXACT`, `HIT_SEMANTIC`, `MISS`, `BYPASS`
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

- `core/response_cache.py` — Cache implementation (L1/L2/L3, stats, store/lookup)
- `core/config.py` — `CacheSettings` class
- `openai/server.py` — Cache integration (check, store, endpoints)
- `openai/notebook_cache.py` — Auto-invalidation callback
