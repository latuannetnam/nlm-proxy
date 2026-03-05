# Drop L3 Cache — Simplify Response Cache to L1 + L2

## Problem

The three-layer Response Cache has precision issues observed in production:

1. **L1 (exact hash)**: Mostly useless because query rewrite changes exact text → hash never matches
2. **L2 (embedding similarity)**: Very efficient when `similarity_exact_threshold` > 0.9
3. **L3 (LLM verification)**: Misjudges HIT/MISS decisions — adds 1-2s latency AND decreases precision

## Solution

**Drop L3 entirely.** Use L2 embedding similarity with a single high threshold (0.93) as the sole semantic matching mechanism. Any L2 match above threshold → cache HIT (create alias, skip LLM).

### Key Changes

- Remove the two-threshold system (`similarity_threshold` + `similarity_exact_threshold`)
- Replace with a single `similarity_threshold` at 0.93
- Remove `_verify_semantic_match`, `_build_verification_prompt`, `_parse_semantic_match`
- Remove `llm_client` parameter from `ResponseCache`
- Remove `l3_hits`, `l3_misses` from stats
- Remove `top_k` parameter (no longer needed without L3 candidate ranking)
- Update default in `CacheSettings`
- Remove L3 LLM client creation in `server.py` cache initialization
- Update all documentation

### What Stays

- L1 exact hash (still useful for non-rewritten queries, pre-routing)
- L2 embedding pre-filter with NumPy cosine similarity
- Alias creation on semantic match
- All LRU, TTL, invalidation mechanics
