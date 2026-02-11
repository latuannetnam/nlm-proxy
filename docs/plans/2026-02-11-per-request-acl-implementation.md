# Per-Request ACL Filtering — Implementation Plan

> **Type**: Feature Enhancement
> **Status**: Ready for Implementation
> **Date**: 2026-02-11
> **Spec**: `docs/plans/per-request-acl-specification.md`
> **Estimated Effort**: ~25 lines production code, ~120 lines tests, ~80 lines docs

---

## Context

The knowledge-finder-bot chatbot already sends `metadata.allowed_notebooks` in OpenAI chat completion requests to restrict which notebooks a user can query (based on Azure AD group membership). However, the nlm-proxy Smart Router **does not yet process this field** — `route()` and `select_notebook()` ignore the metadata entirely. This plan implements the server-side filtering to complete the integration.

---

## Two-Layer Filtering Architecture

```
Layer 1 (Cache): NLM_PROXY_ROUTING_ALLOWED_NOTEBOOKS env var
    → Controls which notebooks get cached at all (server-wide)
    → Already implemented, no changes needed

Layer 2 (Per-Request): metadata.allowed_notebooks in request body
    → Filters cached notebooks for each individual request
    → TO BE IMPLEMENTED
```

These compose naturally: cache holds only server-allowed notebooks, per-request ACL further restricts per user.

---

## Implementation Steps

### Step 1: Add ACL filtering to `select_notebook()` in `router.py`

**File**: `src/nlm_proxy/openai/router.py` (lines 100-111)

- Add `allowed_notebooks: list[str] | None = None` parameter
- After `_ensure_notebooks_cached()`, filter notebooks if `allowed_notebooks` is provided
- Add OTEL span attributes: `acl_filter_applied`, `acl_allowed_count`, `acl_matched_count`
- Return early with "No accessible notebooks for this user" if filter produces empty list
- Move `candidates_count` attribute to after filtering

### Step 2: Add `allowed_notebooks` pass-through to `route()` in `router.py`

**File**: `src/nlm_proxy/openai/router.py` (lines 210-229)

- Add `allowed_notebooks: list[str] | None = None` parameter to `route()`
- Pass it to `self.select_notebook(query, allowed_notebooks)` on line 229
- Classification (`classify_request`) is NOT affected by ACL

### Step 3: Extract `allowed_notebooks` from request metadata in `server.py`

**File**: `src/nlm_proxy/openai/server.py` (after line 287)

- Extract `request.metadata.get("allowed_notebooks")` into `request_allowed_notebooks`
- Handle wildcard: `["*"]` → `None` (all notebooks)
- `null`/missing → `None` (all notebooks, backward compatible)
- Empty list `[]` → passes through as-is (router returns error)

### Step 4: Pass ACL to `router.route()` — streaming path

**File**: `src/nlm_proxy/openai/server.py` (line 309)

- Change `router.route(query)` → `router.route(query, allowed_notebooks=request_allowed_notebooks)`

### Step 5: Pass ACL to `router.route()` — non-streaming path

**File**: `src/nlm_proxy/openai/server.py` (line 336)

- Change `router.route(query)` → `router.route(query, allowed_notebooks=request_allowed_notebooks)`

### Step 6: Verify `types.py` (no changes needed)

**File**: `src/nlm_proxy/openai/types.py` (line 24)

- `metadata: dict | None = None` already exists and accepts `{"allowed_notebooks": [...]}`

### Step 7: Add unit tests

**File to create**: `tests/test_openai_module/test_router_acl.py`

Test cases:
1. `test_select_notebook_no_acl_filter` — all notebooks considered when no ACL
2. `test_select_notebook_with_acl_filter` — only allowed notebooks in LLM prompt
3. `test_select_notebook_acl_filters_all` — returns error when no match
4. `test_select_notebook_empty_acl_list` — empty list returns error
5. `test_route_passes_acl_to_select_notebook` — route() forwards ACL correctly
6. `test_route_llm_task_ignores_acl` — LLM_TASK classification skips ACL

### Step 8: Update documentation

**File**: `docs/smart-routing-architecture.md`

- Add "Per-Request ACL Filtering" section with usage examples, behavior table, and OTEL attributes

---

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| No `metadata` in request | All notebooks (existing behavior) |
| `metadata` without `allowed_notebooks` | All notebooks (existing behavior) |
| `metadata.allowed_notebooks = null` | All notebooks (existing behavior) |
| `metadata.allowed_notebooks = ["*"]` | All notebooks (normalized to null) |
| `metadata.allowed_notebooks = ["nb-1"]` | Only `nb-1` considered |
| `metadata.allowed_notebooks = []` | Returns "No accessible notebooks" |

**Breaking changes**: None. Purely additive.

---

## Critical Files

| File | Action |
|------|--------|
| `src/nlm_proxy/openai/router.py` | Modify: add ACL param + filtering to `select_notebook()` and `route()` |
| `src/nlm_proxy/openai/server.py` | Modify: extract ACL from metadata, pass to router |
| `src/nlm_proxy/openai/types.py` | Verify only (already correct) |
| `tests/test_openai_module/test_router_acl.py` | Create: 6 unit tests |
| `docs/smart-routing-architecture.md` | Modify: add Per-Request ACL section |

---

## Verification

1. Run `uv run pytest tests/test_openai_module/test_router_acl.py` — all 6 tests pass
2. Run `uv run pytest` — no regressions in existing tests
3. Manual curl tests from the spec:
   - Without ACL metadata → all notebooks considered
   - With `allowed_notebooks: ["<valid-id>"]` → only that notebook
   - With `allowed_notebooks: []` → "No accessible notebooks" error
4. Reinstall: `uv cache clean && uv tool install ".[all]" --force`
