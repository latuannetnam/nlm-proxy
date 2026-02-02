# Enhanced NotebookCache with Source-Level Information

## Overview

Enhance the OpenAI Proxy smart router by adding source-level information to `NotebookCache`, enabling more accurate notebook selection based on individual source titles and types.

## Problem Statement

Current limitations:
1. **Ambiguous routing** - Overlapping notebook topics cause misrouting
2. **Missing context** - Notebook summaries don't reflect actual source content
3. **No source-level routing** - Can't route based on specific document names

## Design Decisions

| Decision | Choice |
|----------|--------|
| Fetching Strategy | Eager parallel at startup |
| Selection Approach | Single-stage LLM with source titles/types |
| Concurrency | Configurable via env var (default: 10) |
| Refresh | Full refresh with notebooks at 80% TTL |

## Implementation Plan

### Step 1: Add SourceInfo Dataclass

**File:** `src/nlm_proxy/openai/notebook_cache.py`

```python
@dataclass
class SourceInfo:
    id: str
    title: str
    source_type: str      # "pdf", "url", "text", "gdoc", etc.
    summary: str          # Stored but not passed to selection LLM
    keywords: list[str]
```

### Step 2: Update NotebookInfo Dataclass

**File:** `src/nlm_proxy/openai/notebook_cache.py`

Add `sources: list[SourceInfo]` field and computed properties:
- `source_count` - Total number of sources
- `source_types` - Dict counting sources by type
- `source_titles` - List of source title strings

### Step 3: Implement Parallel Source Fetching

**File:** `src/nlm_proxy/openai/notebook_cache.py`

Modify `_fetch_all_summaries()`:
1. Fetch notebook list
2. For each notebook in parallel (with semaphore):
   - Fetch notebook summary
   - Fetch notebook details (contains source list)
   - For each source in parallel: fetch source summary via `source_describe`
3. Build `NotebookInfo` with populated `sources` list

Use `asyncio.Semaphore` with configurable concurrency (`NLM_SOURCE_FETCH_CONCURRENCY`, default 10).

### Step 4: Update NotebookCache.set() Method

Modify the `set()` method to accept sources parameter and store in cache.

### Step 5: Update Router Selection Logic

**File:** `src/nlm_proxy/openai/router.py`

Modify `select_notebook()` to include in `notebooks_info`:
```python
{
    "id": nb.id,
    "title": nb.title,
    "summary": nb.summary[:500],
    "topics": nb.topics[:5],
    "source_count": nb.source_count,
    "source_types": nb.source_types,
    "source_titles": nb.source_titles[:15]  # Configurable limit
}
```

### Step 6: Update Selection Prompt

**File:** `src/nlm_proxy/openai/prompts/select_notebook.txt`

Add instructions for the LLM to consider:
- Source titles that indicate relevant content
- Source types that match query intent
- Specific document/URL names mentioned in query

### Step 7: Add Configuration

**File:** `src/nlm_proxy/openai/config.py` (or existing config location)

New environment variables:
- `NLM_SOURCE_FETCH_CONCURRENCY` (default: 10)
- `NLM_MAX_SOURCE_TITLES` (default: 15)

### Step 8: Error Handling

Implement graceful degradation:
- If source summary fetch fails, keep source with title only
- Log warnings but don't fail entire notebook fetch
- Truncate long source titles to ~100 chars

### Step 9: Update Claude Memory Documentation

**File:** `.claude/memory/configuration.md`

Add new environment variables section:
- `NLM_SOURCE_FETCH_CONCURRENCY` - Controls parallel source fetching (default: 10)
- `NLM_MAX_SOURCE_TITLES` - Max source titles in selection prompt (default: 15)

**File:** `.claude/memory/smart-routing.md`

Update architecture documentation:
- Document `SourceInfo` dataclass and its role
- Explain enhanced `NotebookInfo` with sources list
- Describe parallel fetching strategy
- Document source-aware selection logic

### Step 10: Update README

**File:** `README.md`

Add/update sections:
- Smart routing now uses source-level information
- New configuration options for source fetching
- Improved notebook selection accuracy

### Step 11: Update Smart Routing Architecture Docs

**File:** `docs/` (relevant smart routing docs)

- Add architecture diagram showing source data flow
- Document the selection algorithm with source awareness
- Add troubleshooting section for source fetching issues

## Files to Modify

### Code Changes

| File | Changes |
|------|---------|
| `src/nlm_proxy/openai/notebook_cache.py` | Add `SourceInfo`, update `NotebookInfo`, parallel fetching |
| `src/nlm_proxy/openai/router.py` | Include source info in selection prompt |
| `src/nlm_proxy/openai/prompts/select_notebook.txt` | Update prompt template |

### Documentation Updates

| File | Changes |
|------|---------|
| `.claude/memory/configuration.md` | Document new env vars (`NLM_SOURCE_FETCH_CONCURRENCY`, `NLM_MAX_SOURCE_TITLES`) |
| `.claude/memory/smart-routing.md` | Update architecture with source-aware routing details |
| `README.md` | Add smart routing enhancements and new config options |

## Verification

1. **Unit Tests:**
   - Test `SourceInfo` and updated `NotebookInfo` dataclasses
   - Test parallel fetching with mock API responses
   - Test graceful degradation when source fetch fails

2. **Integration Tests:**
   - Start server with multiple notebooks containing various sources
   - Verify startup logs show source fetching progress
   - Test routing queries that should match specific source titles

3. **Manual Testing:**
   ```bash
   # Start server with debug logging
   nlm-proxy serve openai --port 8080 --debug

   # Test routing with source-specific queries
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "notebooklm", "messages": [{"role": "user", "content": "What does the Attention Is All You Need paper say about transformers?"}]}'
   ```

4. **Verify improved routing:**
   - Query mentioning specific PDF name → routes to correct notebook
   - Query about URL content → routes to notebook containing that URL
   - Ambiguous topic query → better differentiation between similar notebooks
