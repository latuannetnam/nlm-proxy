# Missing Test Cases — LangChain Refactor Parity

> **Date**: 2026-03-04
> **Context**: After completing the LangChain/LangGraph refactor (design: `2026-03-03-langchain-refactor-design.md`), this plan covers 13 categories of missing test coverage to ensure behavior parity.

## Overview

49 new tests across 9 files (5 new, 4 modified) covering:
- Four-phase pipeline integration (`handle_smart_routing`)
- Streaming (LLM_TASK + NLM + cache store)
- Direct notebook path (cache hit/miss, session)
- Cached response helpers (SSE + JSON format)
- AgentCore session helpers & edge cases
- LLM client message conversion & provider factory
- SessionStore (no tests existed)

## Test File Plan

### NEW: `tests/test_openai_module/test_smart_routing_pipeline.py` (11 tests)
Phase 0/2 cache hits (streaming + non-streaming), bypass_cache, ACL wildcard/empty, chat_id extraction, error cases (503, 400).

### NEW: `tests/test_openai_module/test_streaming_smart.py` (5 tests)
LLM_TASK streaming SSE format, NLM cache store after stream, include_thinking filter, conversation_id extraction, stream end marker.

### NEW: `tests/test_openai_module/test_direct_notebook.py` (5 tests)
Direct path cache hit/miss (streaming + non-streaming), session lookup/save.

### NEW: `tests/test_openai_module/test_cached_response_helpers.py` (5 tests)
`_stream_cached_response` SSE sequence, thinking chunk presence/absence, system_fingerprint format, X-Cache-Status header, JSON body content.

### NEW: `tests/core/test_session_store.py` (7 tests)
set/get, TTL expiration, delete, list_all, cleanup_expired, get_stats.

### MODIFY: `tests/core/test_agent.py` (+9 tests)
Session helpers (get/save conversation_id with None/empty edge cases), route without response_cache, fallback with empty notebooks, fallback with ACL.

### MODIFY: `tests/core/test_llm_client.py` (+6 tests)
`_convert_messages` for system/assistant/mixed/Pydantic objects, `create_chat_model` for Anthropic/Ollama providers.

### MODIFY: `tests/core/test_routing_graph.py` (+1 test)
`select_notebook_node` with source descriptions enabled.

### MODIFY: `tests/conftest.py`
Add `session_store` to reset fixture.

## Verification

```bash
uv run pytest --tb=short  # All tests pass, no regressions
```
