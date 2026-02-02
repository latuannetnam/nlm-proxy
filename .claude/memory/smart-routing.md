# Smart Routing

Smart routing automatically classifies incoming requests and routes them to either NotebookLM (for knowledge queries) or an external LLM (for general tasks).

## Overview

When a client uses `model="smart-router"`, the proxy:
1. Classifies the request using an external LLM
2. For knowledge queries: selects the best notebook and queries NotebookLM
3. For general tasks: passes through to the external LLM

## Key Files

| File | Purpose |
|------|---------|
| `core/config.py` | `SmartRoutingSettings` class with env var bindings |
| `core/llm_client.py` | `ExternalLLMClient` using OpenAI SDK |
| `openai/notebook_cache.py` | `NotebookCache` with TTL expiration |
| `openai/router.py` | `SmartRouter`, `RequestType`, `RoutingDecision` |
| `openai/prompts/__init__.py` | `load_prompt()` function |
| `openai/prompts/classify_request.txt` | Classification prompt template |
| `openai/prompts/select_notebook.txt` | Notebook selection prompt template |
| `openai/server.py` | `handle_smart_routing()`, `stream_smart_response()` |

## Configuration Environment Variables

All use prefix `NLM_PROXY_ROUTING_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | External LLM endpoint |
| `LLM_API_KEY` | (required) | API key for external LLM |
| `LLM_MODEL` | `gpt-4o-mini` | Model for classification |
| `ROUTER_MODEL_NAME` | `smart-router` | Model name triggering routing |
| `ALLOWED_NOTEBOOKS` | (empty = all) | Comma-separated notebook IDs |
| `SUMMARY_CACHE_TTL` | `3600` | Cache TTL in seconds |

## Logging Tags

| Tag | Component |
|-----|-----------|
| `[LLM]` | ExternalLLMClient - external API calls |
| `[ROUTER]` | SmartRouter - classification and selection |
| `[SMART-ROUTER]` | Server - high-level routing decisions |

## Request Types

- `RequestType.NOTEBOOKLM` - Knowledge queries routed to NotebookLM
- `RequestType.LLM_TASK` - General tasks passed to external LLM

## Quick Reference

```python
# Classification flow
router.route(query) -> RoutingDecision
  -> classify_request(query) -> RequestType
  -> if NOTEBOOKLM: select_notebook(query) -> notebook_id
```

See `docs/smart-routing-architecture.md` for detailed architecture diagrams.
