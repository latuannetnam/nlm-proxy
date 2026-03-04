# Smart Routing

Smart routing automatically classifies incoming requests and routes them to either NotebookLM (for knowledge queries) or an external LLM (for general tasks).

## Overview

When a client uses `model="knowledge-finder"`, the proxy:
1. Classifies the request using an external LLM (via LangChain `ChatModel`)
2. For knowledge queries: selects the best notebook and queries NotebookLM
3. For general tasks: passes through to the external LLM

## Architecture (post-LangChain refactor)

The routing pipeline uses **AgentCore** as the shared orchestration layer, with a **LangGraph StateGraph** for routing decisions:

```
Request → AgentCore.route() → LangGraph StateGraph
  ├─ classify_node() → NOTEBOOKLM | LLM_TASK
  ├─ select_notebook_node() → notebook_id (if NOTEBOOKLM)
  └─ RoutingDecision
```

## Key Files

| File | Purpose |
|------|---------|
| `core/config.py` | `SmartRoutingSettings`, `AgentSettings` classes |
| `core/llm_client.py` | `LangChainLLMClient`, `create_chat_model()` factory |
| `core/routing_graph.py` | LangGraph `StateGraph` with classify/select nodes |
| `core/agent.py` | `AgentCore` singleton (shared by OpenAI + MCP) |
| `core/notebook_cache.py` | `NotebookCache` with proactive background refresh |
| `openai/prompts/__init__.py` | `load_prompt()` function |
| `openai/prompts/classify_request.txt` | Classification prompt template |
| `openai/prompts/select_notebook.txt` | Notebook selection prompt template |
| `openai/server.py` | `handle_smart_routing()` 4-phase pipeline |

## Configuration Environment Variables

All use prefix `NLM_PROXY_ROUTING_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | External LLM endpoint |
| `LLM_API_KEY` | (required) | API key for external LLM |
| `LLM_MODEL` | `gpt-4o-mini` | Model for classification |
| `ROUTER_MODEL_NAME` | `knowledge-finder` | Model name triggering routing |
| `ALLOWED_NOTEBOOKS` | (empty = all) | Comma-separated notebook IDs |
| `SUMMARY_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `SOURCE_FETCH_CONCURRENCY` | `10` | Max parallel source summary fetches |
| `MAX_SOURCE_TITLES` | `15` | Max source titles in selection prompt |
| `SOURCE_DESCRIPTIONS_ENABLED` | `true` | Include source keywords and summaries |

Agent-specific settings (prefix `NLM_PROXY_AGENT_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LangChain provider: openai, anthropic, ollama |
| `EMBEDDING_PROVIDER` | `huggingface` | huggingface or openai |
| `FALLBACK_ON_ERROR` | `true` | Fall back to NotebookLM on routing error |

## Logging Tags

| Tag | Component |
|-----|-----------|
| `[LLM]` | LangChainLLMClient - external API calls |
| `[ROUTING]` | LangGraph routing graph - classification and selection |
| `[SMART-ROUTER]` | Server - high-level routing decisions |
| `[CACHE]` | NotebookCache - cache operations and source fetching |

## Request Types

- `"notebooklm"` - Knowledge queries routed to NotebookLM
- `"llm_task"` - General tasks passed to external LLM

## Quick Reference

```python
# Routing (via AgentCore -> LangGraph)
agent_core = AgentCore(nlm_client, notebook_cache, response_cache, chat_model)
decision = await agent_core.route(query, options)
# -> RoutingDecision(request_type, notebook_id, reasoning)

# NotebookCache (shared via app.state or AgentCore)
cache = NotebookCache(
    nlm_client,
    ttl_seconds=3600,
    source_fetch_concurrency=10
)
# - Fetches all notebooks and sources at init (blocking)
# - Refreshes in background before TTL expires
```

See `docs/smart-routing-architecture.md` for detailed architecture diagrams.
