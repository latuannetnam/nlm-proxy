# Architecture

```
src/nlm_proxy/
├── __init__.py           # Package version
├── __main__.py           # Module entry point (python -m nlm_proxy)
├── cli.py                # Typer CLI (serve mcp, serve openai, auth)
├── core/
│   ├── __init__.py       # Public exports
│   ├── agent.py          # AgentCore orchestration (shared by OpenAI + MCP)
│   ├── auth.py           # Token management
│   ├── auth_cli.py       # CLI authentication commands
│   ├── client.py         # NotebookLMClient
│   ├── config.py         # Pydantic settings (Shared, MCP, OpenAI, Auth,
│   │                     #   Logging, SmartRouting, Tracing, Cache, Agent)
│   ├── constants.py      # Code mappings
│   ├── exceptions.py     # Custom exceptions
│   ├── llm_client.py     # LangChainLLMClient + create_chat_model() factory
│   ├── logging.py        # Centralized logging setup
│   ├── notebook_cache.py # NotebookCache with proactive background refresh
│   ├── response_cache.py # Three-layer response cache (L1/L2/L3)
│   ├── routing_graph.py  # LangGraph StateGraph (classify + select nodes)
│   └── session.py        # SessionStore (chat_id → conversation_id mapping)
├── mcp/
│   ├── __init__.py       # Lazy imports
│   └── server.py         # FastMCP tools (uses AgentCore for queries)
└── openai/
    ├── __init__.py       # Lazy imports
    ├── notebook_cache.py # Re-export for backward compat (canonical: core/)
    ├── prompts/          # LLM prompt templates
    │   ├── __init__.py   # load_prompt() function
    │   ├── classify_request.txt
    │   └── select_notebook.txt
    ├── server.py         # FastAPI routes (four-phase smart routing pipeline)
    ├── session.py        # Re-export for backward compat (canonical: core/)
    └── types.py          # Pydantic models
```

## Executables

- `nlm-proxy` - Typer CLI (serve mcp, serve openai, auth commands)

## Key Components

| Module | Purpose |
|--------|---------|
| `core/agent.py` | AgentCore — shared orchestration layer (routing, caching, NLM queries) |
| `core/routing_graph.py` | LangGraph StateGraph with classify + select_notebook nodes |
| `core/llm_client.py` | LangChain ChatModel factory + LLM client wrapper |
| `core/notebook_cache.py` | Proactive cache for notebook/source summaries |
| `core/response_cache.py` | Three-layer response cache (exact → embedding → LLM) |
| `core/session.py` | SessionStore — chat_id to conversation_id mapping with TTL |
| `core/client.py` | NotebookLM API client (batchexecute RPCs) |
| `core/auth.py` | Token extraction and management |
| `core/config.py` | Unified settings (CLI > env > .env > defaults) |
| `mcp/server.py` | MCP tool definitions (uses AgentCore for query tools) |
| `openai/server.py` | OpenAI-compatible proxy with four-phase routing pipeline |

## Architecture Overview

```
                    ┌──────────────┐
                    │   AgentCore  │  ← Shared singleton
                    │ (core/agent) │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼──┐  ┌──────▼──────┐  ┌─▼──────────┐
     │  LangGraph │  │ Response    │  │ Notebook   │
     │  Routing   │  │ Cache       │  │ Cache      │
     │  Graph     │  │ (L1/L2/L3)  │  │ (proactive)│
     └────────────┘  └─────────────┘  └────────────┘
              │
     ┌────────▼────────┐
     │  LangChain      │
     │  ChatModel      │
     │  (multi-provider)│
     └─────────────────┘
```

Both `openai/server.py` and `mcp/server.py` share a single `AgentCore` instance for routing, caching, and NLM query delegation. Transport-specific concerns (SSE streaming, MCP progress) are handled by the callers.

## Configuration

Settings loaded via pydantic-settings with precedence:
1. CLI arguments (highest)
2. Environment variables
3. .env files (~/.nlm-proxy/.env, ./.env)
4. Defaults (lowest)

See `configuration.md` for details.
