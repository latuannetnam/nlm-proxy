# Architecture

```
src/nlm_proxy/
├── __init__.py       # Package version
├── __main__.py       # Module entry point (python -m nlm_proxy)
├── cli.py            # Unified CLI entry point
├── core/
│   ├── __init__.py   # Public exports
│   ├── auth.py       # Token management
│   ├── auth_cli.py   # CLI authentication commands
│   ├── client.py     # NotebookLMClient
│   ├── config.py     # Pydantic settings configuration
│   ├── constants.py  # Code mappings
│   ├── exceptions.py # Custom exceptions
│   └── logging.py    # Centralized logging setup
├── mcp/
│   ├── __init__.py   # Lazy imports
│   └── server.py     # FastMCP tools
└── openai/
    ├── __init__.py   # Lazy imports
    ├── server.py     # FastAPI routes
    ├── session.py    # Session management
    └── types.py      # Pydantic models
```

## Executables

- `nlm-proxy` - Unified CLI (serve mcp, serve openai, auth commands)

## Key Components

| Module | Purpose |
|--------|---------|
| `core/client.py` | NotebookLM API client |
| `core/auth.py` | Token extraction and management |
| `core/config.py` | Environment-based settings |
| `mcp/server.py` | MCP tool definitions |
| `openai/server.py` | OpenAI-compatible proxy |
