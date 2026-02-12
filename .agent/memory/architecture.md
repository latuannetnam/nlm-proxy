# Architecture

```
src/nlm_proxy/
├── __init__.py       # Package version
├── __main__.py       # Module entry point (python -m nlm_proxy)
├── cli.py            # Typer CLI (serve mcp, serve openai, auth)
├── core/
│   ├── __init__.py   # Public exports
│   ├── auth.py       # Token management
│   ├── auth_cli.py   # CLI authentication commands
│   ├── client.py     # NotebookLMClient
│   ├── config.py     # Pydantic settings (Shared, MCP, OpenAI, Auth, Logging)
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

- `nlm-proxy` - Typer CLI (serve mcp, serve openai, auth commands)

## Key Components

| Module | Purpose |
|--------|---------|
| `core/client.py` | NotebookLM API client |
| `core/auth.py` | Token extraction and management |
| `core/config.py` | Unified settings (CLI > env > .env > defaults) |
| `mcp/server.py` | MCP tool definitions |
| `openai/server.py` | OpenAI-compatible proxy |

## Configuration

Settings loaded via pydantic-settings with precedence:
1. CLI arguments (highest)
2. Environment variables
3. .env files (~/.nlm-proxy/.env, ./.env)
4. Defaults (lowest)

See `configuration.md` for details.
