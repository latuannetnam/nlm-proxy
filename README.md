# NLM Proxy

**NotebookLM client library** with MCP server and OpenAI-compatible proxy for **NotebookLM** (notebooklm.google.com).

> This project is a fork of [notebooklm-mcp](https://github.com/jacob-bd/notebooklm-mcp) with significant enhancements, including an OpenAI-compatible API proxy and modular architecture.

## Key Features

### Core Library
Standalone Python library for programmatic NotebookLM access:
- Zero framework dependencies
- Reusable in any Python project
- Clean separation of concerns

### MCP Server
Full programmatic access to NotebookLM through the Model Context Protocol:

| Feature | Description |
|---------|-------------|
| **Notebook Management** | Create, list, rename, delete notebooks |
| **Source Management** | Add URLs, YouTube videos, text, Google Drive documents |
| **AI Queries** | Ask questions and get AI-powered answers from your sources |
| **Research** | Web and Drive research to discover and import new sources |
| **Content Generation** | Audio podcasts, video overviews, infographics, slide decks, reports, flashcards, quizzes, mind maps |
| **Drive Sync** | Check freshness and sync stale Google Drive sources |

### OpenAI-Compatible Proxy
Connect **any OpenAI client** (Open WebUI, Python SDK, etc.) to NotebookLM:

| Feature | Description |
|---------|-------------|
| **Standard API** | `/v1/chat/completions`, `/v1/models` endpoints |
| **Streaming** | Real-time streaming responses |
| **Session Persistence** | Maintains conversation context across queries |
| **Multi-notebook** | Each notebook appears as a separate model |

## Installation

### Basic Installation

```bash
# Using uv (recommended)
uv tool install nlm-proxy

# Using pip
pip install nlm-proxy
```

### Installation with Extras

```bash
# Install with MCP server support
pip install nlm-proxy[mcp]

# Install with OpenAI proxy support
pip install nlm-proxy[openai]

# Install everything
pip install nlm-proxy[all]

# Install from source
git clone https://github.com/latuannetnam/nlm-proxy.git
cd nlm-proxy
uv tool install .
```

**Python requirement:** >=3.11

## Quick Start

### 1. Authentication

Extract authentication tokens from your browser:

```bash
# If installed with uv tool
nlm-proxy auth extract

# Or with uv run (from source)
uv run nlm-proxy auth extract
```

Test your authentication:

```bash
nlm-proxy auth test
# Or: uv run nlm-proxy auth test
```

### 2. Use as Python Library

```python
from nlm_proxy.core import NotebookLMClient, load_cached_tokens

# Load authentication
tokens = load_cached_tokens()
client = NotebookLMClient(tokens)

# List notebooks
notebooks = client.list_notebooks()
for nb in notebooks:
    print(f"{nb.title} ({nb.id})")

# Query a notebook
response = client.query_notebook(
    notebook_id="<notebook-id>",
    query="Summarize the main points"
)
print(response["answer"])
```

### 3. Run MCP Server

```bash
# Standard stdio transport (for Claude Code, Cursor, etc.)
nlm-proxy serve mcp

# Or with uv run (from source)
uv run nlm-proxy serve mcp

# HTTP transport (for remote access)
nlm-proxy serve mcp --transport http --port 8000

# With debug logging
nlm-proxy serve mcp --debug
```

**Add to Claude Code:**
```bash
# If installed with uv tool
claude mcp add --scope user notebooklm-mcp nlm-proxy serve mcp

# If running from source
claude mcp add --scope user notebooklm-mcp uv run nlm-proxy serve mcp
```

### 4. Run OpenAI Proxy

```bash
# Start the proxy server
nlm-proxy serve openai --port 8080

# Or with uv run (from source)
uv run nlm-proxy serve openai --port 8080

# With custom session TTL (1 hour)
nlm-proxy serve openai --port 8080 --session-ttl 3600

# With custom host
nlm-proxy serve openai --host 127.0.0.1 --port 8000
```

## OpenAI Proxy Usage

### Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

# List notebooks as models
for model in client.models.list():
    print(f"{model.id}: {model.name}")

# Chat with a notebook
response = client.chat.completions.create(
    model="<notebook-uuid>",
    messages=[{"role": "user", "content": "Summarize the key points"}],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Open WebUI

1. Set `ENABLE_FORWARD_USER_INFO_HEADERS=true` in Open WebUI
2. Add connection: `http://localhost:8080/v1`
3. Select a notebook as the model

### Custom Parameters

```python
response = client.chat.completions.create(
    model="notebook-id",
    messages=[...],
    extra_body={
        "conversation_id": "prev-conv-id",  # For multi-turn
        "include_thinking": True  # Include reasoning steps
    }
)
```

## API Endpoints (OpenAI Proxy)

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List notebooks as models |
| `POST /v1/chat/completions` | Chat with a notebook (streaming/non-streaming) |
| `POST /v1/embeddings` | Returns 501 (not supported) |
| `GET /health` | Health check |
| `GET /v1/sessions` | List active sessions |
| `DELETE /v1/sessions/{chat_id}` | Delete specific session |
| `GET /v1/sessions/stats` | Session statistics |

## CLI Reference

```bash
# Authentication
nlm-proxy auth extract     # Extract tokens from browser
nlm-proxy auth test        # Test current tokens

# MCP Server
nlm-proxy serve mcp [OPTIONS]
  --debug                  # Enable debug logging
  --transport {stdio,http} # Transport type (default: stdio)
  --port PORT              # Port for HTTP transport (default: 8000)

# OpenAI Proxy
nlm-proxy serve openai [OPTIONS]
  --host HOST              # Host to bind (default: 0.0.0.0)
  --port PORT              # Port to listen (default: 8080)
  --session-ttl SECONDS    # Session TTL (default: 86400 = 24h)
```

## Architecture

```
src/nlm_proxy/
├── __init__.py         # Package version
├── cli.py              # Unified CLI entry point
├── core/               # Standalone library (no framework deps)
│   ├── __init__.py     # Public exports
│   ├── client.py       # NotebookLMClient
│   ├── auth.py         # Token management
│   ├── constants.py    # Code mappings
│   └── exceptions.py   # Custom exceptions
├── mcp/                # MCP server (optional)
│   ├── __init__.py     # Lazy imports
│   └── server.py       # FastMCP tools
└── openai/             # OpenAI proxy (optional)
    ├── __init__.py     # Lazy imports
    ├── server.py       # FastAPI routes
    ├── session.py      # Session management
    └── types.py        # Pydantic models
```

## MCP Tools

<details>
<summary>Click to expand full tool list (31 tools)</summary>

| Tool | Description |
|------|-------------|
| `notebook_list` | List all notebooks |
| `notebook_create` | Create a new notebook |
| `notebook_get` | Get notebook details with sources |
| `notebook_describe` | Get AI-generated summary |
| `notebook_rename` | Rename a notebook |
| `notebook_delete` | Delete a notebook |
| `notebook_add_url` | Add URL/YouTube as source |
| `notebook_add_text` | Add text as source |
| `notebook_add_drive` | Add Google Drive document |
| `notebook_query` | Ask questions, get AI answers |
| `source_describe` | Get AI summary of a source |
| `source_get_content` | Get raw source content |
| `source_list_drive` | List sources with freshness status |
| `source_sync_drive` | Sync stale Drive sources |
| `source_delete` | Delete a source |
| `chat_configure` | Configure chat style/length |
| `research_start` | Start web/Drive research |
| `research_status` | Check research progress |
| `research_import` | Import discovered sources |
| `audio_overview_create` | Generate audio podcasts |
| `video_overview_create` | Generate video overviews |
| `infographic_create` | Generate infographics |
| `slide_deck_create` | Generate slide decks |
| `report_create` | Generate reports |
| `flashcards_create` | Generate flashcards |
| `quiz_create` | Generate quizzes |
| `data_table_create` | Generate data tables |
| `mind_map_create` | Generate mind maps |
| `studio_status` | Check generation status |
| `studio_delete` | Delete studio artifacts |
| `save_auth_tokens` | Save authentication tokens |

</details>

## Development

```bash
# Install dependencies
uv tool install .

# Reinstall after code changes
uv cache clean && uv tool install --force .

# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_file.py::test_function -v
```

## Disclaimer

This project uses **internal/undocumented APIs** that may change without notice. Use at your own risk for personal/experimental purposes.

## Limitations

- **Rate limits**: Free tier ~50 queries/day
- **Cookie expiration**: Re-authenticate every few weeks
- **No official support**: API may change anytime

## Credits

This project is based on [notebooklm-mcp](https://github.com/jacob-bd/notebooklm-mcp) by **Jacob Ben-David** ([@jacob-bd](https://github.com/jacob-bd)).

Contributors:
- **Le Anh Tuan** ([@latuannetnam](https://github.com/latuannetnam)) - OpenAI-compatible proxy, HTTP transport, session management, refactoring
- **David Szabo-Pele** ([@davidszp](https://github.com/davidszp)) - `source_get_content` tool, Linux auth fixes

## License

MIT License
