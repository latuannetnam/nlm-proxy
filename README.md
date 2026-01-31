# NLM Proxy

MCP server and OpenAI-compatible proxy for **NotebookLM** (notebooklm.google.com).

> This project is a fork of [notebooklm-mcp](https://github.com/jacob-bd/notebooklm-mcp) with significant enhancements, including an OpenAI-compatible API proxy that allows any OpenAI client to interact with NotebookLM.

## Key Features

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

```bash
# Using uv (recommended)
uv tool install nlm-proxy

# Using pip
pip install nlm-proxy

# From source
git clone https://github.com/latuannetnam/nlm-proxy.git
cd nlm-proxy
uv tool install .
```

**Python requirement:** >=3.11

## Authentication

Before using, authenticate with NotebookLM:

```bash
# Auto mode: launches Chrome, you log in
notebooklm-mcp-auth

# File mode: manual cookie extraction
notebooklm-mcp-auth --file
```

## Usage

### MCP Server

```bash
# Standard stdio transport (for Claude Code, Cursor, etc.)
notebooklm-mcp

# HTTP transport (for remote access)
notebooklm-mcp --transport http --port 8000

# With debug logging
notebooklm-mcp --debug
```

**Add to Claude Code:**
```bash
claude mcp add --scope user notebooklm-mcp notebooklm-mcp
```

### OpenAI Proxy

```bash
# Start the proxy server
notebooklm-openai --port 8080

# With custom session TTL (1 hour)
notebooklm-openai --port 8080 --session-ttl 3600
```

**Connect with OpenAI Python SDK:**
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

**Connect with Open WebUI:**
1. Set `ENABLE_FORWARD_USER_INFO_HEADERS=true` in Open WebUI
2. Add connection: `http://localhost:8080/v1`
3. Select a notebook as the model

## API Endpoints (OpenAI Proxy)

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List notebooks as models |
| `POST /v1/chat/completions` | Chat with a notebook |
| `GET /health` | Health check |
| `GET /v1/sessions` | List active sessions |
| `GET /v1/sessions/stats` | Session statistics |

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

## Disclaimer

This project uses **internal/undocumented APIs** that may change without notice. Use at your own risk for personal/experimental purposes.

## Limitations

- **Rate limits**: Free tier ~50 queries/day
- **Cookie expiration**: Re-authenticate every few weeks
- **No official support**: API may change anytime

## Credits

This project is based on [notebooklm-mcp](https://github.com/jacob-bd/notebooklm-mcp) by **Jacob Ben-David** ([@jacob-bd](https://github.com/jacob-bd)).

Contributors:
- **Le Anh Tuan** ([@latuannetnam](https://github.com/latuannetnam)) - OpenAI-compatible proxy, HTTP transport, session management, debug logging
- **David Szabo-Pele** ([@davidszp](https://github.com/davidszp)) - `source_get_content` tool, Linux auth fixes

## License

MIT License
