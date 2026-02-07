# NLM Proxy

**NotebookLM client library** with MCP server and OpenAI-compatible proxy for **NotebookLM** (notebooklm.google.com).

> This project is a fork of [notebooklm-mcp](https://github.com/jacob-bd/notebooklm-mcp) with significant enhancements, including an OpenAI-compatible API proxy and modular architecture.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [OpenAI Proxy Usage](#openai-proxy-usage)
- [API Endpoints](#api-endpoints-openai-proxy)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Tracing Infrastructure](#tracing-infrastructure)
- [Architecture](#architecture)
- [MCP Tools](#mcp-tools)
- [Development](#development)
- [Disclaimer](#disclaimer)
- [Limitations](#limitations)
- [Credits](#credits)
- [License](#license)

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

### Smart Request Routing
Automatically route requests to the best backend using AI-powered classification:

| Feature | Description |
|---------|-------------|
| **Auto-Classification** | LLM classifies queries as knowledge lookups or general tasks |
| **Notebook Selection** | Automatically selects the most relevant notebook for knowledge queries |
| **Source-Level Routing** | Routes based on specific document names, URLs, and source types |
| **LLM Passthrough** | General tasks routed to external LLM (e.g., GPT-4o-mini) |
| **Cached Summaries** | Notebook and source info cached with configurable TTL |

### OpenTelemetry Tracing
Monitor and analyze routing decisions with distributed tracing:

| Feature | Description |
|---------|-------------|
| **Request Tracing** | Full visibility into routing decisions and timing |
| **Response Capture** | User queries and LLM/NotebookLM responses stored in traces |
| **Span Hierarchy** | Parent-child spans: handle_request → route → classify → select_notebook |
| **Rich Attributes** | Query text, response content, classification results, notebook selection |
| **Auto-Instrumentation** | FastAPI and httpx automatically traced |
| **ClickHouse Storage** | 90-day retention with efficient querying |
| **Grafana Dashboard** | Pre-built analytics dashboard with response previews |

See [Tracing Infrastructure](#tracing-infrastructure) below for installation instructions (Docker and Native Ubuntu).

For advanced configuration, querying, and troubleshooting, see the [Tracing Guide](docs/TRACING.md).

**Configuration:**
```bash
# Add to ~/.nlm-proxy/.env
NLM_PROXY_ROUTING_LLM_API_KEY=sk-your-openai-key
NLM_PROXY_ROUTING_LLM_MODEL=gpt-4o-mini
NLM_PROXY_ROUTING_SOURCE_FETCH_CONCURRENCY=10  # parallel source fetches
NLM_PROXY_ROUTING_MAX_SOURCE_TITLES=15         # titles in selection prompt
```

**Usage:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="your-key")

# Use "knowledge-finder" model for automatic routing
response = client.chat.completions.create(
    model="knowledge-finder",
    messages=[{"role": "user", "content": "What does the Attention Is All You Need paper say?"}],
    stream=True
)
# Routes to notebook containing "Attention Is All You Need.pdf" source
```

See [Smart Routing Architecture](docs/smart-routing-architecture.md) for details.

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

# Install from source (for development with all extras)
git clone https://github.com/latuannetnam/nlm-proxy.git
cd nlm-proxy
uv pip install -e ".[all]"

# Or install as global tool
uv tool install .
```

**Python requirement:** >=3.11

## Quick Start

### 1. Authentication

**Three methods available:**

#### Method 1: Automated Chrome Extraction (Recommended)

Automatically launches Chrome, extracts tokens, and saves them:

```bash
# Automatic extraction (launches Chrome)
nlm-proxy auth extract

# Or with uv run (from source)
uv run nlm-proxy auth extract

# First time: Log in to Google in the Chrome window
# Next times: Uses saved profile (instant!)
```

**How it works:**
- Launches Chrome with remote debugging
- Navigates to NotebookLM
- Waits for you to log in (first time only)
- Extracts cookies, CSRF token, and session ID automatically
- Saves profile at `~/.notebooklm-mcp/chrome-profile` for future use

#### Method 2: File-Based Import (Most Reliable)

Manually copy cookies from Chrome DevTools and import from file:

```bash
# Shows step-by-step instructions
nlm-proxy auth extract --file

# Direct file import
nlm-proxy auth extract --file ~/cookies.txt
```

**Steps:**
1. Open https://notebooklm.google.com in Chrome
2. Press F12 to open DevTools
3. Go to Network tab, filter "batchexecute"
4. Click on a request, find "cookie:" in Request Headers
5. Copy the cookie value and save to a file
6. Run the command above with your file path

#### Method 3: Environment Variables

```bash
export NOTEBOOKLM_COOKIES="SID=xxx; HSID=xxx; SSID=xxx; ..."
```

**Test your authentication:**

```bash
nlm-proxy auth test
# Or: uv run nlm-proxy auth test
```

**Additional Options:**

```bash
# Custom Chrome DevTools port
nlm-proxy auth extract --port 9223

# Use existing Chrome instance (must have --remote-debugging-port=9222)
nlm-proxy auth extract --no-auto-launch
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

# From source (using uv run)
uv run nlm-proxy serve mcp
# Or using python module
uv run python -m nlm_proxy serve mcp

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

**Required:** Set an API key (can be any value, used for client authentication):

```bash
export NLM_PROXY_OPENAI_API_KEY="your-secret-key"
```

Or add to `~/.nlm-proxy/.env`:
```bash
NLM_PROXY_OPENAI_API_KEY=your-secret-key
```

**Start the server:**

```bash
# Start the proxy server
nlm-proxy serve openai --port 8080

# From source (using uv run)
uv run nlm-proxy serve openai --port 8080
# Or using python module
uv run python -m nlm_proxy serve openai --port 8080

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

## Configuration

Configure via CLI arguments, environment variables, or `.env` files.

**Precedence:** CLI args > Environment variables > .env files > Defaults

### Environment Variables

Create `~/.nlm-proxy/.env` or `.env` in project root:

```bash
# Shared Settings
NLM_PROXY_DEBUG=false
NLM_PROXY_AUTH_DIR=~/.nlm-proxy

# MCP Server
NLM_PROXY_MCP_PORT=8000
NLM_PROXY_MCP_TRANSPORT=stdio

# OpenAI Proxy
NLM_PROXY_OPENAI_HOST=0.0.0.0
NLM_PROXY_OPENAI_PORT=8080
NLM_PROXY_OPENAI_SESSION_TTL=86400

# Authentication
NLM_PROXY_AUTH_CHROME_PORT=9222
NLM_PROXY_AUTH_AUTO_LAUNCH=true

# Logging
NLM_PROXY_LOG_LEVEL=INFO
NLM_PROXY_LOG_FILE=~/.nlm-proxy/logs/nlm-proxy.log
NLM_PROXY_LOG_MAX_SIZE=10485760
NLM_PROXY_LOG_BACKUP_COUNT=5
```

### Usage Examples

```bash
# CLI args override environment variables
export NLM_PROXY_OPENAI_PORT=9000
nlm-proxy serve openai --port 8080  # Uses 8080, not 9000

# Enable debug logging via CLI flag
nlm-proxy serve mcp --debug

# Or via environment variable
export NLM_PROXY_DEBUG=true
nlm-proxy serve mcp
```

See `.env.example` in the project root for all available options.

## Tracing Infrastructure

NLM Proxy uses **OpenTelemetry**, **ClickHouse**, and **Grafana** for comprehensive request tracing.

### 1. Docker Setup (Recommended)

The easiest way to get started is using the provided Docker Compose configuration.

```bash
# Start the infrastructure
docker compose -f docker-compose.otel.yml up -d

# Enable tracing and start proxy
export NLM_PROXY_OTEL_ENABLED=true
nlm-proxy serve openai --port 8080
```

Access Grafana at `http://localhost:3000` (default login: `admin` / `admin`).

### 2. Native Ubuntu 24.04 Setup

For environments where Docker is not available, you can install the components natively.

**Step 1: Install ClickHouse**
```bash
# Install dependencies
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg

# Add GPG key and repository (safe for Ubuntu 24.04)
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
ARCH=$(dpkg --print-architecture)
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb stable main" | sudo tee /etc/apt/sources.list.d/clickhouse.list

# Install and start
sudo apt-get update
sudo apt-get install -y clickhouse-server clickhouse-client
sudo systemctl enable clickhouse-server
sudo systemctl start clickhouse-server

# Initialize database
clickhouse-client < docker/clickhouse/init.sql
```

**Step 2: Install OpenTelemetry Collector**
```bash
# Download Contrib release (required for ClickHouse exporter)
# Check https://github.com/open-telemetry/opentelemetry-collector-releases/releases for latest
wget https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.120.0/otelcol-contrib_0.120.0_linux_amd64.deb
sudo dpkg -i otelcol-contrib_0.120.0_linux_amd64.deb

# Configure
sudo cp docker/otel/config.yaml /etc/otelcol-contrib/config.yaml
# Point to localhost instead of docker hostname
sudo sed -i 's/clickhouse:9000/127.0.0.1:9000/g' /etc/otelcol-contrib/config.yaml

sudo systemctl restart otelcol-contrib
```

**Step 3: Install Grafana**
```bash
# Install dependencies
sudo apt-get install -y apt-transport-https software-properties-common wget

# Add GPG key and repository
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list

# Install Grafana
sudo apt-get update
sudo apt-get install grafana

# Install ClickHouse Plugin
sudo grafana-cli plugins install grafana-clickhouse-datasource

# Configure Provisioning
sudo cp -r docker/grafana/provisioning/* /etc/grafana/provisioning/
# Point to localhost instead of docker hostname
sudo sed -i 's/host: clickhouse/host: 127.0.0.1/g' /etc/grafana/provisioning/datasources/clickhouse.yml

# Start service
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
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
uv tool install .[all] --reinstall

# Reinstall after code changes
uv cache clean && uv tool install --force .[all]

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
