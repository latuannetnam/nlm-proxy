# NotebookLM MCP Tool for Open WebUI

Query NotebookLM notebooks directly from Open WebUI with **real-time streaming** of AI thinking steps and answers via the Model Context Protocol (MCP).

## ✨ Features

- 🔄 **Real-time Streaming**: Watch AI thinking steps appear as they happen
- 💬 **Conversation Context**: Support for follow-up questions
- 📚 **Notebook Management**: List, view, and query your NotebookLM notebooks
- 🎯 **Source Filtering**: Query specific sources within a notebook
- 🔍 **Health Monitoring**: Built-in health check for MCP server connectivity

## 🚀 Quick Start

### 1. Start the MCP Server

First, ensure the NotebookLM MCP server is running with HTTP transport:

```bash
# From the notebooklm-mcp repository
uv run notebooklm-mcp --transport http --port 9888
```

Or with debug logging:

```bash
uv run notebooklm-mcp --transport http --port 9888 --debug
```

### 2. Install Tool in Open WebUI

1. Open Open WebUI in your browser
2. Navigate to **Workspace** → **Tools** → **+ Create New Tool**
3. Copy and paste the contents of `notebooklm_mcp_tool.py`
4. Click **Save**

### 3. Configure Tool Settings (Valves)

Click the **⚙️ gear icon** on the tool and configure:

- **MCP Server URL**: `http://localhost:9888` (default)
- **Timeout**: `120.0` seconds (adjust for longer queries)
- **Enable Debug**: `false` (set to `true` for troubleshooting)

### 4. Enable Default Function Calling Mode

**⚠️ CRITICAL for Streaming:**

For full real-time streaming of thinking steps, you must use **Default** function calling mode:

**Administrator Setup:**
1. Go to **Admin Panel** → **Settings** → **Models**
2. Select your model → **Advanced Parameters**
3. Set **Function Calling** to **"Default"**

**Per-Chat Override:**
- In chat, click **Chat Controls** → **Advanced Params**
- Set `function_calling` to `"default"`

> **Why?** Native (agentic) mode has limited event emitter support and will overwrite streaming content. See [Open WebUI Documentation](https://docs.openwebui.com/features/plugin/tools/development/#event-emitters) for details.

### 5. Use the Tool

Enable the tool in your chat (click the **+** button) and start querying:

```
List my notebooks
```

```
Query notebook abc-123-def: What are the main findings in the research papers?
```

```
Get info about notebook abc-123-def
```

## 📖 Usage Examples

### List All Notebooks

```
@notebooklm List my notebooks
```

Returns a markdown table with:
- Notebook titles
- Notebook IDs (for querying)
- Source counts

### Query a Notebook

```
@notebooklm Query notebook [notebook-id]: What is the main topic discussed?
```

Features:
- Real-time thinking steps (in Default mode)
- Progressive answer delivery
- Returns conversation ID for follow-ups

### Follow-up Questions

```
@notebooklm Query notebook [notebook-id] with conversation [conversation-id]: Can you elaborate on that?
```

The conversation ID is included in the previous response.

### Get Notebook Details

```
@notebooklm Get info about notebook [notebook-id]
```

Shows:
- Notebook title
- All sources with IDs and types
- Source count

### Query Specific Sources

```
@notebooklm Query notebook [notebook-id] using sources ["source-id-1", "source-id-2"]: What do these sources say about X?
```

### Health Check

```
@notebooklm Check server health
```

Verifies:
- MCP server connectivity
- Session initialization
- Configuration settings

## 🎭 Event Types & Modes

### Default Mode (Recommended)

**Full streaming support:**
- ✅ Status updates (progress bar)
- ✅ Thinking steps (real-time in chat)
- ✅ Answer streaming (progressive delivery)
- ✅ Error notifications
- ✅ Conversation context

**Example output:**
```
🔗 Connecting to MCP server...
📝 Sending query to NotebookLM...
💭 Thinking: Analyzing sources...
💭 Thinking: Cross-referencing findings...
💭 Thinking: Synthesizing answer...
📝 Answer:
[Final answer appears here]
✅ Complete - 8 updates, 3 thinking steps
```

### Native Mode (Limited)

**What works:**
- ✅ Status updates
- ✅ Final answer
- ✅ Error handling

**What doesn't work:**
- ❌ Real-time thinking steps (get overwritten)
- ❌ Progressive content streaming
- ❌ Message events

> The tool will warn you if Native mode is detected.

## 🔧 Troubleshooting

### "Cannot connect to MCP server"

**Solutions:**
1. Check if server is running:
   ```bash
   curl http://localhost:9888/health
   ```
2. Verify server URL in tool Valves
3. Check firewall settings
4. Try restarting the MCP server

### "No thinking steps appearing"

**Solutions:**
1. Verify Function Calling mode is set to **"Default"**
2. Check Admin Panel → Settings → Models → Function Calling
3. Try setting it per-chat via Chat Controls → Advanced Params

### "Tool not found" or "Session error"

**Solutions:**
1. Run health check: `@notebooklm Check server health`
2. Check MCP server logs for errors
3. Restart the MCP server
4. Clear browser cache and reload Open WebUI

### "Authentication failed"

**Solutions:**
1. Authenticate with the MCP server:
   ```bash
   notebooklm-mcp-auth
   ```
2. Or save tokens manually via the server's `save_auth_tokens` tool
3. Verify cookies are still valid at https://notebooklm.google.com

## 🏗️ Architecture

```
┌─────────────────┐
│   Open WebUI    │
│  (Browser UI)   │
└────────┬────────┘
         │ Event Emitter API
         │ (__event_emitter__)
         ▼
┌─────────────────┐
│  Python Tool    │
│ MCPClientAdapter│
└────────┬────────┘
         │ HTTP + SSE
         │ (Server-Sent Events)
         ▼
┌─────────────────┐
│   MCP Server    │
│ (FastMCP/HTTP)  │
└────────┬────────┘
         │ API Calls
         │
         ▼
┌─────────────────┐
│  NotebookLM API │
│ (Google Cloud)  │
└─────────────────┘
```

### Key Components

1. **MCPClientAdapter**: Handles MCP protocol communication
   - Session initialization and handshake
   - SSE parsing for real-time progress
   - Tool invocation with proper request/response handling

2. **Tools Class**: Open WebUI tool interface
   - `notebook_query_stream`: Main query tool with streaming
   - `notebook_list`: List available notebooks
   - `notebook_info`: Get notebook details
   - `health_check`: Verify server connectivity

3. **Event Emission**: Real-time UI updates
   - Status events: Progress indicators
   - Message events: Thinking steps and answers (Default mode)
   - Error events: User-friendly error messages
   - Notification events: Mode warnings

## 📊 Event Flow

```
User Query
   ↓
Tool Invoked ─────────→ "🔗 Connecting..." (status event)
   ↓
MCP Handshake ────────→ "📝 Sending query..." (status event)
   ↓
SSE Stream Started
   ↓
Progress Event 1 ─────→ "💭 Thinking: ..." (status + message)
   ↓
Progress Event 2 ─────→ "💭 Thinking: ..." (status + message)
   ↓
Progress Event 3 ─────→ "💭 Thinking: ..." (status + message)
   ↓
Answer Chunk 1 ───────→ "📝 Answer:" (message event)
   ↓
Answer Chunk 2 ───────→ [appended to message]
   ↓
Final Result ─────────→ "✅ Complete" (status done=True)
```

## 🔐 Security Considerations

### Authentication

The tool communicates with the MCP server, which requires NotebookLM authentication:

1. **Server-side auth**: MCP server handles NotebookLM cookies/tokens
2. **Client-side**: Tool only needs MCP server URL (no credentials)
3. **Session isolation**: Each tool invocation gets a unique session ID

### Network Security

- Tool makes HTTP requests to localhost by default
- For remote servers, use HTTPS and proper firewall rules
- Consider network isolation for sensitive notebooks

## 🧪 Testing

### Manual Testing

1. **Health Check**:
   ```
   @notebooklm Check server health
   ```
   Expected: "✅ MCP server is healthy!"

2. **List Notebooks**:
   ```
   @notebooklm List my notebooks
   ```
   Expected: Markdown table with notebooks

3. **Query Test**:
   ```
   @notebooklm Query notebook [id]: Test question
   ```
   Expected: Thinking steps → Answer

### Automated Testing

```python
# test_tool.py
import asyncio
from notebooklm_mcp_tool import Tools

async def test_health_check():
    tool = Tools()
    result = await tool.health_check()
    assert "healthy" in result.lower()

if __name__ == "__main__":
    asyncio.run(test_health_check())
```

## 📝 API Reference

### Tool Methods

#### `notebook_query_stream(notebook_id, query, source_ids=None, conversation_id=None)`

Query a notebook with real-time streaming.

**Parameters:**
- `notebook_id` (str): Notebook UUID from `notebook_list`
- `query` (str): Question to ask
- `source_ids` (list[str], optional): Filter specific sources
- `conversation_id` (str, optional): For follow-up questions

**Returns:** Final answer with conversation ID

---

#### `notebook_list(max_results=10)`

List available notebooks.

**Parameters:**
- `max_results` (int): Maximum notebooks to return (default: 10)

**Returns:** Markdown table of notebooks

---

#### `notebook_info(notebook_id)`

Get detailed notebook information.

**Parameters:**
- `notebook_id` (str): Notebook UUID

**Returns:** Formatted notebook details with sources

---

#### `health_check()`

Verify MCP server connectivity.

**Returns:** Health status message

## 🤝 Contributing

Contributions welcome! Please see the main repository:
https://github.com/jacob-bd/notebooklm-mcp

## 📄 License

MIT License - see main repository for details

## 🔗 Related Documentation

- [Open WebUI Tool Development](https://docs.openwebui.com/features/plugin/tools/development)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [NotebookLM MCP Server](https://github.com/jacob-bd/notebooklm-mcp)
- [Function Calling Modes Guide](https://docs.openwebui.com/features/plugin/tools#tool-calling-modes-default-vs-native)

## 🆘 Support

- **Issues**: https://github.com/jacob-bd/notebooklm-mcp/issues
- **Discussions**: https://github.com/jacob-bd/notebooklm-mcp/discussions
- **Discord**: [Open WebUI Discord](https://discord.gg/5rJgQTnV4s)
