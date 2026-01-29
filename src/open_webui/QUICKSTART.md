# Quick Start Guide

Get the NotebookLM MCP tool running in Open WebUI in under 5 minutes!

## Prerequisites

✅ Open WebUI v0.4.0+ installed and running  
✅ Python 3.11+ with `uv` installed  
✅ NotebookLM account with notebooks  

## 3-Step Setup

### Step 1: Start MCP Server (2 minutes)

```bash
# Clone and install (if not already done)
git clone https://github.com/jacob-bd/notebooklm-mcp.git
cd notebooklm-mcp
uv sync

# Authenticate with NotebookLM
notebooklm-mcp-auth
# Follow prompts to save cookies

# Start HTTP server
uv run notebooklm-mcp --transport http --port 9888
```

**Verify server is running:**
```bash
curl http://localhost:9888/health
# Should return: {"status":"healthy",...}
```

### Step 2: Install Tool in Open WebUI (1 minute)

1. Open Open WebUI in browser
2. Go to **Workspace** → **Tools** → **+ Create New Tool**
3. Copy and paste from: `src/open_webui/notebooklm_mcp_tool.py`
4. Click **Save**

### Step 3: Configure & Test (2 minutes)

**Set Function Calling Mode:**
1. **Admin Panel** → **Settings** → **Models**
2. Select your model → **Function Calling** = **"Default"**
3. Save

**Test the tool:**
1. Create new chat
2. Click **+** button, enable **NotebookLM MCP Client**
3. Type: `Check server health`
4. Should see: ✅ MCP server is healthy!

## First Query

```
List my notebooks
```

Then query one:
```
Query notebook [notebook-id]: What is this about?
```

Watch the thinking steps appear in real-time! 🎉

## Troubleshooting

**"Cannot connect"**: Ensure server is running on port 9888  
**"No streaming"**: Check Function Calling mode is "Default"  
**"Authentication failed"**: Re-run `notebooklm-mcp-auth`  

## Next Steps

📖 Read [full README](README.md) for all features  
🎯 See [examples](examples/) for advanced usage  
🐛 Report issues at [GitHub](https://github.com/jacob-bd/notebooklm-mcp/issues)

---

**That's it!** You now have NotebookLM integrated into Open WebUI with real-time streaming. 🚀
