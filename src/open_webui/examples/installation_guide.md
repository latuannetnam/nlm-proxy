# Installation Guide: NotebookLM MCP Tool for Open WebUI

Complete step-by-step guide to install and configure the NotebookLM MCP tool.

## Prerequisites

- ✅ Open WebUI instance (v0.4.0+)
- ✅ NotebookLM MCP server repository cloned
- ✅ Python 3.11+ (via uv or pip)
- ✅ Valid NotebookLM account with notebooks

## Step 1: Install & Authenticate MCP Server

### 1.1 Clone the Repository (if not already done)

```bash
git clone https://github.com/jacob-bd/notebooklm-mcp.git
cd notebooklm-mcp
```

### 1.2 Install Dependencies

Using `uv` (recommended):
```bash
uv sync
```

Or using `pip`:
```bash
pip install -e .
```

### 1.3 Authenticate with NotebookLM

Run the authentication CLI:
```bash
notebooklm-mcp-auth
```

Follow the prompts:
1. Opens browser to NotebookLM
2. Sign in to your Google account
3. Copy cookies from browser
4. Paste into CLI
5. Tokens saved to `~/.notebooklm-mcp/auth.json`

**Alternative - Manual Token Setup:**
```bash
# Export cookies manually
export NOTEBOOKLM_COOKIES="your-cookie-string"
export NOTEBOOKLM_CSRF_TOKEN="your-csrf-token"
export NOTEBOOKLM_SESSION_ID="your-session-id"
```

### 1.4 Verify Authentication

```bash
# Test the server
uv run notebooklm-mcp
```

Should show: `MCP server started successfully`

## Step 2: Start MCP Server with HTTP Transport

The tool requires HTTP transport (not stdio):

### 2.1 Start Server

```bash
uv run notebooklm-mcp --transport http --port 9888
```

**With debug logging:**
```bash
uv run notebooklm-mcp --transport http --port 9888 --debug
```

### 2.2 Verify Server is Running

Open a new terminal and test:

```bash
# Health check endpoint
curl http://localhost:9888/health

# Expected response:
# {"status":"healthy","service":"notebooklm-mcp","version":"1.0.0"}
```

### 2.3 Keep Server Running

The server must remain running while using the tool.

**Options:**
- Terminal window (manual)
- Screen/tmux session (Linux/Mac)
- Windows Service (Windows)
- Docker container (any platform)
- Systemd service (Linux)

**Example - Using screen:**
```bash
screen -S notebooklm-mcp
uv run notebooklm-mcp --transport http --port 9888
# Press Ctrl+A, then D to detach
```

**Example - Docker (if available):**
```bash
docker run -d -p 9888:9888 \
  -e NOTEBOOKLM_COOKIES="..." \
  -e NOTEBOOKLM_CSRF_TOKEN="..." \
  notebooklm-mcp
```

## Step 3: Install Tool in Open WebUI

### 3.1 Copy Tool Code

1. Navigate to: `src/open_webui/notebooklm_mcp_tool.py`
2. Copy the entire file contents

### 3.2 Create Tool in Open WebUI

1. Open Open WebUI in browser
2. Click **Workspace** (left sidebar)
3. Click **Tools**
4. Click **+ Create New Tool** button
5. Paste the tool code
6. Click **Save** button

### 3.3 Verify Tool Installation

The tool should appear in your tools list:
- **Name**: NotebookLM MCP Client
- **Version**: 1.0.0
- **Status**: Active ✅

## Step 4: Configure Tool Settings (Valves)

### 4.1 Open Tool Settings

1. Find the tool in your tools list
2. Click the **⚙️ gear icon**

### 4.2 Set Configuration

**Required Settings:**

| Setting | Value | Description |
|---------|-------|-------------|
| `mcp_server_url` | `http://localhost:9888` | MCP server URL |
| `timeout` | `120.0` | Request timeout (seconds) |
| `enable_debug` | `false` | Debug logging |

**For Remote Server:**

If your MCP server is on a different machine:
```
mcp_server_url: http://your-server-ip:9888
```

**For Docker:**

If MCP server is in Docker:
```
mcp_server_url: http://host.docker.internal:9888  (Mac/Windows)
mcp_server_url: http://172.17.0.1:9888           (Linux)
```

### 4.3 Save Settings

Click **Save** to apply changes.

## Step 5: Configure Function Calling Mode

**CRITICAL for full streaming support!**

### 5.1 Admin Configuration (Global)

1. Navigate to **Admin Panel** (admin icon)
2. Click **Settings**
3. Select **Models**
4. Choose your model (e.g., GPT-4, Claude)
5. Expand **Advanced Parameters**
6. Find **Function Calling**
7. Select **"Default"** (not "Native")
8. Click **Save**

### 5.2 Per-Chat Configuration (Optional)

To override mode for a specific chat:

1. Open your chat
2. Click **Chat Controls** (⋮ icon)
3. Select **Advanced Params**
4. Add parameter:
   ```json
   {
     "function_calling": "default"
   }
   ```
5. Click **Save**

### 5.3 Verify Mode

The tool will warn you if Native mode is detected:
```
⚠️ Native function calling mode detected. Streaming may be limited.
```

## Step 6: Enable and Test Tool

### 6.1 Enable Tool in Chat

1. Open a chat or create new chat
2. Click the **+** button (add tool)
3. Select **NotebookLM MCP Client**
4. Tool icon should appear in chat

### 6.2 Run Health Check

First command to verify everything works:

```
Check server health
```

**Expected output:**
```
✅ MCP server is healthy!

**Server URL:** http://localhost:9888
**Session ID:** `abc-123-def-456`
**Timeout:** 120.0s
```

**If connection fails:**
```
❌ Cannot connect to MCP server

**Troubleshooting:**
1. Ensure server is running: uv run notebooklm-mcp --transport http --port 9888
2. Check server URL in tool settings
3. Verify no firewall is blocking the connection
```

### 6.3 List Your Notebooks

```
List my notebooks
```

**Expected output:**
```markdown
## 📚 Your NotebookLM Notebooks

| Title | Notebook ID | Sources |
|-------|-------------|----------|
| My Research | `abc-123` | 5 |

*Showing 1 notebook(s)*
```

### 6.4 Test Query with Streaming

```
Query notebook abc-123: What is this notebook about?
```

**Expected behavior (Default mode):**
- Status updates appear in real-time
- Thinking steps stream progressively
- Answer builds incrementally
- Completion message shows final stats

## Troubleshooting Installation

### Issue: "Cannot connect to MCP server"

**Diagnosis:**
```bash
# Check if server is running
curl http://localhost:9888/health

# Check if port is in use
netstat -an | grep 9888  (Linux/Mac)
netstat -an | findstr 9888  (Windows)

# Check MCP server logs
uv run notebooklm-mcp --transport http --port 9888 --debug
```

**Solutions:**
1. Start the MCP server
2. Change port if conflict exists
3. Update tool Valves with new URL/port
4. Check firewall settings

### Issue: "Tool not found in Open WebUI"

**Solutions:**
1. Refresh the page (Ctrl+F5)
2. Check tool code was saved correctly
3. Look for Python syntax errors in tool
4. Verify Open WebUI version is 0.4.0+

### Issue: "Authentication failed"

**Solutions:**
1. Re-run authentication:
   ```bash
   notebooklm-mcp-auth
   ```
2. Check if cookies expired (re-login to NotebookLM)
3. Verify auth tokens in `~/.notebooklm-mcp/auth.json`
4. Test directly with server:
   ```bash
   uv run notebooklm-mcp --transport stdio
   # Type: {"method": "tools/list"}
   ```

### Issue: "No thinking steps appearing"

**Diagnosis:**
- Check which function calling mode is active
- Look for Native mode warning in chat

**Solutions:**
1. Switch to Default mode (see Step 5)
2. Verify setting applied:
   - Reload page
   - Start new chat
   - Check Chat Controls > Advanced Params

### Issue: "Slow or timeout errors"

**Solutions:**
1. Increase timeout in tool Valves (e.g., 180.0)
2. Check MCP server logs for API errors
3. Verify NotebookLM API is accessible
4. Test with smaller queries first

### Issue: "httpx not installed"

**Solution:**
Open WebUI should auto-install, but if not:

1. Add to tool requirements metadata:
   ```python
   requirements: httpx
   ```
2. Save tool (triggers pip install)
3. Wait for installation to complete

## Advanced Configuration

### Running MCP Server as a Service

**Linux (systemd):**
```ini
# /etc/systemd/system/notebooklm-mcp.service
[Unit]
Description=NotebookLM MCP Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/notebooklm-mcp
ExecStart=/usr/bin/uv run notebooklm-mcp --transport http --port 9888
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable notebooklm-mcp
sudo systemctl start notebooklm-mcp
sudo systemctl status notebooklm-mcp
```

**Windows (Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task
3. Action: Start a program
4. Program: `C:\path\to\uv.exe`
5. Arguments: `run notebooklm-mcp --transport http --port 9888`
6. Working directory: `C:\path\to\notebooklm-mcp`

### Remote Server Setup

**Server side:**
```bash
# Bind to all interfaces (⚠️ use with firewall!)
uv run notebooklm-mcp --transport http --port 9888 --host 0.0.0.0
```

**Client side (tool Valves):**
```
mcp_server_url: http://your-server-ip:9888
```

**Security recommendations:**
- Use HTTPS reverse proxy (nginx/caddy)
- Configure firewall rules
- Consider VPN for remote access
- Use authentication if exposing publicly

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

EXPOSE 9888

CMD ["uv", "run", "notebooklm-mcp", "--transport", "http", "--port", "9888", "--host", "0.0.0.0"]
```

**Build and run:**
```bash
docker build -t notebooklm-mcp .
docker run -d -p 9888:9888 \
  -e NOTEBOOKLM_COOKIES="..." \
  notebooklm-mcp
```

## Next Steps

✅ Installation complete!

Now you can:
- 📖 Read the [README](../README.md) for usage examples
- 🎯 See [Default Mode Example](./default_mode_example.md) for full streaming
- ⚙️ Compare with [Native Mode Example](./native_mode_example.md)
- 🧪 Test with your NotebookLM notebooks

## Support

- 📚 [Documentation](../README.md)
- 🐛 [Issues](https://github.com/jacob-bd/notebooklm-mcp/issues)
- 💬 [Discussions](https://github.com/jacob-bd/notebooklm-mcp/discussions)
