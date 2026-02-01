# Development Commands

## Quick Reference

```bash
# Install with all extras
uv pip install -e ".[all]"

# Reinstall after changes (ALWAYS clean cache)
uv cache clean && uv tool install ".[all]" --force

# Run MCP server
nlm-proxy serve mcp
nlm-proxy serve mcp --debug
nlm-proxy serve mcp --transport http --port 8000

# Run OpenAI proxy
nlm-proxy serve openai --port 8080

# Authentication
nlm-proxy auth extract    # Automated (recommended)
nlm-proxy auth test       # Verify tokens

# Run tests
uv run pytest
uv run pytest tests/test_file.py::test_function -v
```

## From Source (bypasses CLI caching)

```bash
uv run python -m nlm_proxy serve mcp
uv run python -m nlm_proxy serve mcp --debug
uv run python -m nlm_proxy serve openai --port 8080
uv run python -m nlm_proxy auth test
```

## Direct Python (bypasses CLI entirely)

```bash
uv run python -c "from nlm_proxy.mcp import run_server; run_server()"
uv run python -c "from nlm_proxy.mcp import run_server; run_server(debug=True)"
uv run python -c "from nlm_proxy.openai import run_server; run_server(port=8080)"
```

**Python requirement:** >=3.11
