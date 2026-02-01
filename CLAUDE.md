# CLAUDE.md

## Project Overview

**NotebookLM MCP Server** - Provides programmatic access to NotebookLM (notebooklm.google.com) using internal APIs. Tested with personal/free tier accounts.

## Quick Commands

```bash
nlm-proxy serve mcp              # Run MCP server
nlm-proxy serve mcp --debug      # With debug logging
nlm-proxy serve openai --port 8080  # OpenAI proxy
nlm-proxy auth extract           # Extract auth tokens
nlm-proxy auth test              # Verify authentication
uv run pytest                    # Run tests
```

**Reinstall after changes:**
```bash
uv cache clean && uv tool install ".[all]" --force
```

## Memory Modules

Detailed documentation in `.claude/memory/`:

| Module | When to Read |
|--------|--------------|
| `commands.md` | Full command reference, from-source execution |
| `architecture.md` | Understanding codebase structure |
| `authentication.md` | Auth issues, setting up tokens |
| `mcp-tools.md` | MCP tool reference, confirmation rules |
| `openai-proxy.md` | OpenAI proxy setup, SDK examples |
| `logging.md` | Configuring logs, debugging |
| `troubleshooting.md` | Common errors and fixes |

## References

| Document | When to Read |
|----------|--------------|
| `docs/API_REFERENCE.md` | Debugging APIs, adding features, RPC details |
| `docs/MCP_TEST_PLAN.md` | Testing MCP tools, validation |

## Contributing

1. Capture network request with Chrome DevTools
2. Document RPC ID in `docs/API_REFERENCE.md`
3. Add method to `core/client.py`
4. Add tool in `mcp/server.py`
5. Add test case to `docs/MCP_TEST_PLAN.md`

## License

MIT License
