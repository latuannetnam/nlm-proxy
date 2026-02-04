# CLAUDE.md

## Project Overview

**NLM Proxy** - Provides programmatic access to NotebookLM (notebooklm.google.com) using internal APIs. Tested with personal/free tier accounts.

## Rules
- After done planing for new feature, always write plan to docs/plan folder

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
| `configuration.md` | Environment variables, .env files, settings precedence |
| `authentication.md` | Auth issues, setting up tokens |
| `mcp-tools.md` | MCP tool reference, confirmation rules |
| `openai-proxy.md` | OpenAI proxy setup, SDK examples |
| `smart-routing.md` | Smart routing configuration, LLM client, router |
| `logging.md` | Configuring logs, debugging |
| `troubleshooting.md` | Common errors and fixes |

## References

| Document | When to Read |
|----------|--------------|
| `docs/API_REFERENCE.md` | Debugging APIs, adding features, RPC details |
| `docs/MCP_TEST_PLAN.md` | Testing MCP tools, validation |
| `docs/ASYNCIO_THREADING_PITFALLS.md` | Asyncio + threading bugs, "Event loop is closed" errors |

## Known Issues & Lessons Learned

Critical issues encountered and resolved in this project:

| Issue | Document | Summary |
|-------|----------|---------|
| Event loop is closed | `docs/ASYNCIO_THREADING_PITFALLS.md` | Asyncio objects (Lock, httpx.AsyncClient) bind to their creation event loop. When reused in a different thread's event loop, they fail. **Fix**: Close async clients before switching event loops. |

When debugging asyncio/threading issues:
1. Check for alternating success/failure patterns → `asyncio.run()` misuse
2. First operation fails, rest succeed → async object bound to wrong loop
3. Always close async clients before event loop closes

## Contributing

1. Capture network request with Chrome DevTools
2. Document RPC ID in `docs/API_REFERENCE.md`
3. Add method to `core/client.py`
4. Add tool in `mcp/server.py`
5. Add test case to `docs/MCP_TEST_PLAN.md`

## License

MIT License
