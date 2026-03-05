# GEMINI.md

## Project Overview

**NLM Proxy** — OpenAI-compatible proxy for NotebookLM. Provides programmatic access to NotebookLM (notebooklm.google.com) via MCP server and OpenAI-compatible REST API. Features smart routing powered by **LangChain/LangGraph** that automatically classifies requests and routes them to NotebookLM (knowledge queries) or external LLM (general tasks) through an **AgentCore** orchestration layer shared by both OpenAI proxy and MCP server. Includes a two-layer response cache (exact hash match → embedding similarity) that eliminates 40-50s latency for repeated queries. Tested with personal/free tier accounts. Relies on internal `batchexecute` RPCs.

## Rules

- After planning for new features, always write plan to `docs/plans/` folder
- **ALWAYS** update following important documents with appropriate contents after implementing new features, fixing bugs, or making important changes to code:
  + `README.md`
  + `docs/smart-routing-architecture.md`
  + `docs/TRACING.md`
  + `.env.example`
  + `GEMINI.md`

## Quick Commands

```bash
# === Installation ===
uv pip install -e ".[all]"              # Install with all extras (incl. dev)
uv cache clean && uv tool install ".[all]" --force  # Reinstall after changes

# === MCP Server ===
nlm-proxy serve mcp                     # Run MCP server
nlm-proxy serve mcp --debug             # With debug logging

# === OpenAI Proxy ===
nlm-proxy serve openai --port 8080      # OpenAI-compatible proxy

# === Authentication ===
nlm-proxy auth extract                  # Extract auth tokens (recommended, one-time)
nlm-proxy auth refresh                  # Refresh CSRF token (~2s, run when tokens expired)
nlm-proxy auth refresh --full           # Full refresh: CSRF + cookies via headless Chrome (~10s)
nlm-proxy auth test                     # Verify authentication

# === Testing ===
uv run pytest                           # Run all tests

# === Cache Monitor ===
.\scripts\cache-stats.ps1               # Cache stats (PowerShell)
.\scripts\cache-stats.ps1 -Watch        # Auto-refresh mode
./scripts/cache-stats.sh                # Cache stats (Bash)
./scripts/cache-stats.sh --watch        # Auto-refresh mode

# === Cache Log Analyzer ===
python scripts/cache-log-analyzer.py --today --queries    # Per-query cache flow
python scripts/cache-log-analyzer.py --today --summary    # Hit-rate summary
python scripts/cache-log-analyzer.py --json --queries     # JSON (for scripts/AI)

# === Tracing Infrastructure ===
docker compose -f docker-compose.otel.yml up -d  # Start basic stack (dev)
docker compose -f docker-compose.otel-secure.yml up -d  # Secure stack (prod)
open http://localhost:3000  # Grafana dashboard (admin/admin)
```

**Workflows:**
- `/update-nlm-proxy-docs` — Auto-analyze changes & update docs

## Memory Modules

Detailed documentation in `.agent/memory/`:

| Module | When to Read |
|--------|--------------|
| `architecture.md` | Understanding codebase structure, key components |
| `commands.md` | Full command reference, from-source execution |
| `configuration.md` | Environment variables, .env files, settings precedence |
| `authentication.md` | Auth issues, setting up tokens |
| `mcp-tools.md` | MCP tool reference, confirmation rules |
| `openai-proxy.md` | OpenAI proxy setup, SDK examples |
| `smart-routing.md` | Smart routing configuration, LLM client, router |
| `logging.md` | Configuring logs, debugging |
| `tracing.md` | OpenTelemetry tracing setup, TLS/auth configuration, known issues |
| `response-cache.md` | Response cache architecture, pre-routing L1, aliases, configuration |
| `troubleshooting.md` | Common errors and fixes |

## References

| Document | When to Read |
|----------|--------------|
| `docs/API_REFERENCE.md` | Debugging APIs, adding features, RPC details |
| `docs/ASYNCIO_THREADING_PITFALLS.md` | Asyncio + threading bugs, "Event loop is closed" errors |
| `docs/TRACING.md` | Setting up tracing, TLS/auth configuration, troubleshooting collector issues |
| `docs/smart-routing-architecture.md` | Understanding smart routing architecture and flow diagrams |

## Contributing

1. Capture network request with Chrome DevTools
2. Document RPC ID in `docs/API_REFERENCE.md`
3. Add method to `core/client.py`
4. Add tool in `mcp/server.py`
5. Add test case to `docs/MCP_TEST_PLAN.md`

## License

MIT License
