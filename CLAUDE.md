# CLAUDE.md

## Project Overview

**NLM Proxy** - Provides programmatic access to NotebookLM (notebooklm.google.com) using internal APIs. Tested with personal/free tier accounts.

## Rules
- After done planing for new feature, always write plan to docs/plans folder
- **ALWAYS** Update follwoing important documents with appropriate contents after done implementating new features, fix bug, make important changes to code:
  + README.md
  + docs\smart-routing-architecture.md
  + docs\TRACING.md
  + .env.example
  + CLAUDE.md

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

**Tracing Infrastructure:**
```bash
# Start basic stack (dev - no TLS/auth)
docker compose -f docker-compose.otel.yml up -d

# Start secure stack (prod - TLS + bearer token auth)
bash docker/otel/generate-certs.sh
docker compose -f docker-compose.otel-secure.yml up -d

# Access Grafana dashboard
open http://localhost:3000  # admin/admin
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
| `tracing.md` | OpenTelemetry tracing setup, TLS/auth configuration |

## References

| Document | When to Read |
|----------|--------------|
| `docs/API_REFERENCE.md` | Debugging APIs, adding features, RPC details |
| `docs/ASYNCIO_THREADING_PITFALLS.md` | Asyncio + threading bugs, "Event loop is closed" errors |
| `docs/TRACING.md` | Setting up tracing, TLS/auth configuration, troubleshooting collector issues |

## Known Issues & Lessons Learned

Critical issues encountered and resolved in this project:

| Issue | Document | Summary |
|-------|----------|---------|
| Event loop is closed | `docs/ASYNCIO_THREADING_PITFALLS.md` | Asyncio objects (Lock, httpx.AsyncClient) bind to their creation event loop. When reused in a different thread's event loop, they fail. **Fix**: Close async clients before switching event loops. |
| OTLP UNAVAILABLE errors | `docs/TRACING.md` | Generic error that masks auth failures, TLS handshake errors, or network issues. **Fix**: Check collector logs for real error, verify bearer token matches on client/collector, ensure TLS config is consistent (INSECURE flag vs actual TLS). |

When debugging asyncio/threading issues:
1. Check for alternating success/failure patterns → `asyncio.run()` misuse
2. First operation fails, rest succeed → async object bound to wrong loop
3. Always close async clients before event loop closes

When debugging OpenTelemetry connection issues:
1. Check collector logs first: `docker logs nlm-otel-collector` or `sudo journalctl -u otelcol-contrib`
2. Verify bearer token matches: `NLM_PROXY_OTEL_API_KEY` == collector's `OTEL_BEARER_TOKEN`
3. Check TLS consistency: If collector has TLS, client needs `INSECURE=false` + CA cert
4. Protocol matters: gRPC doesn't support skip-verify; use HTTP for self-signed certs

## Tracing & Observability

**Development Setup (No Security):**
- Use `docker-compose.otel.yml` for local development
- Plain HTTP/gRPC without TLS or authentication
- Quick start, minimal configuration

**Production Setup (Secure):**
- Use `docker-compose.otel-secure.yml` with TLS + bearer token auth
- Generate certificates: `bash docker/otel/generate-certs.sh`
- Generate token: `openssl rand -base64 32`
- Configure both collector and client with matching tokens
- See `docs/TRACING.md` for complete setup guide

**Key Configuration Files:**
- `docker/otel/config.yaml` - Basic collector config (no auth/TLS)
- `docker/otel/config-secure.yaml` - Secure collector config (TLS + bearertokenauth)
- `docker/otel/generate-certs.sh` - Self-signed certificate generator
- `docker/otel/.env.example` - Environment variables template

**Troubleshooting:**
- `StatusCode.UNAVAILABLE` → Check collector logs, verify token, check TLS config
- Authentication failures → Ensure `OTEL_BEARER_TOKEN` matches on both sides
- TLS handshake errors → Verify `INSECURE` flag matches actual collector config
- For details: `docs/TRACING.md#troubleshooting`

## Contributing

1. Capture network request with Chrome DevTools
2. Document RPC ID in `docs/API_REFERENCE.md`
3. Add method to `core/client.py`
4. Add tool in `mcp/server.py`
5. Add test case to `docs/MCP_TEST_PLAN.md`

## License

MIT License
