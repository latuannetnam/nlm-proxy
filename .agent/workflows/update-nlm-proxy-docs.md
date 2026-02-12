---
description: Auto-analyze code changes and propose documentation updates for NLM Proxy
---

# Update NLM Proxy Documentation

This workflow analyzes recent code changes and proposes documentation updates specific to the NLM Proxy architecture (Smart Routing, Tracing, MCP, OpenAI Proxy).

**Documentation structure:**
- `GEMINI.md` — Concise root file with rules and quick commands
- `.agent/memory/*.md` — Detailed memory modules
- `docs/` — Technical documentation and architecture

## Step 1: Detect Changes

Check what has changed in the codebase:

// turbo
```bash
cd d:/latuan/Programming/nlm-proxy && git status --short
```

// turbo
```bash
cd d:/latuan/Programming/nlm-proxy && git diff HEAD --name-status
```

// turbo
```bash
cd d:/latuan/Programming/nlm-proxy && git log -10 --pretty=format:"%h %s%n%b"
```

// turbo
```bash
cd d:/latuan/Programming/nlm-proxy && git diff HEAD -- pyproject.toml | grep -E "^[+-]" | Select-Object -First 20
```

## Step 2: Classify Change Types

Based on git output, identify which change types occurred:

**Smart Routing & LLM Client**
- Detection: Changes to `core/router.py`, `core/llm_client.py`, `openai/router.py`, or `openai/notebook_cache.py`
- Update:
  - `docs/smart-routing-architecture.md` (flow diagrams/logic)
  - `.agent/memory/smart-routing.md` (configuration, cache, logging)
  - `GEMINI.md` (rules if routing logic changes)

**Tracing & Telemetry**
- Detection: Changes to `core/config.py` (TracingSettings), `docker/otel/`, `*.yml` (docker-compose), or `docs/TRACING.md`
- Update:
  - `docs/TRACING.md` (detailed setup/troubleshooting)
  - `.agent/memory/tracing.md` (quick start, known issues)
  - `.env.example` (if OTEL env vars change)

**MCP Tools**
- Detection: Changes to `mcp/server.py` or new files in `mcp/`
- Update:
  - `docs/API_REFERENCE.md` (detailed tool documentation)
  - `.agent/memory/mcp-tools.md` (tools table, confirmation rules)
  - `README.md` (MCP capabilities)

**OpenAI Proxy Interface**
- Detection: Changes to `openai/server.py`, `core/client.py`, or `openai/types.py`
- Update:
  - `docs/API_REFERENCE.md` (detailed endpoint documentation)
  - `.agent/memory/openai-proxy.md` (SDK examples, session persistence)
  - `README.md` (compatibility list)

**Project Structure & Config**
- Detection: New files, `pyproject.toml`, or `Dockerfile`
- Update:
  - `README.md` (project structure tree, prerequisites)
  - `.agent/memory/architecture.md` (file structure, key components)
  - `.agent/memory/configuration.md` (if env vars change)
  - `GEMINI.md` (rules if fundamental changes)

## Step 3: Generate Update Proposals

For each detected change type, create a checklist:

```
🔍 Detected Changes:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. [Change Type]: [Specific item]
   Files changed:
   - [file path] ([status: new/modified])

   📝 Documentation to update:
   ✓ [doc file 1] ([what to add/update])
   ✓ [doc file 2] ([what to add/update])

2. [Next change type]...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Step 4: Request User Approval

Ask the user which updates to perform:
- "yes" or "approve" → Proceed with all
- "selective" → Ask which specific files
- "no" or "cancel" → Abort

## Step 5: Execute Documentation Updates

For each approved file:

1. Read current content
2. Identify relevant section
3. Update with new information
4. Verify correctness

**Key files to update:**

*Root files:*
- `GEMINI.md` — if rules or quick commands changed
- `README.md` — if project structure or features changed
- `.env.example` — if new environment variables added

*Memory modules (`.agent/memory/`):*
- `architecture.md` — if file structure or key components changed
- `commands.md` — if new commands or execution methods added
- `configuration.md` — if env vars or settings classes changed
- `authentication.md` — if auth methods changed
- `mcp-tools.md` — if new tools added or confirmation rules changed
- `openai-proxy.md` — if endpoints or SDK examples changed
- `smart-routing.md` — if routing config or cache logic changed
- `logging.md` — if logging config changed
- `tracing.md` — if OTEL setup or known issues changed
- `troubleshooting.md` — if new errors/fixes added

*Technical docs (`docs/`):*
- `docs/TRACING.md` — detailed tracing setup/troubleshooting
- `docs/API_REFERENCE.md` — if new MCP tools or OpenAI endpoints added
- `docs/smart-routing-architecture.md` — if routing architecture/flow changed

## Step 6: Final Verification

1. Show summary of files updated
2. Run `git status` to show modified documentation
3. Offer to commit changes:
   ```bash
   git add [modified files]
   git commit -m "docs: update documentation for [feature]"
   ```
