---
name: update-nlm-proxy-docs
description: Analyze code changes and propose documentation updates for NLM Proxy
---

# NLM Proxy Documentation Assistant

This skill analyzes recent code changes and proposes documentation updates specific to the NLM Proxy architecture (Smart Routing, Tracing, MCP, OpenAI Proxy).

## Step 1: Detect Changes

### Git Status
```bash
git status --short
```

### Changed Files
```bash
git diff HEAD --name-status
```

### Recent Commit
```bash
git log -10 --pretty=format:"%h %s%n%b"
```

### Dependency Changes
```bash
git diff HEAD -- pyproject.toml | grep -E "^[+-]" | head -20
```

### Load Update Rules
```bash
cat CLAUDE.md | grep -A 10 "## Rules"
```

### Load Current Phase
```bash
cat .claude/memory/MEMORY.md | grep -A 20 "## Current Phase"
```

## Step 2: Classify Change Types

Based on git output, identify which change types occurred:

**1. Smart Routing & LLM Client**
Detection: Changes to `core/router.py`, `core/llm.py`, `config/`, or `docs/smart-routing-architecture.md`
Map to:
- `docs/smart-routing-architecture.md` (Update flow diagrams/logic)
- `CLAUDE.md` (Update rules if routing logic changes)
- `.claude/memory/smart-routing.md`

**2. Tracing & Telemetry**
Detection: Changes to `core/telemetry.py`, `docker/otel/`, `*.yml` (docker-compose), or `docs/TRACING.md`
Map to:
- `docs/TRACING.md` (Update troubleshooting/setup)
- `.env.example` (If OTEL env vars change)
- `.claude/memory/tracing.md`

**3. MCP Tools**
Detection: Changes to `mcp/server.py` or new files in `mcp/`
Map to:
- `docs/API_REFERENCE.md` (Document new tools/RPCs)
- `.claude/memory/mcp-tools.md`
- `README.md` (Update MCP capabilities)

**4. OpenAI Proxy Interface**
Detection: Changes to `server/openai.py`, `core/client.py`, or `models/`
Map to:
- `docs/API_REFERENCE.md` (Update endpoint documentation)
- `.claude/memory/openai-proxy.md` (Update usage examples)
- `README.md` (Update compatibility list)

**5. Project Structure & Config**
Detection: New files, `pyproject.toml`, or `Dockerfile`
Map to:
- `README.md` (Project structure tree, Prerequisites)
- `.claude/memory/architecture.md`
- `.claude/memory/configuration.md`

## Step 3: Generate Update Proposals

For each detected change type, create a checklist entry:

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

## Step 4: Handle User Approval

**If user responds "yes" or "approve":**
- Proceed to Step 5 (execute all proposed updates)

**If user responds "selective":**
- Ask which specific files to update
- Only proceed with user-selected files

**If user responds "no" or "cancel":**
- Abort and show: "✅ Documentation update cancelled. No changes made."

## Step 5: Execute Documentation Updates

For each approved file, use the Read and Edit tools.

### Template: Update TRACING.md

**When telemetry/OTEL config changes:**

1. **Read current content:** `cat docs/TRACING.md`
2. **Identify section:** Look for "Configuration", "Troubleshooting", or "Architecture".
3. **Update:** Use Edit tool to reflect new env vars or troubleshooting steps.

### Template: Update API_REFERENCE.md

**When new MCP tool or OpenAI endpoint added:**

1. **Read current content:** `cat docs/API_REFERENCE.md`
2. **Generate Entry:**
   ```markdown
   ### [Method/Tool Name]
   - **Purpose:** [Description]
   - **Inputs:** [Args]
   - **Returns:** [Output]
   ```
3. **Update:** Insert into appropriate section (MCP Tools or API Endpoints).

### Template: Update README.md

**When project structure changes:**

1. **Read README.md**
2. **Update Structure Tree:** Look for the `tree` command output or similar structure representation.
3. **Update Status/Features:** Add new features to the list.

### Template: Update Memory Modules

**Always update relevant memory modules:**

1. **Read target memory file** (e.g., `.claude/memory/smart-routing.md`)
2. **Update:** Add new patterns, decisions, or architectural changes.
3. **Verify:** Read back to ensure correctness.

## Step 6: Final Verification

1. **Review changes:** Show a summary of files updated.
2. **Git Status:** Run `git status` to show modified documentation.
3. **Commit Suggestion:** Offer to commit changes with a standard message:
   `docs: update documentation for [feature]`
