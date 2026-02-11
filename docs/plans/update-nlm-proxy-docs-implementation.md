# Implementation Plan - Adapt `update-docs` Skill for NLM Proxy

## Context
The goal is to adapt the existing `update-docs` skill from `knowledge-finder-bot` to `nlm-proxy`, creating an automated documentation workflow tailored to this project's specific architecture (Smart Routing, Tracing, MCP, OpenAI Proxy). The skill will be renamed `update-nlm-proxy-docs` to avoid conflicts.

## Phase 1: Create Plugin Structure
- [ ] Create directory structure: `.claude/plugins/plugins/update-nlm-proxy-docs/skills/update-nlm-proxy-docs/`
- [ ] Create `SKILL.md` with adapted content.

## Phase 2: Adapt Skill Logic
- [ ] **Header:** Update metadata (name: `update-nlm-proxy-docs`).
- [ ] **Step 1 (Detection):** Keep git commands but target `nlm-proxy` directories (`core/`, `mcp/`, `server/`, `tests/`).
- [ ] **Step 2 (Classification):** Implement 5 specific change types:
    1.  **Smart Routing/Client:** `core/router.py`, `core/llm.py` -> `docs/smart-routing-architecture.md`
    2.  **Tracing/Telemetry:** `core/telemetry.py`, `docker/otel/` -> `docs/TRACING.md`
    3.  **MCP Tools:** `mcp/` -> `docs/API_REFERENCE.md`, `.claude/memory/mcp-tools.md`
    4.  **OpenAI Proxy:** `server/openai.py`, `core/client.py` -> `docs/API_REFERENCE.md`, `.claude/memory/openai-proxy.md`
    5.  **Project Structure:** General new files -> `README.md`, `.claude/memory/architecture.md`
- [ ] **Step 3 (Proposals):** Update checklist template to reference `nlm-proxy` docs.
- [ ] **Step 5 (Execution):** Adapt templates for `README.md` (structure updates), `docs/TRACING.md`, and `.env.example`.

## Phase 3: Documentation & Verification
- [ ] Verify file creation and content.
- [ ] Update `CLAUDE.md` to list the new skill.
- [ ] Update `TODO.md` to mark documentation task as complete.
- [ ] Run a manual test (dry run) of the skill's detection logic.

## Verification
- Verify the skill file exists at the correct path.
- Verify the skill contains correct paths for `nlm-proxy`.
- Verify `CLAUDE.md` mentions the new skill.
