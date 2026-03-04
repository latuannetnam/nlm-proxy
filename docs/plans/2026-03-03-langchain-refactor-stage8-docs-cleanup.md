# Stage 8: Documentation & Cleanup

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update documentation to reflect the LangChain/LangGraph refactor and remove dead code.

**Architecture:** Update all project docs per the GEMINI.md rules, add new env vars to `.env.example`, and remove the old `ExternalLLMClient`, `SmartRouter`, legacy test files, and `fastembed` references.

**Inputs:** All stages 0-7 complete and passing.

**Outputs:** Clean codebase with accurate documentation.

---

## Task 8.1: Update documentation

**Files:**
- Modify: `README.md` — update architecture description, dependency list
- Modify: `GEMINI.md` — update Architecture section, add AgentCore references
- Modify: `.env.example` — add `NLM_PROXY_AGENT_*` variables
- Modify: `docs/smart-routing-architecture.md` — update flow diagram with LangGraph
- Modify: `docs/TRACING.md` — document LangGraph span interaction with OTEL

Key changes for `README.md`:
- Replace "hand-rolled SmartRouter" with "LangGraph StateGraph"
- Update dependency list (add langchain, langgraph; remove fastembed)
- Document new `NLM_PROXY_AGENT_*` env vars

Key changes for `.env.example`:
```ini
# === Agent Settings (NEW — LangChain/LangGraph) ===
# NLM_PROXY_AGENT_LLM_PROVIDER=openai          # openai | anthropic | ollama
# NLM_PROXY_AGENT_EMBEDDING_PROVIDER=huggingface # huggingface | openai
# NLM_PROXY_AGENT_MEMORY_BACKEND=memory         # memory | sqlite | postgres
# NLM_PROXY_AGENT_MEMORY_DB_PATH=~/.nlm-proxy/memory.db
# NLM_PROXY_AGENT_MAX_ITERATIONS=10
# NLM_PROXY_AGENT_FALLBACK_ON_ERROR=true
```

**Step 1: Commit**

```bash
git commit -m "docs: update README, GEMINI.md, TRACING.md for LangChain refactor"
```

---

## Task 8.2: Remove dead code

**Files to delete:**
- `tests/test_openai_module/test_router_legacy.py`
- `tests/test_openai_module/test_router_acl_legacy.py`

**Code to remove:**
- `ExternalLLMClient` class from `core/llm_client.py` (replaced by `LangChainLLMClient`)
- All `fastembed` imports and references across the codebase
- `openai/router.py` — delete entirely (replaced by `core/routing_graph.py`)

**Code to keep (for now):**
- `openai/notebook_cache.py` re-export — keep for backward compatibility for 1 release

**Step 1: Remove files**

```bash
rm tests/test_openai_module/test_router_legacy.py
rm tests/test_openai_module/test_router_acl_legacy.py
```

**Step 2: Remove `ExternalLLMClient` from `core/llm_client.py`**

Delete the old `ExternalLLMClient` class. Keep only `LangChainLLMClient`, `create_chat_model()`, and `_convert_messages()`.

**Step 3: Delete `openai/router.py`**

```bash
rm src/nlm_proxy/openai/router.py
```

**Step 4: Search and remove all `fastembed` references**

```bash
# Find remaining references
grep -r "fastembed" src/ tests/
```

Remove any remaining imports or comments referencing fastembed.

**Step 5: Run ALL tests**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git commit -m "chore: remove dead ExternalLLMClient, SmartRouter, fastembed refs, legacy tests"
```

---

## 🔒 Stage 8 Checkpoint (Final)

Run: `uv run pytest -v --tb=long`
Expected: ALL PASS — clean codebase, no dead code.

This completes the LangChain/LangGraph refactor. 🎉
