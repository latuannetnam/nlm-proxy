# Stage 7: MCP Server Unification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire MCP server's `notebook_query` and `notebook_query_stream` tools to use the shared `AgentCore`.

**Architecture:** Add `_agent_core` singleton alongside existing `_client` singleton. Only query tools use `AgentCore` — all CRUD tools continue using `get_client()` directly.

**Inputs:** Stage 6 complete — `AgentCore` is proven in the OpenAI proxy.

**Outputs:** MCP query tools delegate to `AgentCore.query()` / `AgentCore.query_stream()`.

---

## Task 7.1: Add `_agent_core` singleton to MCP server

**Files:**
- Modify: `src/nlm_proxy/mcp/server.py`

**Step 1: Add agent core initialization**

Add alongside existing `_client` singleton:

```python
from nlm_proxy.core.agent import AgentCore, RequestOptions

_agent_core: AgentCore | None = None

async def get_agent_core() -> AgentCore:
    """Get or create the shared AgentCore singleton for MCP query tools."""
    global _agent_core
    if _agent_core is None:
        client = await get_client()

        from nlm_proxy.core.config import get_routing_settings, get_cache_settings, get_agent_settings
        from nlm_proxy.core.llm_client import LangChainLLMClient, create_chat_model

        routing_settings = get_routing_settings()
        agent_settings = get_agent_settings()

        # Only create agent if LLM is configured
        if routing_settings.llm_api_key:
            chat_model = create_chat_model(
                model=routing_settings.llm_model,
                provider=agent_settings.llm_provider,
                base_url=routing_settings.llm_base_url,
                api_key=routing_settings.llm_api_key,
            )

            # Response cache (optional)
            cache_settings = get_cache_settings()
            response_cache = None
            if cache_settings.response_cache_enabled:
                from nlm_proxy.core.response_cache import ResponseCache
                llm_client = LangChainLLMClient(chat_model=chat_model)
                response_cache = ResponseCache(
                    max_entries=cache_settings.response_cache_max_entries,
                    ttl_seconds=cache_settings.response_cache_ttl,
                    semantic_enabled=cache_settings.semantic_match_enabled,
                    llm_client=llm_client,
                    embedding_model=cache_settings.embedding_model,
                    similarity_threshold=cache_settings.similarity_threshold,
                )

            # NotebookCache (optional)
            from nlm_proxy.core.notebook_cache import NotebookCache
            notebook_cache = NotebookCache(
                nlm_client=client,
                ttl_seconds=routing_settings.summary_cache_ttl,
                allowed_notebooks=routing_settings.allowed_notebooks,
                on_sources_changed=response_cache.invalidate_notebook if response_cache else None,
            )

            _agent_core = AgentCore(
                nlm_client=client,
                notebook_cache=notebook_cache,
                response_cache=response_cache,
                chat_model=chat_model,
                routing_settings=routing_settings,
            )
        else:
            # No LLM configured — AgentCore without routing
            _agent_core = AgentCore(
                nlm_client=client,
                notebook_cache=None,
                response_cache=None,
                chat_model=None,
            )

    return _agent_core
```

**Step 2: Commit**

```bash
git add src/nlm_proxy/mcp/server.py
git commit -m "feat: add AgentCore singleton to MCP server"
```

---

## Task 7.2: Update `notebook_query` tool

**Files:**
- Modify: `src/nlm_proxy/mcp/server.py`

Replace the body of `notebook_query` to use `agent.query()`:

```python
@logged_tool()
async def notebook_query(
    notebook_id: str,
    query: str,
    source_ids: list[str] | str | None = None,
    conversation_id: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Ask AI about EXISTING sources already in notebook."""
    try:
        if isinstance(source_ids, str):
            import json
            try:
                source_ids = json.loads(source_ids)
            except json.JSONDecodeError:
                source_ids = [source_ids]

        effective_timeout = timeout if timeout is not None else _query_timeout

        agent = await get_agent_core()
        result = await agent.query(
            notebook_id, query,
            conversation_id=conversation_id,
            source_ids=source_ids,
            timeout=effective_timeout,
        )

        if result:
            return {
                "status": "success",
                "answer": result.get("answer", ""),
                "conversation_id": result.get("conversation_id"),
            }
        return {"status": "error", "error": "Failed to query notebook"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

**Step 1: Commit**

```bash
git add src/nlm_proxy/mcp/server.py
git commit -m "refactor: MCP notebook_query uses AgentCore"
```

---

## Task 7.3: Update `notebook_query_stream` tool

**Files:**
- Modify: `src/nlm_proxy/mcp/server.py`

```python
@logged_tool()
async def notebook_query_stream(
    notebook_id: str,
    query: str,
    source_ids: list[str] | str | None = None,
    conversation_id: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Ask AI with real-time streaming."""
    try:
        if isinstance(source_ids, str):
            import json as json_module
            try:
                source_ids = json_module.loads(source_ids)
            except json_module.JSONDecodeError:
                source_ids = [source_ids]

        agent = await get_agent_core()

        thinking_steps: list[str] = []
        answer_chunks: list[str] = []
        final_conversation_id: str | None = None
        chunk_count = 0

        async for chunk in agent.query_stream(
            notebook_id, query,
            source_ids=source_ids,
            conversation_id=conversation_id,
        ):
            chunk_count += 1
            final_conversation_id = chunk.get("conversation_id")

            if chunk["type"] == "thinking":
                thinking_steps.append(chunk["text"])
                # MCP progress reporting — transport-layer concern, PRESERVED
                if ctx:
                    preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                    await ctx.report_progress(
                        progress=chunk_count,
                        total=chunk_count + 5,
                        message=f"Thinking: {preview}",
                    )
            else:
                answer_chunks.append(chunk["text"])
                if ctx:
                    await ctx.report_progress(
                        progress=chunk_count,
                        total=chunk_count + 1,
                        message=f"Answer:{chunk['text']}",
                    )

        final_answer = max(answer_chunks, key=len) if answer_chunks else ""

        return {
            "status": "success",
            "answer": final_answer,
            "conversation_id": final_conversation_id,
            "thinking_steps": thinking_steps,
            "chunk_count": chunk_count,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

> [!NOTE]
> Only `notebook_query` and `notebook_query_stream` use `AgentCore`. All other MCP tools continue using `get_client()` directly.

**Step 1: Manual verification**

Test with MCP client:
```bash
nlm-proxy serve mcp --debug
# In another terminal, test notebook_query and notebook_query_stream
```

**Step 2: Commit**

```bash
git add src/nlm_proxy/mcp/server.py
git commit -m "refactor: MCP notebook_query_stream uses AgentCore"
```

---

## 🔒 Stage 7 Checkpoint

Run: `uv run pytest -v`
Expected: ALL PASS

Manual verification: test `notebook_query` and `notebook_query_stream` via MCP client.
