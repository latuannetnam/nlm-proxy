# Cross-Notebook Query Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Enable querying multiple notebooks in parallel, streaming the primary response immediately while buffering secondary notebooks for synthesis.

**Architecture:**
- **Router:** Add `route_multi` to select 1 primary + N secondary notebooks in one LLM call.
- **Concurrency:** Global semaphore (`asyncio.Semaphore`) to limit background NotebookLM queries.
- **Streaming:** Stream primary immediately. Run secondary in background.
- **Synthesis:** If secondary results are valuable, synthesize and append to stream.

**Tech Stack:** Python 3.12, asyncio, OpenAI SDK (for router/synthesis), Pydantic/Dataclasses.

---

### Task 1: Configuration & Environment

**Files:**
- Modify: `src/nlm_proxy/core/config.py`
- Test: `tests/core/test_config.py` (create if missing or add to existing)

**Step 1: Write test for new config settings**

```python
# tests/core/test_config.py
from nlm_proxy.core.config import SmartRoutingSettings

def test_cross_notebook_settings_defaults():
    settings = SmartRoutingSettings()
    assert settings.cross_notebook_enabled is True
    assert settings.cross_notebook_max_secondary == 2
    assert settings.cross_notebook_concurrency == 5
    assert "Cross-referenced" in settings.cross_notebook_section_marker

def test_cross_notebook_env_vars(monkeypatch):
    monkeypatch.setenv("NLM_PROXY_ROUTING_CROSS_NOTEBOOK_ENABLED", "false")
    monkeypatch.setenv("NLM_PROXY_ROUTING_CROSS_NOTEBOOK_CONCURRENCY", "10")

    settings = SmartRoutingSettings()
    assert settings.cross_notebook_enabled is False
    assert settings.cross_notebook_concurrency == 10
```

**Step 2: Run test (FAIL)**

Run: `uv run pytest tests/core/test_config.py -v`
Expected: Fail (attributes missing)

**Step 3: Update `SmartRoutingSettings`**

Modify `src/nlm_proxy/core/config.py`:
- Add fields to `SmartRoutingSettings`:
  - `cross_notebook_enabled: bool = True`
  - `cross_notebook_max_secondary: int = 2`
  - `cross_notebook_concurrency: int = 5`
  - `cross_notebook_synthesis_enabled: bool = True`
  - `cross_notebook_section_marker: str = "\n\n---\n\n📚 **Cross-referenced from other sources:**\n\n"`

**Step 4: Run test (PASS)**

Run: `uv run pytest tests/core/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/core/config.py tests/core/test_config.py
git commit -m "feat: add cross-notebook query configuration settings"
```

---

### Task 2: Router Data Models

**Files:**
- Modify: `src/nlm_proxy/openai/router.py`
- Test: `tests/openai/test_router_models.py` (new)

**Step 1: Write test for new data models**

```python
# tests/openai/test_router_models.py
from nlm_proxy.openai.router import NotebookRole, NotebookSelection, MultiRoutingDecision, RequestType

def test_multi_routing_decision_properties():
    primary = NotebookSelection(notebook_id="1", role=NotebookRole.PRIMARY, reasoning="main")
    secondary1 = NotebookSelection(notebook_id="2", role=NotebookRole.SECONDARY, reasoning="supp1")
    secondary2 = NotebookSelection(notebook_id="3", role=NotebookRole.SECONDARY, reasoning="supp2")

    decision = MultiRoutingDecision(
        request_type=RequestType.NOTEBOOKLM,
        notebooks=[secondary1, primary, secondary2],
        reasoning="complex query"
    )

    assert decision.primary_notebook == primary
    assert len(decision.secondary_notebooks) == 2
    assert secondary1 in decision.secondary_notebooks
    assert secondary2 in decision.secondary_notebooks

def test_multi_routing_no_primary():
    decision = MultiRoutingDecision(
        request_type=RequestType.LLM_TASK,
        notebooks=[],
        reasoning="general task"
    )
    assert decision.primary_notebook is None
    assert decision.secondary_notebooks == []
```

**Step 2: Run test (FAIL)**

Run: `uv run pytest tests/openai/test_router_models.py -v`
Expected: Fail (ImportError)

**Step 3: Implement Data Models**

Modify `src/nlm_proxy/openai/router.py`:
- Import `Enum` and `dataclass`.
- Add `NotebookRole(Enum)`.
- Add `NotebookSelection(dataclass)`.
- Add `MultiRoutingDecision(dataclass)` with properties `primary_notebook` and `secondary_notebooks`.

**Step 4: Run test (PASS)**

Run: `uv run pytest tests/openai/test_router_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/router.py tests/openai/test_router_models.py
git commit -m "feat: add data models for multi-notebook routing"
```

---

### Task 3: Prompts

**Files:**
- Create: `src/nlm_proxy/openai/prompts/route_and_select.txt`
- Create: `src/nlm_proxy/openai/prompts/synthesize_cross_notebook.txt`

**Step 1: Create `route_and_select.txt`**

Content:
```text
You are an expert router for a knowledge system.
Your goal is to classify the user's request and select the most relevant notebooks.

AVAILABLE NOTEBOOKS:
{notebook_list}

USER QUERY:
{query}

INSTRUCTIONS:
1. Classify the request type:
   - "notebooklm": Requires specific knowledge from the notebooks.
   - "llm_task": General task, coding, or chit-chat not requiring notebook knowledge.

2. If "notebooklm", select relevant notebooks:
   - Select ONE "primary" notebook that is most likely to contain the core answer.
   - Select up to {max_secondary} "secondary" notebooks that might have supplementary info.
   - If no notebooks are relevant, switch type to "llm_task".

3. Respond in JSON format:
{{
  "type": "notebooklm" | "llm_task",
  "reasoning": "Brief explanation of your decision",
  "notebooks": [
    {{
      "id": "UUID",
      "role": "primary",
      "reason": "Why this is the best primary source",
      "title": "Title"
    }},
    {{
      "id": "UUID",
      "role": "secondary",
      "reason": "What supplementary info this adds",
      "title": "Title"
    }}
  ]
}}
```

**Step 2: Create `synthesize_cross_notebook.txt`**

Content:
```text
You are a research assistant synthesizing information from multiple sources.

USER QUERY:
{query}

PRIMARY ANSWER (already shown to user):
{primary_answer_start}...

SECONDARY SOURCE FINDINGS:
{secondary_results}

INSTRUCTIONS:
1. Determine if the secondary findings add SIGNIFICANT value or new perspectives to the primary answer.
2. If YES:
   - Write a concise synthesis (2-4 sentences).
   - Focus ONLY on the new information.
   - Do NOT repeat what was likely in the primary answer.
   - **DO NOT include citation numbers like [1], [2].**
   - **Answer in under 300 words.**
3. If NO (secondary findings are irrelevant, redundant, or empty):
   - Return exactly: NO_SYNTHESIS_NEEDED

OUTPUT:
```

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/prompts/
git commit -m "feat: add prompts for multi-routing and synthesis"
```

---

### Task 4: Router Logic (`route_multi`)

**Files:**
- Modify: `src/nlm_proxy/openai/router.py`
- Test: `tests/openai/test_router_multi.py`

**Step 1: Write test for `route_multi`**

```python
# tests/openai/test_router_multi.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from nlm_proxy.openai.router import SmartRouter, RequestType, NotebookRole

@pytest.mark.asyncio
async def test_route_multi_notebooklm():
    mock_llm_client = AsyncMock()
    mock_llm_client.chat_completion.return_value = """
    {
        "type": "notebooklm",
        "reasoning": "test",
        "notebooks": [
            {"id": "nb1", "role": "primary", "reason": "main", "title": "NB1"},
            {"id": "nb2", "role": "secondary", "reason": "supp", "title": "NB2"}
        ]
    }
    """

    router = SmartRouter(
        llm_client=mock_llm_client,
        settings=MagicMock(cross_notebook_max_secondary=2)
    )
    router.notebook_cache = MagicMock()
    router.notebook_cache.get_all.return_value = [] # Content mocked in prompt

    decision = await router.route_multi("query")

    assert decision.request_type == RequestType.NOTEBOOKLM
    assert decision.primary_notebook.notebook_id == "nb1"
    assert len(decision.secondary_notebooks) == 1
    assert decision.secondary_notebooks[0].notebook_id == "nb2"

@pytest.mark.asyncio
async def test_route_multi_fallback_on_json_error():
    mock_llm_client = AsyncMock()
    mock_llm_client.chat_completion.return_value = "INVALID JSON"

    router = SmartRouter(mock_llm_client, MagicMock())
    router.notebook_cache = MagicMock()
    router.notebook_cache.get_all.return_value = []

    # Should fall back to LLM_TASK or safe default
    decision = await router.route_multi("query")
    assert decision.request_type == RequestType.LLM_TASK
```

**Step 2: Run test (FAIL)**

Run: `uv run pytest tests/openai/test_router_multi.py -v`
Expected: Fail (method missing)

**Step 3: Implement `route_multi`**

Modify `src/nlm_proxy/openai/router.py`:
- Add `route_multi(query: str) -> MultiRoutingDecision`.
- Load `route_and_select` prompt.
- Format notebook list from cache.
- Call LLM with `json_object` response format (if supported) or just prompt.
- Parse JSON.
- Handle JSONDecodeError -> return `LLM_TASK`.

**Step 4: Run test (PASS)**

Run: `uv run pytest tests/openai/test_router_multi.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/router.py tests/openai/test_router_multi.py
git commit -m "feat: implement route_multi method in SmartRouter"
```

---

### Task 5: Streaming Logic (The Big One)

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`
- Test: `tests/openai/test_server_streaming.py` (mock heavy)

**Step 1: Write test for streaming flow**

```python
# tests/openai/test_server_streaming.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nlm_proxy.openai.server import stream_cross_notebook_response
from nlm_proxy.openai.router import MultiRoutingDecision, NotebookSelection, NotebookRole, RequestType

@pytest.mark.asyncio
async def test_stream_cross_notebook_flow():
    # Setup mocks
    mock_decision = MultiRoutingDecision(
        request_type=RequestType.NOTEBOOKLM,
        notebooks=[
            NotebookSelection("nb1", NotebookRole.PRIMARY, "r1", "T1"),
            NotebookSelection("nb2", NotebookRole.SECONDARY, "r2", "T2")
        ],
        reasoning="reasoning"
    )

    mock_client = AsyncMock()
    # Primary stream
    async def primary_gen():
        yield "Primary Answer"
    mock_client.query_stream.return_value = primary_gen()

    # Secondary query
    mock_client.query.return_value = "Secondary Content"

    mock_settings = MagicMock()
    mock_settings.cross_notebook_section_marker = "\n---\n"

    # Run
    chunks = []
    async for chunk in stream_cross_notebook_response(
        mock_client, mock_decision, "query", AsyncMock(), mock_settings
    ):
        chunks.append(chunk)

    # Verify
    # 1. Reasoning
    assert any("reasoning_content" in c and "reasoning" in c["reasoning_content"] for c in chunks if isinstance(c, dict))
    # 2. Primary Content
    assert any("content" in c and "Primary Answer" in c["content"] for c in chunks if "content" in c)
```

**Step 2: Run test (FAIL)**

Run: `uv run pytest tests/openai/test_server_streaming.py -v`
Expected: Fail (function missing)

**Step 3: Implement `stream_cross_notebook_response`**

Modify `src/nlm_proxy/openai/server.py`:
- Initialize **Global Semaphore** (module level or app state).
- Define `stream_cross_notebook_response`.
- **Primary:** Stream immediately using `_query_notebook_stream`.
- **Secondary:**
  - Launch `asyncio.create_task(_query_notebook_buffered(...))` for each secondary.
  - `_query_notebook_buffered`: Acquire semaphore. Query `client.query` (non-streaming) with appended prompt "Answer in under 300 words". Handle exceptions (return None).
- **Synthesis:**
  - Await secondary tasks.
  - If any results: call `_synthesize_cross_notebook`.
  - If synthesis returns content (not NO_SYNTHESIS_NEEDED), yield section marker + synthesis.

**Step 4: Run test (PASS)**

Run: `uv run pytest tests/openai/test_server_streaming.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nlm_proxy/openai/server.py tests/openai/test_server_streaming.py
git commit -m "feat: implement stream_cross_notebook_response with parallel secondary queries"
```

---

### Task 6: Wiring It Up

**Files:**
- Modify: `src/nlm_proxy/openai/server.py`

**Step 1: Update `handle_smart_routing`**

Modify `src/nlm_proxy/openai/server.py`:
- In `handle_smart_routing`:
- Check `settings.cross_notebook_enabled`.
- If True:
  - Call `router.route_multi()`.
  - Return `StreamingResponse(stream_cross_notebook_response(...))`.
- Else:
  - Keep existing `route()` path.

**Step 2: Manual Verification (End-to-End)**

Run: `nlm-proxy serve openai --port 8080`
- Curl Request 1: Single notebook query (check simple flow).
- Curl Request 2: "Compare X and Y" (check multi flow).
- Check logs for `[CROSS-NOTEBOOK]` tags.

**Step 3: Commit**

```bash
git add src/nlm_proxy/openai/server.py
git commit -m "feat: wire up cross-notebook routing in request handler"
```
