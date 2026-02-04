# Replace Smart Router with LangGraph

## Summary

Replace the custom `SmartRouter` class with a LangGraph-based routing graph, adding LangSmith observability while maintaining the same routing logic.

## Current State

**Files to modify:**
- `src/nlm_proxy/openai/router.py` - Main router logic (166 lines)
- `src/nlm_proxy/core/llm_client.py` - LLM client (will be replaced by LangChain's ChatOpenAI)
- `src/nlm_proxy/openai/server.py` - Server integration point

**Current flow:**
```
Query → classify_request() → [NOTEBOOKLM] → select_notebook() → notebook_id
                           → [LLM_TASK] → passthrough to LLM
```

## Proposed Architecture

### LangGraph State Machine

```
┌─────────────────┐
│   START         │
└────────┬────────┘
         ▼
┌─────────────────┐
│  classify_node  │  ← Calls LLM to classify query
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
[notebooklm] [llm_task]
    │         │
    ▼         ▼
┌─────────┐ ┌─────────────┐
│ select  │ │ passthrough │
│ notebook│ │    END      │
└────┬────┘ └─────────────┘
     ▼
┌─────────────────┐
│      END        │
└─────────────────┘
```

### New Dependencies

```toml
# pyproject.toml
langgraph = "^0.3"
langchain-openai = "^0.3"
langsmith = "^0.3"  # For tracing
```

### Key Components

**1. State Definition (`router.py`)**
```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class RouterState(TypedDict):
    query: str
    request_type: Literal["notebooklm", "llm_task"] | None
    notebook_id: str | None
    reasoning: str
```

**2. Node Functions**
- `classify_node(state)` - Uses ChatOpenAI to classify, updates `state.request_type`
- `select_notebook_node(state)` - Uses cached notebooks + LLM, updates `state.notebook_id`

**3. Conditional Edge**
```python
def route_after_classify(state: RouterState) -> str:
    return state["request_type"]  # "notebooklm" or "llm_task"
```

**4. Graph Assembly**
```python
graph = StateGraph(RouterState)
graph.add_node("classify", classify_node)
graph.add_node("select_notebook", select_notebook_node)
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_after_classify, {
    "notebooklm": "select_notebook",
    "llm_task": END
})
graph.add_edge("select_notebook", END)
router = graph.compile()
```

### LangSmith Integration

```python
# Environment variables for tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=nlm-proxy-routing
```

Benefits:
- Every node execution is traced
- See LLM inputs/outputs for each step
- Debug routing decisions visually
- Track latency per node

## Implementation Steps

1. **Add dependencies** to `pyproject.toml`
2. **Create `src/nlm_proxy/openai/langgraph_router.py`** with:
   - `RouterState` TypedDict
   - `classify_node()` function
   - `select_notebook_node()` function
   - Graph construction and compilation
3. **Update `router.py`** to use the new graph (or replace entirely)
4. **Remove `ExternalLLMClient`** usage, use `ChatOpenAI` from langchain-openai
5. **Update `server.py`** to instantiate new router
6. **Add LangSmith config** to `.env.example` and docs

## Verification

1. Run existing tests: `uv run pytest tests/`
2. Manual test with OpenAI proxy:
   ```bash
   nlm-proxy serve openai --debug
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "knowledge-finder", "messages": [{"role": "user", "content": "What does the attention paper say?"}]}'
   ```
3. Check LangSmith dashboard for trace visibility
4. Verify routing decisions match previous behavior

## Trade-offs

**Pros:**
- Graph-based architecture is more extensible (easy to add synthesis node later)
- Built-in tracing with LangSmith
- Less custom code to maintain
- Industry-standard patterns

**Cons:**
- Adds ~3 new dependencies
- Slight learning curve for LangGraph concepts
- LangSmith requires account (free tier available)
