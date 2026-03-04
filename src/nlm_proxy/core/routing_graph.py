"""LangGraph-based routing graph for smart request classification.

Replaces the linear SmartRouter with a LangGraph StateGraph that:
1. Classifies intent (NOTEBOOKLM vs LLM_TASK)
2. Selects notebook (if NOTEBOOKLM) with ACL filtering

The graph produces a routing DECISION only — it does NOT execute queries
or handle streaming. Those are done by the transport layer.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.tracing import record_span, add_span_attributes
from nlm_proxy.openai.prompts import load_prompt

logger = get_logger(__name__)

# Re-export END as a constant for tests
END_NODE = END

DEFAULT_MAX_SOURCE_TITLES = 15


# --- LangGraph State ---

class RouterState(TypedDict):
    """Internal state for the routing graph."""
    query: str
    messages: list                   # Conversation history (reserved for LangGraph memory)
    request_type: str | None        # "notebooklm" | "llm_task"
    # NOTE: Design uses `notebook_ids: list[str]` for cross-notebook future.
    # Current impl uses singular `notebook_id` since cross-notebook is deferred.
    notebook_id: str | None          # Selected notebook UUID
    reasoning: str                   # Human-readable explanation
    available_notebooks: list[dict]  # Populated by nodes
    allowed_notebooks: list[str] | None  # Per-request ACL filter


# --- Graph Nodes ---

@record_span("smart_router.classify")
async def classify_node(
    state: RouterState,
    *,
    chat_model,
    notebook_cache,
) -> dict:
    """Classify the request as NOTEBOOKLM or LLM_TASK using LLM."""
    query = state["query"]
    logger.debug("[ROUTER] Classifying: %s...", query[:100])

    prompt_template = load_prompt("classify_request")
    prompt = prompt_template.format(query=query)

    response = await chat_model.ainvoke([HumanMessage(content=prompt)])
    response_lower = response.content.lower().strip()

    if "notebooklm" in response_lower:
        logger.info("[ROUTER] Classified as NOTEBOOKLM")
        add_span_attributes(classification_result="NOTEBOOKLM")
        return {"request_type": "notebooklm"}

    logger.info("[ROUTER] Classified as LLM_TASK")
    add_span_attributes(classification_result="LLM_TASK")
    return {
        "request_type": "llm_task",
        "reasoning": "Classified as LLM task (not a notebook query)",
    }


@record_span("smart_router.select_notebook")
async def select_notebook_node(
    state: RouterState,
    *,
    chat_model,
    notebook_cache,
    routing_settings=None,
) -> dict:
    """Select the best notebook for the query, respecting ACL filters."""
    query = state["query"]
    allowed = state.get("allowed_notebooks")

    logger.debug("[ROUTER] Selecting notebook for: %s...", query[:100])

    # Get all cached notebooks
    notebooks = notebook_cache.get_all()
    if not notebooks:
        logger.warning("[ROUTER] No notebooks available")
        add_span_attributes(candidates_count=0)
        return {"notebook_id": None, "reasoning": "No notebooks available"}

    # Apply per-request ACL filtering
    if allowed is not None:
        notebooks = [nb for nb in notebooks if nb.id in allowed]
        add_span_attributes(
            acl_filter_applied=True,
            acl_allowed_count=len(allowed),
            acl_matched_count=len(notebooks),
        )
        if not notebooks:
            logger.warning("[ROUTER] ACL filter matched no notebooks")
            add_span_attributes(candidates_count=0)
            return {
                "notebook_id": None,
                "reasoning": "No accessible notebooks for this user",
            }
    else:
        add_span_attributes(acl_filter_applied=False)

    add_span_attributes(candidates_count=len(notebooks))

    # Build notebook info for LLM prompt
    if routing_settings is None:
        from nlm_proxy.core.config import get_routing_settings
        routing_settings = get_routing_settings()
    max_source_titles = routing_settings.max_source_titles
    source_descriptions_enabled = routing_settings.source_descriptions_enabled
    source_max_keywords = routing_settings.source_max_keywords
    source_summary_max_chars = routing_settings.source_summary_max_chars
    source_descriptions_max_sources = routing_settings.source_descriptions_max_sources

    notebooks_info = []
    for nb in notebooks:
        info: dict = {
            "id": nb.id,
            "title": nb.title,
            "summary": nb.summary[:500] if nb.summary else "",
            "topics": nb.topics[:5] if nb.topics else [],
            "source_count": nb.source_count,
            "source_types": nb.source_types,
        }
        if source_descriptions_enabled:
            info["sources"] = nb.get_source_descriptions(
                max_sources=source_descriptions_max_sources,
                max_keywords=source_max_keywords,
                summary_max_chars=source_summary_max_chars,
            )[:max_source_titles]
        else:
            info["source_titles"] = nb.source_titles[:max_source_titles]
        notebooks_info.append(info)

    # Call LLM to select notebook
    prompt_template = load_prompt("select_notebook")
    prompt = prompt_template.format(
        notebooks_json=json.dumps(notebooks_info, indent=2),
        query=query,
    )

    logger.debug("[ROUTER] Asking LLM to select from %d notebooks", len(notebooks))
    response = await chat_model.ainvoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    # Parse response — expect notebook_id in the response
    for nb in notebooks:
        if nb.id in response_text:
            reasoning = f"Selected notebook: {nb.title} (ID: {nb.id})"
            logger.info("[ROUTER] %s", reasoning)
            add_span_attributes(
                selected_notebook_id=nb.id,
                selected_notebook_title=nb.title,
            )
            return {"notebook_id": nb.id, "reasoning": reasoning}

    # Fallback to first notebook
    if notebooks:
        reasoning = f"Defaulted to notebook: {notebooks[0].title} (ID: {notebooks[0].id})"
        logger.info("[ROUTER] %s", reasoning)
        add_span_attributes(
            selected_notebook_id=notebooks[0].id,
            selected_notebook_title=notebooks[0].title,
            selection_fallback=True,
        )
        return {"notebook_id": notebooks[0].id, "reasoning": reasoning}

    return {"notebook_id": None, "reasoning": "No suitable notebook found"}


# --- Conditional Edge ---

def route_after_classify(state: RouterState) -> str:
    """Route to select_notebook or END based on classification."""
    if state.get("request_type") == "notebooklm":
        return "select_notebook"
    return END


# --- Graph Builder ---

def build_routing_graph(chat_model, notebook_cache, routing_settings=None):
    """Build and compile the LangGraph routing state graph.

    Args:
        chat_model: LangChain ChatModel for LLM calls
        notebook_cache: NotebookCache with cached notebook summaries
        routing_settings: SmartRoutingSettings for notebook display config

    Returns:
        Compiled LangGraph that accepts {"query": str, "allowed_notebooks": ...}
        and returns RouterState with request_type, notebook_id, reasoning.
    """
    # Bind dependencies to node functions via closures
    async def _classify(state):
        return await classify_node(
            state, chat_model=chat_model, notebook_cache=notebook_cache
        )

    async def _select(state):
        return await select_notebook_node(
            state, chat_model=chat_model, notebook_cache=notebook_cache,
            routing_settings=routing_settings,
        )

    graph = StateGraph(RouterState)
    graph.add_node("classify", _classify)
    graph.add_node("select_notebook", _select)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_edge("select_notebook", END)

    return graph.compile()
