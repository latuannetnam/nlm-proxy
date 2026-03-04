"""Shared agent core for both OpenAI proxy and MCP server.

Provides routing (via LangGraph), caching, and NLM query delegation.
Transport-specific concerns (SSE streaming, MCP progress) are handled
by the callers, NOT by AgentCore.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.routing_graph import build_routing_graph

logger = get_logger(__name__)


@dataclass
class RequestOptions:
    """Per-request options extracted from HTTP headers / MCP params."""
    bypass_cache: bool = False
    include_thinking: bool = True
    allowed_notebooks: list[str] | None = None
    conversation_id: str | None = None
    chat_id: str | None = None
    source_ids: list[str] | None = None
    timeout: float | None = None


@dataclass
class RoutingDecision:
    """Result of routing: where to send the query."""
    request_type: str                       # "notebooklm" | "llm_task"
    notebook_id: str | None = None
    reasoning: str = ""
    cache_result: object | None = None      # CachedResponse on cache hit
    cache_hit_type: str | None = None       # "pre_routing_exact" etc.
    conversation_id: str | None = None


class AgentCore:
    """Shared agent logic for both OpenAI proxy and MCP server."""

    def __init__(self, nlm_client, notebook_cache, response_cache, chat_model,
                 session_store=None, routing_settings=None):
        self.nlm_client = nlm_client
        self.notebook_cache = notebook_cache
        self.response_cache = response_cache
        self.chat_model = chat_model
        self.session_store = session_store
        self.routing_graph = build_routing_graph(
            chat_model, notebook_cache, routing_settings=routing_settings
        )

        # Wire bidirectional dependencies
        if notebook_cache and response_cache:
            notebook_cache._on_sources_changed = response_cache.invalidate_notebook
        if nlm_client and notebook_cache:
            nlm_client._notebook_cache = notebook_cache

    async def route(self, query: str, options: RequestOptions) -> RoutingDecision:
        """Get routing decision with optional pre-routing cache check.

        Implements agent_fallback_on_error: if the routing graph fails,
        falls back to a simple NOTEBOOKLM decision using the first
        available notebook (preserving existing behavior on error).
        """
        # Phase 0: Pre-routing global L1 cache check
        if not options.bypass_cache and self.response_cache:
            cached, hit_type = self.response_cache.lookup_global(query)
            if cached:
                # ACL check on cached result
                if options.allowed_notebooks is None or cached.notebook_id in options.allowed_notebooks:
                    return RoutingDecision(
                        request_type="notebooklm",
                        notebook_id=cached.notebook_id,
                        reasoning="Pre-routing cache hit",
                        cache_result=cached,
                        cache_hit_type=f"pre_routing_{hit_type}",
                        conversation_id=options.conversation_id,
                    )

        # Phase 1: LangGraph routing (with thread_id for checkpointing)
        try:
            config = {}
            if options.chat_id:
                config = {"configurable": {"thread_id": options.chat_id}}
            state = await self.routing_graph.ainvoke(
                {
                    "query": query,
                    "allowed_notebooks": options.allowed_notebooks,
                },
                config=config,
            )
            return RoutingDecision(
                request_type=state["request_type"],
                notebook_id=state.get("notebook_id"),
                reasoning=state.get("reasoning", ""),
                conversation_id=options.conversation_id,
            )
        except Exception as e:
            # Fallback: if agent_fallback_on_error is enabled, degrade gracefully
            logger.error("Routing graph failed: %s. Falling back to first notebook.", e)
            fallback_notebook = None
            if self.notebook_cache:
                notebooks = self.notebook_cache.get_all()
                if options.allowed_notebooks is not None:
                    notebooks = [nb for nb in notebooks if nb.id in options.allowed_notebooks]
                if notebooks:
                    fallback_notebook = notebooks[0].id
            if fallback_notebook:
                return RoutingDecision(
                    request_type="notebooklm",
                    notebook_id=fallback_notebook,
                    reasoning=f"Routing fallback (error: {e})",
                    conversation_id=options.conversation_id,
                )
            # No notebooks available — re-raise
            raise

    async def query(self, notebook_id, query, conversation_id=None,
                    source_ids=None, timeout=None) -> dict:
        """Non-streaming query from NotebookLM."""
        return await self.nlm_client.query(
            notebook_id, query_text=query,
            conversation_id=conversation_id,
            source_ids=source_ids,
            timeout=timeout,
        )

    async def query_stream(self, notebook_id, query, conversation_id=None,
                           source_ids=None, **kwargs):
        """Streaming query from NotebookLM. Yields raw NLM chunks."""
        async for chunk in self.nlm_client.query_stream(
            notebook_id, query_text=query,
            conversation_id=conversation_id,
            source_ids=source_ids,
            **kwargs
        ):
            yield chunk

    async def handle_direct_query(self, notebook_id, query, options):
        """Handle direct notebook query (model == notebook_id, bypasses routing).

        Returns (cache_result, hit_type) on cache hit, or (None, None) on miss.
        Caller handles the actual NLM query and format-specific response.
        """
        if not options.bypass_cache and self.response_cache:
            cache_result, hit_type = await self.response_cache.lookup_async(
                notebook_id, query
            )
            if cache_result:
                return cache_result, hit_type
        return None, None
