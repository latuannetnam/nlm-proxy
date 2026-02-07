"""Smart request router using LLM classification."""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.llm_client import ExternalLLMClient
from nlm_proxy.core.tracing import record_span, add_span_attributes
from nlm_proxy.openai.notebook_cache import NotebookCache
from nlm_proxy.openai.prompts import load_prompt

if TYPE_CHECKING:
    from nlm_proxy.core import NotebookLMClient

logger = get_logger(__name__)

# Default max source titles to include in selection prompt
DEFAULT_MAX_SOURCE_TITLES = 15


class RequestType(Enum):
    """Type of request after classification."""
    NOTEBOOKLM = "notebooklm"
    LLM_TASK = "llm_task"


@dataclass
class RoutingDecision:
    """Result of request classification and routing."""
    request_type: RequestType
    notebook_id: str | None = None
    reasoning: str = ""


class SmartRouter:
    """Classifies requests and routes to appropriate backend."""

    def __init__(
        self,
        nlm_client: "NotebookLMClient",
        notebook_cache: NotebookCache,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        allowed_notebooks: list[str] | None = None
    ):
        """Initialize the router with a shared notebook cache.

        Args:
            nlm_client: NotebookLM client for queries
            notebook_cache: Shared cache with proactive refresh (from app.state)
            llm_base_url: Base URL for external LLM
            llm_api_key: API key for external LLM
            llm_model: Model name for external LLM
            allowed_notebooks: Optional list of allowed notebook IDs
        """
        self.nlm_client = nlm_client
        self.notebook_cache = notebook_cache
        self.llm_client = ExternalLLMClient(llm_base_url, llm_api_key, llm_model)
        self.allowed_notebooks = allowed_notebooks or []

    async def _ensure_notebooks_cached(self) -> list:
        """Get cached notebooks - cache is always warm due to proactive refresh."""
        cached = self.notebook_cache.get_all()
        if cached:
            logger.debug(f"[ROUTER] Using {len(cached)} cached notebooks")
            return cached

        # This should rarely happen since cache is proactively refreshed
        logger.warning("[ROUTER] Cache unexpectedly empty - notebooks may not be available")
        return []

    @record_span("smart_router.classify")
    async def classify_request(self, query: str) -> RequestType:
        """Classify the request type using external LLM."""
        logger.debug(f"[ROUTER] Classifying request: {query[:100]}...")
        prompt_template = load_prompt("classify_request")
        prompt = prompt_template.format(query=query)

        response = await self.llm_client.complete(prompt)
        response_lower = response.lower().strip()

        if "notebooklm" in response_lower:
            logger.info(f"[ROUTER] Classified as NOTEBOOKLM query")
            add_span_attributes(
                classification_result="NOTEBOOKLM",
                llm_model=self.llm_client.model
            )
            return RequestType.NOTEBOOKLM
        logger.info(f"[ROUTER] Classified as LLM_TASK")
        add_span_attributes(
            classification_result="LLM_TASK",
            llm_model=self.llm_client.model
        )
        return RequestType.LLM_TASK

    @record_span("smart_router.select_notebook")
    async def select_notebook(self, query: str) -> tuple[str | None, str]:
        """Select best notebook for query. Returns (notebook_id, reasoning)."""
        logger.debug(f"[ROUTER] Selecting notebook for query: {query[:100]}...")
        notebooks = await self._ensure_notebooks_cached()

        if not notebooks:
            logger.warning("[ROUTER] No notebooks available for selection")
            add_span_attributes(candidates_count=0)
            return None, "No notebooks available"

        add_span_attributes(candidates_count=len(notebooks))

        # Get configuration from environment
        max_source_titles = int(
            os.environ.get("NLM_PROXY_ROUTING_MAX_SOURCE_TITLES", DEFAULT_MAX_SOURCE_TITLES)
        )
        source_descriptions_enabled = os.environ.get(
            "NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
        source_max_keywords = int(
            os.environ.get("NLM_PROXY_ROUTING_SOURCE_MAX_KEYWORDS", "5")
        )
        source_summary_max_chars = int(
            os.environ.get("NLM_PROXY_ROUTING_SOURCE_SUMMARY_MAX_CHARS", "80")
        )
        source_descriptions_max_sources = int(
            os.environ.get("NLM_PROXY_ROUTING_SOURCE_DESCRIPTIONS_MAX_SOURCES", "10")
        )

        # Build notebook info for LLM with source-level information
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
                # Include full source descriptions with keywords and summaries
                info["sources"] = nb.get_source_descriptions(
                    max_sources=source_descriptions_max_sources,
                    max_keywords=source_max_keywords,
                    summary_max_chars=source_summary_max_chars
                )[:max_source_titles]
            else:
                # Fallback to title-only mode
                info["source_titles"] = nb.source_titles[:max_source_titles]

            notebooks_info.append(info)

            # Log notebook candidate summary
            sources_key = "sources" if "sources" in info else "source_titles"
            sources_data = info.get(sources_key, [])

            # Extract top keywords from sources (if available)
            top_keywords = []
            if sources_key == "sources":
                for src in sources_data[:3]:
                    top_keywords.extend(src.get("keywords", [])[:2])

            logger.debug(
                f"[ROUTER] Candidate: {info['title']} | "
                f"sources={info['source_count']} | "
                f"types={info['source_types']} | "
                f"keywords={top_keywords[:5]}"
            )

        prompt_template = load_prompt("select_notebook")
        prompt = prompt_template.format(
            notebooks_json=json.dumps(notebooks_info, indent=2),
            query=query
        )

        # Log prompt size metrics
        prompt_chars = len(prompt)
        logger.debug(f"[ROUTER] Prompt size: {prompt_chars} chars (~{prompt_chars // 4} tokens)")
        logger.debug(f"[ROUTER] Selection prompt:\n{prompt}")

        logger.debug(f"[ROUTER] Asking LLM to select from {len(notebooks)} notebooks")
        response = await self.llm_client.complete(prompt, max_tokens=100)

        # Parse response - expect notebook_id
        for nb in notebooks:
            if nb.id in response:
                reasoning = f"Selected notebook: {nb.title} (ID: {nb.id})"
                logger.info(f"[ROUTER] {reasoning}")
                add_span_attributes(
                    selected_notebook_id=nb.id,
                    selected_notebook_title=nb.title
                )
                return nb.id, reasoning

        # Fallback to first notebook
        if notebooks:
            reasoning = f"Defaulted to notebook: {notebooks[0].title} (ID: {notebooks[0].id})"
            logger.info(f"[ROUTER] {reasoning}")
            add_span_attributes(
                selected_notebook_id=notebooks[0].id,
                selected_notebook_title=notebooks[0].title,
                selection_fallback=True
            )
            return notebooks[0].id, reasoning

        return None, "No suitable notebook found"

    @record_span("smart_router.route")
    async def route(self, query: str) -> RoutingDecision:
        """Classify and route the request."""
        logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")
        # user_query attribute moved to smart_router.handle_request span

        request_type = await self.classify_request(query)

        if request_type == RequestType.LLM_TASK:
            logger.info("[ROUTER] Routing to external LLM")
            add_span_attributes(
                request_type="LLM_TASK",
                notebook_id=None
            )
            return RoutingDecision(
                request_type=RequestType.LLM_TASK,
                reasoning="Classified as LLM task (not a notebook query)"
            )

        notebook_id, reasoning = await self.select_notebook(query)
        logger.info(f"[ROUTER] Routing to NotebookLM: {notebook_id}")
        add_span_attributes(
            request_type="NOTEBOOKLM",
            notebook_id=notebook_id,
            routing_reasoning=reasoning
        )
        return RoutingDecision(
            request_type=RequestType.NOTEBOOKLM,
            notebook_id=notebook_id,
            reasoning=reasoning
        )

    async def close(self) -> None:
        """Cleanup resources."""
        logger.debug("[ROUTER] Closing router resources")
        await self.llm_client.close()
