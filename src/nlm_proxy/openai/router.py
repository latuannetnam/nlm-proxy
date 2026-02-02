"""Smart request router using LLM classification."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from nlm_proxy.core.logging import get_logger
from nlm_proxy.core.llm_client import ExternalLLMClient
from nlm_proxy.openai.notebook_cache import NotebookCache
from nlm_proxy.openai.prompts import load_prompt

if TYPE_CHECKING:
    from nlm_proxy.core import NotebookLMClient

logger = get_logger(__name__)


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
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        allowed_notebooks: list[str] | None = None,
        cache_ttl: int = 3600
    ):
        self.nlm_client = nlm_client
        self.llm_client = ExternalLLMClient(llm_base_url, llm_api_key, llm_model)
        self.notebook_cache = NotebookCache(ttl_seconds=cache_ttl)
        self.allowed_notebooks = allowed_notebooks or []

    async def _ensure_notebooks_cached(self) -> list:
        """Ensure notebook summaries are cached, refresh if needed."""
        cached = self.notebook_cache.get_all()
        if cached:
            logger.debug(f"[ROUTER] Using {len(cached)} cached notebooks")
            return cached

        logger.debug("[ROUTER] Cache empty, fetching notebooks from NotebookLM")
        notebooks = await self.nlm_client.list_notebooks()
        logger.debug(f"[ROUTER] Found {len(notebooks)} notebooks")

        # Filter if configured
        if self.allowed_notebooks:
            notebooks = [nb for nb in notebooks if nb.id in self.allowed_notebooks]
            logger.debug(f"[ROUTER] Filtered to {len(notebooks)} allowed notebooks")

        # Get summaries for each notebook
        for nb in notebooks:
            try:
                logger.debug(f"[ROUTER] Fetching summary for notebook: {nb.title} ({nb.id})")
                summary_data = await self.nlm_client.get_notebook_summary(nb.id)
                self.notebook_cache.set(
                    notebook_id=nb.id,
                    title=nb.title,
                    summary=summary_data.get("summary", ""),
                    topics=summary_data.get("suggested_topics", [])
                )
            except Exception as e:
                logger.warning(f"[ROUTER] Failed to get summary for notebook {nb.id}: {e}")
                # Cache with just the title
                self.notebook_cache.set(
                    notebook_id=nb.id,
                    title=nb.title,
                    summary="",
                    topics=[]
                )

        return self.notebook_cache.get_all()

    async def classify_request(self, query: str) -> RequestType:
        """Classify the request type using external LLM."""
        logger.debug(f"[ROUTER] Classifying request: {query[:100]}...")
        prompt_template = load_prompt("classify_request")
        prompt = prompt_template.format(query=query)

        response = await self.llm_client.complete(prompt)
        response_lower = response.lower().strip()

        if "notebooklm" in response_lower:
            logger.info(f"[ROUTER] Classified as NOTEBOOKLM query")
            return RequestType.NOTEBOOKLM
        logger.info(f"[ROUTER] Classified as LLM_TASK")
        return RequestType.LLM_TASK

    async def select_notebook(self, query: str) -> tuple[str | None, str]:
        """Select best notebook for query. Returns (notebook_id, reasoning)."""
        logger.debug(f"[ROUTER] Selecting notebook for query: {query[:100]}...")
        notebooks = await self._ensure_notebooks_cached()

        if not notebooks:
            logger.warning("[ROUTER] No notebooks available for selection")
            return None, "No notebooks available"

        # Build notebook info for LLM
        notebooks_info = [
            {
                "id": nb.id,
                "title": nb.title,
                "summary": nb.summary[:500] if nb.summary else "",
                "topics": nb.topics[:5] if nb.topics else []
            }
            for nb in notebooks
        ]

        prompt_template = load_prompt("select_notebook")
        prompt = prompt_template.format(
            notebooks_json=json.dumps(notebooks_info, indent=2),
            query=query
        )

        logger.debug(f"[ROUTER] Asking LLM to select from {len(notebooks)} notebooks")
        response = await self.llm_client.complete(prompt, max_tokens=100)

        # Parse response - expect notebook_id
        for nb in notebooks:
            if nb.id in response:
                reasoning = f"Selected notebook: {nb.title} (ID: {nb.id})"
                logger.info(f"[ROUTER] {reasoning}")
                return nb.id, reasoning

        # Fallback to first notebook
        if notebooks:
            reasoning = f"Defaulted to notebook: {notebooks[0].title} (ID: {notebooks[0].id})"
            logger.info(f"[ROUTER] {reasoning}")
            return notebooks[0].id, reasoning

        return None, "No suitable notebook found"

    async def route(self, query: str) -> RoutingDecision:
        """Classify and route the request."""
        logger.info(f"[ROUTER] Starting routing for query: {query[:50]}...")
        request_type = await self.classify_request(query)

        if request_type == RequestType.LLM_TASK:
            logger.info("[ROUTER] Routing to external LLM")
            return RoutingDecision(
                request_type=RequestType.LLM_TASK,
                reasoning="Classified as LLM task (not a notebook query)"
            )

        notebook_id, reasoning = await self.select_notebook(query)
        logger.info(f"[ROUTER] Routing to NotebookLM: {notebook_id}")
        return RoutingDecision(
            request_type=RequestType.NOTEBOOKLM,
            notebook_id=notebook_id,
            reasoning=reasoning
        )

    async def close(self) -> None:
        """Cleanup resources."""
        logger.debug("[ROUTER] Closing router resources")
        await self.llm_client.close()
