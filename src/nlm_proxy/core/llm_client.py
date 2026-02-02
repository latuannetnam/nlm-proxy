"""External LLM client for OpenAI-compatible endpoints.

This module is in core/ for reuse across multiple features:
- Smart routing (openai proxy)
- Future MCP tools
- Any feature needing external LLM calls

Uses the official OpenAI SDK for better compatibility and maintainability.
"""

from typing import AsyncIterator

from openai import AsyncOpenAI

from nlm_proxy.core.logging import get_logger

logger = get_logger(__name__)


class ExternalLLMClient:
    """Client for calling external OpenAI-compatible LLM using OpenAI SDK."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client (lazy initialization)."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=30.0
            )
        return self._client

    async def complete(self, prompt: str, max_tokens: int = 50) -> str:
        """Get a simple completion (non-streaming)."""
        logger.debug(f"[LLM] Calling complete: model={self.model}, max_tokens={max_tokens}")
        logger.debug(f"[LLM] Request prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()

        logger.debug(f"[LLM] Response: {result[:200]}{'...' if len(result) > 200 else ''}")
        return result

    async def stream(self, messages: list[dict]) -> AsyncIterator:
        """Stream a completion for LLM task passthrough.

        Returns an async iterator that yields ChatCompletionChunk objects.
        Each chunk has: chunk.choices[0].delta.content
        """
        logger.debug(f"[LLM] Starting stream: model={self.model}, messages={len(messages)}")
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        logger.debug("[LLM] Stream started")
        return stream

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            logger.debug("[LLM] Closing client")
            await self._client.close()
            self._client = None
