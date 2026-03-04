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

    def _uses_max_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens.

        GPT-5.x and reasoning models (o1, o3) require max_completion_tokens.
        Older models (GPT-4.x, GPT-4o, etc.) use max_tokens.
        """
        model_lower = self.model.lower()
        return model_lower.startswith(("gpt-5", "o1", "o3"))

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

        # Use appropriate parameter based on model version
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }

        if self._uses_max_completion_tokens():
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = await self.client.chat.completions.create(**params)
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


# ── LangChain-based LLM Client ──────────────────────────────────────────


def create_chat_model(
    model: str,
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
):
    """Factory to create a LangChain ChatModel for any provider.

    Supports: openai, anthropic, ollama, azure via LangChain provider packages.
    """
    from langchain.chat_models import init_chat_model

    kwargs = {"model": model, "temperature": temperature}
    if provider == "openai":
        kwargs["model_provider"] = "openai"
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
    elif provider == "anthropic":
        kwargs["model_provider"] = "anthropic"
        if api_key:
            kwargs["api_key"] = api_key
    elif provider == "ollama":
        kwargs["model_provider"] = "ollama"
        if base_url:
            kwargs["base_url"] = base_url
    else:
        kwargs["model_provider"] = provider
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

    logger.info(
        "Initializing ChatModel: provider=%s, model=%s", provider, model
    )
    return init_chat_model(**kwargs)


class LangChainLLMClient:
    """Wrapper around LangChain ChatModel with simplified interface.

    Used by:
    - SmartRouter / LangGraph nodes: classify_request(), select_notebook()
    - ResponseCache: L3 semantic verification
    - OpenAI proxy: LLM_TASK passthrough (streaming + non-streaming)
    """

    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def complete(self, prompt: str, max_tokens: int = 50) -> str:
        """Simple completion (non-streaming). Used by L3 cache verification."""
        from langchain_core.messages import HumanMessage

        logger.debug("[LLM] complete: prompt=%s...", prompt[:200])
        response = await self.chat_model.ainvoke([HumanMessage(content=prompt)])
        result = response.content.strip()
        logger.debug("[LLM] complete result: %s...", result[:200])
        return result

    async def ainvoke(self, messages: list[dict]):
        """Non-streaming invoke with messages list. Returns AIMessage."""
        lc_messages = _convert_messages(messages)
        return await self.chat_model.ainvoke(lc_messages)

    async def astream(self, messages: list[dict]):
        """Stream completion. Yields AIMessageChunk objects."""
        lc_messages = _convert_messages(messages)
        async for chunk in self.chat_model.astream(lc_messages):
            yield chunk


def _convert_messages(messages: list[dict]) -> list:
    """Convert OpenAI-style message dicts to LangChain message objects."""
    from langchain_core.messages import (
        AIMessage, HumanMessage, SystemMessage,
    )
    mapping = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    result = []
    for msg in messages:
        role = msg.get("role", "user") if isinstance(msg, dict) else msg.role
        content = msg.get("content", "") if isinstance(msg, dict) else msg.content
        cls = mapping.get(role, HumanMessage)
        result.append(cls(content=content))
    return result
