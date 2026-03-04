"""LLM client for external AI providers via LangChain.

This module is in core/ for reuse across multiple features:
- LangGraph routing nodes: classify_request(), select_notebook()
- ResponseCache: L3 semantic verification
- OpenAI proxy: LLM_TASK passthrough (streaming + non-streaming)
- MCP server: AgentCore query tools
"""

from nlm_proxy.core.logging import get_logger

logger = get_logger(__name__)


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
    - LangGraph routing nodes: classify_request(), select_notebook()
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
