"""Pydantic models for OpenAI-compatible API."""

from typing import Literal
from pydantic import BaseModel


class Message(BaseModel):
    """A single message in the conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str  # notebook_id
    messages: list[Message]
    stream: bool = False
    # Ignored by NotebookLM but accepted for compatibility
    temperature: float | None = None
    max_tokens: int | None = None
    # Custom extensions
    conversation_id: str | None = None
    include_thinking: bool = False
