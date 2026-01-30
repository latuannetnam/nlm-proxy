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


class DeltaContent(BaseModel):
    """Delta content in streaming response."""
    role: str | None = None
    content: str | None = None


class Choice(BaseModel):
    """A single choice in chat completion response."""
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[Choice]
    system_fingerprint: str | None = None
