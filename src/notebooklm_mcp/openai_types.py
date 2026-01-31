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
    include_thinking: bool = True


class DeltaContent(BaseModel):
    """Delta content in streaming response."""
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None  # OpenAI o1/o3 style thinking


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


class ResponseMessage(BaseModel):
    """Message in non-streaming response."""
    role: str = "assistant"
    content: str
    reasoning_content: str | None = None  # OpenAI o1/o3 style thinking


class ResponseChoice(BaseModel):
    """Choice in non-streaming response."""
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible non-streaming response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ResponseChoice]
    usage: Usage
    system_fingerprint: str | None = None
