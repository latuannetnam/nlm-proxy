"""Pydantic models for OpenAI-compatible API."""

from typing import Literal
from pydantic import BaseModel


class Message(BaseModel):
    """A single message in the conversation."""
    role: Literal["system", "user", "assistant"]
    content: str
