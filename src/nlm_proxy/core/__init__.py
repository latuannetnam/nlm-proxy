"""Core NotebookLM client library."""

from .exceptions import (
    NLMProxyError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    APIError,
)

__all__ = [
    "NLMProxyError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "APIError",
]
