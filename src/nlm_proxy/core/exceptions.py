"""Custom exceptions for NLM Proxy."""


class NLMProxyError(Exception):
    """Base exception for all NLM Proxy errors."""
    pass


class AuthenticationError(NLMProxyError):
    """Token expired, invalid, or missing."""
    pass


class RateLimitError(NLMProxyError):
    """API rate limit exceeded."""
    pass


class NotFoundError(NLMProxyError):
    """Notebook, source, or resource not found."""
    pass


class APIError(NLMProxyError):
    """Generic API error from NotebookLM."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
