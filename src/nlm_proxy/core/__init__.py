"""Core NotebookLM client library."""

from .constants import CodeMapper
from .exceptions import (
    NLMProxyError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    APIError,
)
from .auth import (
    AuthTokens,
    get_cache_path,
    load_cached_tokens,
    save_tokens_to_cache,
    extract_tokens_via_chrome_devtools,
    extract_csrf_from_page_source,
    extract_session_id_from_page,
    parse_cookies_from_chrome_format,
    validate_cookies,
    REQUIRED_COOKIES,
)
from .client import NotebookLMClient, Notebook, ConversationTurn

__all__ = [
    "CodeMapper",
    "NLMProxyError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "APIError",
    "AuthTokens",
    "get_cache_path",
    "load_cached_tokens",
    "save_tokens_to_cache",
    "extract_tokens_via_chrome_devtools",
    "extract_csrf_from_page_source",
    "extract_session_id_from_page",
    "parse_cookies_from_chrome_format",
    "validate_cookies",
    "REQUIRED_COOKIES",
    "NotebookLMClient",
    "Notebook",
    "ConversationTurn",
]
