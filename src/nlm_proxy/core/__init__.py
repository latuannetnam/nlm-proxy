"""Core NotebookLM client library."""

from .config import Settings, LoggingSettings, get_settings, get_logging_settings
from .logging import setup_logging, get_logger, reset_logging
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
from .client import NotebookLMClient, Notebook, ConversationTurn, SOURCE_ADD_TIMEOUT

__all__ = [
    # Config
    "Settings",
    "LoggingSettings",
    "get_settings",
    "get_logging_settings",
    # Logging
    "setup_logging",
    "get_logger",
    "reset_logging",
    # Constants
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
    "SOURCE_ADD_TIMEOUT",
]
