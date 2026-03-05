"""Configuration management using pydantic-settings.

Loads settings from environment variables and .env files.
Priority: CLI args > environment variables > ~/.nlm-proxy/.env > .env in project root > defaults
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_env_files() -> list[str]:
    """Return list of .env files to load (earlier files take priority)."""
    return [".env", str(Path.home() / ".nlm-proxy" / ".env")]


class SharedSettings(BaseSettings):
    """Shared settings across all commands."""

    debug: bool = Field(default=False, description="Enable debug logging")
    auth_dir: Path = Field(
        default_factory=lambda: Path.home() / ".nlm-proxy",
        description="Directory for auth cache and config",
    )

    @field_validator("auth_dir", mode="before")
    @classmethod
    def expand_auth_dir(cls, v: str | Path) -> Path:
        """Expand ~ in auth_dir path."""
        if isinstance(v, str):
            return Path(os.path.expanduser(v))
        return v

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    level: str = "INFO"
    file: str = "~/.nlm-proxy/logs/nlm-proxy.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_size: int = 10485760  # 10 MB
    backup_count: int = 5

    @field_validator("file", mode="before")
    @classmethod
    def expand_log_file(cls, v: str) -> str:
        """Expand ~ in log file path."""
        if v and isinstance(v, str):
            return os.path.expanduser(v)
        return v

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_LOG_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class MCPSettings(BaseSettings):
    """MCP server settings."""

    port: int = Field(default=8000, description="Port for HTTP transport")
    transport: Literal["stdio", "http"] = Field(
        default="stdio", description="Transport type"
    )

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_MCP_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class OpenAISettings(BaseSettings):
    """OpenAI proxy server settings."""

    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8080, description="Port to listen on")
    session_ttl: int = Field(
        default=86400, description="Session TTL in seconds (default: 24h)"
    )
    api_key: str = Field(description="API key for authentication (required)")

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_OPENAI_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AuthSettings(BaseSettings):
    """Authentication settings."""

    chrome_port: int = Field(default=9222, description="Chrome DevTools port")
    auto_launch: bool = Field(default=True, description="Auto-launch Chrome for auth")

    # Auto-refresh service settings
    auto_refresh_enabled: bool = Field(
        default=True,
        description="Enable background auth token auto-refresh service",
    )
    csrf_refresh_interval: int = Field(
        default=1800,
        description="Seconds between background CSRF/session token refreshes (default: 30 min)",
    )
    cookie_refresh_interval: int = Field(
        default=21600,
        description="Seconds between background cookie refreshes via headless Chrome (default: 6 h)",
    )
    headless_port: int = Field(
        default=9223,
        description="Chrome DevTools port for headless auth (keep separate from chrome_port to avoid conflicts)",
    )

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_AUTH_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )



class SmartRoutingSettings(BaseSettings):
    """Smart routing configuration for LLM-based request classification."""

    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for external OpenAI-compatible LLM"
    )
    llm_api_key: str = Field(
        default="",
        description="API key for external LLM"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for classification and routing"
    )
    router_model_name: str = Field(
        default="knowledge-finder",
        description="Model name that triggers smart routing"
    )
    allowed_notebooks: list[str] = Field(
        default_factory=list,
        description="List of notebook IDs to include (empty = all)"
    )
    summary_cache_ttl: int = Field(
        default=3600,
        description="TTL for notebook summary cache in seconds"
    )
    source_fetch_concurrency: int = Field(
        default=10,
        description="Max concurrent source summary fetches"
    )
    max_source_titles: int = Field(
        default=15,
        description="Max source titles to include in selection prompt"
    )
    source_descriptions_enabled: bool = Field(
        default=True,
        description="Include source keywords and summaries in selection prompt"
    )
    source_max_keywords: int = Field(
        default=5,
        description="Max keywords per source to include"
    )
    source_summary_max_chars: int = Field(
        default=80,
        description="Max chars of source summary (first sentence or truncated)"
    )
    source_descriptions_max_sources: int = Field(
        default=10,
        description="Max sources with descriptions (others get title only)"
    )

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_ROUTING_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class TracingSettings(BaseSettings):
    """OpenTelemetry tracing configuration."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    endpoint: str = Field(
        default="localhost:4317",
        description="OTLP collector endpoint (host:port)"
    )
    service_name: str = Field(
        default="nlm-proxy",
        description="Service name in traces"
    )
    protocol: Literal["grpc", "http"] = Field(
        default="grpc",
        description="Exporter protocol: grpc or http"
    )
    api_key: str | None = Field(
        default=None,
        description="Bearer token for collector authentication"
    )
    ca_cert_path: str | None = Field(
        default=None,
        description="Path to CA certificate for TLS verification"
    )
    verify_cert: bool = Field(
        default=True,
        description="Verify server certificate (HTTP only, gRPC always verifies)"
    )
    insecure: bool = Field(
        default=True,
        description="Use plain text (no TLS). Set to false for TLS."
    )
    export_timeout: int = Field(
        default=2,
        description="Export timeout in seconds (default: 2s for fast failure)"
    )
    max_queue_size: int = Field(
        default=2048,
        description="Max span queue size (default: 2048, drops oldest when full)"
    )
    request_max_length: int = Field(
        default=500,
        description="Max chars of user query to store in trace (0 to disable)"
    )
    response_max_length: int = Field(
        default=1000,
        description="Max chars of response to store in trace (0 to disable)"
    )

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_OTEL_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class CacheSettings(BaseSettings):
    """Response cache configuration."""

    response_cache_enabled: bool = Field(
        default=True, description="Enable response caching"
    )
    response_cache_ttl: int = Field(
        default=14400, description="Response cache TTL in seconds (4h)"
    )
    response_cache_max_entries: int = Field(
        default=1000, description="Max cached responses (LRU)"
    )
    semantic_match_enabled: bool = Field(
        default=True, description="Enable semantic matching"
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Embedding model for L2 semantic pre-filter",
    )
    similarity_threshold: float = Field(
        default=0.5, description="Min cosine similarity for L2 pre-filter"
    )
    similarity_exact_threshold: float = Field(
        default=0.90, description="Skip LLM verification threshold"
    )
    semantic_match_top_k: int = Field(
        default=10, description="Max candidates sent to LLM"
    )

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_CACHE_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AgentSettings(BaseSettings):
    """LangChain/LangGraph agent configuration (additive — does not replace existing)."""

    llm_provider: str = Field(default="openai", description="LLM provider")
    embedding_provider: str = Field(default="huggingface", description="Embedding provider")
    memory_backend: str = Field(default="memory", description="memory | sqlite | postgres")
    memory_db_path: str = Field(default="~/.nlm-proxy/memory.db")
    agent_max_iterations: int = Field(default=10)
    agent_verbose: bool = Field(default=False)
    agent_fallback_on_error: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_AGENT_",
        env_file=_get_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton instances
_shared: SharedSettings | None = None
_logging: LoggingSettings | None = None
_mcp: MCPSettings | None = None
_openai: OpenAISettings | None = None
_auth: AuthSettings | None = None
_routing: SmartRoutingSettings | None = None
_tracing: TracingSettings | None = None
_cache: CacheSettings | None = None
_agent: AgentSettings | None = None


def get_shared_settings() -> SharedSettings:
    """Get the shared settings instance."""
    global _shared
    if _shared is None:
        _shared = SharedSettings()
    return _shared


def get_logging_settings() -> LoggingSettings:
    """Get the logging settings instance."""
    global _logging
    if _logging is None:
        _logging = LoggingSettings()
    return _logging


def get_mcp_settings() -> MCPSettings:
    """Get the MCP settings instance."""
    global _mcp
    if _mcp is None:
        _mcp = MCPSettings()
    return _mcp


def get_openai_settings() -> OpenAISettings:
    """Get the OpenAI settings instance."""
    global _openai
    if _openai is None:
        _openai = OpenAISettings()
    return _openai


def get_auth_settings() -> AuthSettings:
    """Get the auth settings instance."""
    global _auth
    if _auth is None:
        _auth = AuthSettings()
    return _auth


def get_routing_settings() -> SmartRoutingSettings:
    """Get the smart routing settings instance."""
    global _routing
    if _routing is None:
        _routing = SmartRoutingSettings()
    return _routing


def get_tracing_settings() -> TracingSettings:
    """Get the tracing settings instance."""
    global _tracing
    if _tracing is None:
        _tracing = TracingSettings()
    return _tracing


def get_cache_settings() -> CacheSettings:
    """Get the cache settings instance."""
    global _cache
    if _cache is None:
        _cache = CacheSettings()
    return _cache


def get_agent_settings() -> "AgentSettings":
    """Get the agent settings instance."""
    global _agent
    if _agent is None:
        _agent = AgentSettings()
    return _agent


# Keep backward compatibility aliases
Settings = SharedSettings
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (alias for get_shared_settings)."""
    return get_shared_settings()
