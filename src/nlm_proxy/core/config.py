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

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_ROUTING_",
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


# Keep backward compatibility aliases
Settings = SharedSettings
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (alias for get_shared_settings)."""
    return get_shared_settings()
