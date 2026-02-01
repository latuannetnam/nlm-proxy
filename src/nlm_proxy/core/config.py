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


# Singleton instances
_shared: SharedSettings | None = None
_logging: LoggingSettings | None = None
_mcp: MCPSettings | None = None
_openai: OpenAISettings | None = None
_auth: AuthSettings | None = None


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


# Keep backward compatibility aliases
Settings = SharedSettings
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (alias for get_shared_settings)."""
    return get_shared_settings()
