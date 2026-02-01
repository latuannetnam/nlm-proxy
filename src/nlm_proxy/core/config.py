"""Configuration management using pydantic-settings.

Loads settings from environment variables and .env files.
Priority: environment variables > ~/.nlm-proxy/.env > .env in project root
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    level: str = "INFO"
    file: str = "~/.nlm-proxy/logs/nlm-proxy.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_size: int = 10485760  # 10 MB
    backup_count: int = 5

    model_config = SettingsConfigDict(
        env_prefix="NLM_PROXY_LOG_",
        env_file=["~/.nlm-proxy/.env", ".env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings(BaseSettings):
    """Main settings container.

    Extensible for future configuration beyond logging.
    """

    model_config = SettingsConfigDict(
        env_file=["~/.nlm-proxy/.env", ".env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton instances
_settings: Settings | None = None
_logging_settings: LoggingSettings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_logging_settings() -> LoggingSettings:
    """Get the logging settings instance."""
    global _logging_settings
    if _logging_settings is None:
        _logging_settings = LoggingSettings()
    return _logging_settings
