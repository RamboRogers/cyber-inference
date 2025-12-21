"""
Configuration management for Cyber-Inference.

Provides:
- Environment-based configuration with defaults
- Database-backed configuration persistence
- Runtime configuration updates
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables prefixed with CYBER_INFERENCE_.
    """

    model_config = SettingsConfigDict(
        env_prefix="CYBER_INFERENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8337, description="Port to bind to")

    # Directory paths
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data", description="Data directory")
    models_dir: Path = Field(default_factory=lambda: Path.cwd() / "models", description="Models directory")
    bin_dir: Path = Field(default_factory=lambda: Path.cwd() / "bin", description="Binary directory for llama.cpp")

    # Database
    database_name: str = Field(default="cyber-inference.db", description="Database filename")

    # Logging
    log_level: str = Field(default="DEBUG", description="Log level")

    # Model management
    default_context_size: int = Field(default=4096, description="Default context size for models")
    max_context_size: int = Field(default=32768, description="Maximum allowed context size")
    model_idle_timeout: int = Field(default=300, description="Seconds before unloading idle model")
    max_loaded_models: int = Field(default=3, description="Maximum number of simultaneously loaded models")

    # Resource limits
    max_memory_percent: float = Field(default=80.0, description="Maximum memory usage percentage")
    max_gpu_memory_percent: float = Field(default=90.0, description="Maximum GPU memory usage percentage")

    # llama.cpp settings
    llama_server_base_port: int = Field(default=8338, description="Base port for llama.cpp servers")
    llama_threads: Optional[int] = Field(default=None, description="Number of threads (auto if None)")
    llama_gpu_layers: int = Field(default=-1, description="GPU layers (-1 for auto)")

    # Security
    admin_password: Optional[str] = Field(default=None, description="Admin password (optional)")
    jwt_secret: str = Field(default="cyber-inference-secret-change-me", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(default=24, description="JWT token expiry in hours")

    # HuggingFace
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token")

    @property
    def database_path(self) -> Path:
        """Full path to the database file."""
        return self.data_dir / self.database_name

    @property
    def log_dir(self) -> Path:
        """Path to the log directory."""
        return self.data_dir / "logs"

    @property
    def log_level_int(self) -> int:
        """Log level as logging constant."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return levels.get(self.log_level.upper(), logging.DEBUG)

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.bin_dir,
            self.log_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to ensure settings are only loaded once.
    """
    logger.debug("Loading application settings")
    settings = Settings()
    settings.ensure_directories()

    logger.debug(f"Settings loaded:")
    logger.debug(f"  Host: {settings.host}")
    logger.debug(f"  Port: {settings.port}")
    logger.debug(f"  Data dir: {settings.data_dir}")
    logger.debug(f"  Models dir: {settings.models_dir}")
    logger.debug(f"  Log level: {settings.log_level}")

    return settings


def reload_settings() -> Settings:
    """
    Force reload of settings (clears cache).
    """
    logger.info("Reloading application settings")
    get_settings.cache_clear()
    return get_settings()

