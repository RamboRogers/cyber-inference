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
from sqlalchemy import select

from cyber_inference.core.database import get_db_session
from cyber_inference.core.logging import get_logger
from cyber_inference.models.db_models import Configuration

logger = get_logger(__name__)

CONFIG_DB_CASTS = {
    "default_context_size": int,
    "max_context_size": int,
    "model_idle_timeout": int,
    "max_loaded_models": int,
    "max_memory_percent": float,
    "llama_gpu_layers": int,
    "admin_password": str,
    "sglang_mem_fraction": float,
    "sglang_tp_size": int,
}


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
    log_level: str = Field(default="INFO", description="Log level")

    # Model management
    default_context_size: int = Field(default=8192, description="Default context size for models")
    max_context_size: int = Field(default=32768, description="Maximum allowed context size")
    model_idle_timeout: int = Field(default=300, description="Seconds before unloading idle model")
    max_loaded_models: int = Field(default=1, description="Maximum number of simultaneously loaded models")

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

    # SGLang settings
    sglang_enabled: bool = Field(
        default=False,
        description="Whether SGLang engine is available (auto-detected at startup)",
    )
    sglang_mem_fraction: float = Field(
        default=0.85,
        description="SGLang --mem-fraction-static for KV cache pool memory",
    )
    sglang_tp_size: int = Field(
        default=1,
        description="SGLang tensor parallelism degree (number of GPUs)",
    )
    sglang_base_port: int = Field(
        default=8350,
        description="Base port for SGLang servers (separate range from llama.cpp)",
    )

    @property
    def database_path(self) -> Path:
        """Full path to the database file."""
        return self.data_dir / self.database_name

    @property
    def log_dir(self) -> Path:
        """Path to the log directory."""
        return self.data_dir / "logs"

    @property
    def sglang_models_dir(self) -> Path:
        """Path to the SGLang models directory."""
        return self.models_dir / "sglang"

    @property
    def transformers_models_dir(self) -> Path:
        """Path to the transformers models directory."""
        return self.models_dir / "transformers"

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
            self.sglang_models_dir,
            self.transformers_models_dir,
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


async def load_db_config_overrides() -> dict[str, object]:
    """
    Load runtime configuration overrides from the database.
    """
    if not CONFIG_DB_CASTS:
        return {}

    try:
        async with get_db_session() as session:
            result = await session.execute(
                select(Configuration).where(Configuration.key.in_(list(CONFIG_DB_CASTS.keys())))
            )
            configs = result.scalars().all()
    except Exception as exc:
        logger.warning(f"Could not load config overrides from database: {exc}")
        return {}

    overrides: dict[str, object] = {}
    for config in configs:
        cast = CONFIG_DB_CASTS.get(config.key, str)
        try:
            overrides[config.key] = cast(config.value)
        except (TypeError, ValueError):
            overrides[config.key] = config.value
        if config.key == "admin_password":
            value = overrides[config.key]
            if isinstance(value, str) and not value.strip():
                overrides[config.key] = None

    return overrides


async def apply_db_config_overrides(settings: Settings) -> dict[str, object]:
    """
    Apply database overrides to the in-memory settings instance.
    """
    overrides = await load_db_config_overrides()
    for key, value in overrides.items():
        if key == "admin_password" and isinstance(value, str) and not value.strip():
            value = None
        if hasattr(settings, key):
            setattr(settings, key, value)
    if overrides:
        logger.info(f"Applied {len(overrides)} config overrides from database")
    return overrides
