"""
SQLAlchemy ORM models for Cyber-Inference database.

Tables:
- models: Registered GGUF models
- model_sessions: Active model loading sessions
- configurations: Key-value configuration storage
- api_keys: API key management
- usage_logs: Request logging and analytics
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cyber_inference.core.database import Base
from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)


class Model(Base):
    """
    Registered model in the system.

    Tracks downloaded GGUF models and their metadata.
    """
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Model identification
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    # HuggingFace metadata
    hf_repo_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    hf_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Model properties
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    quantization: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    context_length: Mapped[int] = mapped_column(Integer, default=4096)

    # Multimodal support - path to mmproj file for vision models
    mmproj_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # Engine type: which inference backend to use
    engine_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, default="llama"
    )  # "llama", "whisper", "transformers"

    # Model card info (cached from HuggingFace)
    model_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # chat, completion, embedding
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Per-model inference defaults (NULL = use global/request defaults)
    default_context_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    default_temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    default_top_p: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    default_top_k: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    default_max_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    default_repeat_penalty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Status
    is_downloaded: Mapped[bool] = mapped_column(Boolean, default=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    download_progress: Mapped[float] = mapped_column(Float, default=0.0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    sessions: Mapped[list["ModelSession"]] = relationship(
        "ModelSession", back_populates="model", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name={self.name}, downloaded={self.is_downloaded})>"


class ModelSession(Base):
    """
    Active model loading session.

    Tracks currently loaded models and their resource usage.
    """
    __tablename__ = "model_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Model reference
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id"), nullable=False)

    # Server info
    port: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(50), default="starting")  # starting, running, stopping, stopped, error
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Resource usage
    memory_mb: Mapped[float] = mapped_column(Float, default=0.0)
    gpu_memory_mb: Mapped[float] = mapped_column(Float, default=0.0)

    # Configuration
    context_size: Mapped[int] = mapped_column(Integer, default=4096)
    gpu_layers: Mapped[int] = mapped_column(Integer, default=-1)
    threads: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_request_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Request stats
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens_generated: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    model: Mapped["Model"] = relationship("Model", back_populates="sessions")

    def __repr__(self) -> str:
        return f"<ModelSession(id={self.id}, model_id={self.model_id}, port={self.port}, status={self.status})>"


class Configuration(Base):
    """
    Key-value configuration storage.

    Stores runtime-configurable settings that persist across restarts.
    """
    __tablename__ = "configurations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(50), default="string")  # string, int, float, bool, json

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Configuration(key={self.key}, value={self.value[:50]}...)>"


class ApiKey(Base):
    """
    API key for authentication.

    Optional API key management for securing endpoints.
    """
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(10), nullable=False)  # First 8 chars for display

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Permissions
    can_inference: Mapped[bool] = mapped_column(Boolean, default=True)
    can_admin: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<ApiKey(name={self.name}, prefix={self.key_prefix}...)>"


class UsageLog(Base):
    """
    Usage logging for analytics and debugging.

    Records API requests for monitoring and troubleshooting.
    """
    __tablename__ = "usage_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Request info
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)

    # Request details
    request_body: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Response
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # Token usage
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Client info
    client_ip: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    api_key_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("api_keys.id"), nullable=True)

    # Error info
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    def __repr__(self) -> str:
        return f"<UsageLog(endpoint={self.endpoint}, model={self.model_name}, status={self.status_code})>"
