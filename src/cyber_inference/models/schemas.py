"""
Pydantic schemas for API validation and serialization.

Provides request/response models for:
- V1 OpenAI-compatible API
- Admin API
- Internal data transfer
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ModelType(str, Enum):
    """Type of model."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


class SessionStatus(str, Enum):
    """Status of a model session."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# =============================================================================
# OpenAI-Compatible Schemas (V1 API)
# =============================================================================

class ChatContentPart(BaseModel):
    """Structured chat content part (text, image, etc.)."""
    type: str = Field(..., description="Content part type (text, image_url, etc.)")
    text: Optional[str] = Field(None, description="Text content")
    image_url: Optional[Union[str, dict[str, Any]]] = Field(
        None, description="Image payload (url or base64)"
    )

    model_config = {"extra": "allow"}


class ChatMessage(BaseModel):
    """Chat message in a conversation."""
    role: str = Field(..., description="Role of the message author (system, user, assistant)")
    content: Union[str, list[ChatContentPart], dict[str, Any]] = Field(
        ..., description="Content of the message"
    )
    name: Optional[str] = Field(None, description="Optional name of the author")


class ChatCompletionRequest(BaseModel):
    """Request for chat completion (POST /v1/chat/completions)."""
    model: str = Field(..., description="Model to use for completion")
    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    n: int = Field(1, ge=1, le=10, description="Number of completions to generate")
    stream: bool = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, list[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(None, description="User identifier")


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response from chat completion endpoint."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for chat completion."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict]


class CompletionRequest(BaseModel):
    """Request for text completion (POST /v1/completions)."""
    model: str = Field(..., description="Model to use")
    prompt: Union[str, list[str]] = Field(..., description="Prompt(s) to complete")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=10)
    stream: bool = Field(False)
    stop: Optional[Union[str, list[str]]] = Field(None)
    echo: bool = Field(False, description="Echo the prompt in the response")


class CompletionChoice(BaseModel):
    """A single choice in a completion response."""
    text: str
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """Response from completion endpoint."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class EmbeddingRequest(BaseModel):
    """Request for embeddings (POST /v1/embeddings)."""
    model: str = Field(..., description="Model to use")
    input: Union[str, list[str]] = Field(..., description="Text(s) to embed")
    encoding_format: str = Field("float", description="Encoding format (float or base64)")


class EmbeddingData(BaseModel):
    """Single embedding result."""
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Response from embeddings endpoint."""
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "cyber-inference"
    permission: list[dict] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response from /v1/models endpoint."""
    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# Admin API Schemas
# =============================================================================

class ModelCreate(BaseModel):
    """Request to register a new model."""
    name: Optional[str] = Field(None, description="Unique model name (auto-generated if not provided)")
    hf_repo_id: Optional[str] = Field(None, description="HuggingFace repository ID")
    hf_filename: Optional[str] = Field(None, description="Specific filename to download")
    hf_mmproj_filename: Optional[str] = Field(None, description="Specific mmproj filename to download for vision models")
    model_type: ModelType = Field(ModelType.CHAT, description="Type of model")
    context_length: int = Field(4096, description="Context length")


class ModelUpdate(BaseModel):
    """Request to update a model."""
    name: Optional[str] = None
    is_enabled: Optional[bool] = None
    context_length: Optional[int] = None
    model_type: Optional[ModelType] = None


class ModelResponse(BaseModel):
    """Response with model details."""
    id: int
    name: str
    filename: str
    file_path: str
    hf_repo_id: Optional[str]
    size_bytes: int
    quantization: Optional[str]
    context_length: int
    model_type: Optional[str]
    mmproj_path: Optional[str] = None
    is_downloaded: bool
    is_enabled: bool
    download_progress: float
    created_at: datetime
    last_used_at: Optional[datetime]

    class Config:
        from_attributes = True


class RepoFileInfo(BaseModel):
    """Information about a file in a HuggingFace repository."""
    filename: str
    size_bytes: int
    quantization: Optional[str] = None
    is_mmproj: bool = False


class RepoFilesResponse(BaseModel):
    """Response with repository files information."""
    repo_id: str
    model_files: list[RepoFileInfo] = Field(default_factory=list, description="Main model GGUF files")
    mmproj_files: list[RepoFileInfo] = Field(default_factory=list, description="mmproj files for vision models")
    is_multimodal: bool = Field(False, description="Whether repo contains vision/multimodal model files")
    suggested_model: Optional[str] = Field(None, description="Auto-suggested model file to download")
    suggested_mmproj: Optional[str] = Field(None, description="Auto-suggested mmproj file for selected model")


class ModelSessionResponse(BaseModel):
    """Response with session details."""
    id: int
    model_id: int
    model_name: str
    port: int
    pid: Optional[int]
    status: str
    memory_mb: float
    gpu_memory_mb: float
    context_size: int
    started_at: datetime
    last_request_at: Optional[datetime]
    request_count: int

    class Config:
        from_attributes = True


class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_name: str = Field(..., description="Name of the model to load")
    context_size: Optional[int] = Field(None, description="Context size override")
    gpu_layers: Optional[int] = Field(None, description="GPU layers override")


class SystemResourcesResponse(BaseModel):
    """Response with system resource information."""
    platform: str
    cpu_count: int
    cpu_percent: float
    total_memory_gb: float
    available_memory_gb: float
    memory_percent: float
    gpu_info: Optional[str]
    gpu_memory_total_gb: Optional[float]
    gpu_memory_used_gb: Optional[float]
    gpu_memory_note: Optional[str] = None


class ConfigurationResponse(BaseModel):
    """Response with configuration value."""
    key: str
    value: Any
    value_type: str
    description: Optional[str]


class ConfigurationUpdate(BaseModel):
    """Request to update configuration."""
    value: Any
    description: Optional[str] = None


class AdminLoginRequest(BaseModel):
    """Request for admin login."""
    password: str


class AdminLoginResponse(BaseModel):
    """Response with JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# =============================================================================
# WebSocket Schemas
# =============================================================================

class LogMessage(BaseModel):
    """Log message for WebSocket streaming."""
    timestamp: datetime
    level: str
    module: str
    message: str


class StatusUpdate(BaseModel):
    """Status update for WebSocket streaming."""
    type: str  # model_loaded, model_unloaded, resource_update, etc.
    data: dict
