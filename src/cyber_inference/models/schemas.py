"""
Pydantic schemas for API validation and serialization.

Provides request/response models for:
- V1 OpenAI-compatible API
- Admin API
- Internal data transfer
"""

from datetime import datetime
from enum import Enum
from typing import Any

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
    text: str | None = Field(None, description="Text content")
    image_url: str | dict[str, Any] | None = Field(
        None, description="Image payload (url or base64)"
    )

    model_config = {"extra": "allow"}


class ChatMessage(BaseModel):
    """Chat message in a conversation."""
    role: str = Field(..., description="Role of the message author (system, user, assistant)")
    content: str | list[ChatContentPart] | dict[str, Any] | None = Field(
        ..., description="Content of the message"
    )
    name: str | None = Field(None, description="Optional name of the author")
    tool_calls: list[dict[str, Any]] | None = Field(None, description="Tool calls emitted by the assistant")
    tool_call_id: str | None = Field(None, description="Tool call id for tool-role messages")

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    """Request for chat completion (POST /v1/chat/completions)."""
    model: str = Field(..., description="Model to use for completion")
    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int | None = Field(None, ge=0, description="Top-k sampling")
    n: int = Field(1, ge=1, le=10, description="Number of completions to generate")
    stream: bool = Field(False, description="Whether to stream the response")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    repeat_penalty: float | None = Field(None, ge=0.0, description="Repeat penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    user: str | None = Field(None, description="User identifier")
    tools: list[dict[str, Any]] | None = Field(None, description="Available tools")
    tool_choice: str | dict[str, Any] | None = Field(None, description="Tool choice policy")
    parallel_tool_calls: bool | None = Field(
        None,
        description="Whether parallel tool calling is enabled",
    )


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: str | None = None


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
    prompt: str | list[str] = Field(..., description="Prompt(s) to complete")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0)
    repeat_penalty: float | None = Field(None, ge=0.0)
    n: int = Field(1, ge=1, le=10)
    stream: bool = Field(False)
    stop: str | list[str] | None = Field(None)
    echo: bool = Field(False, description="Echo the prompt in the response")


class CompletionChoice(BaseModel):
    """A single choice in a completion response."""
    text: str
    index: int
    finish_reason: str | None = None


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
    input: str | list[str] = Field(..., description="Text(s) to embed")
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


class ModelContext(BaseModel):
    """Context-window metadata for model discovery."""
    length: int | None = None
    window: int | None = None
    configured_length: int | None = None
    native_length: int | None = None
    source: str | None = None


class ModelCapabilities(BaseModel):
    """Simple capability metadata for a model."""
    vision: bool = False
    tool_calling: str = "unsupported"
    embedding: bool = False
    transcription: bool = False


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "cyber-inference"
    permission: list[dict] = Field(default_factory=list)
    root: str | None = None
    parent: str | None = None
    server_type: str | None = None
    is_loaded: bool | None = None
    status: str | None = None
    capabilities: ModelCapabilities | None = None
    context: ModelContext | None = None
    context_length: int | None = None
    max_context_length: int | None = None
    context_window: int | None = None


class ModelsResponse(BaseModel):
    """Response from /v1/models endpoint."""
    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# Admin API Schemas
# =============================================================================

class ModelCreate(BaseModel):
    """Request to register a new model."""
    name: str | None = Field(None, description="Unique model name (auto-generated if not provided)")
    hf_repo_id: str | None = Field(None, description="HuggingFace repository ID")
    hf_filename: str | None = Field(None, description="Specific filename to download")
    hf_mmproj_filename: str | None = Field(None, description="Specific mmproj filename to download for vision models")
    download_id: str | None = Field(None, description="Client-generated download session identifier")
    force: bool = Field(False, description="Force redownload even when local files are complete")
    model_type: ModelType = Field(ModelType.CHAT, description="Type of model")
    context_length: int = Field(4096, description="Context length")


class ModelUpdate(BaseModel):
    """Request to update a model."""
    name: str | None = None
    is_enabled: bool | None = None
    context_length: int | None = None
    model_type: ModelType | None = None


class ModelResponse(BaseModel):
    """Response with model details."""
    id: int
    name: str
    filename: str
    file_path: str
    hf_repo_id: str | None
    size_bytes: int
    quantization: str | None
    context_length: int
    model_type: str | None
    engine_type: str | None = "llama"
    mmproj_path: str | None = None
    is_split_gguf: bool | None = None
    gguf_shard_count: int | None = None
    gguf_shard_filenames: list[str] | None = None
    tool_template_mode: str | None = None
    tool_template_name: str | None = None
    tool_template_path: str | None = None
    tool_jinja_enabled: bool | None = None
    is_downloaded: bool
    is_enabled: bool
    download_progress: float
    created_at: datetime
    last_used_at: datetime | None

    class Config:
        from_attributes = True


class RepoFileInfo(BaseModel):
    """Information about a file in a HuggingFace repository."""
    filename: str
    size_bytes: int
    quantization: str | None = None
    is_mmproj: bool = False
    is_split: bool = False
    shard_count: int | None = None
    shard_total_size_bytes: int | None = None
    shard_filenames: list[str] = Field(default_factory=list)
    primary_filename: str | None = None
    is_complete: bool = True
    missing_shard_filenames: list[str] = Field(default_factory=list)


class RepoFilesResponse(BaseModel):
    """Response with repository files information."""
    repo_id: str
    model_files: list[RepoFileInfo] = Field(default_factory=list, description="Main model GGUF files")
    mmproj_files: list[RepoFileInfo] = Field(default_factory=list, description="mmproj files for vision models")
    is_multimodal: bool = Field(False, description="Whether repo contains vision/multimodal model files")
    suggested_model: str | None = Field(None, description="Auto-suggested model file to download")
    suggested_mmproj: str | None = Field(None, description="Auto-suggested mmproj file for selected model")


class ModelSessionResponse(BaseModel):
    """Response with session details."""
    id: int
    model_id: int
    model_name: str
    port: int
    pid: int | None
    status: str
    memory_mb: float
    gpu_memory_mb: float
    context_size: int
    started_at: datetime
    last_request_at: datetime | None
    request_count: int
    server_type: str | None = None
    last_transition_reason: str | None = None
    reload_count: int = 0
    effective_config: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_name: str = Field(..., description="Name of the model to load")
    context_size: int | None = Field(None, description="Context size override")
    gpu_layers: int | None = Field(None, description="GPU layers override")


class SystemResourcesResponse(BaseModel):
    """Response with system resource information."""
    platform: str
    cpu_count: int
    cpu_percent: float
    total_memory_gb: float
    available_memory_gb: float
    memory_percent: float
    gpu_info: str | None
    gpu_memory_total_gb: float | None
    gpu_memory_used_gb: float | None
    gpu_memory_note: str | None = None


class LlamaCppReleaseInfo(BaseModel):
    """Selected llama.cpp latest-release metadata for admin display."""
    tag_name: str | None = None
    name: str | None = None
    html_url: str | None = None
    published_at: str | None = None
    compatible_asset: str | None = None


class LlamaCppStatusResponse(BaseModel):
    """llama.cpp binary provenance and update status."""
    source: str
    binary_path: str | None = None
    managed_binary_path: str
    installed_version: str | None = None
    is_system_managed: bool
    update_allowed: bool
    update_blocked_reason: str | None = None
    latest_release: LlamaCppReleaseInfo | None = None
    latest_release_error: str | None = None
    update_available: bool | None = None
    running_llama_sessions: list[str] = Field(default_factory=list)


class ConfigurationResponse(BaseModel):
    """Response with configuration value."""
    key: str
    value: Any
    value_type: str
    description: str | None
    applied_live: bool = False
    reload_triggered: bool = False
    reloaded_models: list[str] = Field(default_factory=list)
    restart_required: bool = False
    message: str | None = None


class ConfigurationUpdate(BaseModel):
    """Request to update configuration."""
    value: Any
    description: str | None = None


class AdminLoginRequest(BaseModel):
    """Request for admin login."""
    password: str


class AdminLoginResponse(BaseModel):
    """Response with JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# =============================================================================
# Audio Transcription Schemas (OpenAI-compatible)
# =============================================================================

class TranscriptionRequest(BaseModel):
    """Request for audio transcription (POST /v1/audio/transcriptions)."""
    model: str = Field(..., description="Model to use for transcription")
    language: str | None = Field(None, description="Language of the audio (ISO-639-1 code)")
    prompt: str | None = Field(None, description="Optional prompt to guide transcription")
    response_format: str = Field("json", description="Response format: json, text, srt, verbose_json, vtt")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Sampling temperature")


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio with timing."""
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionResponse(BaseModel):
    """Response from transcription endpoint."""
    text: str
    task: str = "transcribe"
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] | None = None


class TranslationResponse(BaseModel):
    """Response from translation endpoint (audio to English text)."""
    text: str
    task: str = "translate"
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] | None = None


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
