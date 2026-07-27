"""
Admin API endpoints for Cyber-Inference.

Provides management endpoints for:
- Model management (list, download, delete)
- Server control (load, unload, restart)
- Configuration management
- System status and resources
- Authentication (optional)
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt  # type: ignore[import-untyped]
from sqlalchemy import select

from cyber_inference.core.auth import verify_admin_token_value
from cyber_inference.core.config import CONFIG_DB_CASTS, MIN_CONTEXT_SIZE, get_settings
from cyber_inference.core.database import get_db_session
from cyber_inference.core.logging import get_logger
from cyber_inference.models.db_models import Configuration, Model
from cyber_inference.models.schemas import (
    AdminLoginRequest,
    AdminLoginResponse,
    ConfigurationResponse,
    ConfigurationUpdate,
    LlamaCppReleaseInfo,
    LlamaCppStatusResponse,
    LoadModelRequest,
    ModelCreate,
    ModelResponse,
    ModelSessionResponse,
    RepoFileInfo,
    RepoFilesResponse,
    SystemResourcesResponse,
)
from cyber_inference.services.auto_loader import AutoLoader
from cyber_inference.services.llama_installer import LlamaInstaller
from cyber_inference.services.model_manager import ModelManager, normalize_huggingface_reference

logger = get_logger(__name__)

router = APIRouter()
security = HTTPBearer(auto_error=False)

ModelConfigCaster = Callable[[object], object] | type[int] | type[float] | type[str]


def _parse_bool_config(value: object) -> bool:
    """Parse bool-like values for admin config writes."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_optional_string(value: object) -> str | None:
    """Normalize optional string values from admin requests."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_global_config_value(key: str, value: object) -> None:
    """Validate admin-configurable global settings before persistence."""
    if key in {"default_context_size", "max_context_size"}:
        try:
            context_size = int(str(value))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{key} must be an integer")
        if context_size < MIN_CONTEXT_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"{key} must be at least {MIN_CONTEXT_SIZE}",
            )
        settings = get_settings()
        if key == "default_context_size" and context_size > settings.max_context_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"default_context_size cannot exceed max_context_size "
                    f"({settings.max_context_size})"
                ),
            )
        if key == "max_context_size" and context_size < settings.default_context_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"max_context_size cannot be lower than default_context_size "
                    f"({settings.default_context_size})"
                ),
            )
    elif key == "model_load_timeout":
        try:
            timeout = int(str(value))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="model_load_timeout must be an integer")
        if timeout < 30 or timeout > 3600:
            raise HTTPException(
                status_code=400,
                detail="model_load_timeout must be between 30 and 3600 seconds",
            )
    elif key == "pre_model_load_command_timeout":
        try:
            timeout = int(str(value))
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="pre_model_load_command_timeout must be an integer",
            )
        if timeout < 1 or timeout > 300:
            raise HTTPException(
                status_code=400,
                detail="pre_model_load_command_timeout must be between 1 and 300 seconds",
            )
    elif key == "pre_model_load_command_enabled" and _parse_bool_config(value):
        settings = get_settings()
        if not settings.admin_password:
            raise HTTPException(
                status_code=400,
                detail="Admin protection must be enabled before pre-model load commands can run.",
            )
    elif key == "llama_mtp_default_draft_n_max":
        try:
            draft_n_max = int(str(value))
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="llama_mtp_default_draft_n_max must be an integer",
            )
        if draft_n_max < 1 or draft_n_max > 64:
            raise HTTPException(
                status_code=400,
                detail="llama_mtp_default_draft_n_max must be between 1 and 64",
            )


def _get_auto_loader() -> AutoLoader:
    """Get the auto-loader instance."""
    from cyber_inference.api.v1 import get_auto_loader
    return get_auto_loader()


def _get_llama_installer() -> LlamaInstaller:
    """Get a llama.cpp installer instance."""
    return LlamaInstaller()


def _get_running_llama_session_names() -> list[str]:
    """Return active llama.cpp session names that block binary replacement."""
    from cyber_inference.main import get_process_manager

    try:
        processes = get_process_manager().get_all_processes()
    except RuntimeError:
        return []

    active_statuses = {"starting", "running", "stopping"}
    return [
        process.model_name
        for process in processes
        if process.server_type == "llama" and process.status in active_statuses
    ]


async def _build_llama_cpp_status(installer: LlamaInstaller) -> LlamaCppStatusResponse:
    """Build admin-facing llama.cpp binary status."""
    binary_status = await installer.get_binary_status()
    latest_release = None
    latest_release_error = None
    update_available = None

    try:
        latest_release_info = await installer.get_latest_release_info()
        latest_release = LlamaCppReleaseInfo(**latest_release_info)
        update_available = installer.get_update_available(
            binary_status.get("installed_version"),
            latest_release.tag_name,
        )
    except Exception as e:
        latest_release_error = str(e)

    running_sessions = _get_running_llama_session_names()
    update_allowed = bool(binary_status["update_allowed"])
    update_blocked_reason = binary_status.get("update_blocked_reason")
    if running_sessions and update_allowed:
        update_allowed = False
        update_blocked_reason = (
            "Unload llama.cpp models before updating: "
            + ", ".join(sorted(running_sessions))
        )

    return LlamaCppStatusResponse(
        source=str(binary_status["source"]),
        binary_path=binary_status.get("binary_path"),
        managed_binary_path=str(binary_status["managed_binary_path"]),
        installed_version=binary_status.get("installed_version"),
        is_system_managed=bool(binary_status["is_system_managed"]),
        update_allowed=update_allowed,
        update_blocked_reason=update_blocked_reason,
        latest_release=latest_release,
        latest_release_error=latest_release_error,
        update_available=update_available,
        running_llama_sessions=sorted(running_sessions),
        supports_draft_mtp=bool(binary_status.get("supports_draft_mtp", False)),
    )


async def verify_admin_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> bool:
    """
    Verify admin authentication token.

    If admin password is not configured, all requests are allowed.
    """
    settings = get_settings()

    # If no admin password configured, allow access
    if not settings.admin_password:
        return True

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_admin_token_value(credentials.credentials):
        logger.warning("[warning]Invalid admin token[/warning]")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


@router.post("/login")
async def admin_login(request: AdminLoginRequest) -> AdminLoginResponse:
    """
    Authenticate as admin and get a JWT token.
    """
    logger.info("[info]Admin login attempt[/info]")

    settings = get_settings()

    if not settings.admin_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin password not configured",
        )

    # Verify password
    if request.password != settings.admin_password:
        logger.warning("[warning]Invalid admin password[/warning]")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    # Generate JWT token
    expiry = datetime.utcnow() + timedelta(hours=settings.jwt_expiry_hours)
    token = jwt.encode(
        {
            "type": "admin",
            "exp": expiry,
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    logger.info("[success]Admin login successful[/success]")

    return AdminLoginResponse(
        access_token=token,
        expires_in=settings.jwt_expiry_hours * 3600,
    )


@router.get("/llama-cpp/status")
async def get_llama_cpp_status(
    _: bool = Depends(verify_admin_token),
) -> LlamaCppStatusResponse:
    """
    Get llama.cpp binary provenance, version, and latest release status.
    """
    logger.debug("GET /admin/llama-cpp/status")
    return await _build_llama_cpp_status(_get_llama_installer())


@router.post("/llama-cpp/update")
async def update_llama_cpp(
    _: bool = Depends(verify_admin_token),
) -> LlamaCppStatusResponse:
    """
    Update the Cyber-Inference-managed llama.cpp binary.
    """
    logger.info("[highlight]POST /admin/llama-cpp/update[/highlight]")
    installer = _get_llama_installer()
    status_info = await _build_llama_cpp_status(installer)

    if not status_info.update_allowed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=status_info.update_blocked_reason
            or "The active llama-server cannot be updated by Cyber-Inference.",
        )

    await installer.install(force=True)
    return await _build_llama_cpp_status(installer)


@router.get("/status")
async def get_status(
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Get overall system status.
    """
    logger.debug("GET /admin/status")

    from cyber_inference import __version__
    from cyber_inference.main import get_process_manager, get_resource_monitor

    pm = get_process_manager()
    rm = get_resource_monitor()

    resources = await rm.get_resources()
    running_models = pm.get_running_models()

    return {
        "version": __version__,
        "status": "running",
        "uptime_seconds": 0,  # TODO: Track uptime
        "running_models": running_models,
        "loaded_model_count": len(running_models),
        "cpu_percent": resources.cpu_percent,
        "memory_percent": resources.memory_percent,
        "gpu_available": rm.has_gpu(),
    }


@router.get("/resources")
async def get_resources(
    _: bool = Depends(verify_admin_token),
) -> SystemResourcesResponse:
    """
    Get detailed system resource information.
    """
    logger.debug("GET /admin/resources")

    from cyber_inference.main import get_resource_monitor

    rm = get_resource_monitor()
    resources = await rm.get_resources()

    return SystemResourcesResponse(
        platform=f"{resources.timestamp}",  # Will be replaced with actual platform
        cpu_count=resources.cpu_count,
        cpu_percent=resources.cpu_percent,
        total_memory_gb=resources.total_memory_mb / 1024,
        available_memory_gb=resources.available_memory_mb / 1024,
        memory_percent=resources.memory_percent,
        gpu_info=resources.gpu.name if resources.gpu else None,
        gpu_memory_total_gb=(
            resources.gpu.total_memory_mb / 1024
            if resources.gpu and resources.gpu.total_memory_mb > 0
            else None
        ),
        gpu_memory_used_gb=(
            resources.gpu.used_memory_mb / 1024
            if resources.gpu
            and resources.gpu.total_memory_mb > 0
            and resources.gpu.used_memory_mb is not None
            else None
        ),
        gpu_memory_note=resources.gpu.memory_note if resources.gpu else None,
    )


@router.get("/models")
async def list_models(
    _: bool = Depends(verify_admin_token),
) -> list[ModelResponse]:
    """
    List all registered models.
    """
    logger.info("[info]GET /admin/models[/info]")

    mm = ModelManager()
    models = await mm.list_models(include_file_metadata=False)

    return [
        ModelResponse(
            id=m["id"],
            name=m["name"],
            filename=m["filename"],
            file_path=m["path"],
            hf_repo_id=m["hf_repo_id"],
            size_bytes=m["size_bytes"],
            quantization=m["quantization"],
            context_length=m["context_length"],
            model_type=m["model_type"],
            engine_type=m["engine_type"],
            mmproj_path=m["mmproj_path"],
            mtp_draft_path=m.get("mtp_draft_path"),
            mtp_capable=m.get("mtp_capable"),
            mtp_mode=m.get("mtp_mode"),
            mtp_detection_source=m.get("mtp_detection_source"),
            mtp_nextn_predict_layers=m.get("mtp_nextn_predict_layers"),
            mtp_spec_draft_n_max=m.get("mtp_spec_draft_n_max"),
            is_split_gguf=m["is_split_gguf"],
            gguf_shard_count=m["gguf_shard_count"],
            gguf_shard_filenames=m["gguf_shard_filenames"],
            tool_template_mode=m["tool_template_mode"],
            tool_template_name=m["tool_template_name"],
            tool_template_path=m["tool_template_path"],
            tool_jinja_enabled=m["tool_jinja_enabled"],
            is_downloaded=m["is_downloaded"],
            is_enabled=m["is_enabled"],
            download_progress=m["download_progress"],
            created_at=m["created_at"],
            last_used_at=m["last_used_at"],
        )
        for m in models
        if m["registered"] and m["id"] is not None
    ]


@router.get("/models/repo-files")
async def list_repo_files(
    repo_id: str,
    _: bool = Depends(verify_admin_token),
) -> RepoFilesResponse:
    """
    List available GGUF files in a HuggingFace repository.

    Returns model files and mmproj files separately, with suggestions
    for which files to download.
    """
    mm = ModelManager()

    try:
        repo_id, _direct_filename = normalize_huggingface_reference(repo_id)
        logger.info(f"[info]GET /admin/models/repo-files?repo_id={repo_id}[/info]")
        result = await mm.list_repo_files_detailed(repo_id)

        return RepoFilesResponse(
            repo_id=result["repo_id"],
            model_files=[
                RepoFileInfo(
                    filename=f["filename"],
                    size_bytes=f["size_bytes"],
                    quantization=f.get("quantization"),
                    artifact_type=f.get("artifact_type", "model"),
                    is_mmproj=f.get("is_mmproj", False),
                    is_split=f.get("is_split", False),
                    shard_count=f.get("shard_count"),
                    shard_total_size_bytes=f.get("shard_total_size_bytes"),
                    shard_filenames=f.get("shard_filenames", []),
                    primary_filename=f.get("primary_filename"),
                    is_complete=f.get("is_complete", True),
                    missing_shard_filenames=f.get("missing_shard_filenames", []),
                )
                for f in result["model_files"]
            ],
            mmproj_files=[
                RepoFileInfo(
                    filename=f["filename"],
                    size_bytes=f["size_bytes"],
                    quantization=f.get("quantization"),
                    artifact_type=f.get("artifact_type", "mmproj"),
                    is_mmproj=True,
                    is_split=f.get("is_split", False),
                    shard_count=f.get("shard_count"),
                    shard_total_size_bytes=f.get("shard_total_size_bytes"),
                    shard_filenames=f.get("shard_filenames", []),
                    primary_filename=f.get("primary_filename"),
                    is_complete=f.get("is_complete", True),
                    missing_shard_filenames=f.get("missing_shard_filenames", []),
                )
                for f in result["mmproj_files"]
            ],
            mtp_files=[
                RepoFileInfo(
                    filename=f["filename"],
                    size_bytes=f["size_bytes"],
                    quantization=f.get("quantization"),
                    artifact_type=f.get("artifact_type", "mtp"),
                    is_mmproj=False,
                    is_split=f.get("is_split", False),
                    shard_count=f.get("shard_count"),
                    shard_total_size_bytes=f.get("shard_total_size_bytes"),
                    shard_filenames=f.get("shard_filenames", []),
                    primary_filename=f.get("primary_filename"),
                    is_complete=f.get("is_complete", True),
                    missing_shard_filenames=f.get("missing_shard_filenames", []),
                )
                for f in result.get("mtp_files", [])
            ],
            is_multimodal=result["is_multimodal"],
            suggested_model=result.get("suggested_model"),
            suggested_mmproj=result.get("suggested_mmproj"),
            suggested_mtp=result.get("suggested_mtp"),
            is_mtp_candidate=bool(result.get("is_mtp_candidate", False)),
            mtp_default_enabled=bool(result.get("mtp_default_enabled", False)),
            suggested_mtp_mode=result.get("suggested_mtp_mode"),
            suggested_spec_draft_n_max=result.get("suggested_spec_draft_n_max"),
        )
    except Exception as e:
        logger.error(f"[error]Failed to list repo files: {e}[/error]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/models/suggest-mmproj")
async def suggest_mmproj(
    model_filename: str,
    mmproj_files: str,  # Comma-separated list of mmproj filenames
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Get the suggested mmproj file for a given model filename.

    Args:
        model_filename: The model filename to match
        mmproj_files: Comma-separated list of available mmproj filenames
    """
    logger.debug(f"GET /admin/models/suggest-mmproj?model_filename={model_filename}")

    mm = ModelManager()
    mmproj_list = [f.strip() for f in mmproj_files.split(",") if f.strip()]

    suggestion = mm.get_suggested_mmproj(model_filename, mmproj_list)

    return {"suggested_mmproj": suggestion}


@router.post("/models/download")
async def download_model(
    request: ModelCreate,
    _: bool = Depends(verify_admin_token),
) -> ModelResponse:
    """
    Download a model from HuggingFace.

    Progress updates are sent via WebSocket to /ws/status.

    For multimodal/vision models, you can specify hf_mmproj_filename to download
    the specific mmproj file. If not specified, it will be auto-selected.
    """
    # Validate that hf_repo_id is provided
    if not request.hf_repo_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="hf_repo_id is required for model download",
        )

    mm = ModelManager()

    try:
        repo_id, filename = normalize_huggingface_reference(
            request.hf_repo_id,
            request.hf_filename,
        )
        logger.info(f"[highlight]POST /admin/models/download: {repo_id}[/highlight]")
        logger.info(f"  Filename: {filename or 'auto-select'}")
        if request.hf_mmproj_filename:
            logger.info(f"  mmproj Filename: {request.hf_mmproj_filename}")
        if request.hf_mtp_filename:
            logger.info(f"  MTP draft filename: {request.hf_mtp_filename}")

        path = await mm.download_model(
            repo_id=repo_id,
            filename=filename,
            mmproj_filename=request.hf_mmproj_filename,
            mtp_filename=request.hf_mtp_filename,
            force=request.force,
            download_id=request.download_id,
        )

        model_name = path.model_name

        model = await mm.get_model(model_name)

        if not model:
            # Model was downloaded but not found in DB, create minimal response
            logger.warning(f"Model downloaded but not found in DB: {model_name}")
            return ModelResponse(
                id=0,
                name=model_name,
                filename=path.filename,
                file_path=str(path.path),
                hf_repo_id=repo_id,
                size_bytes=path.size_bytes,
                quantization=None,
                context_length=4096,
                model_type=None,
                mmproj_path=None,
                mtp_draft_path=None,
                mtp_capable=None,
                mtp_mode=None,
                mtp_detection_source=None,
                mtp_nextn_predict_layers=None,
                mtp_spec_draft_n_max=None,
                is_split_gguf=path.is_split_gguf,
                gguf_shard_count=len(path.shard_filenames or []) if path.is_split_gguf else None,
                gguf_shard_filenames=path.shard_filenames,
                tool_template_mode=None,
                tool_template_name=None,
                tool_template_path=None,
                tool_jinja_enabled=None,
                is_downloaded=True,
                is_enabled=True,
                download_progress=100.0,
                created_at=datetime.now(),
                last_used_at=None,
            )

        return ModelResponse(
            id=model.get("id", 0),
            name=model["name"],
            filename=model["filename"],
            file_path=model["path"],
            hf_repo_id=model.get("hf_repo_id"),
            size_bytes=model["size_bytes"],
            quantization=model.get("quantization"),
            context_length=model["context_length"],
            model_type=model.get("model_type"),
            mmproj_path=model.get("mmproj_path"),
            mtp_draft_path=model.get("mtp_draft_path"),
            mtp_capable=model.get("mtp_capable"),
            mtp_mode=model.get("mtp_mode"),
            mtp_detection_source=model.get("mtp_detection_source"),
            mtp_nextn_predict_layers=model.get("mtp_nextn_predict_layers"),
            mtp_spec_draft_n_max=model.get("mtp_spec_draft_n_max"),
            is_split_gguf=model.get("is_split_gguf"),
            gguf_shard_count=model.get("gguf_shard_count"),
            gguf_shard_filenames=model.get("gguf_shard_filenames"),
            tool_template_mode=model.get("tool_template_mode"),
            tool_template_name=model.get("tool_template_name"),
            tool_template_path=model.get("tool_template_path"),
            tool_jinja_enabled=model.get("tool_jinja_enabled"),
            is_downloaded=True,
            is_enabled=True,
            download_progress=100.0,
            created_at=datetime.now(),
            last_used_at=None,
        )

    except ValueError as e:
        # User error (e.g., repo not found, no GGUF files)
        logger.error(f"[error]Download failed (user error): {e}[/error]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[error]Download failed: {e}[/error]")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}",
        )


@router.post("/models/download-transformers")
async def download_transformers_model(
    request: ModelCreate,
    _: bool = Depends(verify_admin_token),
) -> ModelResponse:
    """
    Download a HuggingFace model for use with the transformers engine.

    Downloads the full model repository (safetensors, config, tokenizer, etc.)
    to models/transformers/. Uses AutoModelForCausalLM + model.generate().

    Progress updates are sent via WebSocket to /ws/status.
    """
    if not request.hf_repo_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="hf_repo_id is required for model download",
        )

    mm = ModelManager()

    try:
        repo_id, direct_filename = normalize_huggingface_reference(request.hf_repo_id)
        if direct_filename:
            raise ValueError(
                "Transformers downloads require a HuggingFace repository URL, "
                "not a direct model file URL"
            )
        logger.info(f"[highlight]POST /admin/models/download-transformers: {repo_id}[/highlight]")

        path = await mm.download_transformers_model(
            repo_id=repo_id,
            force=False,
            download_id=request.download_id,
        )

        model_name = mm._sanitize_repo_name(repo_id)
        model = await mm.get_model(model_name)

        if not model:
            return ModelResponse(
                id=0,
                name=model_name,
                filename=model_name,
                file_path=str(path),
                hf_repo_id=repo_id,
                size_bytes=0,
                quantization=None,
                context_length=4096,
                model_type=None,
                mmproj_path=None,
                tool_template_mode=None,
                tool_template_name=None,
                tool_template_path=None,
                tool_jinja_enabled=None,
                is_downloaded=True,
                is_enabled=True,
                download_progress=100.0,
                created_at=datetime.now(),
                last_used_at=None,
            )

        return ModelResponse(
            id=model.get("id", 0),
            name=model["name"],
            filename=model["filename"],
            file_path=model["path"],
            hf_repo_id=model.get("hf_repo_id"),
            size_bytes=model["size_bytes"],
            quantization=model.get("quantization"),
            context_length=model["context_length"],
            model_type=model.get("model_type"),
            mmproj_path=model.get("mmproj_path"),
            tool_template_mode=model.get("tool_template_mode"),
            tool_template_name=model.get("tool_template_name"),
            tool_template_path=model.get("tool_template_path"),
            tool_jinja_enabled=model.get("tool_jinja_enabled"),
            is_downloaded=True,
            is_enabled=True,
            download_progress=100.0,
            created_at=datetime.now(),
            last_used_at=None,
        )

    except ValueError as e:
        logger.error(f"[error]Transformers download failed (user error): {e}[/error]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[error]Transformers download failed: {e}[/error]")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}",
        )


@router.delete("/models/{model_name:path}")
async def delete_model(
    model_name: str,
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Delete a model.
    """
    logger.info(f"[warning]DELETE /admin/models/{model_name}[/warning]")

    # First unload if loaded
    auto_loader = _get_auto_loader()
    loaded = await auto_loader.get_loaded_models()

    if model_name in loaded:
        await auto_loader.unload_model(model_name)

    # Delete from storage
    mm = ModelManager()
    deleted = await mm.delete_model(model_name)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}",
        )

    return {"status": "deleted", "model": model_name}


@router.post("/models/{model_name:path}/load")
async def load_model(
    model_name: str,
    request: LoadModelRequest | None = None,
    _: bool = Depends(verify_admin_token),
) -> ModelSessionResponse:
    """
    Load a model into memory.
    """
    logger.info(f"[highlight]POST /admin/models/{model_name}/load[/highlight]")

    auto_loader = _get_auto_loader()

    try:
        await auto_loader.load_model(
            model_name,
            context_size_override=request.context_size if request else None,
            gpu_layers_override=request.gpu_layers if request else None,
            load_trigger="admin_manual",
        )
        status_info = await auto_loader.get_model_status(model_name)

        return ModelSessionResponse(
            id=0,
            model_id=0,
            model_name=model_name,
            port=status_info.get("port", 0),
            pid=None,
            status=status_info.get("status", "unknown"),
            memory_mb=status_info.get("memory_mb", 0),
            gpu_memory_mb=0,
            context_size=status_info.get("effective_config", {}).get("launch_config", {}).get("context_size", 4096),
            started_at=datetime.now(),
            last_request_at=None,
            request_count=status_info.get("request_count", 0),
            server_type=status_info.get("server_type"),
            last_transition_reason=status_info.get("last_transition_reason"),
            reload_count=status_info.get("reload_count", 0),
            effective_config=status_info.get("effective_config"),
        )

    except ValueError as e:
        logger.warning(f"[warning]Load rejected: {e}[/warning]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[error]Load failed: {e}[/error]")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/models/{model_name:path}/unload")
async def unload_model(
    model_name: str,
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Unload a model from memory.
    """
    logger.info(f"[warning]POST /admin/models/{model_name}/unload[/warning]")

    auto_loader = _get_auto_loader()
    await auto_loader.unload_model(model_name)

    return {"status": "unloaded", "model": model_name}


@router.get("/models/{model_name:path}/config")
async def get_model_config(
    model_name: str,
    _: bool = Depends(verify_admin_token),
) -> dict:
    """Get per-model inference defaults."""
    async with get_db_session() as session:
        result = await session.execute(select(Model).where(Model.name == model_name))
        model = result.scalar_one_or_none()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        status_info = await _get_auto_loader().get_model_status(model_name)
        return {
            "model": model_name,
            "context_length": model.context_length,
            "default_context_size": model.default_context_size,
            "default_temperature": model.default_temperature,
            "default_top_p": model.default_top_p,
            "default_top_k": model.default_top_k,
            "default_max_tokens": model.default_max_tokens,
            "default_repeat_penalty": model.default_repeat_penalty,
            "tool_template_mode": model.tool_template_mode,
            "tool_template_name": model.tool_template_name,
            "tool_template_path": model.tool_template_path,
            "tool_jinja_enabled": model.tool_jinja_enabled,
            "mtp_capable": model.mtp_capable,
            "mtp_mode": model.mtp_mode,
            "mtp_detection_source": model.mtp_detection_source,
            "mtp_nextn_predict_layers": model.mtp_nextn_predict_layers,
            "mtp_spec_draft_n_max": model.mtp_spec_draft_n_max,
            "runtime": status_info,
        }


@router.put("/models/{model_name:path}/config")
async def update_model_config(
    model_name: str,
    config: dict,
    _: bool = Depends(verify_admin_token),
) -> dict:
    """Update per-model inference defaults.

    Pass null/None for any field to clear it (revert to global default).
    """
    logger.info(f"[info]PUT /admin/models/{model_name}/config[/info]")

    allowed_fields: dict[str, ModelConfigCaster] = {
        "default_context_size": int,
        "default_temperature": float,
        "default_top_p": float,
        "default_top_k": int,
        "default_max_tokens": int,
        "default_repeat_penalty": float,
        "tool_template_mode": str,
        "tool_template_name": str,
        "tool_template_path": str,
        "tool_jinja_enabled": _parse_bool_config,
        "mtp_mode": str,
        "mtp_spec_draft_n_max": int,
    }

    async with get_db_session() as session:
        result = await session.execute(select(Model).where(Model.name == model_name))
        model = result.scalar_one_or_none()
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        for field, cast_type in allowed_fields.items():
            if field in config:
                value = config[field]
                if value is None:
                    setattr(model, field, None)
                elif field in {"tool_template_name", "tool_template_path"}:
                    setattr(model, field, _normalize_optional_string(value))
                elif field == "tool_template_mode":
                    normalized = _normalize_optional_string(value)
                    if normalized not in {"inherit", "disabled", "explicit", "auto"}:
                        raise HTTPException(status_code=400, detail="Invalid tool_template_mode")
                    setattr(model, field, normalized)
                elif field == "mtp_mode":
                    normalized = _normalize_optional_string(value)
                    if normalized not in {"auto", "enabled", "disabled"}:
                        raise HTTPException(status_code=400, detail="Invalid mtp_mode")
                    setattr(model, field, normalized)
                elif field == "mtp_spec_draft_n_max":
                    draft_n_max = int(value)
                    if draft_n_max < 1 or draft_n_max > 64:
                        raise HTTPException(
                            status_code=400,
                            detail="mtp_spec_draft_n_max must be between 1 and 64",
                        )
                    setattr(model, field, draft_n_max)
                elif field == "default_context_size":
                    context_size = int(value)
                    maximum = get_settings().max_context_size
                    if context_size < MIN_CONTEXT_SIZE or context_size > maximum:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "default_context_size must be between "
                                f"{MIN_CONTEXT_SIZE} and max_context_size ({maximum})"
                            ),
                        )
                    setattr(model, field, context_size)
                else:
                    setattr(model, field, cast_type(value))

        await session.commit()
        logger.info(f"[success]Updated config for {model_name}[/success]")

        runtime = await _get_auto_loader().reconcile_model_config_change(model_name)
        return {
            "model": model_name,
            "default_context_size": model.default_context_size,
            "default_temperature": model.default_temperature,
            "default_top_p": model.default_top_p,
            "default_top_k": model.default_top_k,
            "default_max_tokens": model.default_max_tokens,
            "default_repeat_penalty": model.default_repeat_penalty,
            "tool_template_mode": model.tool_template_mode,
            "tool_template_name": model.tool_template_name,
            "tool_template_path": model.tool_template_path,
            "tool_jinja_enabled": model.tool_jinja_enabled,
            "mtp_capable": model.mtp_capable,
            "mtp_mode": model.mtp_mode,
            "mtp_detection_source": model.mtp_detection_source,
            "mtp_nextn_predict_layers": model.mtp_nextn_predict_layers,
            "mtp_spec_draft_n_max": model.mtp_spec_draft_n_max,
            "runtime": runtime,
            "reload_triggered": runtime.get("reload_triggered", False),
            "message": runtime.get("message"),
        }


@router.get("/sessions")
async def list_sessions(
    _: bool = Depends(verify_admin_token),
) -> list[ModelSessionResponse]:
    """
    List all active model sessions.
    """
    logger.debug("GET /admin/sessions")

    from cyber_inference.main import get_process_manager

    pm = get_process_manager()
    processes = pm.get_all_processes()

    return [
        ModelSessionResponse(
            id=0,
            model_id=0,
            model_name=p.model_name,
            port=p.port,
            pid=p.pid,
            status=p.status,
            memory_mb=p.memory_mb,
            gpu_memory_mb=p.gpu_memory_mb,
            context_size=p.context_size,
            started_at=p.started_at,
            last_request_at=p.last_request_at,
            request_count=p.request_count,
            server_type=p.server_type,
            last_transition_reason=p.last_transition_reason,
            reload_count=p.reload_count,
            effective_config=p.effective_config or None,
        )
        for p in processes
    ]


@router.get("/config")
async def get_config(
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Get current configuration.
    """
    logger.debug("GET /admin/config")

    settings = get_settings()

    return {
        "host": settings.host,
        "port": settings.port,
        "log_level": settings.log_level,
        "default_context_size": settings.default_context_size,
        "max_context_size": settings.max_context_size,
        "model_idle_unload_enabled": settings.model_idle_unload_enabled,
        "model_idle_timeout": settings.model_idle_timeout,
        "model_load_timeout": settings.model_load_timeout,
        "pre_model_load_command_enabled": settings.pre_model_load_command_enabled,
        "pre_model_load_command": settings.pre_model_load_command,
        "pre_model_load_command_timeout": settings.pre_model_load_command_timeout,
        "max_loaded_models": settings.max_loaded_models,
        "max_memory_percent": settings.max_memory_percent,
        "llama_gpu_layers": settings.llama_gpu_layers,
        "llama_tool_template": settings.llama_tool_template,
        "llama_tool_template_file": settings.llama_tool_template_file,
        "llama_mtp_auto_enable": settings.llama_mtp_auto_enable,
        "llama_mtp_default_draft_n_max": settings.llama_mtp_default_draft_n_max,
        "admin_password_set": settings.admin_password is not None,
        "live_apply_keys": [
            "default_context_size",
            "max_context_size",
            "model_idle_unload_enabled",
            "model_idle_timeout",
            "model_load_timeout",
            "pre_model_load_command_enabled",
            "pre_model_load_command",
            "pre_model_load_command_timeout",
            "max_loaded_models",
            "max_memory_percent",
            "llama_gpu_layers",
            "llama_tool_template",
            "llama_tool_template_file",
            "llama_mtp_auto_enable",
            "llama_mtp_default_draft_n_max",
        ],
        "reload_on_save_keys": [
            "default_context_size",
            "max_context_size",
            "llama_gpu_layers",
            "llama_tool_template",
            "llama_tool_template_file",
            "llama_mtp_auto_enable",
            "llama_mtp_default_draft_n_max",
        ],
        "restart_only_keys": ["host", "port", "log_level"],
    }


@router.put("/config/{key}")
async def update_config(
    key: str,
    update: ConfigurationUpdate,
    _: bool = Depends(verify_admin_token),
) -> ConfigurationResponse:
    """
    Update a configuration value.
    """
    logger.info(f"[info]PUT /admin/config/{key}[/info]")

    _validate_global_config_value(key, update.value)

    async with get_db_session() as session:
        if key == "max_context_size":
            maximum = int(str(update.value))
            oversized_models = await session.execute(
                select(Model.name, Model.default_context_size).where(
                    Model.default_context_size.is_not(None),
                    Model.default_context_size > maximum,
                )
            )
            violations = oversized_models.all()
            if violations:
                names = ", ".join(str(row[0]) for row in violations[:5])
                extra = f" and {len(violations) - 5} more" if len(violations) > 5 else ""
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "max_context_size cannot be lower than existing per-model defaults: "
                        f"{names}{extra}"
                    ),
                )

        result = await session.execute(
            select(Configuration).where(Configuration.key == key)
        )
        config = result.scalar_one_or_none()
        optional_string_keys = {
            "admin_password",
            "llama_tool_template",
            "llama_tool_template_file",
            "pre_model_load_command",
        }
        stored_value = (
            ""
            if key in optional_string_keys and update.value is None
            else str(update.value)
        )

        if config:
            config.value = stored_value
            if update.description:
                config.description = update.description
        else:
            config = Configuration(
                key=key,
                value=stored_value,
                description=update.description,
            )
            session.add(config)

        await session.commit()

        typed_value = update.value
        cast_fn = CONFIG_DB_CASTS.get(key)
        if cast_fn is not None and update.value is not None:
            typed_value = cast_fn(update.value)
        if key in optional_string_keys and isinstance(typed_value, str) and not typed_value.strip():
            typed_value = None

        settings = get_settings()
        if hasattr(settings, key):
            setattr(settings, key, typed_value)

        auto_loader = _get_auto_loader()
        runtime_change = {
            "applied_live": False,
            "reload_triggered": False,
            "reloaded_models": [],
            "restart_required": True,
            "message": None,
        }
        if key in CONFIG_DB_CASTS:
            if key == "admin_password":
                runtime_change = {
                    "applied_live": True,
                    "reload_triggered": False,
                    "reloaded_models": [],
                    "restart_required": False,
                    "message": "Admin setting applied live.",
                }
            else:
                runtime_change = await auto_loader.reconcile_global_config_change(key)
                runtime_change["message"] = (
                    "Settings saved and running models reloaded."
                    if runtime_change["reload_triggered"]
                    else "Settings saved and applied live."
                )

        reloaded_models = runtime_change["reloaded_models"]

        return ConfigurationResponse(
            key=config.key,
            value=typed_value,
            value_type=config.value_type,
            description=config.description,
            applied_live=bool(runtime_change["applied_live"]),
            reload_triggered=bool(runtime_change["reload_triggered"]),
            reloaded_models=(
                [str(name) for name in reloaded_models]
                if isinstance(reloaded_models, list)
                else []
            ),
            restart_required=bool(runtime_change["restart_required"]),
            message=(
                str(runtime_change["message"])
                if runtime_change["message"] is not None
                else None
            ),
        )


@router.post("/shutdown")
async def shutdown_server(
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Gracefully shutdown the server.
    """
    logger.warning("[warning]POST /admin/shutdown - Initiating shutdown[/warning]")

    import asyncio
    import os
    import signal

    # Schedule shutdown after response
    async def delayed_shutdown():
        await asyncio.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(delayed_shutdown())

    return {"status": "shutting_down"}
