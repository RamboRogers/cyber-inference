"""
Admin API endpoints for Cyber-Inference.

Provides management endpoints for:
- Model management (list, download, delete)
- Server control (load, unload, restart)
- Configuration management
- System status and resources
- Authentication (optional)
"""

from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cyber_inference.core.auth import verify_admin_token_value
from cyber_inference.core.config import get_settings
from cyber_inference.core.database import get_db, get_db_session
from cyber_inference.core.logging import get_logger
from cyber_inference.models.db_models import Configuration, Model
from cyber_inference.models.schemas import (
    AdminLoginRequest,
    AdminLoginResponse,
    ConfigurationResponse,
    ConfigurationUpdate,
    LoadModelRequest,
    ModelCreate,
    ModelResponse,
    ModelSessionResponse,
    ModelUpdate,
    RepoFileInfo,
    RepoFilesResponse,
    SystemResourcesResponse,
)
from cyber_inference.services.auto_loader import AutoLoader
from cyber_inference.services.model_manager import ModelManager

logger = get_logger(__name__)

router = APIRouter()
security = HTTPBearer(auto_error=False)


def _get_auto_loader() -> AutoLoader:
    """Get the auto-loader instance."""
    from cyber_inference.api.v1 import get_auto_loader
    return get_auto_loader()


async def verify_admin_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
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


@router.get("/status")
async def get_status(
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Get overall system status.
    """
    logger.debug("GET /admin/status")

    from cyber_inference.main import get_process_manager, get_resource_monitor
    from cyber_inference import __version__

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

    async with get_db_session() as session:
        result = await session.execute(select(Model))
        models = result.scalars().all()

        return [
            ModelResponse(
                id=m.id,
                name=m.name,
                filename=m.filename,
                file_path=m.file_path,
                hf_repo_id=m.hf_repo_id,
                size_bytes=m.size_bytes,
                quantization=m.quantization,
                context_length=m.context_length,
                model_type=m.model_type,
                engine_type=m.engine_type,
                mmproj_path=m.mmproj_path,
                is_downloaded=m.is_downloaded,
                is_enabled=m.is_enabled,
                download_progress=m.download_progress,
                created_at=m.created_at,
                last_used_at=m.last_used_at,
            )
            for m in models
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
    logger.info(f"[info]GET /admin/models/repo-files?repo_id={repo_id}[/info]")

    mm = ModelManager()

    try:
        result = await mm.list_repo_files_detailed(repo_id)

        return RepoFilesResponse(
            repo_id=result["repo_id"],
            model_files=[
                RepoFileInfo(
                    filename=f["filename"],
                    size_bytes=f["size_bytes"],
                    quantization=f.get("quantization"),
                    is_mmproj=f.get("is_mmproj", False),
                )
                for f in result["model_files"]
            ],
            mmproj_files=[
                RepoFileInfo(
                    filename=f["filename"],
                    size_bytes=f["size_bytes"],
                    quantization=f.get("quantization"),
                    is_mmproj=True,
                )
                for f in result["mmproj_files"]
            ],
            is_multimodal=result["is_multimodal"],
            suggested_model=result.get("suggested_model"),
            suggested_mmproj=result.get("suggested_mmproj"),
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

    logger.info(f"[highlight]POST /admin/models/download: {request.hf_repo_id}[/highlight]")
    logger.info(f"  Filename: {request.hf_filename or 'auto-select'}")
    if request.hf_mmproj_filename:
        logger.info(f"  mmproj Filename: {request.hf_mmproj_filename}")

    mm = ModelManager()

    try:
        path = await mm.download_model(
            repo_id=request.hf_repo_id,
            filename=request.hf_filename,
            mmproj_filename=request.hf_mmproj_filename,
        )

        # Auto-generate model name if not provided
        model_name = request.name
        if not model_name:
            if path:
                # Use filename stem (without extension)
                model_name = path.stem
            elif request.hf_filename:
                # Use filename without extension
                from pathlib import Path as PathLib
                model_name = PathLib(request.hf_filename).stem
            else:
                # Fallback to repo_id slug
                model_name = request.hf_repo_id.split("/")[-1].replace("-", "_")

        model = await mm.get_model(model_name)

        if not model:
            # Model was downloaded but not found in DB, create minimal response
            logger.warning(f"Model downloaded but not found in DB: {model_name}")
            return ModelResponse(
                id=0,
                name=model_name,
                filename=path.name if path else request.hf_filename or "unknown",
                file_path=str(path) if path else "",
                hf_repo_id=request.hf_repo_id,
                size_bytes=path.stat().st_size if path and path.exists() else 0,
                quantization=None,
                context_length=4096,
                model_type=None,
                mmproj_path=None,
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


@router.post("/models/download-sglang")
async def download_sglang_model(
    request: ModelCreate,
    _: bool = Depends(verify_admin_token),
) -> ModelResponse:
    """
    Download a HuggingFace model for use with the SGLang engine.

    Unlike GGUF downloads, SGLang downloads the full model repository
    (safetensors, config, tokenizer, etc.) to models/sglang/.

    Progress updates are sent via WebSocket to /ws/status.
    """
    if not request.hf_repo_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="hf_repo_id is required for model download",
        )

    logger.info(f"[highlight]POST /admin/models/download-sglang: {request.hf_repo_id}[/highlight]")

    # Check if SGLang is available
    from cyber_inference.services.sglang_manager import SGLangManager
    sglang_mgr = SGLangManager.get_instance()
    if not sglang_mgr.is_available():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SGLang is not installed. Enable it with: CYBER_INFERENCE_ENABLE_SGLANG=1 ./start.sh",
        )

    mm = ModelManager()

    try:
        path = await mm.download_sglang_model(
            repo_id=request.hf_repo_id,
            force=False,
        )

        model_name = mm._sanitize_repo_name(request.hf_repo_id)
        model = await mm.get_model(model_name)

        if not model:
            return ModelResponse(
                id=0,
                name=model_name,
                filename=model_name,
                file_path=str(path),
                hf_repo_id=request.hf_repo_id,
                size_bytes=0,
                quantization=None,
                context_length=4096,
                model_type=None,
                mmproj_path=None,
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
            is_downloaded=True,
            is_enabled=True,
            download_progress=100.0,
            created_at=datetime.now(),
            last_used_at=None,
        )

    except ValueError as e:
        logger.error(f"[error]SGLang download failed (user error): {e}[/error]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[error]SGLang download failed: {e}[/error]")
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

    logger.info(f"[highlight]POST /admin/models/download-transformers: {request.hf_repo_id}[/highlight]")

    mm = ModelManager()

    try:
        path = await mm.download_transformers_model(
            repo_id=request.hf_repo_id,
            force=False,
        )

        model_name = mm._sanitize_repo_name(request.hf_repo_id)
        model = await mm.get_model(model_name)

        if not model:
            return ModelResponse(
                id=0,
                name=model_name,
                filename=model_name,
                file_path=str(path),
                hf_repo_id=request.hf_repo_id,
                size_bytes=0,
                quantization=None,
                context_length=4096,
                model_type=None,
                mmproj_path=None,
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


@router.get("/sglang/status")
async def get_sglang_status(
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Get SGLang engine status information.

    Returns availability, version, CUDA info, and installed status.
    """
    logger.debug("GET /admin/sglang/status")

    from cyber_inference.services.sglang_manager import SGLangManager
    sglang_mgr = SGLangManager.get_instance()

    return sglang_mgr.get_status()


@router.get("/sglang/repo-info")
async def get_sglang_repo_info(
    repo_id: str,
    _: bool = Depends(verify_admin_token),
) -> dict:
    """
    Get information about a HuggingFace repo for SGLang download.

    Returns file count, total size, and whether it's a valid model.
    """
    logger.info(f"[info]GET /admin/sglang/repo-info?repo_id={repo_id}[/info]")

    mm = ModelManager()

    try:
        return await mm.list_sglang_repo_info(repo_id)
    except Exception as e:
        logger.error(f"[error]Failed to get repo info: {e}[/error]")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
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
    request: Optional[LoadModelRequest] = None,
    _: bool = Depends(verify_admin_token),
) -> ModelSessionResponse:
    """
    Load a model into memory.
    """
    logger.info(f"[highlight]POST /admin/models/{model_name}/load[/highlight]")

    auto_loader = _get_auto_loader()

    try:
        url = await auto_loader.load_model(model_name)
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
            context_size=4096,
            started_at=datetime.now(),
            last_request_at=None,
            request_count=status_info.get("request_count", 0),
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
        "model_idle_timeout": settings.model_idle_timeout,
        "max_loaded_models": settings.max_loaded_models,
        "max_memory_percent": settings.max_memory_percent,
        "llama_gpu_layers": settings.llama_gpu_layers,
        "admin_password_set": settings.admin_password is not None,
        "sglang_enabled": settings.sglang_enabled,
        "sglang_mem_fraction": settings.sglang_mem_fraction,
        "sglang_tp_size": settings.sglang_tp_size,
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

    async with get_db_session() as session:
        result = await session.execute(
            select(Configuration).where(Configuration.key == key)
        )
        config = result.scalar_one_or_none()

        if config:
            config.value = str(update.value)
            if update.description:
                config.description = update.description
        else:
            config = Configuration(
                key=key,
                value=str(update.value),
                description=update.description,
            )
            session.add(config)

        await session.commit()

        return ConfigurationResponse(
            key=config.key,
            value=update.value,
            value_type=config.value_type,
            description=config.description,
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
    import signal
    import os

    # Schedule shutdown after response
    async def delayed_shutdown():
        await asyncio.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(delayed_shutdown())

    return {"status": "shutting_down"}
