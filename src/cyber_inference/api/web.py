"""
Web GUI routes for Cyber-Inference.

Serves the HTML templates for:
- Dashboard (/)
- Models management (/models)
- Settings (/settings)
- Logs (/logs)
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from cyber_inference import __version__
from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Setup templates
_templates_dir = Path(__file__).parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=_templates_dir) if _templates_dir.exists() else None


def _template_context(request: Request, **kwargs) -> dict:
    """Build common template context."""
    settings = get_settings()
    return {
        "request": request,
        "version": __version__,
        "app_name": "Cyber-Inference",
        "admin_enabled": settings.admin_password is not None,
        **kwargs,
    }


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """
    Main dashboard page.

    Shows:
    - System status overview
    - Resource usage
    - Active models
    - Recent activity
    """
    logger.debug("GET / - Dashboard")

    if not templates:
        return HTMLResponse(
            content="<h1>Cyber-Inference</h1><p>Templates not found. API is running.</p>",
            status_code=200,
        )

    # Get data for dashboard
    try:
        from cyber_inference.main import get_process_manager, get_resource_monitor

        pm = get_process_manager()
        rm = get_resource_monitor()

        resources = await rm.get_resources()
        running_models = pm.get_running_models()

        context = _template_context(
            request,
            page="dashboard",
            resources={
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "memory_used_gb": resources.used_memory_mb / 1024,
                "memory_total_gb": resources.total_memory_mb / 1024,
                "gpu_info": resources.gpu.name if resources.gpu else None,
                "gpu_memory_used": resources.gpu.used_memory_mb / 1024 if resources.gpu else None,
                "gpu_memory_total": resources.gpu.total_memory_mb / 1024 if resources.gpu else None,
            },
            running_models=running_models,
            model_count=len(running_models),
        )
    except Exception as e:
        logger.warning(f"Could not get dashboard data: {e}")
        context = _template_context(
            request,
            page="dashboard",
            resources=None,
            running_models=[],
            model_count=0,
        )

    return templates.TemplateResponse("dashboard.html", context)


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    """
    Models management page.

    Shows:
    - Downloaded models
    - Model status (loaded/unloaded)
    - Download new models
    """
    logger.debug("GET /models - Models page")

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    try:
        from cyber_inference.services.model_manager import ModelManager
        from cyber_inference.api.v1 import get_auto_loader

        mm = ModelManager()
        auto_loader = get_auto_loader()

        models = await mm.list_models()
        loaded = await auto_loader.get_loaded_models()

        # Add loaded status to each model
        for model in models:
            model["is_loaded"] = model["name"] in loaded

        context = _template_context(
            request,
            page="models",
            models=models,
            loaded_models=loaded,
        )
    except Exception as e:
        logger.warning(f"Could not get models data: {e}")
        context = _template_context(
            request,
            page="models",
            models=[],
            loaded_models=[],
        )

    return templates.TemplateResponse("models.html", context)


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    """
    Settings page.

    Shows:
    - Server configuration
    - Resource limits
    - Admin settings
    """
    logger.debug("GET /settings - Settings page")

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    settings = get_settings()

    context = _template_context(
        request,
        page="settings",
        settings={
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "default_context_size": settings.default_context_size,
            "max_context_size": settings.max_context_size,
            "model_idle_timeout": settings.model_idle_timeout,
            "max_loaded_models": settings.max_loaded_models,
            "max_memory_percent": settings.max_memory_percent,
            "llama_gpu_layers": settings.llama_gpu_layers,
        },
    )

    return templates.TemplateResponse("settings.html", context)


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request) -> HTMLResponse:
    """
    Real-time logs page.

    Shows:
    - Live log stream via WebSocket
    - Log filtering
    - Log level selection
    """
    logger.debug("GET /logs - Logs page")

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    context = _template_context(
        request,
        page="logs",
    )

    return templates.TemplateResponse("logs.html", context)


@router.get("/api-docs", response_class=HTMLResponse)
async def api_docs_page(request: Request) -> HTMLResponse:
    """
    API documentation page.
    """
    logger.debug("GET /api-docs")

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    context = _template_context(
        request,
        page="api-docs",
    )

    return templates.TemplateResponse("api_docs.html", context)

