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
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from urllib.parse import quote

from cyber_inference import __version__
from cyber_inference.core.auth import extract_bearer_token, verify_admin_token_value
from cyber_inference.core.config import get_settings, load_db_config_overrides
from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Setup templates
_templates_dir = Path(__file__).parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=_templates_dir) if _templates_dir.exists() else None

_CONFIG_UI_LABELS = {
    "default_context_size": "Default Context Size",
    "max_context_size": "Max Context Size",
    "model_idle_timeout": "Idle Timeout (seconds)",
    "max_loaded_models": "Max Loaded Models",
    "max_memory_percent": "Max Memory Usage (%)",
    "llama_gpu_layers": "GPU Layers",
    "admin_password": "Admin Password",
}


def _template_context(request: Request, **kwargs) -> dict:
    """Build common template context."""
    settings = get_settings()
    return {
        "request": request,
        "version": __version__,
        "app_name": "Cyber-Inference",
        "admin_enabled": bool(settings.admin_password),
        **kwargs,
    }


def _build_next_url(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"
    return path


async def _require_admin(request: Request) -> RedirectResponse | None:
    settings = get_settings()
    if not settings.admin_password:
        return None

    token = request.cookies.get("admin_token") or extract_bearer_token(
        request.headers.get("authorization")
    )
    invalid_token = False
    if token:
        if verify_admin_token_value(token):
            return None
        invalid_token = True

    next_url = quote(_build_next_url(request))
    error_param = "&error=invalid" if invalid_token else ""
    response = RedirectResponse(
        url=f"/login?next={next_url}{error_param}",
        status_code=303,
    )
    response.delete_cookie("admin_token")
    return response


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

    redirect = await _require_admin(request)
    if redirect:
        return redirect

    if not templates:
        return HTMLResponse(
            content="<h1>Cyber-Inference</h1><p>Templates not found. API is running.</p>",
            status_code=200,
        )

    # Get SGLang status
    from cyber_inference.services.sglang_manager import SGLangManager
    sglang_mgr = SGLangManager.get_instance()
    sglang_available = sglang_mgr.is_available()
    sglang_version = sglang_mgr.get_version() if sglang_available else None

    # Get data for dashboard
    try:
        from cyber_inference.main import get_process_manager, get_resource_monitor

        pm = get_process_manager()
        rm = get_resource_monitor()

        resources = await rm.get_resources()
        running_models = pm.get_running_models()

        # Build model info with server type for the dashboard
        running_models_info = []
        for model_name in running_models:
            proc = pm.get_process(model_name)
            running_models_info.append({
                "name": model_name,
                "server_type": proc.server_type if proc else "llama",
            })

        gpu_info = resources.gpu.name if resources.gpu else None
        gpu_memory_total = None
        gpu_memory_used = None
        gpu_memory_note = resources.gpu.memory_note if resources.gpu else None
        if resources.gpu and resources.gpu.total_memory_mb > 0:
            gpu_memory_total = resources.gpu.total_memory_mb / 1024
            if resources.gpu.used_memory_mb is not None:
                gpu_memory_used = resources.gpu.used_memory_mb / 1024

        context = _template_context(
            request,
            page="dashboard",
            resources={
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "memory_used_gb": resources.used_memory_mb / 1024,
                "memory_total_gb": resources.total_memory_mb / 1024,
                "gpu_info": gpu_info,
                "gpu_memory_total": gpu_memory_total,
                "gpu_memory_used": gpu_memory_used,
                "gpu_memory_note": gpu_memory_note,
            },
            running_models=running_models_info,
            model_count=len(running_models),
            sglang_available=sglang_available,
            sglang_version=sglang_version,
            transformers_available=True,
        )
    except Exception as e:
        logger.warning(f"Could not get dashboard data: {e}")
        context = _template_context(
            request,
            page="dashboard",
            resources=None,
            running_models=[],
            model_count=0,
            sglang_available=sglang_available,
            sglang_version=sglang_version,
            transformers_available=True,
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

    redirect = await _require_admin(request)
    if redirect:
        return redirect

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    from cyber_inference.services.sglang_manager import SGLangManager
    sglang_available = SGLangManager.get_instance().is_available()

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
            sglang_available=sglang_available,
            transformers_available=True,
        )
    except Exception as e:
        logger.warning(f"Could not get models data: {e}")
        context = _template_context(
            request,
            page="models",
            models=[],
            loaded_models=[],
            sglang_available=sglang_available,
            transformers_available=True,
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

    redirect = await _require_admin(request)
    if redirect:
        return redirect

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    settings = get_settings()
    overrides = await load_db_config_overrides()

    from cyber_inference.services.sglang_manager import SGLangManager
    sglang_mgr = SGLangManager.get_instance()
    sglang_available = sglang_mgr.is_available()
    sglang_version = sglang_mgr.get_version() if sglang_available else None

    runtime_settings = {
        "default_context_size": settings.default_context_size,
        "max_context_size": settings.max_context_size,
        "model_idle_timeout": settings.model_idle_timeout,
        "max_loaded_models": settings.max_loaded_models,
        "max_memory_percent": settings.max_memory_percent,
        "llama_gpu_layers": settings.llama_gpu_layers,
    }
    saved_settings = dict(runtime_settings)
    for key, value in overrides.items():
        if key in saved_settings:
            saved_settings[key] = value

    pending_restart_items = []
    for key, saved_value in overrides.items():
        if key == "admin_password":
            current_value = settings.admin_password
            if saved_value == current_value:
                continue
            current_display = "set" if current_value else "not set"
            if saved_value and current_value:
                saved_display = "updated"
            else:
                saved_display = "set" if saved_value else "not set"
            pending_restart_items.append(
                {
                    "key": key,
                    "label": _CONFIG_UI_LABELS.get(key, key.replace("_", " ").title()),
                    "current": current_display,
                    "saved": saved_display,
                }
            )
            continue

        current_value = runtime_settings.get(key)
        if current_value is None:
            continue
        if saved_value != current_value:
            pending_restart_items.append(
                {
                    "key": key,
                    "label": _CONFIG_UI_LABELS.get(key, key.replace("_", " ").title()),
                    "current": current_value,
                    "saved": saved_value,
                }
            )

    context = _template_context(
        request,
        page="settings",
        settings={
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "default_context_size": saved_settings["default_context_size"],
            "max_context_size": saved_settings["max_context_size"],
            "model_idle_timeout": saved_settings["model_idle_timeout"],
            "max_loaded_models": saved_settings["max_loaded_models"],
            "max_memory_percent": saved_settings["max_memory_percent"],
            "llama_gpu_layers": saved_settings["llama_gpu_layers"],
            "sglang_mem_fraction": settings.sglang_mem_fraction,
            "sglang_tp_size": settings.sglang_tp_size,
        },
        pending_restart_items=pending_restart_items,
        sglang_available=sglang_available,
        sglang_version=sglang_version,
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

    redirect = await _require_admin(request)
    if redirect:
        return redirect

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

    redirect = await _require_admin(request)
    if redirect:
        return redirect

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    context = _template_context(
        request,
        page="api-docs",
    )

    return templates.TemplateResponse("api_docs.html", context)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """
    Admin login page.
    """
    logger.debug("GET /login - Login page")

    settings = get_settings()
    if not settings.admin_password:
        return RedirectResponse(url="/", status_code=303)

    token = request.cookies.get("admin_token")
    if token and verify_admin_token_value(token):
        next_url = request.query_params.get("next") or "/"
        return RedirectResponse(url=next_url, status_code=303)

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    context = _template_context(
        request,
        page="login",
        next_url=request.query_params.get("next") or "/",
        error=request.query_params.get("error"),
    )

    return templates.TemplateResponse("login.html", context)
