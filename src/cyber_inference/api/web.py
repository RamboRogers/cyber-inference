"""
Web GUI routes for Cyber-Inference.

Serves the HTML templates for:
- Dashboard (/)
- Models management (/models)
- Settings (/settings)
- Logs (/logs)
"""

from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

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
    "model_idle_unload_enabled": "Idle Unload Timer",
    "model_idle_timeout": "Idle Timeout (seconds)",
    "model_load_timeout": "Model Load Timeout (seconds)",
    "pre_model_load_command_enabled": "Pre-Model Load Command",
    "pre_model_load_command": "Pre-Model Load Command",
    "pre_model_load_command_timeout": "Pre-Model Load Command Timeout",
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


def _copy_forward_headers(request: Request) -> dict[str, str]:
    forwarded_headers: dict[str, str] = {}
    for header_name in ("accept", "content-type", "origin", "referer", "user-agent"):
        header_value = request.headers.get(header_name)
        if header_value:
            forwarded_headers[header_name] = header_value
    return forwarded_headers


def _copy_response_headers(response: httpx.Response) -> dict[str, str]:
    proxied_headers: dict[str, str] = {}
    for header_name in ("content-type", "cache-control", "etag", "last-modified"):
        header_value = response.headers.get(header_name)
        if header_value:
            proxied_headers[header_name] = header_value
    return proxied_headers


async def _resolve_llama_chat_backend(
    model_name: str,
    load_trigger: str = "public_autoload",
) -> tuple[str, dict]:
    from cyber_inference.api.v1 import get_auto_loader

    auto_loader = get_auto_loader()
    server_url = await auto_loader.ensure_model_loaded(
        model_name,
        load_trigger=load_trigger,
    )
    status_info = await auto_loader.get_model_status(model_name)

    if status_info.get("server_type") != "llama":
        raise HTTPException(status_code=400, detail="Chat UI is only available for llama.cpp models.")

    return server_url, status_info


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
async def dashboard(request: Request) -> Response:
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
            transformers_available=True,
        )

    return templates.TemplateResponse("dashboard.html", context)


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> Response:
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

    try:
        from cyber_inference.api.v1 import get_auto_loader
        from cyber_inference.services.model_manager import ModelManager

        mm = ModelManager()
        auto_loader = get_auto_loader()

        models = await mm.list_models(include_file_metadata=False)
        loaded = await auto_loader.get_loaded_models()
        model_statuses = await auto_loader.get_models_status(models)

        # Add runtime and simple capability status to each model.
        for model in models:
            status_info = model_statuses.get(model["name"], {"status": "not_loaded"})
            effective_config = status_info.get("effective_config", {})
            tool_info = effective_config.get("tool_calling", {})
            vision_info = effective_config.get("vision", {})

            model["is_loaded"] = bool(status_info.get("is_loaded", model["name"] in loaded))
            model["status"] = status_info.get("status", "not_loaded")
            model["server_type"] = status_info.get("server_type", model.get("engine_type", "llama"))
            model["supports_tools"] = tool_info.get("status") == "supported"
            model["tool_calling_status"] = tool_info.get("status", "unknown")
            model["supports_vision"] = bool(
                vision_info.get("enabled") or model.get("mmproj_path") or model.get("is_vlm")
            )

        context = _template_context(
            request,
            page="models",
            models=models,
            loaded_models=loaded,
            transformers_available=True,
        )
    except Exception as e:
        logger.warning(f"Could not get models data: {e}")
        context = _template_context(
            request,
            page="models",
            models=[],
            loaded_models=[],
            transformers_available=True,
        )

    return templates.TemplateResponse("models.html", context)


@router.api_route(
    "/chat/{model_name:path}/ui/{proxy_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def llama_chat_ui_proxy(request: Request, model_name: str, proxy_path: str) -> Response:
    """Proxy the llama.cpp built-in web UI under the Cyber-Inference origin."""
    redirect = await _require_admin(request)
    if redirect:
        return redirect

    server_url, _ = await _resolve_llama_chat_backend(
        model_name,
        load_trigger="admin_manual",
    )
    upstream_path = proxy_path or ""
    upstream_url = f"{server_url}/{upstream_path}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        upstream_response = await client.request(
            request.method,
            upstream_url,
            content=await request.body(),
            headers=_copy_forward_headers(request),
        )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_copy_response_headers(upstream_response),
    )


@router.get("/chat/{model_name:path}", response_class=HTMLResponse)
async def chat_page(request: Request, model_name: str) -> Response:
    """GUI-hosted entry point for the llama.cpp chat interface."""
    redirect = await _require_admin(request)
    if redirect:
        return redirect

    if not templates:
        return HTMLResponse(content="Templates not found", status_code=500)

    _, status_info = await _resolve_llama_chat_backend(
        model_name,
        load_trigger="admin_manual",
    )
    context = _template_context(
        request,
        page="chat",
        model_name=model_name,
        chat_ui_path=f"/chat/{quote(model_name, safe='')}/ui/",
        server_type=status_info.get("server_type", "llama"),
    )
    return templates.TemplateResponse("chat.html", context)


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> Response:
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

    runtime_settings: dict[str, object] = {
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
    }
    saved_settings: dict[str, object] = dict(runtime_settings)
    for key, value in overrides.items():
        if key in saved_settings:
            saved_settings[key] = value

    pending_restart_items: list[dict[str, object]] = []
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

        runtime_value: object = runtime_settings.get(key)
        if runtime_value is None:
            continue
        if saved_value != runtime_value:
            pending_restart_items.append(
                {
                    "key": key,
                    "label": _CONFIG_UI_LABELS.get(key, key.replace("_", " ").title()),
                    "current": runtime_value,
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
            "model_idle_unload_enabled": saved_settings["model_idle_unload_enabled"],
            "model_idle_timeout": saved_settings["model_idle_timeout"],
            "model_load_timeout": saved_settings["model_load_timeout"],
            "pre_model_load_command_enabled": saved_settings["pre_model_load_command_enabled"],
            "pre_model_load_command": saved_settings["pre_model_load_command"],
            "pre_model_load_command_timeout": saved_settings["pre_model_load_command_timeout"],
            "max_loaded_models": saved_settings["max_loaded_models"],
            "max_memory_percent": saved_settings["max_memory_percent"],
            "llama_gpu_layers": saved_settings["llama_gpu_layers"],
        },
        pending_restart_items=pending_restart_items,
    )

    return templates.TemplateResponse("settings.html", context)


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request) -> Response:
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
async def api_docs_page(request: Request) -> Response:
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
async def login_page(request: Request) -> Response:
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
