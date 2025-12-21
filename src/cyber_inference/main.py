"""
Main FastAPI application for Cyber-Inference.

This module initializes and configures the FastAPI application with:
- CORS middleware
- Static file serving
- Template rendering
- API routers (v1, admin, web)
- Lifespan management for startup/shutdown
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cyber_inference import __version__
from cyber_inference.core.config import get_settings
from cyber_inference.core.database import init_database
from cyber_inference.core.logging import get_logger, setup_logging
from cyber_inference.services.process_manager import ProcessManager
from cyber_inference.services.resource_monitor import ResourceMonitor
from cyber_inference.services.auto_loader import AutoLoader
from cyber_inference.api.websocket import setup_log_handler

logger = get_logger(__name__)

# Global service instances
process_manager: ProcessManager | None = None
resource_monitor: ResourceMonitor | None = None
auto_loader: AutoLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifespan - startup and shutdown.
    """
    global process_manager, resource_monitor, auto_loader

    logger.info("[highlight]═══════════════════════════════════════════════════════════[/highlight]")
    logger.info("[highlight]           Cyber-Inference Starting Up                      [/highlight]")
    logger.info("[highlight]═══════════════════════════════════════════════════════════[/highlight]")

    settings = get_settings()

    # Setup logging
    setup_logging(log_dir=settings.log_dir, level=settings.log_level_int)

    # Setup WebSocket log handler for real-time log streaming
    setup_log_handler()

    # Initialize database
    logger.info("[info]Initializing database...[/info]")
    await init_database(settings.database_path)
    logger.info("[success]Database initialized successfully[/success]")

    # Initialize resource monitor
    logger.info("[info]Starting resource monitor...[/info]")
    resource_monitor = ResourceMonitor()
    await resource_monitor.start()
    logger.info("[success]Resource monitor started[/success]")

    # Initialize process manager
    logger.info("[info]Initializing process manager...[/info]")
    process_manager = ProcessManager(
        models_dir=settings.models_dir,
        bin_dir=settings.bin_dir,
    )
    await process_manager.initialize()
    logger.info("[success]Process manager initialized[/success]")

    # Initialize and start auto-loader (for idle model unloading)
    logger.info("[info]Starting auto-loader...[/info]")
    auto_loader = AutoLoader(
        process_manager=process_manager,
        resource_monitor=resource_monitor,
    )
    await auto_loader.start()
    logger.info(f"[success]Auto-loader started (idle timeout: {settings.model_idle_timeout}s)[/success]")

    # Log system info
    sys_info = await resource_monitor.get_system_info()
    logger.info(f"[info]System Information:[/info]")
    logger.info(f"  Platform: {sys_info['platform']}")
    logger.info(f"  CPU Cores: {sys_info['cpu_count']}")
    logger.info(f"  Total RAM: {sys_info['total_memory_gb']:.1f} GB")
    logger.info(f"  GPU: {sys_info.get('gpu_info', 'Not detected')}")

    logger.info("[success]═══════════════════════════════════════════════════════════[/success]")
    logger.info("[success]           Cyber-Inference Ready!                           [/success]")
    logger.info(f"[success]           Web GUI: http://localhost:{settings.port}/            [/success]")
    logger.info(f"[success]           API: http://localhost:{settings.port}/v1/             [/success]")
    logger.info("[success]═══════════════════════════════════════════════════════════[/success]")

    yield

    # Shutdown
    logger.info("[warning]═══════════════════════════════════════════════════════════[/warning]")
    logger.info("[warning]           Cyber-Inference Shutting Down                    [/warning]")
    logger.info("[warning]═══════════════════════════════════════════════════════════[/warning]")

    # Stop auto-loader first
    if auto_loader:
        logger.info("[info]Stopping auto-loader...[/info]")
        await auto_loader.stop()
        logger.info("[success]Auto-loader stopped[/success]")

    # Stop all running inference servers
    if process_manager:
        logger.info("[info]Stopping all inference servers...[/info]")
        await process_manager.shutdown()
        logger.info("[success]All inference servers stopped[/success]")

    # Stop resource monitor
    if resource_monitor:
        logger.info("[info]Stopping resource monitor...[/info]")
        await resource_monitor.stop()
        logger.info("[success]Resource monitor stopped[/success]")

    logger.info("[success]Cyber-Inference shutdown complete[/success]")


# Create FastAPI application
app = FastAPI(
    title="Cyber-Inference",
    description="Edge Inference Server Management with OpenAI-compatible API",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine paths
_src_dir = Path(__file__).parent
_web_dir = _src_dir / "web"
_templates_dir = _web_dir / "templates"
_static_dir = _web_dir / "static"

# Mount static files
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=_templates_dir) if _templates_dir.exists() else None


# Import and include routers
from cyber_inference.api.v1 import router as v1_router
from cyber_inference.api.admin import router as admin_router
from cyber_inference.api.web import router as web_router
from cyber_inference.api.websocket import router as ws_router

app.include_router(v1_router, prefix="/v1", tags=["OpenAI API"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])
app.include_router(web_router, tags=["Web GUI"])


@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint for monitoring.
    """
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "version": __version__,
        "service": "cyber-inference",
    }


def get_process_manager() -> ProcessManager:
    """Get the global process manager instance."""
    if process_manager is None:
        raise RuntimeError("Process manager not initialized")
    return process_manager


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    if resource_monitor is None:
        raise RuntimeError("Resource monitor not initialized")
    return resource_monitor


def get_auto_loader() -> AutoLoader:
    """Get the global auto-loader instance."""
    if auto_loader is None:
        raise RuntimeError("Auto-loader not initialized")
    return auto_loader

