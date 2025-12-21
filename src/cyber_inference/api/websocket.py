"""
WebSocket endpoints for real-time updates.

Provides:
- Log streaming
- Resource updates
- Model status changes
"""

import asyncio
import json
import logging
import re
from collections import deque
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cyber_inference.core.logging import get_logger

# Regex to strip Rich markup tags like [info], [/info], [highlight], etc.
RICH_MARKUP_PATTERN = re.compile(r'\[/?[a-zA-Z_]+\]')

logger = get_logger(__name__)

router = APIRouter()

# Store for recent logs (ring buffer)
_log_buffer: deque = deque(maxlen=1000)

# Connected WebSocket clients
_log_clients: list[WebSocket] = []
_status_clients: list[WebSocket] = []


class WebSocketLogHandler(logging.Handler):
    """Custom log handler that broadcasts to WebSocket clients."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Get the message and strip Rich markup tags
            message = record.getMessage()
            message = RICH_MARKUP_PATTERN.sub('', message)

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "module": record.name,
                "message": message,
            }

            # Add to buffer
            _log_buffer.append(log_entry)

            # Broadcast to clients (non-blocking)
            if _log_clients:
                asyncio.create_task(_broadcast_log(log_entry))

        except Exception:
            pass  # Don't let logging errors break things


async def _broadcast_log(log_entry: dict) -> None:
    """Broadcast a log entry to all connected clients."""
    message = json.dumps(log_entry)

    for client in _log_clients.copy():
        try:
            await client.send_text(message)
        except Exception:
            _log_clients.remove(client)


async def _broadcast_status(status: dict) -> None:
    """Broadcast a status update to all connected clients."""
    message = json.dumps(status)

    for client in _status_clients.copy():
        try:
            await client.send_text(message)
        except Exception:
            _status_clients.remove(client)


def setup_log_handler() -> None:
    """Setup the WebSocket log handler."""
    handler = WebSocketLogHandler()
    handler.setLevel(logging.DEBUG)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    logger.info("[success]WebSocket log handler installed[/success]")


@router.websocket("/logs")
async def websocket_logs(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time log streaming.

    Clients connect here to receive log messages as they're generated.
    """
    await websocket.accept()
    logger.info("[info]WebSocket client connected to /ws/logs[/info]")

    _log_clients.append(websocket)

    try:
        # Send recent logs
        for log_entry in list(_log_buffer):
            await websocket.send_text(json.dumps(log_entry))

        # Keep connection alive
        while True:
            try:
                # Wait for any messages (for keepalive/commands)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )

                # Handle commands
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text(json.dumps({"type": "keepalive"}))

    except WebSocketDisconnect:
        logger.info("[info]WebSocket client disconnected from /ws/logs[/info]")
    except Exception as e:
        logger.error(f"[error]WebSocket error: {e}[/error]")
    finally:
        if websocket in _log_clients:
            _log_clients.remove(websocket)


@router.websocket("/status")
async def websocket_status(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time status updates.

    Provides:
    - Resource usage updates
    - Model load/unload events
    - Health status
    """
    await websocket.accept()
    logger.info("[info]WebSocket client connected to /ws/status[/info]")

    _status_clients.append(websocket)

    try:
        # Start sending periodic updates
        while True:
            try:
                from cyber_inference.main import get_process_manager, get_resource_monitor

                rm = get_resource_monitor()
                pm = get_process_manager()

                resources = await rm.get_resources()
                running_models = pm.get_running_models()

                status = {
                    "type": "status_update",
                    "timestamp": datetime.now().isoformat(),
                    "resources": {
                        "cpu_percent": resources.cpu_percent,
                        "memory_percent": resources.memory_percent,
                        "memory_used_gb": resources.used_memory_mb / 1024,
                        "memory_total_gb": resources.total_memory_mb / 1024,
                    },
                    "models": {
                        "loaded": running_models,
                        "count": len(running_models),
                    },
                }

                if resources.gpu:
                    status["resources"]["gpu"] = {
                        "name": resources.gpu.name,
                        "memory_used_gb": resources.gpu.used_memory_mb / 1024,
                        "memory_total_gb": resources.gpu.total_memory_mb / 1024,
                    }

                await websocket.send_text(json.dumps(status))

            except Exception as e:
                logger.debug(f"Status update error: {e}")

            # Check for incoming messages (ping/pong)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=2.0,
                )
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.info("[info]WebSocket client disconnected from /ws/status[/info]")
    except Exception as e:
        logger.error(f"[error]WebSocket error: {e}[/error]")
    finally:
        if websocket in _status_clients:
            _status_clients.remove(websocket)


async def notify_model_event(event_type: str, model_name: str, details: Optional[dict] = None) -> None:
    """
    Notify connected clients of a model event.

    Args:
        event_type: Type of event (loaded, unloaded, error)
        model_name: Name of the model
        details: Additional event details
    """
    event = {
        "type": f"model_{event_type}",
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        **(details or {}),
    }

    await _broadcast_status(event)


async def notify_download_progress(
    repo_id: str,
    filename: str,
    progress: float,
    status: str = "downloading",
    error: Optional[str] = None,
) -> None:
    """
    Notify connected clients of download progress.

    Args:
        repo_id: HuggingFace repo ID
        filename: File being downloaded
        progress: Progress percentage (0-100)
        status: Status string (downloading, complete, error)
        error: Error message if status is error
    """
    event = {
        "type": "download_progress",
        "timestamp": datetime.now().isoformat(),
        "repo_id": repo_id,
        "filename": filename,
        "progress": progress,
        "status": status,
    }

    if error:
        event["error"] = error

    await _broadcast_status(event)

