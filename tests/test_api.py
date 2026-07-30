"""
Integration tests for Cyber-Inference API endpoints.

Tests cover:
- Health endpoint
- V1 models endpoint
- Admin endpoints
- Error handling
"""

import json
import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from cyber_inference.core.database import Base, get_db
from cyber_inference.main import app
from cyber_inference.models.db_models import Configuration, Model
from cyber_inference.models.schemas import ChatCompletionRequest, ChatMessage


def make_test_client() -> AsyncClient:
    """Create a basic ASGI-backed test client."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health check endpoint."""
    async with make_test_client() as client:
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["service"] == "cyber-inference"


@pytest.mark.asyncio
async def test_health_endpoint_remains_public_when_admin_password_is_set(monkeypatch):
    """Container probes should not require admin credentials."""
    from cyber_inference.core.config import reload_settings

    monkeypatch.setenv("CYBER_INFERENCE_ADMIN_PASSWORD", "secret")
    reload_settings()

    try:
        async with make_test_client() as client:
            health_response = await client.get("/health")
            admin_response = await client.get("/admin/status")
    finally:
        app.dependency_overrides.clear()
        monkeypatch.delenv("CYBER_INFERENCE_ADMIN_PASSWORD", raising=False)
        reload_settings()

    assert health_response.status_code == 200
    health_payload = health_response.json()
    assert health_payload["status"] == "healthy"
    assert health_payload["service"] == "cyber-inference"
    assert "admin_password" not in health_payload
    assert "jwt_secret" not in health_payload
    assert admin_response.status_code == 401


@pytest.mark.asyncio
async def test_v1_models_endpoint():
    """Test the /v1/models endpoint."""
    async def override_get_db():
        yield MagicMock()

    auto_loader = MagicMock()
    auto_loader.list_available_models = AsyncMock(return_value=[])
    auto_loader.get_models_status = AsyncMock(return_value={})

    app.dependency_overrides[get_db] = override_get_db
    try:
        with patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader):
            async with make_test_client() as client:
                response = await client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_v1_models_endpoint_includes_runtime_and_capabilities():
    """The model list should expose loaded state and simple capability metadata."""
    async def override_get_db():
        yield MagicMock()

    auto_loader = MagicMock()
    auto_loader.list_available_models = AsyncMock(
        return_value=[
            {
                "name": "demo-vlm",
                "engine_type": "llama",
                "model_type": "chat",
                "mmproj_path": "/tmp/mmproj-demo.gguf",
                "is_vlm": True,
                "context_length": 131072,
                "default_context_size": 65536,
            }
        ]
    )
    auto_loader.get_models_status = AsyncMock(
        return_value={
            "demo-vlm": {
                "is_loaded": True,
                "status": "running",
                "server_type": "llama",
                "effective_config": {
                    "tool_calling": {"status": "supported"},
                    "vision": {"enabled": True},
                    "launch_config": {
                        "context_size": 65536,
                        "configured_context_size": 65536,
                        "native_context_size": 131072,
                        "context_source": "configured_default",
                    },
                },
            }
        }
    )

    app.dependency_overrides[get_db] = override_get_db
    try:
        with patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader):
            async with make_test_client() as client:
                response = await client.get("/v1/models")

        assert response.status_code == 200
        payload = response.json()
        model = payload["data"][0]
        assert model["id"] == "demo-vlm"
        assert model["is_loaded"] is True
        assert model["status"] == "running"
        assert model["server_type"] == "llama"
        assert model["capabilities"]["vision"] is True
        assert model["capabilities"]["tool_calling"] == "supported"
        assert model["context"] == {
            "length": 131072,
            "window": 131072,
            "effective_length": 65536,
            "configured_length": 65536,
            "native_length": 131072,
            "source": "configured_default",
        }
        assert model["context_length"] == 131072
        assert model["max_context_length"] == 131072
        assert model["context_window"] == 131072
        assert model["effective_context_length"] == 65536
        auto_loader.get_models_status.assert_awaited_once()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_v1_model_detail_includes_context_metadata():
    """The model detail endpoint should expose the same context metadata as the list endpoint."""
    auto_loader = MagicMock()
    auto_loader.get_model_info = AsyncMock(
        return_value={
            "name": "demo-model",
            "engine_type": "llama",
            "model_type": "chat",
            "context_length": 32768,
            "default_context_size": None,
        }
    )
    auto_loader.get_model_status = AsyncMock(
        return_value={
            "is_loaded": True,
            "status": "running",
            "server_type": "llama",
            "effective_config": {
                "tool_calling": {"status": "unknown"},
                "launch_config": {
                    "context_size": 8192,
                    "configured_context_size": None,
                    "native_context_size": 32768,
                    "context_source": "running",
                },
            },
        }
    )

    with patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader):
        async with make_test_client() as client:
            response = await client.get("/v1/models/demo-model")

    assert response.status_code == 200
    model = response.json()
    assert model["id"] == "demo-model"
    assert model["context"]["length"] == 32768
    assert model["context"]["window"] == 32768
    assert model["context"]["effective_length"] == 8192
    assert model["context"]["configured_length"] is None
    assert model["context"]["native_length"] == 32768
    assert model["context"]["source"] == "running"
    assert model["context_length"] == 32768
    assert model["max_context_length"] == 32768
    assert model["context_window"] == 32768
    assert model["effective_context_length"] == 8192


def test_v1_model_info_reports_native_context_when_runtime_context_is_capped():
    """Every discovery alias should expose native capacity without hiding the runtime limit."""
    from cyber_inference.api.v1 import _build_model_info

    model = {
        "name": "qwen-256k",
        "engine_type": "llama",
        "model_type": "chat",
        "context_length": 262144,
        "default_context_size": None,
    }
    status_info = {
        "is_loaded": True,
        "status": "running",
        "server_type": "llama",
        "effective_config": {
            "launch_config": {
                "context_size": 262144,
                "configured_context_size": None,
                "native_context_size": 262144,
                "context_source": "model_native_max",
            },
        },
    }

    model_info = _build_model_info(model, status_info)

    assert model_info.context is not None
    assert model_info.context.length == 262144
    assert model_info.context.window == 262144
    assert model_info.context.effective_length == 262144
    assert model_info.context.native_length == 262144
    assert model_info.context.source == "model_native_max"
    assert model_info.context_length == 262144
    assert model_info.max_context_length == 262144
    assert model_info.context_window == 262144
    assert model_info.effective_context_length == 262144

    fallback_info = _build_model_info(
        {"name": "unknown-native", "engine_type": "llama", "model_type": "chat"},
        {
            "effective_config": {
                "launch_config": {
                    "context_size": 8192,
                    "native_context_size": None,
                },
            },
        },
    )

    assert fallback_info.context is not None
    assert fallback_info.context.native_length is None
    assert fallback_info.context.length == 8192
    assert fallback_info.context.window == 8192
    assert fallback_info.context.effective_length == 8192
    assert fallback_info.context_length == 8192
    assert fallback_info.max_context_length == 8192
    assert fallback_info.context_window == 8192
    assert fallback_info.effective_context_length == 8192


@pytest.mark.asyncio
async def test_v1_models_context_metadata_tolerates_missing_launch_config():
    """Missing launch config should not break model-list serialization."""
    async def override_get_db():
        yield MagicMock()

    auto_loader = MagicMock()
    auto_loader.list_available_models = AsyncMock(
        return_value=[
            {
                "name": "minimal-model",
                "engine_type": "llama",
                "model_type": "chat",
            }
        ]
    )
    auto_loader.get_models_status = AsyncMock(
        return_value={
            "minimal-model": {
                "is_loaded": False,
                "status": "not_loaded",
                "server_type": "llama",
                "effective_config": {"tool_calling": {"status": "unsupported"}},
            }
        }
    )

    app.dependency_overrides[get_db] = override_get_db
    try:
        with patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader):
            async with make_test_client() as client:
                response = await client.get("/v1/models")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["context"] == {
        "length": None,
        "window": None,
        "effective_length": None,
        "configured_length": None,
        "native_length": None,
        "source": None,
    }
    assert model["context_length"] is None
    assert model["max_context_length"] is None
    assert model["context_window"] is None
    assert model["effective_context_length"] is None


@pytest.mark.asyncio
async def test_v1_chat_completions_no_model():
    """Test chat completions with non-existent model."""
    async with make_test_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "non-existent-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Should return 503 when model not available
        assert response.status_code in [503, 404, 500]


@pytest.mark.asyncio
async def test_admin_status_endpoint():
    """Test the admin status endpoint."""
    resource_monitor = MagicMock()
    resource_monitor.get_resources = AsyncMock(
        return_value=MagicMock(cpu_percent=1.0, memory_percent=2.0)
    )
    resource_monitor.has_gpu.return_value = False
    process_manager = MagicMock()
    process_manager.get_running_models.return_value = []

    with (
        patch("cyber_inference.main.get_resource_monitor", return_value=resource_monitor),
        patch("cyber_inference.main.get_process_manager", return_value=process_manager),
    ):
        async with make_test_client() as client:
            response = await client.get("/admin/status")

    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "status" in data
    assert "running_models" in data


@pytest.mark.asyncio
async def test_admin_resources_endpoint():
    """Test the admin resources endpoint."""
    resource_monitor = MagicMock()
    resource_monitor.get_resources = AsyncMock(
        return_value=MagicMock(
            timestamp="test",
            cpu_count=8,
            cpu_percent=1.0,
            total_memory_mb=1024,
            available_memory_mb=512,
            memory_percent=50.0,
            gpu=None,
        )
    )

    with patch("cyber_inference.main.get_resource_monitor", return_value=resource_monitor):
        async with make_test_client() as client:
            response = await client.get("/admin/resources")

    assert response.status_code == 200
    data = response.json()
    assert "cpu_count" in data
    assert "cpu_percent" in data
    assert "memory_percent" in data


@pytest.mark.asyncio
async def test_admin_models_list():
    """Test listing models via admin API."""
    manager = MagicMock()
    manager.list_models = AsyncMock(return_value=[])

    with patch("cyber_inference.api.admin.ModelManager", return_value=manager):
        async with make_test_client() as client:
            response = await client.get("/admin/models")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_admin_models_list_resolves_relative_paths(tmp_path: Path):
    """Admin models response should expose resolved absolute paths even when DB storage is relative."""
    from cyber_inference.services.model_manager import ModelManager

    models_dir = tmp_path / "models"
    nested_dir = models_dir / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    model_path = nested_dir / "demo.gguf"
    mmproj_path = nested_dir / "mmproj-demo.gguf"
    model_path.write_bytes(b"demo")
    mmproj_path.write_bytes(b"mmproj")

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api-relative.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def db_session():
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async with db_session() as session:
        session.add(
            Model(
                name="demo-model",
                filename="demo.gguf",
                file_path="nested/demo.gguf",
                mmproj_path="nested/mmproj-demo.gguf",
                size_bytes=4,
                context_length=4096,
                is_downloaded=True,
                is_enabled=True,
                download_progress=100.0,
            )
        )

    manager = ModelManager(models_dir=models_dir)

    with (
        patch("cyber_inference.api.admin.ModelManager", return_value=manager),
        patch("cyber_inference.services.model_manager.get_db_session", db_session),
    ):
        async with make_test_client() as client:
            response = await client.get("/admin/models")

    await engine.dispose()

    assert response.status_code == 200
    data = response.json()
    assert Path(data[0]["file_path"]) == model_path.resolve(strict=False)
    assert Path(data[0]["mmproj_path"]) == mmproj_path.resolve(strict=False)


@pytest.mark.asyncio
async def test_admin_sessions_list():
    """Test listing active sessions."""
    process_manager = MagicMock()
    process_manager.get_all_processes.return_value = []

    with patch("cyber_inference.main.get_process_manager", return_value=process_manager):
        async with make_test_client() as client:
            response = await client.get("/admin/sessions")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_admin_config():
    """Test getting configuration."""
    async with make_test_client() as client:
        response = await client.get("/admin/config")

        assert response.status_code == 200
        data = response.json()
        assert "host" in data
        assert "port" in data
        assert "log_level" in data
        assert "model_idle_unload_enabled" in data
        assert data["model_load_timeout"] == 300


def _llama_release_info() -> dict[str, str | None]:
    return {
        "tag_name": "b1001",
        "name": "Build 1001",
        "html_url": "https://github.com/ggerganov/llama.cpp/releases/tag/b1001",
        "published_at": "2026-04-11T00:00:00Z",
        "compatible_asset": "llama-b1001-macos-arm64.zip",
    }


def _llama_binary_status(source: str = "managed") -> dict[str, object]:
    is_system = source == "system"
    is_bundled = source == "bundled"
    binary_path = "/usr/local/bin/llama-server" if is_system else "/tmp/bin/llama-server"
    return {
        "source": source,
        "binary_path": None if source == "missing" else binary_path,
        "managed_binary_path": "/tmp/bin/llama-server",
        "installed_version": None if source == "missing" else "version: 1000 (abc123)",
        "supports_draft_mtp": source != "missing",
        "is_system_managed": is_system,
        "update_allowed": not (is_system or is_bundled),
        "update_blocked_reason": (
            "System-managed binary detected."
            if is_system
            else "Pull the thor-arm64 image to update the bundled CUDA binary."
            if is_bundled
            else None
        ),
    }


def _llama_installer_mock(source: str = "managed") -> MagicMock:
    installer = MagicMock()
    installer.get_binary_status = AsyncMock(return_value=_llama_binary_status(source))
    installer.get_latest_release_info = AsyncMock(return_value=_llama_release_info())
    installer.get_update_available = MagicMock(return_value=True)
    installer.install = AsyncMock(return_value=Path("/tmp/bin/llama-server"))
    return installer


@pytest.mark.asyncio
async def test_admin_llama_cpp_status_reports_system_managed_binary():
    """llama.cpp status should disclose system-managed provenance."""
    installer = _llama_installer_mock("system")

    with (
        patch("cyber_inference.api.admin._get_llama_installer", return_value=installer),
        patch("cyber_inference.api.admin._get_running_llama_session_names", return_value=[]),
    ):
        async with make_test_client() as client:
            response = await client.get("/admin/llama-cpp/status")

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "system"
    assert data["binary_path"] == "/usr/local/bin/llama-server"
    assert data["managed_binary_path"] == "/tmp/bin/llama-server"
    assert data["installed_version"] == "version: 1000 (abc123)"
    assert data["supports_draft_mtp"] is True
    assert data["latest_release"]["tag_name"] == "b1001"
    assert data["update_allowed"] is False
    assert data["update_blocked_reason"] == "System-managed binary detected."


@pytest.mark.asyncio
async def test_admin_llama_cpp_status_survives_latest_release_failure():
    """Latest-release errors should not hide local llama.cpp status."""
    installer = _llama_installer_mock("managed")
    installer.get_latest_release_info = AsyncMock(side_effect=RuntimeError("GitHub unavailable"))

    with (
        patch("cyber_inference.api.admin._get_llama_installer", return_value=installer),
        patch("cyber_inference.api.admin._get_running_llama_session_names", return_value=[]),
    ):
        async with make_test_client() as client:
            response = await client.get("/admin/llama-cpp/status")

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "managed"
    assert data["latest_release"] is None
    assert data["latest_release_error"] == "GitHub unavailable"
    assert data["update_available"] is None


@pytest.mark.asyncio
async def test_admin_llama_cpp_update_refuses_system_managed_binary():
    """The admin updater must not overwrite PATH/system binaries."""
    installer = _llama_installer_mock("system")

    with (
        patch("cyber_inference.api.admin._get_llama_installer", return_value=installer),
        patch("cyber_inference.api.admin._get_running_llama_session_names", return_value=[]),
    ):
        async with make_test_client() as client:
            response = await client.post("/admin/llama-cpp/update")

    assert response.status_code == 409
    installer.install.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_llama_cpp_update_refuses_bundled_thor_binary():
    """The admin updater must preserve the CUDA binary supplied by the Thor image."""
    installer = _llama_installer_mock("bundled")

    with (
        patch("cyber_inference.api.admin._get_llama_installer", return_value=installer),
        patch("cyber_inference.api.admin._get_running_llama_session_names", return_value=[]),
    ):
        async with make_test_client() as client:
            response = await client.post("/admin/llama-cpp/update")

    assert response.status_code == 409
    assert "thor-arm64" in response.json()["detail"]
    installer.install.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_llama_cpp_update_refuses_running_llama_sessions():
    """The admin updater should block binary replacement while llama sessions run."""
    installer = _llama_installer_mock("managed")

    with (
        patch("cyber_inference.api.admin._get_llama_installer", return_value=installer),
        patch(
            "cyber_inference.api.admin._get_running_llama_session_names",
            return_value=["demo-model"],
        ),
    ):
        async with make_test_client() as client:
            response = await client.post("/admin/llama-cpp/update")

    assert response.status_code == 409
    assert "Unload llama.cpp models before updating" in response.json()["detail"]
    installer.install.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_llama_cpp_update_installs_managed_binary():
    """Managed or missing installs should run force update and refresh status."""
    installer = _llama_installer_mock("missing")
    installer.get_binary_status = AsyncMock(
        side_effect=[
            _llama_binary_status("missing"),
            _llama_binary_status("managed"),
        ]
    )

    with (
        patch("cyber_inference.api.admin._get_llama_installer", return_value=installer),
        patch("cyber_inference.api.admin._get_running_llama_session_names", return_value=[]),
    ):
        async with make_test_client() as client:
            response = await client.post("/admin/llama-cpp/update")

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "managed"
    assert data["binary_path"] == "/tmp/bin/llama-server"
    installer.install.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
async def test_chat_completion_validation():
    """Test request validation for chat completions."""
    async with make_test_client() as client:
        # Missing model
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 422

        # Missing messages
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
            },
        )
        assert response.status_code == 422

        # Invalid temperature
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 5.0,  # Invalid: > 2.0
            },
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_apply_model_defaults_uses_supported_saved_fields():
    """Saved model defaults should populate supported request fields only."""
    from cyber_inference.api.v1 import _apply_model_defaults

    request = ChatCompletionRequest(
        model="demo",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    auto_loader = MagicMock()
    auto_loader.get_request_defaults = AsyncMock(
        return_value={
            "temperature": 0.33,
            "top_p": 0.91,
            "top_k": 24,
            "max_tokens": 128,
            "repeat_penalty": 1.07,
        }
    )

    with patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader):
        await _apply_model_defaults(request, "demo", "llama")

    assert request.temperature == 0.33
    assert request.top_p == 0.91
    assert request.top_k == 24
    assert request.max_tokens == 128
    assert request.repeat_penalty == 1.07


@pytest.mark.asyncio
async def test_chat_completions_llama_forwards_raw_request_payload():
    """The public llama chat path should forward the caller payload without local shaping."""
    from cyber_inference.api.v1 import chat_completions

    captured_request = {}

    class DummyResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured_request["url"] = url
            captured_request["json"] = json
            return DummyResponse()

    auto_loader = MagicMock()
    auto_loader.ensure_model_loaded = AsyncMock(return_value="http://127.0.0.1:9999")
    auto_loader.record_request = AsyncMock()

    raw_payload = {
        "model": "demo",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.25,
        "top_k": 42,
        "repeat_penalty": 1.15,
        "max_tokens": 128,
    }
    request = ChatCompletionRequest(
        **raw_payload,
    )
    http_request = MagicMock()
    http_request.json = AsyncMock(return_value=raw_payload)

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.api.v1._get_server_type", return_value="llama"),
        patch("cyber_inference.api.v1.httpx.AsyncClient", return_value=DummyClient()),
    ):
        await chat_completions(request, http_request)

    assert captured_request["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert captured_request["json"] == raw_payload
    assert "n_predict" not in captured_request["json"]
    auto_loader.ensure_model_loaded.assert_awaited_once_with(
        "demo",
        load_trigger="public_autoload",
    )


@pytest.mark.asyncio
async def test_chat_completions_forwards_tool_payload_and_preserves_tool_calls():
    """Tool fields should reach llama.cpp and tool-call responses should survive shaping."""
    from cyber_inference.api.v1 import chat_completions

    captured_request = {}

    class DummyResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": "{\"city\":\"Boston\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured_request["url"] = url
            captured_request["json"] = json
            return DummyResponse()

    auto_loader = MagicMock()
    auto_loader.ensure_model_loaded = AsyncMock(return_value="http://127.0.0.1:9999")
    auto_loader.record_request = AsyncMock()

    raw_payload = {
        "model": "demo",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    request = ChatCompletionRequest(**raw_payload)
    http_request = MagicMock()
    http_request.json = AsyncMock(return_value=raw_payload)

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.api.v1._get_server_type", return_value="llama"),
        patch("cyber_inference.api.v1.httpx.AsyncClient", return_value=DummyClient()),
    ):
        response = await chat_completions(request, http_request)

    assert captured_request["json"]["tools"] == raw_payload["tools"]
    assert captured_request["json"]["tool_choice"] == "auto"
    assert captured_request["json"]["parallel_tool_calls"] is True
    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls[0]["id"] == "call_1"
    assert response.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_chat_completions_llama_preserves_image_payload_shape():
    """The llama pass-through branch should preserve the caller's image payload shape."""
    from cyber_inference.api.v1 import chat_completions

    captured_request = {}

    class DummyResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "green"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured_request["url"] = url
            captured_request["json"] = json
            return DummyResponse()

    auto_loader = MagicMock()
    auto_loader.ensure_model_loaded = AsyncMock(return_value="http://127.0.0.1:9999")
    auto_loader.record_request = AsyncMock()

    raw_payload = {
        "model": "demo-vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                ],
            }
        ],
    }
    request = ChatCompletionRequest(**raw_payload)
    http_request = MagicMock()
    http_request.json = AsyncMock(return_value=raw_payload)

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.api.v1._get_server_type", return_value="llama"),
        patch("cyber_inference.api.v1.httpx.AsyncClient", return_value=DummyClient()),
    ):
        response = await chat_completions(request, http_request)

    content = captured_request["json"]["messages"][0]["content"]
    assert content == raw_payload["messages"][0]["content"]
    assert response.choices[0].message.content == "green"


@pytest.mark.asyncio
async def test_chat_completions_rejects_tool_payload_when_capability_missing_for_non_llama():
    """Tool capability enforcement stays in place for non-llama branches."""
    from cyber_inference.api.v1 import chat_completions

    auto_loader = MagicMock()
    auto_loader.ensure_model_loaded = AsyncMock(return_value="http://127.0.0.1:9999")
    auto_loader.get_request_defaults = AsyncMock(return_value={})
    auto_loader.get_model_status = AsyncMock(
        return_value={
            "effective_config": {
                "tool_calling": {
                    "status": "probe_failed",
                    "warnings": ["Runtime capability probe failed"],
                }
            }
        }
    )

    request = ChatCompletionRequest(
        model="demo",
        messages=[ChatMessage(role="user", content="Hello")],
        tools=[{"type": "function", "function": {"name": "lookup_weather"}}],
    )

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.api.v1._get_server_type", return_value="transformers"),
    ):
        http_request = MagicMock()
        http_request.json = AsyncMock(
            return_value={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "function", "function": {"name": "lookup_weather"}}],
            }
        )
        with pytest.raises(HTTPException) as exc_info:
                await chat_completions(request, http_request)

    assert exc_info.value.status_code == 400
    assert "only available for llama-backed chat models" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_stream_chat_completion_preserves_tool_call_deltas():
    """Streaming tool-call deltas should pass through without text normalization."""
    from cyber_inference.api.v1 import _stream_chat_completion

    class DummyStreamResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return b""

        async def aiter_lines(self):
            payloads = [
                {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup_weather",
                                            "arguments": "{",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": "\"city\":\"Boston\"}",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
            for payload in payloads:
                yield f"data: {json.dumps(payload)}"
            yield "data: [DONE]"

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json):
            return DummyStreamResponse()

    auto_loader = MagicMock()
    auto_loader.touch_request = AsyncMock()
    auto_loader.record_request = AsyncMock()

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.api.v1.httpx.AsyncClient", return_value=DummyClient()),
    ):
        chunks = []
        async for item in _stream_chat_completion(
            "http://127.0.0.1:9999",
            {"messages": []},
            "demo",
        ):
            chunks.append(item)

    decoded = [json.loads(item["data"]) for item in chunks if item["data"] != "[DONE]"]
    assert decoded[0]["choices"][0]["delta"]["role"] == "assistant"
    assert decoded[1]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"
    assert decoded[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "\"city\":\"Boston\"}"
    assert decoded[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_update_model_config_returns_runtime_metadata(test_db):
    """Saving model config should surface authoritative runtime metadata in the response."""
    model = Model(
        name="demo-model",
        filename="demo.gguf",
        file_path="/tmp/demo.gguf",
        size_bytes=1,
        context_length=4096,
        is_downloaded=True,
        is_enabled=True,
        created_at=datetime.now(),
    )
    test_db.add(model)
    await test_db.commit()

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    auto_loader = MagicMock()
    auto_loader.reconcile_model_config_change = AsyncMock(
        return_value={
            "reload_triggered": True,
            "message": "Running model reloaded with the updated settings.",
            "status": "running",
        }
    )

    with (
        patch("cyber_inference.api.admin.get_db_session", override_get_db_session),
        patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader),
    ):
        async with make_test_client() as client:
            response = await client.put(
                "/admin/models/demo-model/config",
                json={"default_top_k": 48},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["default_top_k"] == 48
    assert data["reload_triggered"] is True
    assert data["runtime"]["status"] == "running"


@pytest.mark.asyncio
async def test_update_model_config_accepts_tool_template_fields(test_db):
    """Per-model tool template settings should be persisted through the admin API."""
    model = Model(
        name="demo-model",
        filename="demo.gguf",
        file_path="/tmp/demo.gguf",
        size_bytes=1,
        context_length=4096,
        is_downloaded=True,
        is_enabled=True,
        created_at=datetime.now(),
    )
    test_db.add(model)
    await test_db.commit()

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    auto_loader = MagicMock()
    auto_loader.reconcile_model_config_change = AsyncMock(
        return_value={"reload_triggered": False, "message": "Saved", "status": "not_loaded"}
    )

    with (
        patch("cyber_inference.api.admin.get_db_session", override_get_db_session),
        patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader),
    ):
        async with make_test_client() as client:
            response = await client.put(
                "/admin/models/demo-model/config",
                json={
                    "tool_template_mode": "explicit",
                    "tool_template_path": "/tmp/tool-use.jinja",
                    "tool_jinja_enabled": True,
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["tool_template_mode"] == "explicit"
    assert data["tool_template_path"] == "/tmp/tool-use.jinja"
    assert data["tool_jinja_enabled"] is True


@pytest.mark.asyncio
async def test_update_model_config_rejects_context_above_global_maximum(test_db):
    """Per-model defaults cannot bypass the global context ceiling."""
    model = Model(
        name="context-model",
        filename="context.gguf",
        file_path="/tmp/context.gguf",
        size_bytes=1,
        context_length=262144,
        is_downloaded=True,
        is_enabled=True,
        created_at=datetime.now(),
    )
    test_db.add(model)
    await test_db.commit()

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    with patch("cyber_inference.api.admin.get_db_session", override_get_db_session):
        async with make_test_client() as client:
            response = await client.put(
                "/admin/models/context-model/config",
                json={"default_context_size": 65536},
            )

    assert response.status_code == 400
    assert "max_context_size" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_config_returns_runtime_apply_metadata(test_db):
    """Saving global runtime config should disclose live-apply metadata."""

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    auto_loader = MagicMock()
    auto_loader.reconcile_global_config_change = AsyncMock(
        return_value={
            "applied_live": True,
            "reload_triggered": True,
            "reloaded_models": ["demo-model"],
            "restart_required": False,
        }
    )

    with (
        patch("cyber_inference.api.admin.get_db_session", override_get_db_session),
        patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader),
    ):
        async with make_test_client() as client:
            response = await client.put(
                "/admin/config/default_context_size",
                json={"value": 12288},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["value"] == 12288
    assert data["applied_live"] is True
    assert data["reload_triggered"] is True
    assert data["restart_required"] is False
    assert data["reloaded_models"] == ["demo-model"]


@pytest.mark.asyncio
async def test_update_config_rejects_invalid_context_relationship():
    """Global context settings must stay inside the configured ceiling."""
    async with make_test_client() as client:
        response = await client.put(
            "/admin/config/default_context_size",
            json={"value": 65536},
        )

    assert response.status_code == 400
    assert "cannot exceed max_context_size" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_max_context_rejects_existing_oversized_model_default(test_db):
    """Lowering the ceiling must not strand a configured model during live reload."""
    model = Model(
        name="large-context-model",
        filename="large-context.gguf",
        file_path="/tmp/large-context.gguf",
        size_bytes=1,
        context_length=262144,
        default_context_size=24576,
        is_downloaded=True,
        is_enabled=True,
        created_at=datetime.now(),
    )
    test_db.add(model)
    await test_db.commit()

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    with patch("cyber_inference.api.admin.get_db_session", override_get_db_session):
        async with make_test_client() as client:
            response = await client.put(
                "/admin/config/max_context_size",
                json={"value": 16384},
            )

    assert response.status_code == 400
    assert "large-context-model" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_idle_toggle_returns_live_apply_without_reload(test_db):
    """Idle timer GUI toggle should apply live without forcing model reloads."""

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    auto_loader = MagicMock()
    auto_loader.reconcile_global_config_change = AsyncMock(
        return_value={
            "applied_live": True,
            "reload_triggered": False,
            "reloaded_models": [],
            "restart_required": False,
        }
    )

    with (
        patch("cyber_inference.api.admin.get_db_session", override_get_db_session),
        patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader),
    ):
        async with make_test_client() as client:
            response = await client.put(
                "/admin/config/model_idle_unload_enabled",
                json={"value": True},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["value"] is True
    assert data["applied_live"] is True
    assert data["reload_triggered"] is False
    assert data["restart_required"] is False


@pytest.mark.asyncio
async def test_update_model_load_timeout_returns_live_apply_without_reload(test_db):
    """Model load timeout changes should apply to future loads without reload."""

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    auto_loader = MagicMock()
    auto_loader.reconcile_global_config_change = AsyncMock(
        return_value={
            "applied_live": True,
            "reload_triggered": False,
            "reloaded_models": [],
            "restart_required": False,
        }
    )

    with (
        patch("cyber_inference.api.admin.get_db_session", override_get_db_session),
        patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader),
    ):
        async with make_test_client() as client:
            response = await client.put(
                "/admin/config/model_load_timeout",
                json={"value": 600},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["value"] == 600
    assert data["applied_live"] is True
    assert data["reload_triggered"] is False
    assert data["restart_required"] is False


@pytest.mark.asyncio
async def test_update_model_load_timeout_rejects_out_of_range_values(test_db):
    """Model load timeout bounds should be enforced before persistence."""

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    auto_loader = MagicMock()

    with (
        patch("cyber_inference.api.admin.get_db_session", override_get_db_session),
        patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader),
    ):
        async with make_test_client() as client:
            low = await client.put("/admin/config/model_load_timeout", json={"value": 29})
            high = await client.put("/admin/config/model_load_timeout", json={"value": 3601})

    assert low.status_code == 400
    assert high.status_code == 400
    result = await test_db.execute(select(Configuration).where(Configuration.key == "model_load_timeout"))
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_pre_model_load_command_requires_admin_password_to_enable(test_db):
    """Pre-model load command cannot be armed without admin protection."""

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    auto_loader = MagicMock()

    with (
        patch("cyber_inference.api.admin.get_db_session", override_get_db_session),
        patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader),
    ):
        async with make_test_client() as client:
            response = await client.put(
                "/admin/config/pre_model_load_command_enabled",
                json={"value": True},
            )

    assert response.status_code == 400
    result = await test_db.execute(
        select(Configuration).where(Configuration.key == "pre_model_load_command_enabled")
    )
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_pre_model_load_command_config_updates_apply_live(test_db, monkeypatch):
    """Command text can be configured while disabled and applies to future loads."""
    from cyber_inference.core.config import reload_settings

    monkeypatch.setenv("CYBER_INFERENCE_ADMIN_PASSWORD", "secret")
    reload_settings()

    @asynccontextmanager
    async def override_get_db_session():
        yield test_db

    auto_loader = MagicMock()
    auto_loader.reconcile_global_config_change = AsyncMock(
        return_value={
            "applied_live": True,
            "reload_triggered": False,
            "reloaded_models": [],
            "restart_required": False,
        }
    )

    try:
        from cyber_inference.api.admin import verify_admin_token

        app.dependency_overrides[verify_admin_token] = lambda: True
        with (
            patch("cyber_inference.api.admin.get_db_session", override_get_db_session),
            patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader),
        ):
            async with make_test_client() as client:
                command_response = await client.put(
                    "/admin/config/pre_model_load_command",
                    json={"value": "sudo sysctl -w vm.drop_caches=3"},
                )
                enabled_response = await client.put(
                    "/admin/config/pre_model_load_command_enabled",
                    json={"value": True},
                )
    finally:
        app.dependency_overrides.clear()
        monkeypatch.delenv("CYBER_INFERENCE_ADMIN_PASSWORD", raising=False)
        reload_settings()

    assert command_response.status_code == 200
    assert enabled_response.status_code == 200
    assert enabled_response.json()["applied_live"] is True
    assert enabled_response.json()["reload_triggered"] is False


@pytest.mark.asyncio
async def test_load_model_endpoint_honors_context_override():
    """Manual load requests should be able to override the load-time context size."""
    auto_loader = MagicMock()
    auto_loader.load_model = AsyncMock(return_value="http://127.0.0.1:9338")
    auto_loader.get_model_status = AsyncMock(
        return_value={
            "port": 9338,
            "status": "running",
            "request_count": 0,
            "server_type": "llama",
            "reload_count": 0,
            "last_transition_reason": "manual_load",
            "effective_config": {"launch_config": {"context_size": 16384}},
        }
    )

    with patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader):
        async with make_test_client() as client:
            response = await client.post(
                "/admin/models/demo-model/load",
                json={"model_name": "demo-model", "context_size": 16384},
            )

    assert response.status_code == 200
    auto_loader.load_model.assert_awaited_once()
    assert auto_loader.load_model.await_args.kwargs["context_size_override"] == 16384


@pytest.mark.asyncio
async def test_load_model_endpoint_rejects_context_above_global_maximum():
    """Manual context validation errors should be returned as client errors."""
    auto_loader = MagicMock()
    auto_loader.load_model = AsyncMock(
        side_effect=ValueError(
            "Context size 65536 exceeds the configured maximum of 32768."
        )
    )

    with patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader):
        async with make_test_client() as client:
            response = await client.post(
                "/admin/models/demo-model/load",
                json={"model_name": "demo-model", "context_size": 65536},
            )

    assert response.status_code == 400
    assert "configured maximum" in response.json()["detail"]


@pytest.mark.asyncio
async def test_download_model_endpoint_passes_download_id_to_manager():
    """Download endpoints should preserve the client download session id."""
    from cyber_inference.services.model_manager import ModelDownloadResult

    model_manager = MagicMock()
    model_manager.download_model = AsyncMock(
        return_value=ModelDownloadResult(
            path=MagicMock(name="demo.gguf", stat=lambda: MagicMock(st_size=1), exists=lambda: True),
            model_name="demo",
            filename="demo.gguf",
            size_bytes=1,
        )
    )
    model_manager.get_model = AsyncMock(
        return_value={
            "id": 1,
            "name": "demo",
            "filename": "demo.gguf",
            "path": "/tmp/demo.gguf",
            "hf_repo_id": "demo/repo",
            "size_bytes": 1,
            "context_length": 4096,
            "model_type": "chat",
            "is_downloaded": True,
            "is_enabled": True,
        }
    )

    with patch("cyber_inference.api.admin.ModelManager", return_value=model_manager):
        async with make_test_client() as client:
            response = await client.post(
                "/admin/models/download",
                json={
                    "hf_repo_id": "https://huggingface.co/demo/repo/",
                    "hf_filename": "demo.gguf",
                    "download_id": "download-123",
                },
            )

    assert response.status_code == 200
    assert model_manager.download_model.await_args.kwargs["download_id"] == "download-123"
    assert model_manager.download_model.await_args.kwargs["repo_id"] == "demo/repo"
    assert response.json()["hf_repo_id"] == "demo/repo"


@pytest.mark.asyncio
async def test_download_transformers_endpoint_normalizes_full_repo_url():
    """Transformers API downloads should pass and return the canonical HuggingFace repo ID."""
    model_manager = MagicMock()
    model_manager.download_transformers_model = AsyncMock(return_value=Path("/tmp/Fable-GGUF"))
    model_manager._sanitize_repo_name.return_value = "Fable-GGUF"
    model_manager.get_model = AsyncMock(
        return_value={
            "id": 1,
            "name": "Fable-GGUF",
            "filename": "Fable-GGUF",
            "path": "/tmp/Fable-GGUF",
            "hf_repo_id": "DavidAU/Fable-GGUF",
            "size_bytes": 1,
            "context_length": 4096,
            "model_type": "chat",
            "is_downloaded": True,
            "is_enabled": True,
        }
    )

    with patch("cyber_inference.api.admin.ModelManager", return_value=model_manager):
        async with make_test_client() as client:
            response = await client.post(
                "/admin/models/download-transformers",
                json={
                    "hf_repo_id": "https://huggingface.co/DavidAU/Fable-GGUF/?download=true",
                    "download_id": "download-transformers-123",
                },
            )

    assert response.status_code == 200
    assert model_manager.download_transformers_model.await_args.kwargs == {
        "repo_id": "DavidAU/Fable-GGUF",
        "force": False,
        "download_id": "download-transformers-123",
    }
    assert response.json()["hf_repo_id"] == "DavidAU/Fable-GGUF"


@pytest.mark.asyncio
async def test_download_model_endpoint_uses_canonical_split_result_name_and_force():
    """Split downloads should use the manager's canonical result name for response lookup."""
    from cyber_inference.services.model_manager import ModelDownloadResult

    model_manager = MagicMock()
    model_manager.download_model = AsyncMock(
        return_value=ModelDownloadResult(
            path=MagicMock(),
            model_name="Model-Q4_K_M",
            filename="Model-Q4_K_M-00001-of-00003.gguf",
            size_bytes=60,
            is_split_gguf=True,
            shard_filenames=[
                "Model-Q4_K_M-00001-of-00003.gguf",
                "Model-Q4_K_M-00002-of-00003.gguf",
                "Model-Q4_K_M-00003-of-00003.gguf",
            ],
        )
    )
    model_manager.get_model = AsyncMock(
        return_value={
            "id": 1,
            "name": "Model-Q4_K_M",
            "filename": "Model-Q4_K_M-00001-of-00003.gguf",
            "path": "/tmp/Model-Q4_K_M-00001-of-00003.gguf",
            "hf_repo_id": "demo/repo",
            "size_bytes": 60,
            "quantization": "q4_k_m",
            "context_length": 4096,
            "model_type": "chat",
            "is_split_gguf": True,
            "gguf_shard_count": 3,
            "gguf_shard_filenames": [
                "Model-Q4_K_M-00001-of-00003.gguf",
                "Model-Q4_K_M-00002-of-00003.gguf",
                "Model-Q4_K_M-00003-of-00003.gguf",
            ],
            "is_downloaded": True,
            "is_enabled": True,
        }
    )

    with patch("cyber_inference.api.admin.ModelManager", return_value=model_manager):
        async with make_test_client() as client:
            response = await client.post(
                "/admin/models/download",
                json={
                    "hf_repo_id": "demo/repo",
                    "hf_filename": "Model-Q4_K_M-00002-of-00003.gguf",
                    "force": True,
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Model-Q4_K_M"
    assert payload["size_bytes"] == 60
    assert payload["is_split_gguf"] is True
    assert payload["gguf_shard_count"] == 3
    assert model_manager.get_model.await_args.args == ("Model-Q4_K_M",)
    assert model_manager.download_model.await_args.kwargs["force"] is True


@pytest.mark.asyncio
async def test_repo_files_endpoint_returns_split_metadata():
    """Repo file listing should pass split GGUF metadata through the API schema."""
    model_manager = MagicMock()
    model_manager.list_repo_files_detailed = AsyncMock(
        return_value={
            "repo_id": "demo/repo",
            "model_files": [
                {
                    "filename": "Model-Q4_K_M-00001-of-00003.gguf",
                    "size_bytes": 60,
                    "quantization": "q4_k_m",
                    "is_mmproj": False,
                    "is_split": True,
                    "shard_count": 3,
                    "shard_total_size_bytes": 60,
                    "shard_filenames": [
                        "Model-Q4_K_M-00001-of-00003.gguf",
                        "Model-Q4_K_M-00002-of-00003.gguf",
                        "Model-Q4_K_M-00003-of-00003.gguf",
                    ],
                    "primary_filename": "Model-Q4_K_M-00001-of-00003.gguf",
                    "is_complete": True,
                    "missing_shard_filenames": [],
                }
            ],
            "mmproj_files": [],
            "is_multimodal": False,
            "suggested_model": "Model-Q4_K_M-00001-of-00003.gguf",
            "suggested_mmproj": None,
        }
    )

    with patch("cyber_inference.api.admin.ModelManager", return_value=model_manager):
        async with make_test_client() as client:
            response = await client.get(
                "/admin/models/repo-files",
                params={"repo_id": "https://huggingface.co/demo/repo/?download=true"},
            )

    assert response.status_code == 200
    model_manager.list_repo_files_detailed.assert_awaited_once_with("demo/repo")
    payload = response.json()
    assert payload["repo_id"] == "demo/repo"
    assert len(payload["model_files"]) == 1
    model_file = payload["model_files"][0]
    assert model_file["is_split"] is True
    assert model_file["shard_count"] == 3
    assert model_file["primary_filename"] == "Model-Q4_K_M-00001-of-00003.gguf"


@pytest.mark.asyncio
async def test_web_dashboard():
    """Test the web dashboard page."""
    async with make_test_client() as client:
        response = await client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_web_models_page():
    """Test the web models page."""
    async with make_test_client() as client:
        response = await client.get("/models")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


def test_models_template_consumes_split_gguf_metadata():
    """Downloader UI should render one logical split option from mocked metadata."""
    if not shutil.which("node"):
        pytest.skip("node is required for the focused downloader JavaScript harness")

    template = Path("src/cyber_inference/web/templates/models.html").read_text()
    script = re.search(r"<script>([\s\S]*)</script>", template).group(1)
    harness = f"""
class Element {{
  constructor(id) {{
    this.id = id;
    this.children = [];
    this.value = '';
    this.textContent = '';
    this.disabled = false;
    this.selected = false;
    this.className = '';
    this.style = {{}};
    this.classList = {{
      add: (...names) => {{ this.className += ' ' + names.join(' '); }},
      remove: (...names) => {{
        const remove = new Set(names);
        this.className = this.className.split(/\\s+/).filter(x => !remove.has(x)).join(' ');
      }},
    }};
  }}
  set innerHTML(value) {{
    this._innerHTML = value;
    this.children = [];
  }}
  get innerHTML() {{ return this._innerHTML || ''; }}
  appendChild(child) {{
    this.children.push(child);
    if (child.selected) this.value = child.value;
  }}
  addEventListener() {{}}
  reset() {{}}
}}
const elements = {{}};
globalThis.document = {{
  getElementById: (id) => elements[id] || (elements[id] = new Element(id)),
  createElement: (tag) => new Element(tag),
}};
globalThis.window = {{ location: {{ protocol: 'http:', host: 'test' }} }};
globalThis.location = {{ reload: () => {{}} }};
globalThis.WebSocket = function() {{}};
globalThis.setTimeout = () => {{}};
globalThis.alert = (message) => {{ throw new Error(message); }};
globalThis.console = console;
{script}
repoData = {{
  is_multimodal: false,
  suggested_model: 'Model-Q4_K_M-00001-of-00003.gguf',
  suggested_mmproj: null,
  mmproj_files: [],
  model_files: [
    {{
      filename: 'Model-Q4_K_M-00001-of-00003.gguf',
      size_bytes: 60,
      quantization: 'q4_k_m',
      is_split: true,
      shard_count: 3,
      shard_filenames: [
        'Model-Q4_K_M-00001-of-00003.gguf',
        'Model-Q4_K_M-00002-of-00003.gguf',
        'Model-Q4_K_M-00003-of-00003.gguf'
      ],
      is_complete: true,
      missing_shard_filenames: [],
    }},
    {{
      filename: 'Broken-00001-of-00002.gguf',
      size_bytes: 10,
      quantization: null,
      is_split: true,
      shard_count: 2,
      shard_filenames: ['Broken-00001-of-00002.gguf'],
      is_complete: false,
      missing_shard_filenames: ['Broken-00002-of-00002.gguf'],
    }}
  ],
}};
populateFileSelections(repoData);
const options = document.getElementById('modelFileSelect').children;
if (options.length !== 2) throw new Error(`expected 2 logical options, got ${{options.length}}`);
if (!options[0].textContent.includes('3 files')) throw new Error(options[0].textContent);
if (!options[0].textContent.includes('60 B')) throw new Error(options[0].textContent);
if (options[0].textContent.includes('00002-of-00003')) throw new Error(options[0].textContent);
if (!options[1].disabled) throw new Error('incomplete option should be disabled');
if (!document.getElementById('shardInfo').textContent.includes('3/3 files')) {{
  throw new Error(document.getElementById('shardInfo').textContent);
}}
"""

    result = subprocess.run(["node", "-e", harness], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_settings_template_saves_model_load_timeout():
    """Settings UI should expose and save model load timeout."""
    template = Path("src/cyber_inference/web/templates/settings.html").read_text()

    assert "Model Load Timeout (seconds)" in template
    assert 'id="model_load_timeout"' in template
    assert 'min="30"' in template
    assert 'max="3600"' in template
    assert "Large GGUF models may need several minutes" in template
    assert "const modelLoadTimeout" in template
    assert "model_load_timeout" in template
    assert "Model load timeout must be between 30 and 3600 seconds." in template


def test_settings_template_updates_context_limits_without_racing():
    """Related context settings should be persisted in a server-valid order."""
    template = Path("src/cyber_inference/web/templates/settings.html").read_text()

    assert "const currentDefaultContextSize" in template
    assert "const contextSettings = maxContextSize < currentDefaultContextSize" in template
    assert "for (const setting of settings)" in template
    assert "Promise.allSettled" not in template


def test_settings_template_saves_pre_model_load_command():
    """Settings UI should expose and save pre-model load command controls."""
    template = Path("src/cyber_inference/web/templates/settings.html").read_text()

    assert "Run pre-model load command" in template
    assert "Thor/DGX Spark" in template
    assert "sudo sysctl -w vm.drop_caches=3" in template
    assert 'id="pre_model_load_command_enabled"' in template
    assert 'id="pre_model_load_command"' in template
    assert 'id="pre_model_load_command_timeout"' in template
    assert "preModelLoadCommandEnabled" in template
    assert "pre_model_load_command_enabled" in template
    assert "pre_model_load_command_timeout" in template


def test_settings_template_shows_llama_cpp_runtime_controls():
    """Settings UI should expose llama.cpp version and managed update controls."""
    template = Path("src/cyber_inference/web/templates/settings.html").read_text()

    assert "LLAMA.CPP RUNTIME" in template
    assert 'id="llamaCppRuntimePanel"' in template
    assert 'id="llamaCppSource"' in template
    assert 'id="llamaCppBinaryPath"' in template
    assert 'id="llamaCppInstalledVersion"' in template
    assert 'id="llamaCppMtpSupport"' in template
    assert 'id="llamaCppLatestRelease"' in template
    assert 'id="llamaCppUpdateBtn"' in template
    assert "/admin/llama-cpp/status" in template
    assert "/admin/llama-cpp/update" in template
    assert "System-managed binary detected" in template
    assert "Unload llama.cpp models before updating" in template
    assert "updateBtn.disabled = !data.update_allowed" in template
    assert "llama_mtp_auto_enable" in template
    assert "llama_mtp_default_draft_n_max" in template


@pytest.mark.asyncio
async def test_web_models_page_shows_vision_badge_for_mmproj_models():
    """Installed models with mmproj should advertise Vision in the UI even when unloaded."""
    model_manager = MagicMock()
    model_manager.list_models = AsyncMock(
        return_value=[
            {
                "name": "demo-vlm",
                "engine_type": "llama",
                "quantization": "q4_k_m",
                "size_bytes": 1024,
                "context_length": 8192,
                "hf_repo_id": "demo/repo",
                "mmproj_path": "/tmp/mmproj-demo.gguf",
                "is_vlm": True,
            }
        ]
    )
    auto_loader = MagicMock()
    auto_loader.get_loaded_models = AsyncMock(return_value=[])
    auto_loader.get_models_status = AsyncMock(
        return_value={
            "demo-vlm": {
                "is_loaded": False,
                "status": "not_loaded",
                "server_type": "llama",
                "effective_config": {
                    "tool_calling": {"status": "unknown"},
                    "vision": {"enabled": True},
                },
            }
        }
    )

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.services.model_manager.ModelManager", return_value=model_manager),
    ):
        async with make_test_client() as client:
            response = await client.get("/models")

    assert response.status_code == 200
    assert re.search(r"demo-vlm[\s\S]{0,1500}title=\"Vision enabled\"", response.text)
    auto_loader.get_models_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_web_models_page_shows_mtp_badge_for_detected_models():
    """Installed MTP models should advertise speculative decoding in the models UI."""
    model_manager = MagicMock()
    model_manager.list_models = AsyncMock(
        return_value=[
            {
                "name": "qwen-mtp",
                "engine_type": "llama",
                "quantization": "q4_k_xl",
                "size_bytes": 1024,
                "context_length": 131072,
                "hf_repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
                "mmproj_path": None,
                "is_vlm": False,
                "mtp_capable": True,
            }
        ]
    )
    auto_loader = MagicMock()
    auto_loader.get_loaded_models = AsyncMock(return_value=[])
    auto_loader.get_models_status = AsyncMock(
        return_value={
            "qwen-mtp": {
                "is_loaded": False,
                "status": "not_loaded",
                "server_type": "llama",
                "effective_config": {
                    "tool_calling": {"status": "unknown"},
                    "vision": {"enabled": False},
                    "mtp": {"capable": True, "enabled": True},
                },
            }
        }
    )

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.services.model_manager.ModelManager", return_value=model_manager),
    ):
        async with make_test_client() as client:
            response = await client.get("/models")

    assert response.status_code == 200
    assert re.search(r"qwen-mtp[\s\S]{0,1500}title=\"MTP speculative decoding enabled\"", response.text)


@pytest.mark.asyncio
async def test_chat_page_renders_iframe_for_loaded_llama_model():
    auto_loader = MagicMock()
    auto_loader.ensure_model_loaded = AsyncMock(return_value="http://127.0.0.1:9999")
    auto_loader.get_model_status = AsyncMock(
        return_value={
            "server_type": "llama",
            "status": "running",
            "is_loaded": True,
        }
    )

    with patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader):
        async with make_test_client() as client:
            response = await client.get("/chat/demo-model")

    assert response.status_code == 200
    assert '/chat/demo-model/ui/' in response.text
    assert "demo-model" in response.text
    assert "chat-shell" in response.text
    assert "chat-frame" in response.text


@pytest.mark.asyncio
async def test_chat_ui_proxy_passes_through_upstream_html():
    class DummyResponse:
        status_code = 200
        content = b"<html><body>llama ui</body></html>"
        headers = {"content-type": "text/html; charset=utf-8"}

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, content, headers):
            assert method == "GET"
            assert url == "http://127.0.0.1:9999/"
            return DummyResponse()

    auto_loader = MagicMock()
    auto_loader.ensure_model_loaded = AsyncMock(return_value="http://127.0.0.1:9999")
    auto_loader.get_model_status = AsyncMock(
        return_value={
            "server_type": "llama",
            "status": "running",
            "is_loaded": True,
        }
    )

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.api.web.httpx.AsyncClient", return_value=DummyClient()),
    ):
        async with make_test_client() as client:
            response = await client.get("/chat/demo-model/ui/")

    assert response.status_code == 200
    assert "llama ui" in response.text


@pytest.mark.asyncio
async def test_chat_ui_proxy_passes_through_post_requests():
    class DummyResponse:
        status_code = 200
        content = b'{"ok":true}'
        headers = {"content-type": "application/json"}

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, content, headers):
            assert method == "POST"
            assert url == "http://127.0.0.1:9999/v1/chat/completions"
            assert content
            return DummyResponse()

    auto_loader = MagicMock()
    auto_loader.ensure_model_loaded = AsyncMock(return_value="http://127.0.0.1:9999")
    auto_loader.get_model_status = AsyncMock(
        return_value={
            "server_type": "llama",
            "status": "running",
            "is_loaded": True,
        }
    )

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.api.web.httpx.AsyncClient", return_value=DummyClient()),
    ):
        async with make_test_client() as client:
            response = await client.post(
                "/chat/demo-model/ui/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_chat_page_rejects_non_llama_models():
    auto_loader = MagicMock()
    auto_loader.ensure_model_loaded = AsyncMock(return_value="http://127.0.0.1:9999")
    auto_loader.get_model_status = AsyncMock(
        return_value={
            "server_type": "transformers",
            "status": "running",
            "is_loaded": True,
        }
    )

    with patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader):
        async with make_test_client() as client:
            response = await client.get("/chat/demo-model")

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_web_settings_page():
    """Test the web settings page."""
    async with make_test_client() as client:
        response = await client.get("/settings")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_web_logs_page():
    """Test the web logs page."""
    async with make_test_client() as client:
        response = await client.get("/logs")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_base_layout_uses_local_assets_only():
    """The shared web layout should not require external runtime CSS/JS/font/image assets."""
    async with make_test_client() as client:
        response = await client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "/static/css/app.css" in html
    assert "/static/images/ramborogers.png" in html

    for blocked_asset_host in (
        "cdn.tailwindcss.com",
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "raw.githubusercontent.com",
    ):
        assert blocked_asset_host not in html
