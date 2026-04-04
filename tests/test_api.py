"""
Integration tests for Cyber-Inference API endpoints.

Tests cover:
- Health endpoint
- V1 models endpoint
- Admin endpoints
- Error handling
"""

from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from cyber_inference.core.database import get_db
from cyber_inference.main import app
from cyber_inference.models.db_models import Model
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
async def test_v1_models_endpoint():
    """Test the /v1/models endpoint."""
    async def override_get_db():
        yield MagicMock()

    auto_loader = MagicMock()
    auto_loader.list_available_models = AsyncMock(return_value=[])

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
    session = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    @asynccontextmanager
    async def override_get_db_session():
        yield session

    with patch("cyber_inference.api.admin.get_db_session", override_get_db_session):
        async with make_test_client() as client:
            response = await client.get("/admin/models")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


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
async def test_chat_completions_forwards_top_k_and_repeat_penalty():
    """The llama chat proxy should forward saved/request generation fields it claims to support."""
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
    auto_loader.get_request_defaults = AsyncMock(
        return_value={"top_k": 42, "repeat_penalty": 1.15}
    )
    auto_loader.record_request = AsyncMock()

    request = ChatCompletionRequest(
        model="demo",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    with (
        patch("cyber_inference.api.v1.get_auto_loader", return_value=auto_loader),
        patch("cyber_inference.api.v1._get_server_type", return_value="llama"),
        patch("cyber_inference.api.v1.httpx.AsyncClient", return_value=DummyClient()),
    ):
        await chat_completions(request, MagicMock())

    assert captured_request["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert captured_request["json"]["top_k"] == 42
    assert captured_request["json"]["repeat_penalty"] == 1.15


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
            "effective_config": {"launch_config": {"context_size": 65536}},
        }
    )

    with patch("cyber_inference.api.admin._get_auto_loader", return_value=auto_loader):
        async with make_test_client() as client:
            response = await client.post(
                "/admin/models/demo-model/load",
                json={"model_name": "demo-model", "context_size": 65536},
            )

    assert response.status_code == 200
    auto_loader.load_model.assert_awaited_once()
    assert auto_loader.load_model.await_args.kwargs["context_size_override"] == 65536


@pytest.mark.asyncio
async def test_download_model_endpoint_passes_download_id_to_manager():
    """Download endpoints should preserve the client download session id."""
    model_manager = MagicMock()
    model_manager.download_model = AsyncMock(return_value=MagicMock(name="demo.gguf", stat=lambda: MagicMock(st_size=1), exists=lambda: True))
    model_manager.get_model = AsyncMock(
        return_value={
            "id": 1,
            "name": "demo",
            "filename": "demo.gguf",
            "path": "/tmp/demo.gguf",
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
                    "hf_repo_id": "demo/repo",
                    "hf_filename": "demo.gguf",
                    "download_id": "download-123",
                },
            )

    assert response.status_code == 200
    assert model_manager.download_model.await_args.kwargs["download_id"] == "download-123"


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
