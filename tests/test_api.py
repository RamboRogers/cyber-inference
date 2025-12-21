"""
Integration tests for Cyber-Inference API endpoints.

Tests cover:
- Health endpoint
- V1 models endpoint
- Admin endpoints
- Error handling
"""

import pytest
from httpx import AsyncClient

from cyber_inference.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["service"] == "cyber-inference"


@pytest.mark.asyncio
async def test_v1_models_endpoint():
    """Test the /v1/models endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_v1_chat_completions_no_model():
    """Test chat completions with non-existent model."""
    async with AsyncClient(app=app, base_url="http://test") as client:
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
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/admin/status")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "status" in data
        assert "running_models" in data


@pytest.mark.asyncio
async def test_admin_resources_endpoint():
    """Test the admin resources endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/admin/resources")

        assert response.status_code == 200
        data = response.json()
        assert "cpu_count" in data
        assert "cpu_percent" in data
        assert "memory_percent" in data


@pytest.mark.asyncio
async def test_admin_models_list():
    """Test listing models via admin API."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/admin/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_admin_sessions_list():
    """Test listing active sessions."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/admin/sessions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_admin_config():
    """Test getting configuration."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/admin/config")

        assert response.status_code == 200
        data = response.json()
        assert "host" in data
        assert "port" in data
        assert "log_level" in data


@pytest.mark.asyncio
async def test_chat_completion_validation():
    """Test request validation for chat completions."""
    async with AsyncClient(app=app, base_url="http://test") as client:
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
async def test_web_dashboard():
    """Test the web dashboard page."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_web_models_page():
    """Test the web models page."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/models")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_web_settings_page():
    """Test the web settings page."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/settings")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_web_logs_page():
    """Test the web logs page."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/logs")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

