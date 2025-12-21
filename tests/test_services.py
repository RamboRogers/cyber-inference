"""
Unit tests for Cyber-Inference services.

Tests cover:
- Resource monitor
- Configuration management
- Model manager
- Process manager (mock)
"""

import asyncio
import platform
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cyber_inference.core.config import Settings, get_settings, reload_settings
from cyber_inference.services.resource_monitor import ResourceMonitor, SystemResources


class TestResourceMonitor:
    """Tests for the ResourceMonitor service."""

    @pytest.fixture
    def monitor(self):
        """Create a resource monitor instance."""
        return ResourceMonitor(update_interval=1.0)

    @pytest.mark.asyncio
    async def test_collect_resources(self, monitor):
        """Test resource collection."""
        resources = await monitor._collect_resources()

        assert isinstance(resources, SystemResources)
        assert resources.cpu_count > 0
        assert 0 <= resources.cpu_percent <= 100
        assert resources.total_memory_mb > 0
        assert resources.available_memory_mb > 0
        assert 0 <= resources.memory_percent <= 100

    @pytest.mark.asyncio
    async def test_get_system_info(self, monitor):
        """Test system info retrieval."""
        info = await monitor.get_system_info()

        assert "platform" in info
        assert "cpu_count" in info
        assert "total_memory_gb" in info
        assert info["cpu_count"] > 0
        assert info["total_memory_gb"] > 0

    @pytest.mark.asyncio
    async def test_check_memory_available(self, monitor):
        """Test memory availability check."""
        # Small requirement should pass
        is_available = await monitor.check_memory_available(100)
        assert is_available is True

        # Huge requirement should fail
        is_available = await monitor.check_memory_available(1000000000)  # 1 PB
        assert is_available is False

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test monitor start and stop."""
        await monitor.start()
        assert monitor._running is True

        await monitor.stop()
        assert monitor._running is False

    def test_has_gpu(self, monitor):
        """Test GPU detection."""
        # Just verify it returns a boolean
        has_gpu = monitor.has_gpu()
        assert isinstance(has_gpu, bool)

    def test_get_gpu_vendor(self, monitor):
        """Test GPU vendor detection."""
        vendor = monitor.get_gpu_vendor()
        assert vendor is None or vendor in ["nvidia", "apple", "amd"]


class TestSettings:
    """Tests for configuration management."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 8337
        assert settings.log_level == "DEBUG"
        assert settings.default_context_size == 4096
        assert settings.max_loaded_models == 3

    def test_settings_from_env(self, monkeypatch):
        """Test settings from environment variables."""
        monkeypatch.setenv("CYBER_INFERENCE_PORT", "9999")
        monkeypatch.setenv("CYBER_INFERENCE_LOG_LEVEL", "WARNING")

        # Clear cache to reload
        reload_settings()
        settings = get_settings()

        assert settings.port == 9999
        assert settings.log_level == "WARNING"

    def test_database_path(self):
        """Test database path property."""
        settings = Settings()

        db_path = settings.database_path
        assert db_path.name == settings.database_name
        assert db_path.parent == settings.data_dir

    def test_log_level_int(self):
        """Test log level conversion."""
        import logging

        settings = Settings()
        settings.log_level = "DEBUG"
        assert settings.log_level_int == logging.DEBUG

        settings.log_level = "INFO"
        assert settings.log_level_int == logging.INFO

        settings.log_level = "ERROR"
        assert settings.log_level_int == logging.ERROR


class TestModelManager:
    """Tests for model manager."""

    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_list_models_empty(self, temp_models_dir):
        """Test listing models in empty directory."""
        from cyber_inference.services.model_manager import ModelManager

        # Need to mock the database
        with patch('cyber_inference.services.model_manager.get_db_session'):
            manager = ModelManager(models_dir=temp_models_dir)

            # This would need a mock database session
            # For now, just verify the manager initializes
            assert manager.models_dir == temp_models_dir

    def test_model_manager_init(self, temp_models_dir):
        """Test model manager initialization."""
        from cyber_inference.services.model_manager import ModelManager

        manager = ModelManager(models_dir=temp_models_dir)

        assert manager.models_dir == temp_models_dir
        assert manager.models_dir.exists()


class TestProcessManager:
    """Tests for process manager."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        with tempfile.TemporaryDirectory() as models_dir:
            with tempfile.TemporaryDirectory() as bin_dir:
                yield Path(models_dir), Path(bin_dir)

    def test_process_manager_init(self, temp_dirs):
        """Test process manager initialization."""
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)

        assert pm.models_dir == models_dir
        assert pm.bin_dir == bin_dir
        assert pm.base_port == 8338

    def test_find_available_port(self, temp_dirs):
        """Test port allocation."""
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)

        port = pm._find_available_port()
        assert port >= pm.base_port
        assert port in pm._port_allocations

        # Release and reallocate
        pm._release_port(port)
        assert port not in pm._port_allocations

    def test_get_running_models_empty(self, temp_dirs):
        """Test getting running models when none are running."""
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)

        running = pm.get_running_models()
        assert running == []

    def test_get_all_processes_empty(self, temp_dirs):
        """Test getting all processes when none exist."""
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)

        processes = pm.get_all_processes()
        assert processes == []


class TestAutoLoader:
    """Tests for auto-loader service."""

    @pytest.mark.asyncio
    async def test_auto_loader_init(self):
        """Test auto-loader initialization."""
        from cyber_inference.services.auto_loader import AutoLoader

        loader = AutoLoader()

        assert loader._idle_timeout > 0
        assert loader._max_loaded > 0
        assert loader._running is False

    @pytest.mark.asyncio
    async def test_auto_loader_start_stop(self):
        """Test auto-loader start and stop."""
        from cyber_inference.services.auto_loader import AutoLoader

        loader = AutoLoader()

        await loader.start()
        assert loader._running is True

        await loader.stop()
        assert loader._running is False

