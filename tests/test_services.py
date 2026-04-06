"""
Unit tests for Cyber-Inference services.

Tests cover:
- Resource monitor
- Configuration management
- Model manager
- Process manager (mock)
"""

import tempfile
from datetime import datetime, timedelta
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
        assert settings.log_level == "INFO"
        assert settings.default_context_size == 8192
        assert settings.model_idle_unload_enabled is False
        assert settings.max_loaded_models == 1

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
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir, base_port=48338)

        fake_socket = MagicMock()
        fake_socket.__enter__.return_value = fake_socket
        fake_socket.__exit__.return_value = False

        with patch("cyber_inference.services.process_manager.socket.socket", return_value=fake_socket):
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

    def test_build_llama_server_command_includes_tool_flags(self, temp_dirs):
        """Chat launches should include jinja and template overrides when configured."""
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)
        template_path = models_dir / "tool-use.jinja"
        template_path.write_text("{{ messages }}")

        cmd = pm._build_llama_server_command(
            Path("/tmp/llama-server"),
            Path("/tmp/demo.gguf"),
            9338,
            8192,
            -1,
            8,
            False,
            None,
            {
                "tool_template_path": str(template_path),
            },
        )

        assert "--jinja" in cmd
        assert "--chat-template-file" in cmd
        assert str(template_path) in cmd

    def test_build_llama_server_command_includes_mmproj_for_vision_chat(self, temp_dirs):
        """Vision GGUF launches should include both mmproj and jinja chat handling."""
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)
        mmproj_path = models_dir / "mmproj-demo.gguf"
        mmproj_path.write_text("mmproj")

        cmd = pm._build_llama_server_command(
            Path("/tmp/llama-server"),
            Path("/tmp/demo.gguf"),
            9338,
            8192,
            -1,
            8,
            False,
            mmproj_path,
            {},
        )

        assert "--mmproj" in cmd
        assert str(mmproj_path) in cmd
        assert "--jinja" in cmd

    def test_build_llama_server_command_skips_tool_flags_for_embeddings(self, temp_dirs):
        """Embedding launches should keep embedding mode and skip chat tool flags."""
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)

        cmd = pm._build_llama_server_command(
            Path("/tmp/llama-server"),
            Path("/tmp/demo.gguf"),
            9338,
            8192,
            -1,
            8,
            True,
            None,
            {
                "tool_template_name": "chatml",
            },
        )

        assert "--embedding" in cmd
        assert "--jinja" not in cmd
        assert "--chat-template" not in cmd

    def test_build_llama_server_command_jinja_always_enabled(self, temp_dirs):
        """--jinja should always be passed for non-embedding models, even without launch_config."""
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)

        cmd = pm._build_llama_server_command(
            Path("/tmp/llama-server"),
            Path("/tmp/demo.gguf"),
            9338,
            8192,
            -1,
            8,
            False,
            None,
            {},
        )

        assert "--jinja" in cmd


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

    @pytest.mark.asyncio
    async def test_check_idle_models_is_disabled(self):
        """Idle timeout checks should be disabled by the resident-model policy."""
        from cyber_inference.services.auto_loader import AutoLoader
        from cyber_inference.services.process_manager import LlamaProcess

        proc = LlamaProcess(
            model_name="demo",
            model_path=Path("/tmp/demo.gguf"),
            port=9000,
            status="running",
            started_at=datetime.now() - timedelta(minutes=20),
            last_request_at=datetime.now() - timedelta(minutes=20),
        )
        process_manager = MagicMock()
        process_manager.get_all_processes.return_value = [proc]

        loader = AutoLoader(process_manager=process_manager)
        loader.unload_model = AsyncMock()

        await loader._check_idle_models()

        loader.unload_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_check_idle_models_unloads_when_enabled(self):
        """Idle timeout checks should still work when the GUI option is enabled."""
        from cyber_inference.services.auto_loader import AutoLoader
        from cyber_inference.services.process_manager import LlamaProcess

        proc = LlamaProcess(
            model_name="demo",
            model_path=Path("/tmp/demo.gguf"),
            port=9000,
            status="running",
            started_at=datetime.now() - timedelta(minutes=20),
            last_request_at=datetime.now() - timedelta(minutes=20),
        )
        process_manager = MagicMock()
        process_manager.get_all_processes.return_value = [proc]

        loader = AutoLoader(process_manager=process_manager)
        loader._idle_unload_enabled = True
        loader.unload_model = AsyncMock()

        await loader._check_idle_models()

        loader.unload_model.assert_awaited_once_with("demo", reason="idle_timeout")

    @pytest.mark.asyncio
    async def test_get_request_defaults_respects_backend_support(self):
        """Only backend-supported saved defaults should be exposed as request defaults."""
        from cyber_inference.services.auto_loader import AutoLoader

        loader = AutoLoader()
        loader.get_model_info = AsyncMock(
            return_value={
                "name": "demo",
                "engine_type": "transformers",
                "default_temperature": 0.4,
                "default_top_p": 0.8,
                "default_top_k": 32,
                "default_max_tokens": 256,
                "default_repeat_penalty": 1.15,
            }
        )

        defaults = await loader.get_request_defaults("demo", "transformers")

        assert defaults == {
            "temperature": 0.4,
            "top_p": 0.8,
            "max_tokens": 256,
        }

    @pytest.mark.asyncio
    async def test_reconcile_global_config_change_reloads_running_llama_models(self):
        """Live runtime reconciliation should reload only affected llama models."""
        from cyber_inference.services.auto_loader import AutoLoader
        from cyber_inference.services.process_manager import LlamaProcess

        llama_proc = LlamaProcess(
            model_name="llama-model",
            model_path=Path("/tmp/llama.gguf"),
            port=9001,
            status="running",
            server_type="llama",
        )
        transformers_proc = LlamaProcess(
            model_name="transformer-model",
            model_path=Path("/tmp/transformers"),
            port=9002,
            status="running",
            server_type="transformers",
        )

        process_manager = MagicMock()
        process_manager.get_all_processes.return_value = [llama_proc, transformers_proc]

        loader = AutoLoader(process_manager=process_manager)
        loader.reload_model = AsyncMock(return_value={"reload_triggered": True})

        result = await loader.reconcile_global_config_change("default_context_size")

        loader.reload_model.assert_awaited_once_with(
            "llama-model",
            reason="global_config:default_context_size",
        )
        assert result["reload_triggered"] is True
        assert result["reloaded_models"] == ["llama-model"]

    @pytest.mark.asyncio
    async def test_reconcile_global_toggle_updates_runtime_without_reload(self):
        """Idle-timer toggle changes should apply live without forcing reloads."""
        from cyber_inference.services.auto_loader import AutoLoader

        loader = AutoLoader()
        loader.refresh_runtime_settings = MagicMock(
            return_value={
                "idle_timeout": 300,
                "idle_unload_enabled": True,
                "max_loaded_models": 1,
                "max_memory_percent": 80.0,
            }
        )

        result = await loader.reconcile_global_config_change("model_idle_unload_enabled")

        assert result["applied_live"] is True
        assert result["reload_triggered"] is False
        assert result["restart_required"] is False

    @pytest.mark.asyncio
    async def test_load_model_prefers_native_context_when_no_override(self):
        """A model's detected native context should beat the low global default."""
        from cyber_inference.services.auto_loader import AutoLoader

        process_manager = MagicMock()
        process_manager.start_server = AsyncMock(
            return_value=MagicMock(
                status="running",
                port=9338,
                server_type="llama",
                effective_config={},
            )
        )
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "demo",
                "engine_type": "llama",
                "context_length": 131072,
                "default_context_size": None,
                "model_type": "chat",
                "mmproj_path": None,
                "hf_repo_id": "demo/repo",
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)

        await loader.load_model("demo")

        assert process_manager.start_server.await_args.kwargs["context_size"] == 131072

    @pytest.mark.asyncio
    async def test_load_model_caches_supported_tool_capability(self):
        """Successful llama props probing should mark tool calling as supported."""
        from cyber_inference.services.auto_loader import AutoLoader

        process_manager = MagicMock()
        proc = MagicMock(status="running", port=9338, server_type="llama", effective_config={})
        process_manager.start_server = AsyncMock(return_value=proc)
        process_manager.get_server_props = AsyncMock(
            return_value={"chat_template_tool_use": "builtin"}
        )
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "demo",
                "engine_type": "llama",
                "context_length": 131072,
                "default_context_size": None,
                "model_type": "chat",
                "mmproj_path": None,
                "hf_repo_id": "demo/repo",
                "tool_template_mode": None,
                "tool_template_name": None,
                "tool_template_path": None,
                "tool_jinja_enabled": None,
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)

        await loader.load_model("demo")

        assert proc.effective_config["launch_config"]["jinja_enabled"] is True
        assert proc.effective_config["tool_calling"]["status"] == "supported"
        assert proc.effective_config["tool_calling"]["source"] == "detected_runtime"

    @pytest.mark.asyncio
    async def test_load_model_promotes_qwen35_family_metadata_to_supported(self):
        """Qwen 3.5 GGUF tool markers should be enough when runtime exposes a chat template."""
        from cyber_inference.services.auto_loader import AutoLoader

        process_manager = MagicMock()
        proc = MagicMock(status="running", port=9338, server_type="llama", effective_config={})
        process_manager.start_server = AsyncMock(return_value=proc)
        process_manager.get_server_props = AsyncMock(
            return_value={"chat_template": "{% for message in messages %}<tool_call>{% endfor %}"}
        )
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "Qwen3.5-35B-A3B-MXFP4_MOE",
                "engine_type": "llama",
                "context_length": 131072,
                "default_context_size": None,
                "model_type": "chat",
                "mmproj_path": None,
                "hf_repo_id": "unsloth/Qwen3.5-35B-A3B-GGUF",
                "tool_template_mode": None,
                "tool_template_name": None,
                "tool_template_path": None,
                "tool_jinja_enabled": None,
                "gguf_has_chat_template": True,
                "gguf_has_tool_call_tokens": True,
                "gguf_has_tool_response_tokens": True,
                "gguf_has_response_schema_tool_calls": False,
                "gguf_has_gemma4_tool_parser": False,
                "gguf_architecture": "qwen3moe",
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)

        await loader.load_model("Qwen3.5-35B-A3B-MXFP4_MOE")

        assert proc.effective_config["tool_calling"]["status"] == "supported"
        assert proc.effective_config["tool_calling"]["source"] == "detected_family_metadata"

    @pytest.mark.asyncio
    async def test_load_model_promotes_gemma4_family_metadata_to_supported(self):
        """Gemma 4 GGUF tool markers should be enough when runtime exposes a chat template."""
        from cyber_inference.services.auto_loader import AutoLoader

        process_manager = MagicMock()
        proc = MagicMock(status="running", port=9338, server_type="llama", effective_config={})
        process_manager.start_server = AsyncMock(return_value=proc)
        process_manager.get_server_props = AsyncMock(
            return_value={"chat_template": "{% if tools %}<|tool_call>{% endif %}"}
        )
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "gemma-4-27b-it",
                "engine_type": "llama",
                "context_length": 131072,
                "default_context_size": None,
                "model_type": "chat",
                "mmproj_path": None,
                "hf_repo_id": "google/gemma-4-27b-it-gguf",
                "tool_template_mode": None,
                "tool_template_name": None,
                "tool_template_path": None,
                "tool_jinja_enabled": None,
                "gguf_has_chat_template": True,
                "gguf_has_tool_call_tokens": True,
                "gguf_has_tool_response_tokens": False,
                "gguf_has_response_schema_tool_calls": True,
                "gguf_has_gemma4_tool_parser": True,
                "gguf_architecture": "gemma4",
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)

        await loader.load_model("gemma-4-27b-it")

        assert proc.effective_config["tool_calling"]["status"] == "supported"
        assert proc.effective_config["tool_calling"]["source"] == "detected_family_metadata"

    @pytest.mark.asyncio
    async def test_load_model_marks_probe_failures_without_failing_chat_load(self):
        """Tool capability probe failures should not block normal llama chat loads."""
        from cyber_inference.services.auto_loader import AutoLoader

        process_manager = MagicMock()
        proc = MagicMock(status="running", port=9338, server_type="llama", effective_config={})
        process_manager.start_server = AsyncMock(return_value=proc)
        process_manager.get_server_props = AsyncMock(side_effect=RuntimeError("props unavailable"))
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "demo",
                "engine_type": "llama",
                "context_length": 8192,
                "default_context_size": None,
                "model_type": "chat",
                "mmproj_path": None,
                "hf_repo_id": "demo/repo",
                "tool_template_mode": None,
                "tool_template_name": None,
                "tool_template_path": None,
                "tool_jinja_enabled": None,
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)

        url = await loader.load_model("demo")

        assert url == "http://127.0.0.1:9338"
        assert proc.effective_config["tool_calling"]["status"] == "probe_failed"
        assert proc.effective_config["tool_calling"]["warnings"]

    @pytest.mark.asyncio
    async def test_non_llama_load_ignores_global_jinja_disable(self, monkeypatch):
        """Disabling llama jinja globally should not break non-llama backends."""
        from cyber_inference.services.auto_loader import AutoLoader

        monkeypatch.setenv("CYBER_INFERENCE_LLAMA_ENABLE_JINJA", "false")
        reload_settings()

        process_manager = MagicMock()
        proc = MagicMock(status="running", port=9444, server_type="transformers", effective_config={})
        process_manager.start_transformers_server = AsyncMock(return_value=proc)
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "demo",
                "engine_type": "transformers",
                "context_length": 8192,
                "default_context_size": None,
                "model_type": "chat",
                "mmproj_path": None,
                "hf_repo_id": "demo/repo",
                "tool_template_mode": None,
                "tool_template_name": None,
                "tool_template_path": None,
                "tool_jinja_enabled": None,
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo"))
        model_manager.update_last_used = AsyncMock()

        try:
            loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)
            url = await loader.load_model("demo")
        finally:
            monkeypatch.delenv("CYBER_INFERENCE_LLAMA_ENABLE_JINJA", raising=False)
            reload_settings()

        assert url == "http://127.0.0.1:9444"


def test_vendored_web_assets_exist():
    """Required vendored frontend assets should exist in the repo."""
    root = Path("src/cyber_inference/web/static")
    for asset_path in (
        "css/app.css",
        "images/ramborogers.png",
        "fonts/orbitron-400.ttf",
    ):
        assert (root / asset_path).exists()


class TestDownloadProgressEvents:
    """Tests for download progress event shaping."""

    def test_build_download_progress_event_includes_session_and_phase(self):
        """Download progress events should expose the session envelope and phase fields."""
        from cyber_inference.api.websocket import build_download_progress_event

        event = build_download_progress_event(
            repo_id="demo/repo",
            filename="demo.gguf",
            progress=42.0,
            status="downloading",
            message="Downloading model",
            download_id="download-123",
            phase="downloading_model",
            downloaded_bytes=420,
            total_bytes=1000,
        )

        assert event["repo_id"] == "demo/repo"
        assert event["download_id"] == "download-123"
        assert event["phase"] == "downloading_model"
        assert event["message"] == "Downloading model"
        assert event["downloaded_bytes"] == 420
