"""
Unit tests for Cyber-Inference services.

Tests cover:
- Resource monitor
- Configuration management
- Model manager
- Process manager (mock)
"""

import sqlite3
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from cyber_inference.core.config import Settings, get_settings, reload_settings
from cyber_inference.core.database import Base, close_database, init_database
from cyber_inference.models.db_models import Model
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
        assert settings.model_load_timeout == 300
        assert settings.pre_model_load_command_enabled is False
        assert settings.pre_model_load_command == "sudo sysctl -w vm.drop_caches=3"
        assert settings.pre_model_load_command_timeout == 15
        assert settings.max_loaded_models == 1
        assert settings.llama_mtp_default_draft_n_max == 2

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

    def test_parse_split_gguf_filename_preserves_canonical_quant_name(self):
        """Split GGUF names should strip only the terminal shard suffix."""
        from cyber_inference.services.model_manager import ModelManager

        filename = "Qwen3.5-122B-A10B-MXFP4_MOE-00001-of-00003.gguf"
        shard = ModelManager._parse_gguf_shard_filename(filename)

        assert shard is not None
        assert shard.index == 1
        assert shard.total == 3
        assert shard.is_primary is True
        assert ModelManager._canonical_split_gguf_name(filename) == (
            "Qwen3.5-122B-A10B-MXFP4_MOE"
        )
        assert ModelManager._parse_gguf_shard_filename("Model.00002-of-00003.gguf").index == 2
        assert ModelManager._parse_gguf_shard_filename("Model_00003-of-00003.gguf").index == 3
        assert ModelManager._parse_gguf_shard_filename("mmproj-00001-of-00002.gguf") is None

    def test_group_gguf_model_files_consolidates_complete_shards(self):
        """Complete split GGUF shard sets should appear as one logical file."""
        from cyber_inference.services.model_manager import ModelManager

        grouped = ModelManager._group_gguf_model_files(
            [
                {
                    "filename": "Model-Q4_K_M-00002-of-00003.gguf",
                    "size_bytes": 20,
                    "quantization": None,
                },
                {
                    "filename": "Model-Q4_K_M-00001-of-00003.gguf",
                    "size_bytes": 10,
                    "quantization": None,
                },
                {
                    "filename": "Model-Q4_K_M-00003-of-00003.gguf",
                    "size_bytes": 30,
                    "quantization": None,
                },
                {
                    "filename": "mmproj-Model-Q4_K_M.gguf",
                    "size_bytes": 5,
                    "quantization": None,
                    "is_mmproj": True,
                },
            ]
        )

        model = next(item for item in grouped if item["is_split"])
        assert model["filename"] == "Model-Q4_K_M-00001-of-00003.gguf"
        assert model["primary_filename"] == "Model-Q4_K_M-00001-of-00003.gguf"
        assert model["shard_count"] == 3
        assert model["size_bytes"] == 60
        assert model["shard_filenames"] == [
            "Model-Q4_K_M-00001-of-00003.gguf",
            "Model-Q4_K_M-00002-of-00003.gguf",
            "Model-Q4_K_M-00003-of-00003.gguf",
        ]
        assert model["is_complete"] is True
        assert model["quantization"] == "q4_k_m"

    def test_group_gguf_model_files_marks_incomplete_groups(self):
        """Incomplete split groups should stay logical and report missing shards."""
        from cyber_inference.services.model_manager import ModelManager

        grouped = ModelManager._group_gguf_model_files(
            [
                {
                    "filename": "Model-Q4_K_M-00001-of-00003.gguf",
                    "size_bytes": 10,
                    "quantization": None,
                },
                {
                    "filename": "Model-Q4_K_M-00003-of-00003.gguf",
                    "size_bytes": 30,
                    "quantization": None,
                },
            ]
        )

        assert len(grouped) == 1
        assert grouped[0]["is_split"] is True
        assert grouped[0]["is_complete"] is False
        assert grouped[0]["missing_shard_filenames"] == [
            "Model-Q4_K_M-00002-of-00003.gguf"
        ]

    def test_group_gguf_model_files_keeps_single_and_multiple_quant_groups(self):
        """Single GGUFs and separate split quantizations should not be collapsed together."""
        from cyber_inference.services.model_manager import ModelManager

        grouped = ModelManager._group_gguf_model_files(
            [
                {"filename": "Single-Q8_0.gguf", "size_bytes": 8, "quantization": "q8_0"},
                {"filename": "Model-Q4_K_M-00001-of-00002.gguf", "size_bytes": 10},
                {"filename": "Model-Q4_K_M-00002-of-00002.gguf", "size_bytes": 20},
                {"filename": "Model-Q8_0-00001-of-00002.gguf", "size_bytes": 30},
                {"filename": "Model-Q8_0-00002-of-00002.gguf", "size_bytes": 40},
                {"filename": "ggml-small.bin", "size_bytes": 5, "quantization": None},
            ]
        )

        names = [item["filename"] for item in grouped]
        assert names == [
            "ggml-small.bin",
            "Model-Q4_K_M-00001-of-00002.gguf",
            "Model-Q8_0-00001-of-00002.gguf",
            "Single-Q8_0.gguf",
        ]
        split_sizes = {item["filename"]: item["size_bytes"] for item in grouped if item["is_split"]}
        assert split_sizes == {
            "Model-Q4_K_M-00001-of-00002.gguf": 30,
            "Model-Q8_0-00001-of-00002.gguf": 70,
        }

    @pytest.mark.asyncio
    async def test_download_split_gguf_fetches_missing_and_wrong_sized_shards(
        self,
        temp_models_dir,
    ):
        """Split downloads should fetch missing shards and redownload wrong-sized shards."""
        from cyber_inference.services.model_manager import ModelManager

        manager = ModelManager(models_dir=temp_models_dir)
        (temp_models_dir / "Model-Q4_K_M-00001-of-00003.gguf").write_bytes(b"1" * 10)
        (temp_models_dir / "Model-Q4_K_M-00002-of-00003.gguf").write_bytes(b"2" * 19)

        repo_files = [
            "Model-Q4_K_M-00001-of-00003.gguf",
            "Model-Q4_K_M-00002-of-00003.gguf",
            "Model-Q4_K_M-00003-of-00003.gguf",
        ]
        detailed_files = [
            {"path": repo_files[0], "size": 10},
            {"path": repo_files[1], "size": 20},
            {"path": repo_files[2], "size": 30},
        ]
        downloaded: list[str] = []

        async def fake_download(**kwargs):
            filename = kwargs["filename"]
            downloaded.append(filename)
            kwargs["local_path"].write_bytes(b"x" * kwargs["expected_size"])
            return kwargs["local_path"]

        with (
            patch("cyber_inference.services.model_manager.list_repo_files", return_value=repo_files),
            patch.object(
                manager._hf_api,
                "list_repo_tree",
                return_value=[MagicMock(path=item["path"], size=item["size"]) for item in detailed_files],
            ),
            patch.object(manager, "_download_file_with_progress", AsyncMock(side_effect=fake_download)),
            patch.object(manager, "_download_mmproj", AsyncMock(return_value=None)),
            patch.object(manager, "_register_model", AsyncMock()),
            patch.object(manager, "_notify_progress", AsyncMock()),
        ):
            result = await manager.download_model(
                "demo/repo",
                filename="Model-Q4_K_M-00002-of-00003.gguf",
            )

        assert downloaded == [
            "Model-Q4_K_M-00002-of-00003.gguf",
            "Model-Q4_K_M-00003-of-00003.gguf",
        ]
        assert result.model_name == "Model-Q4_K_M"
        assert result.path == temp_models_dir / "Model-Q4_K_M-00001-of-00003.gguf"
        assert result.size_bytes == 60
        assert result.is_split_gguf is True

    @pytest.mark.asyncio
    async def test_download_split_gguf_force_redownloads_all_shards(self, temp_models_dir):
        """Force mode should redownload every shard in the resolved split group."""
        from cyber_inference.services.model_manager import ModelManager

        manager = ModelManager(models_dir=temp_models_dir)
        for index in range(1, 4):
            (temp_models_dir / f"Model-Q4_K_M-0000{index}-of-00003.gguf").write_bytes(b"x" * 10)

        repo_files = [
            "Model-Q4_K_M-00001-of-00003.gguf",
            "Model-Q4_K_M-00002-of-00003.gguf",
            "Model-Q4_K_M-00003-of-00003.gguf",
        ]
        downloaded: list[str] = []

        async def fake_download(**kwargs):
            downloaded.append(kwargs["filename"])
            return kwargs["local_path"]

        with (
            patch("cyber_inference.services.model_manager.list_repo_files", return_value=repo_files),
            patch.object(
                manager._hf_api,
                "list_repo_tree",
                return_value=[MagicMock(path=filename, size=10) for filename in repo_files],
            ),
            patch.object(manager, "_download_file_with_progress", AsyncMock(side_effect=fake_download)),
            patch.object(manager, "_download_mmproj", AsyncMock(return_value=None)),
            patch.object(manager, "_register_model", AsyncMock()),
            patch.object(manager, "_notify_progress", AsyncMock()),
        ):
            await manager.download_model(
                "demo/repo",
                filename="Model-Q4_K_M-00001-of-00003.gguf",
                force=True,
            )

        assert downloaded == repo_files

    @pytest.mark.asyncio
    async def test_list_repo_files_detailed_marks_mtp_and_suppresses_mmproj_suggestion(
        self,
        temp_models_dir,
    ):
        """MTP repos should default to text MTP mode instead of vision projector selection."""
        from cyber_inference.services.model_manager import ModelManager

        manager = ModelManager(models_dir=temp_models_dir)

        with patch.object(
            manager._hf_api,
            "list_repo_tree",
            return_value=[
                MagicMock(path="Qwen3.6-27B-Q4_K_M.gguf", size=10),
                MagicMock(path="Qwen3.6-27B-UD-Q4_K_XL.gguf", size=20),
                MagicMock(path="mmproj-BF16.gguf", size=2),
            ],
        ):
            result = await manager.list_repo_files_detailed("unsloth/Qwen3.6-27B-MTP-GGUF")

        assert result["is_mtp_candidate"] is True
        assert result["mtp_default_enabled"] is True
        assert result["suggested_model"] == "Qwen3.6-27B-UD-Q4_K_XL.gguf"
        assert result["suggested_mmproj"] is None
        assert result["suggested_spec_draft_n_max"] == get_settings().llama_mtp_default_draft_n_max

    @pytest.mark.asyncio
    async def test_download_mtp_model_skips_implicit_mmproj(self, temp_models_dir):
        """Detected MTP downloads should not auto-download mmproj unless explicitly selected."""
        from cyber_inference.services.model_manager import ModelManager

        manager = ModelManager(models_dir=temp_models_dir)
        repo_files = [
            "Qwen3.6-27B-Q4_K_M.gguf",
            "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            "mmproj-BF16.gguf",
        ]

        async def fake_download(**kwargs):
            kwargs["local_path"].write_bytes(b"x" * kwargs["expected_size"])
            return kwargs["local_path"]

        with (
            patch("cyber_inference.services.model_manager.list_repo_files", return_value=repo_files),
            patch.object(
                manager._hf_api,
                "list_repo_tree",
                return_value=[MagicMock(path=filename, size=10) for filename in repo_files],
            ),
            patch.object(manager, "_download_file_with_progress", AsyncMock(side_effect=fake_download)),
            patch.object(manager, "_download_mmproj", AsyncMock(return_value=temp_models_dir / "mmproj.gguf")) as download_mmproj,
            patch.object(manager, "_register_model", AsyncMock()),
            patch.object(manager, "_notify_progress", AsyncMock()),
        ):
            result = await manager.download_model("unsloth/Qwen3.6-27B-MTP-GGUF")

        assert result.filename == "Qwen3.6-27B-UD-Q4_K_XL.gguf"
        download_mmproj.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_download_split_gguf_rejects_incomplete_non_primary_submission(
        self,
        temp_models_dir,
    ):
        """A non-primary shard in an incomplete group should fail clearly."""
        from cyber_inference.services.model_manager import ModelManager

        manager = ModelManager(models_dir=temp_models_dir)
        repo_files = [
            "Model-Q4_K_M-00002-of-00003.gguf",
            "Model-Q4_K_M-00003-of-00003.gguf",
        ]

        with (
            patch("cyber_inference.services.model_manager.list_repo_files", return_value=repo_files),
            patch.object(
                manager._hf_api,
                "list_repo_tree",
                return_value=[MagicMock(path=filename, size=10) for filename in repo_files],
            ),
        ):
            with pytest.raises(ValueError, match="incomplete"):
                await manager.download_model(
                    "demo/repo",
                    filename="Model-Q4_K_M-00002-of-00003.gguf",
                )

    @pytest.mark.asyncio
    async def test_list_models_suppresses_local_secondary_shard_without_primary(
        self,
        temp_models_dir,
    ):
        """A lone secondary shard should not appear as an installed model."""
        from cyber_inference.services.model_manager import ModelManager

        (temp_models_dir / "Model-Q4_K_M-00002-of-00003.gguf").write_bytes(b"x")
        manager = ModelManager(models_dir=temp_models_dir)

        @asynccontextmanager
        async def empty_db_session():
            class EmptySession:
                async def execute(self, *_args, **_kwargs):
                    result = MagicMock()
                    result.scalars.return_value.all.return_value = []
                    return result

                async def commit(self):
                    return None

            yield EmptySession()

        with patch("cyber_inference.services.model_manager.get_db_session", empty_db_session):
            models = await manager.list_models(include_file_metadata=False)

        assert models == []

    @pytest.mark.asyncio
    async def test_register_split_gguf_reconciles_raw_primary_row(self, temp_models_dir):
        """Legacy first-shard DB rows should upgrade to the canonical split model name."""
        from cyber_inference.services.model_manager import ModelManager

        shard_paths = [
            temp_models_dir / "Model-Q4_K_M-00001-of-00003.gguf",
            temp_models_dir / "Model-Q4_K_M-00002-of-00003.gguf",
            temp_models_dir / "Model-Q4_K_M-00003-of-00003.gguf",
        ]
        for index, shard_path in enumerate(shard_paths, start=1):
            shard_path.write_bytes(str(index).encode() * 10)

        engine = create_async_engine(f"sqlite+aiosqlite:///{temp_models_dir / 'test.db'}")
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
                    name="Model-Q4_K_M-00001-of-00003",
                    filename="Model-Q4_K_M-00001-of-00003.gguf",
                    file_path=str(shard_paths[0]),
                    size_bytes=10,
                    context_length=4096,
                    is_downloaded=True,
                )
            )

        manager = ModelManager(models_dir=temp_models_dir)
        with patch("cyber_inference.services.model_manager.get_db_session", db_session):
            await manager._register_model(
                repo_id="demo/repo",
                filename="Model-Q4_K_M-00001-of-00003.gguf",
                file_path=shard_paths[0],
                model_name_override="Model-Q4_K_M",
                size_bytes_override=30,
                is_split_gguf=True,
                gguf_shard_filenames=[path.name for path in shard_paths],
            )
            models = await manager.list_models(include_file_metadata=False)

        await engine.dispose()

        assert [model["name"] for model in models] == ["Model-Q4_K_M"]
        model = models[0]
        assert model["filename"] == "Model-Q4_K_M-00001-of-00003.gguf"
        assert model["size_bytes"] == 30
        assert model["is_split_gguf"] is True
        assert model["gguf_shard_count"] == 3

    @pytest.mark.asyncio
    async def test_list_models_suppresses_raw_shard_rows_when_canonical_exists(
        self,
        temp_models_dir,
    ):
        """Canonical split rows should hide legacy raw shard rows without deleting them."""
        from cyber_inference.services.model_manager import ModelManager

        shard_paths = [
            temp_models_dir / "Model-Q4_K_M-00001-of-00002.gguf",
            temp_models_dir / "Model-Q4_K_M-00002-of-00002.gguf",
        ]
        for shard_path in shard_paths:
            shard_path.write_bytes(b"x" * 10)

        engine = create_async_engine(f"sqlite+aiosqlite:///{temp_models_dir / 'test.db'}")
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
            session.add_all(
                [
                    Model(
                        name="Model-Q4_K_M",
                        filename=shard_paths[0].name,
                        file_path=str(shard_paths[0]),
                        size_bytes=20,
                        context_length=4096,
                        is_split_gguf=True,
                        gguf_shard_count=2,
                        gguf_shard_filenames=[path.name for path in shard_paths],
                        is_downloaded=True,
                    ),
                    Model(
                        name="Model-Q4_K_M-00002-of-00002",
                        filename=shard_paths[1].name,
                        file_path=str(shard_paths[1]),
                        size_bytes=10,
                        context_length=4096,
                        is_downloaded=True,
                    ),
                ]
            )

        manager = ModelManager(models_dir=temp_models_dir)
        with patch("cyber_inference.services.model_manager.get_db_session", db_session):
            models = await manager.list_models(include_file_metadata=False)

        async with db_session() as session:
            result = await session.execute(select(Model))
            stored_rows = result.scalars().all()

        await engine.dispose()

        assert [model["name"] for model in models] == ["Model-Q4_K_M"]
        assert len(stored_rows) == 2

    @pytest.mark.asyncio
    async def test_register_model_stores_relative_paths_and_lists_resolved_paths(self, temp_models_dir):
        """Models under models_dir should be stored relatively and listed as resolved paths."""
        from cyber_inference.services.model_manager import ModelManager

        model_path = temp_models_dir / "nested" / "demo.gguf"
        mmproj_path = temp_models_dir / "nested" / "mmproj-demo.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"demo")
        mmproj_path.write_bytes(b"mmproj")

        engine = create_async_engine(f"sqlite+aiosqlite:///{temp_models_dir / 'relative.db'}")
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

        manager = ModelManager(models_dir=temp_models_dir)
        with patch("cyber_inference.services.model_manager.get_db_session", db_session):
            await manager._register_model(
                repo_id="demo/repo",
                filename=model_path.name,
                file_path=model_path,
                mmproj_path=mmproj_path,
            )

            async with db_session() as session:
                result = await session.execute(select(Model))
                stored = result.scalar_one()
                assert stored.file_path == "nested/demo.gguf"
                assert stored.mmproj_path == "nested/mmproj-demo.gguf"

            models = await manager.list_models(include_file_metadata=False)
            model = models[0]
            assert Path(model["path"]) == model_path.resolve(strict=False)
            assert Path(model["mmproj_path"]) == mmproj_path.resolve(strict=False)
            assert await manager.get_model_path(model["name"]) == model_path.resolve(strict=False)

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_list_models_keeps_split_paths_relative_in_db_but_resolves_for_runtime(
        self,
        temp_models_dir,
    ):
        """Nested split GGUF rows should stay relative in DB while runtime paths resolve absolutely."""
        from cyber_inference.services.model_manager import ModelManager

        shard_dir = temp_models_dir / "MXFP4_MOE"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [
            shard_dir / "Model-00001-of-00002.gguf",
            shard_dir / "Model-00002-of-00002.gguf",
        ]
        for shard_path in shard_paths:
            shard_path.write_bytes(b"x" * 10)

        engine = create_async_engine(f"sqlite+aiosqlite:///{temp_models_dir / 'split-relative.db'}")
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
                    name="Model",
                    filename=shard_paths[0].name,
                    file_path="MXFP4_MOE/Model-00001-of-00002.gguf",
                    size_bytes=20,
                    context_length=4096,
                    is_split_gguf=True,
                    gguf_shard_count=2,
                    gguf_shard_filenames=[path.name for path in shard_paths],
                    is_downloaded=True,
                )
            )

        manager = ModelManager(models_dir=temp_models_dir)
        with patch("cyber_inference.services.model_manager.get_db_session", db_session):
            models = await manager.list_models(include_file_metadata=False)
            assert Path(models[0]["path"]) == shard_paths[0].resolve(strict=False)

            async with db_session() as session:
                result = await session.execute(select(Model))
                stored = result.scalar_one()
                assert stored.file_path == "MXFP4_MOE/Model-00001-of-00002.gguf"

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_database_init_adds_split_gguf_columns_idempotently(self, temp_models_dir):
        """Startup migration should add split metadata columns and tolerate repeat runs."""
        db_path = temp_models_dir / "migration.db"

        await init_database(db_path)
        await close_database()
        await init_database(db_path)
        await close_database()

        with sqlite3.connect(db_path) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info('models')").fetchall()
            }

        assert {
            "is_split_gguf",
            "gguf_shard_count",
            "gguf_shard_filenames",
            "mtp_capable",
            "mtp_mode",
            "mtp_detection_source",
            "mtp_nextn_predict_layers",
            "mtp_spec_draft_n_max",
        } <= columns

    @pytest.mark.asyncio
    async def test_database_init_migrates_legacy_mtp_draft_defaults(self, temp_models_dir):
        """Startup migration should retune old auto-populated MTP draft settings."""
        db_path = temp_models_dir / "mtp-defaults.db"

        await init_database(db_path)
        await close_database()

        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                INSERT INTO configurations (key, value, value_type)
                VALUES ('llama_mtp_default_draft_n_max', '6', 'int')
                """
            )
            connection.execute(
                """
                INSERT INTO models (
                    name, filename, file_path, size_bytes, context_length,
                    mtp_capable, mtp_mode, mtp_spec_draft_n_max,
                    is_downloaded, is_enabled, download_progress
                )
                VALUES (
                    'qwen-mtp', 'qwen-mtp.gguf', '/tmp/qwen-mtp.gguf', 10, 4096,
                    1, 'auto', 6, 1, 1, 100.0
                )
                """
            )

        await init_database(db_path)
        await close_database()

        with sqlite3.connect(db_path) as connection:
            config_value = connection.execute(
                "SELECT value FROM configurations WHERE key = 'llama_mtp_default_draft_n_max'"
            ).fetchone()[0]
            model_value = connection.execute(
                "SELECT mtp_spec_draft_n_max FROM models WHERE name = 'qwen-mtp'"
            ).fetchone()[0]

        assert config_value == "2"
        assert model_value == 2


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

    def test_build_llama_server_command_prioritizes_mtp_over_mmproj(self, temp_dirs):
        """MTP launches should force draft-mtp mode and avoid incompatible mmproj flags."""
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
            {
                "mtp_enabled": True,
                "mtp_spec_type": "draft-mtp",
                "mtp_spec_draft_n_max": 2,
                "parallel": 1,
                "flash_attn": "on",
                "chat_template_kwargs": {"preserve_thinking": True},
            },
        )

        assert "--mmproj" not in cmd
        assert cmd[cmd.index("--parallel") + 1] == "1"
        assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "2"
        assert cmd[cmd.index("--flash-attn") + 1] == "on"
        assert cmd[cmd.index("--chat-template-kwargs") + 1] == '{"preserve_thinking":true}'

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

    @pytest.mark.asyncio
    async def test_start_server_uses_configured_model_load_timeout(self, temp_dirs):
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)
        process = MagicMock(pid=1234, returncode=None, stdout=None)
        settings = MagicMock(
            default_context_size=8192,
            llama_gpu_layers=-1,
            llama_threads=None,
            model_load_timeout=444,
        )
        wait_for_ready = AsyncMock()

        with (
            patch("cyber_inference.services.process_manager.get_settings", return_value=settings),
            patch.object(pm._installer, "get_binary_path", return_value=Path("/tmp/llama-server")),
            patch.object(pm, "_find_available_port", return_value=9338),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)),
            patch.object(pm, "_wait_for_ready", wait_for_ready),
        ):
            await pm.start_server("demo", Path("/tmp/demo.gguf"))

        wait_for_ready.assert_awaited_once_with("demo", 9338, timeout=444.0)

    @pytest.mark.asyncio
    async def test_start_whisper_and_transformers_use_configured_model_load_timeout(self, temp_dirs):
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)
        settings = MagicMock(llama_threads=None, model_load_timeout=555)
        wait_for_whisper_ready = AsyncMock()

        with (
            patch("cyber_inference.services.process_manager.get_settings", return_value=settings),
            patch.object(pm._whisper_installer, "is_installed", return_value=True),
            patch.object(pm._whisper_installer, "get_binary_path", return_value=Path("/tmp/whisper-server")),
            patch.object(pm, "_find_available_port", return_value=9339),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=MagicMock(pid=1, returncode=None, stdout=None))),
            patch.object(pm, "_wait_for_whisper_ready", wait_for_whisper_ready),
        ):
            await pm.start_whisper_server("whisper", Path("/tmp/whisper.bin"))

        wait_for_whisper_ready.assert_awaited_once_with("whisper", 9339, timeout=555.0)

        wait_for_server_ready = AsyncMock()
        with (
            patch("cyber_inference.services.process_manager.get_settings", return_value=settings),
            patch.object(pm, "_find_available_port", return_value=9340),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=MagicMock(pid=2, returncode=None, stdout=None))),
            patch.object(pm, "_wait_for_server_ready", wait_for_server_ready),
        ):
            await pm.start_transformers_server("tf", Path("/tmp/tf"))

        wait_for_server_ready.assert_awaited_once_with(
            "tf",
            9340,
            timeout=555.0,
            server_label="Transformers",
        )

    @pytest.mark.asyncio
    async def test_start_whisper_server_skips_install_when_binary_present(self, temp_dirs):
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir)
        settings = MagicMock(llama_threads=None, model_load_timeout=555)
        wait_for_whisper_ready = AsyncMock()
        install = AsyncMock()

        with (
            patch("cyber_inference.services.process_manager.get_settings", return_value=settings),
            patch.object(pm._whisper_installer, "is_installed", return_value=True),
            patch.object(pm._whisper_installer, "install", install),
            patch.object(pm._whisper_installer, "get_binary_path", return_value=Path("/tmp/whisper-server")),
            patch.object(pm, "_find_available_port", return_value=9339),
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=MagicMock(pid=1, returncode=None, stdout=None)),
            ),
            patch.object(pm, "_wait_for_whisper_ready", wait_for_whisper_ready),
        ):
            await pm.start_whisper_server("whisper", Path("/tmp/whisper.bin"))

        install.assert_not_awaited()
        wait_for_whisper_ready.assert_awaited_once_with("whisper", 9339, timeout=555.0)

    @pytest.mark.asyncio
    async def test_cleanup_failed_start_terminates_and_removes_owned_process(self, temp_dirs):
        from cyber_inference.services.process_manager import LlamaProcess, ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir, base_port=49338)
        process = MagicMock(returncode=None)
        process.wait = AsyncMock(return_value=0)
        proc = LlamaProcess(
            model_name="demo",
            model_path=Path("/tmp/demo.gguf"),
            port=49338,
            process=process,
        )
        pm._processes["demo"] = proc
        pm._port_allocations.add(49338)

        await pm._cleanup_failed_start("demo", proc, 49338)

        process.terminate.assert_called_once()
        process.kill.assert_not_called()
        assert "demo" not in pm._processes
        assert 49338 not in pm._port_allocations

    @pytest.mark.asyncio
    async def test_cleanup_failed_start_preserves_newer_process_entry(self, temp_dirs):
        from cyber_inference.services.process_manager import LlamaProcess, ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir, base_port=49339)
        old_proc = LlamaProcess(
            model_name="demo",
            model_path=Path("/tmp/old.gguf"),
            port=49339,
            process=None,
        )
        new_proc = LlamaProcess(
            model_name="demo",
            model_path=Path("/tmp/new.gguf"),
            port=49340,
            process=None,
        )
        pm._processes["demo"] = new_proc
        pm._port_allocations.add(49339)

        await pm._cleanup_failed_start("demo", old_proc, 49339)

        assert pm._processes["demo"] is new_proc
        assert 49339 not in pm._port_allocations

    @pytest.mark.asyncio
    async def test_start_server_timeout_cleans_up_failed_process(self, temp_dirs):
        from cyber_inference.services.process_manager import ProcessManager

        models_dir, bin_dir = temp_dirs
        pm = ProcessManager(models_dir=models_dir, bin_dir=bin_dir, base_port=49340)
        process = MagicMock(pid=4321, returncode=None, stdout=None)
        process.wait = AsyncMock(return_value=0)
        settings = MagicMock(
            default_context_size=8192,
            llama_gpu_layers=-1,
            llama_threads=None,
            model_load_timeout=30,
        )

        with (
            patch("cyber_inference.services.process_manager.get_settings", return_value=settings),
            patch.object(pm._installer, "get_binary_path", return_value=Path("/tmp/llama-server")),
            patch.object(pm, "_find_available_port", return_value=49340),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)),
            patch.object(pm, "_wait_for_ready", AsyncMock(side_effect=TimeoutError("Server failed to start within 30s"))),
        ):
            with pytest.raises(TimeoutError, match="30s"):
                await pm.start_server("demo", Path("/tmp/demo.gguf"))

        process.terminate.assert_called_once()
        assert "demo" not in pm._processes
        assert 49340 not in pm._port_allocations


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
    async def test_model_load_timeout_applies_live_without_reload(self):
        from cyber_inference.services.auto_loader import AutoLoader

        loader = AutoLoader()
        loader.refresh_runtime_settings = MagicMock(
            return_value={
                "idle_timeout": 300,
                "load_timeout": 600,
                "idle_unload_enabled": False,
                "max_loaded_models": 1,
                "max_memory_percent": 80.0,
            }
        )

        result = await loader.reconcile_global_config_change("model_load_timeout")

        assert result["applied_live"] is True
        assert result["reload_triggered"] is False
        assert result["restart_required"] is False
        assert result["runtime_policy"]["load_timeout"] == 600

    @pytest.mark.asyncio
    async def test_pre_model_load_command_runs_before_server_start_when_enabled(self, monkeypatch):
        from cyber_inference.core.config import reload_settings
        from cyber_inference.services.auto_loader import AutoLoader

        monkeypatch.setenv("CYBER_INFERENCE_ADMIN_PASSWORD", "secret")
        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_ENABLED", "true")
        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND", "echo ready")
        reload_settings()

        process = MagicMock()
        process.wait = AsyncMock(return_value=0)
        process.stderr = None
        create_process = AsyncMock(return_value=process)

        process_manager = MagicMock()
        process_manager.start_server = AsyncMock(
            return_value=MagicMock(status="running", port=9338, server_type="llama", effective_config={})
        )
        process_manager.get_server_props = AsyncMock(return_value={"chat_template": "{{ messages }}"})
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "demo",
                "engine_type": "llama",
                "context_length": 4096,
                "model_type": "chat",
                "hf_repo_id": "demo/repo",
                "is_downloaded": True,
                "is_enabled": True,
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        try:
            with patch("asyncio.create_subprocess_exec", create_process):
                loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)
                await loader.load_model("demo", load_trigger="admin_manual")
        finally:
            monkeypatch.delenv("CYBER_INFERENCE_ADMIN_PASSWORD", raising=False)
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_ENABLED", raising=False)
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND", raising=False)
            reload_settings()

        create_process.assert_awaited_once()
        process_manager.start_server.assert_awaited_once()
        effective_config = process_manager.start_server.await_args.kwargs["effective_config"]
        assert effective_config["pre_model_load_command"]["status"] == "succeeded"
        assert loader._model_events["demo"]["pre_model_load_command"]["status"] == "succeeded"

    @pytest.mark.asyncio
    async def test_pre_model_load_command_skips_public_autoload(self, monkeypatch):
        from cyber_inference.core.config import reload_settings
        from cyber_inference.services.auto_loader import AutoLoader

        monkeypatch.setenv("CYBER_INFERENCE_ADMIN_PASSWORD", "secret")
        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_ENABLED", "true")
        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND", "echo ready")
        reload_settings()

        process_manager = MagicMock()
        process_manager.get_server_url = AsyncMock(return_value=None)
        process_manager.get_running_models.return_value = []
        process_manager.start_server = AsyncMock(
            return_value=MagicMock(status="running", port=9338, server_type="llama", effective_config={})
        )
        process_manager.get_server_props = AsyncMock(return_value={"chat_template": "{{ messages }}"})
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "demo",
                "engine_type": "llama",
                "context_length": 4096,
                "model_type": "chat",
                "hf_repo_id": "demo/repo",
                "is_downloaded": True,
                "is_enabled": True,
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        try:
            with patch("asyncio.create_subprocess_exec", AsyncMock()) as create_process:
                loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)
                await loader.ensure_model_loaded("demo")
        finally:
            monkeypatch.delenv("CYBER_INFERENCE_ADMIN_PASSWORD", raising=False)
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_ENABLED", raising=False)
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND", raising=False)
            reload_settings()

        create_process.assert_not_called()
        effective_config = process_manager.start_server.await_args.kwargs["effective_config"]
        assert effective_config["pre_model_load_command"]["status"] == "skipped_public_autoload"

    @pytest.mark.asyncio
    async def test_pre_model_load_command_blocks_without_admin_password(self, monkeypatch):
        from cyber_inference.core.config import reload_settings
        from cyber_inference.services.auto_loader import AutoLoader

        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_ENABLED", "true")
        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND", "echo ready")
        reload_settings()

        try:
            loader = AutoLoader()
            result = await loader._run_pre_model_load_command("demo", "admin_manual")
        finally:
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_ENABLED", raising=False)
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND", raising=False)
            reload_settings()

        assert result["status"] == "blocked"

    @pytest.mark.asyncio
    async def test_pre_model_load_command_timeout_does_not_raise(self, monkeypatch):
        from cyber_inference.core.config import reload_settings
        from cyber_inference.services.auto_loader import AutoLoader

        monkeypatch.setenv("CYBER_INFERENCE_ADMIN_PASSWORD", "secret")
        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_ENABLED", "true")
        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND", "sleep 10")
        monkeypatch.setenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_TIMEOUT", "1")
        reload_settings()

        process = MagicMock(returncode=None)
        process.wait = AsyncMock(side_effect=[TimeoutError(), 0])
        process.stderr = None

        try:
            with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
                loader = AutoLoader()
                result = await loader._run_pre_model_load_command("demo", "admin_manual")
        finally:
            monkeypatch.delenv("CYBER_INFERENCE_ADMIN_PASSWORD", raising=False)
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_ENABLED", raising=False)
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND", raising=False)
            monkeypatch.delenv("CYBER_INFERENCE_PRE_MODEL_LOAD_COMMAND_TIMEOUT", raising=False)
            reload_settings()

        assert result["status"] == "timeout"
        process.terminate.assert_called_once()

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
    async def test_load_model_enables_mtp_and_suppresses_mmproj(self):
        """Detected MTP models should probe binary support and launch without mmproj."""
        from cyber_inference.services.auto_loader import AutoLoader

        process_manager = MagicMock()
        proc = MagicMock(status="running", port=9338, server_type="llama", effective_config={})
        process_manager.ensure_draft_mtp_support = AsyncMock()
        process_manager.start_server = AsyncMock(return_value=proc)
        process_manager.get_server_props = AsyncMock(return_value={"chat_template": "{{ messages }}"})
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "Qwen3.6-27B-UD-Q4_K_XL",
                "filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
                "engine_type": "llama",
                "context_length": 131072,
                "default_context_size": None,
                "model_type": "chat",
                "mmproj_path": "/tmp/mmproj-demo.gguf",
                "hf_repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
                "mtp_capable": True,
                "mtp_mode": "auto",
                "mtp_spec_draft_n_max": 6,
                "tool_template_mode": None,
                "tool_template_name": None,
                "tool_template_path": None,
                "tool_jinja_enabled": None,
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)

        await loader.load_model("Qwen3.6-27B-UD-Q4_K_XL")

        process_manager.ensure_draft_mtp_support.assert_awaited_once()
        assert process_manager.start_server.await_args.kwargs["mmproj_path"] is None
        effective_config = process_manager.start_server.await_args.kwargs["effective_config"]
        assert effective_config["mtp"]["enabled"] is True
        assert effective_config["launch_config"]["mtp_spec_type"] == "draft-mtp"
        assert effective_config["launch_config"]["mtp_spec_draft_n_max"] == 2
        assert effective_config["launch_config"]["flash_attn"] == "on"
        assert effective_config["launch_config"]["chat_template_kwargs"] == {"preserve_thinking": True}
        assert effective_config["vision"]["suppressed_by_mtp"] is True

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
    async def test_load_model_detects_tools_for_unknown_family_with_chat_template(self):
        """Models with a chat template but no recognized family or tool-use metadata should be supported."""
        from cyber_inference.services.auto_loader import AutoLoader

        process_manager = MagicMock()
        proc = MagicMock(status="running", port=9338, server_type="llama", effective_config={})
        process_manager.start_server = AsyncMock(return_value=proc)
        process_manager.get_server_props = AsyncMock(
            return_value={"chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}"}
        )
        model_manager = MagicMock()
        model_manager.get_model = AsyncMock(
            return_value={
                "name": "Qwopus-MoE-35B-A3B-Q4_K_M",
                "engine_type": "llama",
                "context_length": 131072,
                "default_context_size": None,
                "model_type": "chat",
                "mmproj_path": None,
                "hf_repo_id": "some/repo",
                "tool_template_mode": None,
                "tool_template_name": None,
                "tool_template_path": None,
                "tool_jinja_enabled": None,
            }
        )
        model_manager.get_model_path = AsyncMock(return_value=Path("/tmp/demo.gguf"))
        model_manager.update_last_used = AsyncMock()

        loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)
        await loader.load_model("Qwopus-MoE-35B-A3B-Q4_K_M")

        assert proc.effective_config["tool_calling"]["status"] == "supported"
        assert proc.effective_config["tool_calling"]["source"] == "detected_runtime"

    @pytest.mark.asyncio
    async def test_load_model_detects_tools_ignoring_global_jinja_disable(self, monkeypatch):
        """Tool detection should work even when LLAMA_ENABLE_JINJA is disabled globally."""
        from cyber_inference.services.auto_loader import AutoLoader

        monkeypatch.setenv("CYBER_INFERENCE_LLAMA_ENABLE_JINJA", "false")
        reload_settings()

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

        try:
            loader = AutoLoader(process_manager=process_manager, model_manager=model_manager)
            await loader.load_model("demo")
        finally:
            monkeypatch.delenv("CYBER_INFERENCE_LLAMA_ENABLE_JINJA", raising=False)
            reload_settings()

        assert proc.effective_config["launch_config"]["jinja_enabled"] is True
        assert proc.effective_config["tool_calling"]["status"] == "supported"

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
            items_complete=1,
            items_total=3,
        )

        assert event["repo_id"] == "demo/repo"
        assert event["download_id"] == "download-123"
        assert event["phase"] == "downloading_model"
        assert event["message"] == "Downloading model"
        assert event["downloaded_bytes"] == 420
        assert event["items_complete"] == 1
        assert event["items_total"] == 3
