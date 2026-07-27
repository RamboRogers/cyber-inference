"""Focused coverage for separate Qwen MTP draft-model artifacts."""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from cyber_inference.api import admin
from cyber_inference.models.db_models import Model
from cyber_inference.models.schemas import ModelCreate
from cyber_inference.services.model_manager import ModelDownloadResult, ModelManager


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "target_size", "draft_size"),
    [
        ("Qwen3.6-27B", 19_100_000_000, 1_680_000_000),
        ("Qwen3.6-35B-A3B", 20_400_000_000, 1_060_000_000),
    ],
)
async def test_repo_discovery_separates_qwen_mtp_artifacts(
    tmp_path: Path,
    model_name: str,
    target_size: int,
    draft_size: int,
) -> None:
    manager = ModelManager(models_dir=tmp_path)
    target = f"{model_name}-Q4_K_M.gguf"
    mtp = f"mtp-{model_name}-Q4_0.gguf"

    with patch.object(
        manager._hf_api,
        "list_repo_tree",
        return_value=[
            MagicMock(path=target, size=target_size),
            MagicMock(path=mtp, size=draft_size),
            MagicMock(path=f"dflash-{model_name}-Q4_0.gguf", size=100),
            MagicMock(path="mmproj-model-f16.gguf", size=200),
        ],
    ):
        result = await manager.list_repo_files_detailed(f"ggml-org/{model_name}-GGUF")

    assert [item["filename"] for item in result["model_files"]] == [target]
    assert result["model_files"][0]["artifact_type"] == "model"
    assert [item["filename"] for item in result["mtp_files"]] == [mtp]
    assert result["mtp_files"][0]["artifact_type"] == "mtp"
    assert result["suggested_model"] == target
    assert result["suggested_mtp"] == mtp
    assert result["suggested_mmproj"] is None
    assert result["is_mtp_candidate"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["Qwen3.6-27B", "Qwen3.6-35B-A3B"])
async def test_download_pairs_target_and_separate_mtp_before_registration(
    tmp_path: Path,
    model_name: str,
) -> None:
    manager = ModelManager(models_dir=tmp_path)
    target = f"{model_name}-Q4_K_M.gguf"
    mtp = f"mtp-{model_name}-Q4_0.gguf"
    repo_files = [target, mtp, "mmproj-model-f16.gguf"]
    downloaded: list[str] = []

    async def fake_download(**kwargs):
        downloaded.append(kwargs["filename"])
        kwargs["local_path"].write_bytes(b"gguf")
        return kwargs["local_path"]

    def fake_metadata(path: Path):
        if path.name == mtp:
            return {
                "mtp_capable": True,
                "mtp_nextn_predict_layers": 1,
            }
        return {"mtp_capable": False}

    with (
        patch(
            "cyber_inference.services.model_manager.list_repo_files",
            return_value=repo_files,
        ),
        patch.object(
            manager._hf_api,
            "list_repo_tree",
            return_value=[MagicMock(path=name, size=4) for name in repo_files],
        ),
        patch.object(
            manager,
            "_download_file_with_progress",
            AsyncMock(side_effect=fake_download),
        ),
        patch.object(
            manager,
            "_read_gguf_metadata_summary_cached",
            side_effect=fake_metadata,
        ),
        patch.object(manager, "_download_mmproj", AsyncMock()) as download_mmproj,
        patch.object(manager, "_register_model", AsyncMock()) as register_model,
        patch.object(manager, "_notify_progress", AsyncMock()),
    ):
        await manager.download_model(f"ggml-org/{model_name}-GGUF")

    assert downloaded == [target, mtp]
    download_mmproj.assert_not_awaited()
    assert register_model.await_args.kwargs["mtp_draft_path"] == tmp_path / mtp


def test_split_primary_shard_matches_separate_mtp_head(tmp_path: Path) -> None:
    """Split-GGUF shard suffixes must not change the model-family identity."""
    manager = ModelManager(models_dir=tmp_path)
    target = tmp_path / "Qwen3.6-27B-Q4_K_M-00001-of-00002.gguf"
    draft = tmp_path / "mtp-Qwen3.6-27B-Q4_0.gguf"
    target.write_bytes(b"target shard")
    draft.write_bytes(b"draft")

    assert manager._select_mtp_file([draft.name], target.name) == draft.name
    with patch.object(
        manager,
        "_read_gguf_metadata_summary_cached",
        return_value={"mtp_capable": True, "mtp_nextn_predict_layers": 1},
    ):
        metadata = manager._validate_mtp_draft_path(target, draft)

    assert metadata["mtp_capable"] is True


@pytest.mark.asyncio
async def test_local_scan_excludes_auxiliaries_and_pairs_valid_mtp(
    tmp_path: Path,
) -> None:
    manager = ModelManager(models_dir=tmp_path)
    target = tmp_path / "Qwen3.6-27B-Q4_K_M.gguf"
    mtp = tmp_path / "mtp-Qwen3.6-27B-Q4_0.gguf"
    dflash = tmp_path / "dflash-Qwen3.6-27B-Q4_0.gguf"
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    for path in (target, mtp, dflash, mmproj):
        path.write_bytes(b"gguf")

    @asynccontextmanager
    async def empty_db_session():
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session = MagicMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        yield session

    def fake_metadata(path: Path):
        return {
            "mtp_capable": path.name == mtp.name,
            "mtp_nextn_predict_layers": 1 if path.name == mtp.name else None,
        }

    with (
        patch(
            "cyber_inference.services.model_manager.get_db_session",
            empty_db_session,
        ),
        patch.object(
            manager,
            "_read_gguf_metadata_summary_cached",
            side_effect=fake_metadata,
        ),
    ):
        models = await manager.list_models()

    assert [model["filename"] for model in models] == [target.name]
    assert models[0]["mtp_draft_path"] == str(mtp)
    assert models[0]["mtp_capable"] is True
    assert models[0]["mtp_detection_source"] == "separate"


@pytest.mark.asyncio
async def test_download_api_passes_explicit_mtp_head_and_returns_path(
    tmp_path: Path,
) -> None:
    target = "Qwen3.6-27B-Q4_K_M.gguf"
    mtp = "mtp-Qwen3.6-27B-Q4_0.gguf"
    manager = MagicMock()
    manager.download_model = AsyncMock(
        return_value=ModelDownloadResult(
            path=tmp_path / target,
            model_name=Path(target).stem,
            filename=target,
            size_bytes=4,
        )
    )
    manager.get_model = AsyncMock(
        return_value={
            "id": 1,
            "name": Path(target).stem,
            "filename": target,
            "path": str(tmp_path / target),
            "hf_repo_id": "ggml-org/Qwen3.6-27B-GGUF",
            "size_bytes": 4,
            "quantization": "q4_k_m",
            "context_length": 4096,
            "model_type": "chat",
            "mmproj_path": None,
            "mtp_draft_path": str(tmp_path / mtp),
            "mtp_capable": True,
            "mtp_detection_source": "separate",
            "is_split_gguf": False,
            "is_downloaded": True,
            "is_enabled": True,
        }
    )

    with patch.object(admin, "ModelManager", return_value=manager):
        response = await admin.download_model(
            ModelCreate(
                hf_repo_id="ggml-org/Qwen3.6-27B-GGUF",
                hf_filename=target,
                hf_mtp_filename=mtp,
            ),
            True,
        )

    assert manager.download_model.await_args.kwargs["mtp_filename"] == mtp
    assert response.mtp_draft_path == str(tmp_path / mtp)


@pytest.mark.asyncio
async def test_registered_wrong_model_head_is_cleared(
    tmp_path: Path,
    test_db,
) -> None:
    """Persisted draft associations must match the target model identity."""
    target = tmp_path / "Qwen3.6-27B-Q4_K_M.gguf"
    wrong_draft = tmp_path / "mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
    target.write_bytes(b"target")
    wrong_draft.write_bytes(b"draft")
    stored = Model(
        name=target.stem,
        filename=target.name,
        file_path=str(target),
        hf_repo_id="ggml-org/Qwen3.6-27B-GGUF",
        size_bytes=target.stat().st_size,
        context_length=262144,
        mtp_draft_path=str(wrong_draft),
        mtp_capable=True,
        mtp_detection_source="separate",
        is_downloaded=True,
        is_enabled=True,
    )
    test_db.add(stored)
    await test_db.commit()

    @asynccontextmanager
    async def db_session():
        yield test_db

    manager = ModelManager(models_dir=tmp_path)
    with (
        patch(
            "cyber_inference.services.model_manager.get_db_session",
            db_session,
        ),
        patch.object(
            manager,
            "_read_gguf_metadata_summary_cached",
            side_effect=lambda path: {
                "mtp_capable": path == wrong_draft,
                "mtp_nextn_predict_layers": 1 if path == wrong_draft else None,
            },
        ),
    ):
        models = await manager.list_models(include_file_metadata=True)

    assert models[0]["mtp_draft_path"] is None
    await test_db.refresh(stored)
    assert stored.mtp_draft_path is None


@pytest.mark.asyncio
async def test_delete_model_removes_final_managed_mtp_head(
    tmp_path: Path,
    test_db,
) -> None:
    """A managed draft head should be removed with its final target reference."""
    target = tmp_path / "Qwen3.6-27B-Q4_K_M.gguf"
    draft = tmp_path / "mtp-Qwen3.6-27B-Q4_0.gguf"
    target.write_bytes(b"target")
    draft.write_bytes(b"draft")
    stored = Model(
        name=target.stem,
        filename=target.name,
        file_path=str(target),
        size_bytes=target.stat().st_size,
        context_length=262144,
        mtp_draft_path=str(draft),
        mtp_capable=True,
        mtp_detection_source="separate",
        is_downloaded=True,
        is_enabled=True,
    )
    test_db.add(stored)
    await test_db.commit()
    stored_id = stored.id

    @asynccontextmanager
    async def db_session():
        yield test_db

    manager = ModelManager(models_dir=tmp_path)
    model_info = {
        "id": stored_id,
        "name": stored.name,
        "path": str(target),
        "mtp_draft_path": str(draft),
    }
    with (
        patch(
            "cyber_inference.services.model_manager.get_db_session",
            db_session,
        ),
        patch.object(manager, "get_model", AsyncMock(return_value=model_info)),
    ):
        assert await manager.delete_model(stored.name) is True

    assert not target.exists()
    assert not draft.exists()
    result = await test_db.execute(select(Model).where(Model.id == stored_id))
    assert result.scalar_one_or_none() is None
