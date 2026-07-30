"""Focused runtime policy tests for MTP and context-size handling."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cyber_inference.services.auto_loader import AutoLoader
from cyber_inference.services.model_manager import ModelManager
from cyber_inference.services.process_manager import ProcessManager


def test_mtp_command_includes_separate_draft_and_explicit_projector(tmp_path: Path) -> None:
    """Separate MTP and explicitly selected vision artifacts can launch together."""
    manager = ProcessManager(models_dir=tmp_path, bin_dir=tmp_path / "bin")
    projector = tmp_path / "mmproj-Qwen3.6-27B.gguf"
    projector.write_bytes(b"projector")
    draft = tmp_path / "mtp-Qwen3.6-27B-Q4_0.gguf"

    command = manager._build_llama_server_command(
        Path("/tmp/llama-server"),
        tmp_path / "Qwen3.6-27B-Q4_K_M.gguf",
        9338,
        32768,
        -1,
        None,
        False,
        projector,
        {
            "mtp_enabled": True,
            "mtp_spec_type": "draft-mtp",
            "mtp_spec_draft_n_max": 2,
            "mtp_draft_path": str(draft),
            "parallel": 1,
            "flash_attn": "on",
        },
    )

    assert command[command.index("--mmproj") + 1] == str(projector)
    assert command[command.index("--spec-draft-model") + 1] == str(draft)
    assert command[command.index("--spec-draft-n-max") + 1] == "2"


def test_process_manager_validates_separate_mtp_metadata(tmp_path: Path) -> None:
    """A separate draft must be distinct, present, and metadata-confirmed."""
    manager = ProcessManager(models_dir=tmp_path, bin_dir=tmp_path / "bin")
    target = tmp_path / "Qwen3.6-27B-Q4_K_M.gguf"
    draft = tmp_path / "mtp-Qwen3.6-27B-Q4_0.gguf"
    target.write_bytes(b"target")
    draft.write_bytes(b"draft")

    with patch.object(
        ModelManager,
        "_read_gguf_metadata_summary_cached",
        return_value={"mtp_capable": True, "mtp_nextn_predict_layers": 1},
    ):
        result = manager._validate_mtp_draft_model(
            target,
            {"mtp_enabled": True, "mtp_draft_path": str(draft)},
        )

    assert result == draft.resolve()

    with pytest.raises(ValueError, match="separate GGUF"):
        manager._validate_mtp_draft_model(
            target,
            {"mtp_enabled": True, "mtp_draft_path": str(target)},
        )

    with pytest.raises(ValueError, match="not found"):
        manager._validate_mtp_draft_model(
            target,
            {"mtp_enabled": True, "mtp_draft_path": str(tmp_path / "missing.gguf")},
        )

    wrong_draft = tmp_path / "mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
    wrong_draft.write_bytes(b"wrong draft")
    with pytest.raises(ValueError, match="does not match target model"):
        manager._validate_mtp_draft_model(
            target,
            {"mtp_enabled": True, "mtp_draft_path": str(wrong_draft)},
        )

    with (
        patch.object(
            ModelManager,
            "_read_gguf_metadata_summary_cached",
            return_value={"mtp_capable": False},
        ),
        pytest.raises(ValueError, match="not a valid MTP GGUF"),
    ):
        manager._validate_mtp_draft_model(
            target,
            {"mtp_enabled": True, "mtp_draft_path": str(draft)},
        )


def test_context_resolution_uses_native_and_rejects_oversized_explicit_values() -> None:
    """Native context wins by default, but explicit choices retain their safety ceiling."""
    assert AutoLoader._resolve_context_config(
        {"context_length": 262144, "default_context_size": None},
        strict=True,
    ) == (262144, "model_native_max")
    assert AutoLoader._resolve_context_config(
        {"context_length": 262144, "default_context_size": 8192},
        strict=True,
    ) == (8192, "configured_default")
    assert AutoLoader._resolve_context_config(
        {"context_length": 262144},
        context_size_override=16384,
        strict=True,
    ) == (16384, "load_override")

    with pytest.raises(ValueError, match="exceeds the configured maximum"):
        AutoLoader._resolve_context_config(
            {"context_length": 262144},
            context_size_override=65536,
            strict=True,
        )
    with pytest.raises(ValueError, match="Configured context size"):
        AutoLoader._resolve_context_config(
            {"context_length": 262144, "default_context_size": 65536},
            strict=True,
        )
    with pytest.raises(ValueError, match="at least 1024"):
        AutoLoader._resolve_context_config(
            {"context_length": 262144},
            context_size_override=512,
            strict=True,
        )


def test_mtp_runtime_rejects_wrong_head_but_disabled_mode_can_run(
    tmp_path: Path,
) -> None:
    """Split-head identity is strict unless the operator disables MTP."""
    target = tmp_path / "Qwen3.6-27B-Q4_K_M.gguf"
    wrong_draft = tmp_path / "mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
    target.write_bytes(b"target")
    wrong_draft.write_bytes(b"draft")
    model_info = {
        "name": target.stem,
        "filename": target.name,
        "path": str(target),
        "engine_type": "llama",
        "model_type": "chat",
        "mtp_capable": True,
        "mtp_detection_source": "separate",
        "mtp_draft_path": str(wrong_draft),
    }
    loader = AutoLoader(process_manager=MagicMock(), model_manager=MagicMock())

    with (
        patch.object(
            ModelManager,
            "_read_gguf_metadata_summary_cached",
            return_value={"mtp_capable": True, "mtp_nextn_predict_layers": 1},
        ),
        pytest.raises(ValueError, match="does not match target model"),
    ):
        loader._resolve_mtp_config(
            model_info,
            strict=True,
            model_path=target,
        )

    model_info["mtp_mode"] = "disabled"
    launch, status = loader._resolve_mtp_config(
        model_info,
        strict=True,
        model_path=target,
    )

    assert launch == {"mtp_enabled": False}
    assert status["enabled"] is False
    assert any("does not match target model" in warning for warning in status["warnings"])


@pytest.mark.asyncio
async def test_load_model_uses_separate_mtp_with_explicit_projector(tmp_path: Path) -> None:
    """AutoLoader validates and forwards a separate head without dropping vision."""
    target = tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    draft = tmp_path / "mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
    projector = tmp_path / "mmproj-Qwen3.6-35B-A3B.gguf"
    target.write_bytes(b"target")
    draft.write_bytes(b"draft")
    projector.write_bytes(b"projector")

    process_manager = MagicMock()
    process_manager.ensure_draft_mtp_support = AsyncMock()
    process_manager.start_server = AsyncMock(
        return_value=MagicMock(
            status="running",
            port=9338,
            server_type="llama",
            effective_config={},
        )
    )
    process_manager.get_server_props = AsyncMock(
        return_value={"chat_template": "{{ messages }}"}
    )
    model_manager = MagicMock()
    model_manager.get_model = AsyncMock(
        return_value={
            "name": "Qwen3.6-35B-A3B-Q4_K_M",
            "filename": target.name,
            "file_path": str(target),
            "engine_type": "llama",
            "context_length": 262144,
            "default_context_size": None,
            "model_type": "chat",
            "mmproj_path": str(projector),
            "mtp_draft_path": str(draft),
            "hf_repo_id": "ggml-org/Qwen3.6-35B-A3B-GGUF",
            "mtp_capable": True,
            "mtp_mode": "auto",
        }
    )
    model_manager.get_model_path = AsyncMock(return_value=target)
    model_manager.update_last_used = AsyncMock()
    loader = AutoLoader(
        process_manager=process_manager,
        model_manager=model_manager,
    )

    with patch.object(
        ModelManager,
        "_read_gguf_metadata_summary_cached",
        return_value={"mtp_capable": True, "mtp_nextn_predict_layers": 1},
    ):
        await loader.load_model("Qwen3.6-35B-A3B-Q4_K_M")

    process_manager.ensure_draft_mtp_support.assert_awaited_once()
    call = process_manager.start_server.await_args.kwargs
    assert call["context_size"] == 262144
    assert call["mmproj_path"] == projector
    launch = call["effective_config"]["launch_config"]
    assert launch["context_source"] == "model_native_max"
    assert launch["mtp_draft_path"] == str(draft.resolve())
    assert call["effective_config"]["mtp"]["source"] == "separate"
    assert call["effective_config"]["vision"] == {
        "available": True,
        "enabled": True,
        "source": "mmproj",
        "suppressed_by_mtp": False,
    }


@pytest.mark.asyncio
async def test_process_manager_rejects_context_above_maximum(tmp_path: Path) -> None:
    """Direct process starts cannot bypass the context ceiling."""
    manager = ProcessManager(models_dir=tmp_path, bin_dir=tmp_path / "bin")

    with pytest.raises(ValueError, match="exceeds the configured maximum"):
        await manager.start_server(
            "demo",
            tmp_path / "demo.gguf",
            context_size=65536,
        )

    with pytest.raises(ValueError, match="exceeds the configured maximum"):
        await manager.start_server(
            "mismatched-native",
            tmp_path / "mismatched-native.gguf",
            context_size=65536,
            effective_config={
                "launch_config": {
                    "native_context_size": 262144,
                    "context_source": "model_native_max",
                }
            },
        )

    assert manager._port_allocations == set()


@pytest.mark.asyncio
async def test_process_manager_allows_detected_native_context_above_explicit_maximum(
    tmp_path: Path,
) -> None:
    """A metadata-derived native context reaches the exact llama-server command."""
    manager = ProcessManager(models_dir=tmp_path, bin_dir=tmp_path / "bin")
    process = MagicMock(pid=1234, returncode=None, stdout=None)
    create_process = AsyncMock(return_value=process)
    effective_config = {
        "launch_config": {
            "context_size": 262144,
            "native_context_size": 262144,
            "context_source": "model_native_max",
        }
    }

    with (
        patch.object(
            manager._installer,
            "get_binary_path",
            return_value=Path("/tmp/llama-server"),
        ),
        patch.object(manager, "_find_available_port", return_value=9338),
        patch("asyncio.create_subprocess_exec", create_process),
        patch.object(manager, "_wait_for_ready", AsyncMock()),
    ):
        loaded = await manager.start_server(
            "native-256k",
            tmp_path / "native-256k.gguf",
            context_size=262144,
            effective_config=effective_config,
        )

    command = create_process.await_args.args
    assert command[command.index("--ctx-size") + 1] == "262144"
    assert loaded.context_size == 262144
    assert loaded.effective_config == effective_config
