"""Tests for llama.cpp installer fallback behavior."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cyber_inference.services.llama_installer import LlamaInstaller


def _version_result() -> MagicMock:
    return MagicMock(stdout="version: 1000 (abc123)", stderr="")


def test_select_github_asset_does_not_choose_wrong_arch(tmp_path: Path) -> None:
    """Linux ARM should not select a Linux x64 generic asset."""
    installer = LlamaInstaller(bin_dir=tmp_path)
    installer._platform = "linux"
    installer._arch = "arm64"

    release = {
        "assets": [
            {"name": "llama-b9999-linux-x64.tar.gz", "browser_download_url": "https://example/x64"},
        ]
    }

    selected = installer._select_github_asset(release, backend="cpu")
    assert selected is None


def test_get_managed_binary_path_ignores_system_path(tmp_path: Path) -> None:
    """Managed binary path should always point inside bin_dir."""
    installer = LlamaInstaller(bin_dir=tmp_path)

    assert installer.get_managed_binary_path() == tmp_path / "llama-server"


@pytest.mark.asyncio
async def test_binary_status_reports_system_managed_path(tmp_path: Path) -> None:
    """A PATH binary outside bin_dir should be reported but not updateable."""
    installer = LlamaInstaller(bin_dir=tmp_path)
    system_binary = tmp_path.parent / "system-llama-server"
    system_binary.write_text("#!/bin/sh\n")

    with (
        patch("cyber_inference.services.llama_installer.shutil.which", return_value=str(system_binary)),
        patch("cyber_inference.services.llama_installer.subprocess.run", return_value=_version_result()),
    ):
        status = await installer.get_binary_status()

    assert status["source"] == "system"
    assert status["binary_path"] == str(system_binary)
    assert status["is_system_managed"] is True
    assert status["update_allowed"] is False
    assert "system" in status["update_blocked_reason"]


@pytest.mark.asyncio
async def test_binary_status_treats_path_to_managed_binary_as_managed(tmp_path: Path) -> None:
    """A managed binary found through PATH should remain Cyber-Inference managed."""
    installer = LlamaInstaller(bin_dir=tmp_path)
    managed_binary = installer.get_managed_binary_path()
    managed_binary.write_text("#!/bin/sh\n")

    with (
        patch("cyber_inference.services.llama_installer.shutil.which", return_value=str(managed_binary)),
        patch("cyber_inference.services.llama_installer.subprocess.run", return_value=_version_result()),
    ):
        status = await installer.get_binary_status()

    assert status["source"] == "managed"
    assert status["binary_path"] == str(managed_binary)
    assert status["is_system_managed"] is False
    assert status["update_allowed"] is True


@pytest.mark.asyncio
async def test_binary_status_reports_managed_or_missing_without_path(tmp_path: Path) -> None:
    """Local managed state should be clear when PATH has no llama-server."""
    installer = LlamaInstaller(bin_dir=tmp_path)

    with patch("cyber_inference.services.llama_installer.shutil.which", return_value=None):
        missing = await installer.get_binary_status()

    assert missing["source"] == "missing"
    assert missing["binary_path"] is None
    assert missing["managed_binary_path"] == str(tmp_path / "llama-server")
    assert missing["update_allowed"] is True

    managed_binary = installer.get_managed_binary_path()
    managed_binary.write_text("#!/bin/sh\n")
    with (
        patch("cyber_inference.services.llama_installer.shutil.which", return_value=None),
        patch("cyber_inference.services.llama_installer.subprocess.run", return_value=_version_result()),
    ):
        managed = await installer.get_binary_status()

    assert managed["source"] == "managed"
    assert managed["binary_path"] == str(managed_binary)
    assert managed["installed_version"] == "version: 1000 (abc123)"


def test_update_available_only_when_versions_are_comparable(tmp_path: Path) -> None:
    """Version comparison should be conservative."""
    installer = LlamaInstaller(bin_dir=tmp_path)

    assert installer.get_update_available("version: 1000 (abc123)", "b1001") is True
    assert installer.get_update_available("version: 1000 (abc123)", "b1000") is False
    assert installer.get_update_available("llama build unknown", "b1001") is None


@pytest.mark.asyncio
async def test_get_latest_release_info_serializes_selected_asset(tmp_path: Path) -> None:
    """Release status should expose compact metadata and matching asset name."""
    installer = LlamaInstaller(bin_dir=tmp_path)
    release = {
        "tag_name": "b1001",
        "name": "Build 1001",
        "html_url": "https://github.com/ggerganov/llama.cpp/releases/tag/b1001",
        "published_at": "2026-04-11T00:00:00Z",
        "assets": [
            {
                "name": "llama-b1001-macos-arm64.zip",
                "browser_download_url": "https://example/llama.zip",
            }
        ],
    }
    installer._platform = "darwin"
    installer._arch = "arm64"

    with (
        patch.object(installer, "get_latest_release", AsyncMock(return_value=release)),
        patch.object(installer, "detect_gpu_backend", AsyncMock(return_value="metal")),
    ):
        info = await installer.get_latest_release_info()

    assert info["tag_name"] == "b1001"
    assert info["name"] == "Build 1001"
    assert info["html_url"] == "https://github.com/ggerganov/llama.cpp/releases/tag/b1001"
    assert info["published_at"] == "2026-04-11T00:00:00Z"
    assert info["compatible_asset"] == "llama-b1001-macos-arm64.zip"


@pytest.mark.asyncio
async def test_get_homebrew_arm64_linux_bottle_parses_formula(tmp_path: Path) -> None:
    """Formula metadata should resolve to URL + SHA256 for arm64_linux."""
    installer = LlamaInstaller(bin_dir=tmp_path)

    formula_payload = {
        "versions": {"stable": "8070"},
        "bottle": {
            "stable": {
                "files": {
                    "arm64_linux": {
                        "url": "https://ghcr.io/v2/homebrew/core/llama.cpp/blobs/sha256:abc",
                        "sha256": "abc123",
                    }
                }
            }
        },
    }

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = formula_payload

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get.return_value = response

    with patch("cyber_inference.services.llama_installer.httpx.AsyncClient", return_value=client):
        bottle = await installer._get_homebrew_arm64_linux_bottle()

    assert bottle is not None
    assert bottle["url"] == "https://ghcr.io/v2/homebrew/core/llama.cpp/blobs/sha256:abc"
    assert bottle["sha256"] == "abc123"
    assert bottle["name"] == "llama.cpp-8070-arm64_linux.bottle.tar.gz"


@pytest.mark.asyncio
async def test_install_uses_homebrew_fallback_on_linux_arm(tmp_path: Path) -> None:
    """Installer should use Homebrew fallback when Linux ARM GitHub assets are missing."""
    installer = LlamaInstaller(bin_dir=tmp_path)
    installer._platform = "linux"
    installer._arch = "arm64"

    dummy_server = tmp_path / "dummy-llama-server"
    dummy_server.write_text("#!/bin/sh\necho test\n")

    async def _fake_download(
        _url: str,
        dest: Path,
        expected_size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        del expected_size, headers
        dest.write_bytes(b"fake-archive")

    with (
        patch.object(installer, "detect_gpu_backend", AsyncMock(return_value="cpu")),
        patch.object(
            installer,
            "get_latest_release",
            AsyncMock(return_value={"tag_name": "b9999", "published_at": "now", "assets": []}),
        ),
        patch.object(
            installer,
            "_get_homebrew_arm64_linux_bottle",
            AsyncMock(
                return_value={
                    "name": "llama.cpp-8070-arm64_linux.bottle.tar.gz",
                    "url": "https://ghcr.io/v2/homebrew/core/llama.cpp/blobs/sha256:abc",
                    "sha256": "abc123",
                }
            ),
        ),
        patch.object(
            installer,
            "_get_download_headers",
            AsyncMock(return_value={"Authorization": "Bearer test-token"}),
        ) as headers_mock,
        patch.object(installer, "download_file", AsyncMock(side_effect=_fake_download)) as download_mock,
        patch.object(installer, "_verify_sha256", MagicMock()) as verify_mock,
        patch.object(installer, "extract_archive", AsyncMock(return_value=None)),
        patch.object(installer, "_find_llama_server", MagicMock(return_value=dummy_server)),
        patch.object(installer, "_copy_dynamic_libraries", MagicMock(return_value=0)),
        patch("cyber_inference.services.llama_installer.subprocess.run") as run_mock,
    ):
        run_mock.return_value = MagicMock(stdout="version: test", stderr="")
        installed_path = await installer.install(force=True)

    assert installed_path == tmp_path / "llama-server"
    assert installed_path.exists()
    headers_mock.assert_awaited_once()
    verify_mock.assert_called_once()
    download_mock.assert_awaited_once()
