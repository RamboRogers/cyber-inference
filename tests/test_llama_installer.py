"""Tests for llama.cpp installer fallback behavior."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cyber_inference.services.llama_installer import LlamaInstaller


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
