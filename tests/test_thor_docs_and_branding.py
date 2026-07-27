"""Regression coverage for Thor operator docs and shared browser branding."""

from pathlib import Path
from xml.etree import ElementTree

import pytest
from httpx import ASGITransport, AsyncClient

from cyber_inference.main import app

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
STATIC_ROOT = REPOSITORY_ROOT / "src/cyber_inference/web/static"
TEMPLATE_ROOT = REPOSITORY_ROOT / "src/cyber_inference/web/templates"
THOR_IMAGE = "ghcr.io/ramborogers/cyber-inference:thor-arm64"


def test_thor_readme_commands_use_the_thor_image_and_runtime():
    """Every Thor command block should be explicit, persistent, and health checked."""
    readme = README.read_text()
    quick_start = readme.split("### One-shot startup", maxsplit=1)[1].split(
        "### Local development", maxsplit=1
    )[0]
    docker_run = readme.split("### Thor / DGX Spark ARM64 NVIDIA", maxsplit=1)[1].split(
        "### Upgrade while preserving local state", maxsplit=1
    )[0]
    thor_upgrade = readme.split("#### Thor / DGX Spark ARM64 NVIDIA", maxsplit=1)[1].split(
        "#### Linux AMD64 NVIDIA", maxsplit=1
    )[0]

    for section in (quick_start, docker_run, thor_upgrade):
        assert "ghcr.io/ramborogers/cyber-inference:latest" not in section
        assert THOR_IMAGE in section
        assert "--runtime nvidia" in section
        assert '-v "$PWD/data:/app/data"' in section
        assert '-v "$PWD/models:/app/models"' in section
        assert "http://localhost:8337/health" in section

    assert quick_start.index(f"docker pull {THOR_IMAGE}") < quick_start.index(
        "docker rm -f cyber-inference"
    )
    assert thor_upgrade.index(f"docker pull {THOR_IMAGE}") < thor_upgrade.index(
        "docker rm -f cyber-inference"
    )


def test_favicon_is_vendored_and_linked_from_the_shared_layout():
    """The favicon should be valid local SVG referenced by every inherited page."""
    favicon_path = STATIC_ROOT / "images/favicon.svg"
    favicon = ElementTree.parse(favicon_path).getroot()
    base_template = (TEMPLATE_ROOT / "base.html").read_text()

    assert favicon.tag == "{http://www.w3.org/2000/svg}svg"
    assert 'rel="icon" type="image/svg+xml" href="/static/images/favicon.svg"' in base_template


@pytest.mark.asyncio
async def test_favicon_is_served_with_svg_content_type():
    """The mounted static app should expose the favicon with its browser MIME type."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/static/images/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
