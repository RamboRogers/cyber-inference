"""
Cyber-Inference: A web GUI management tool for v1 compatible inference servers.

This package provides:
- Web GUI for managing llama.cpp inference servers
- OpenAI-compatible V1 API endpoints
- Dynamic model management and resource allocation
- Automatic model loading/unloading
- HuggingFace integration for model downloads
"""

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

__author__ = "Matthew Rogers"
__license__ = "GPL-3.0"


def _load_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject_path.exists():
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)
        version = data.get("project", {}).get("version")
        if isinstance(version, str):
            return version

    try:
        return package_version("cyber-inference")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _load_version()

from cyber_inference.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)
logger.info(f"Cyber-Inference v{__version__} initializing...")
