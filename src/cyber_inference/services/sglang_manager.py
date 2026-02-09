"""
SGLang availability and runtime management.

Handles:
- Detecting whether SGLang is installed and importable
- Querying SGLang version information
- Resolving the Python executable for subprocess launches
- CUDA availability detection
"""

import sys
from typing import Optional

from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)


class SGLangManager:
    """
    Manages SGLang availability detection and runtime helpers.

    This does NOT install SGLang - installation is handled by uv
    via the optional `sglang` dependency group.
    """

    _instance: Optional["SGLangManager"] = None
    _available: Optional[bool] = None
    _version: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "SGLangManager":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_available(self) -> bool:
        """
        Check if SGLang is installed and importable.

        Caches the result after the first check.
        """
        if self._available is not None:
            return self._available

        try:
            import importlib

            spec = importlib.util.find_spec("sglang")
            self._available = spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            self._available = False

        if self._available:
            logger.info("[success]SGLang is available[/success]")
        else:
            logger.debug("SGLang is not installed")

        return self._available

    def get_version(self) -> Optional[str]:
        """
        Get the installed SGLang version.

        Returns:
            Version string or None if not installed.
        """
        if self._version is not None:
            return self._version

        if not self.is_available():
            return None

        try:
            from importlib.metadata import version

            self._version = version("sglang")
            return self._version
        except Exception:
            return None

    def get_python_executable(self) -> str:
        """
        Get the Python executable path for the current venv.

        Used when spawning SGLang server as a subprocess via
        `python -m sglang.launch_server`.

        Returns:
            Path to the Python executable.
        """
        return sys.executable

    def get_cuda_info(self) -> dict:
        """
        Get CUDA availability and device information.

        Returns:
            Dict with cuda_available, device_count, and devices list.
        """
        info: dict = {
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }

        try:
            import torch

            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["device_count"] = torch.cuda.device_count()
                for i in range(info["device_count"]):
                    # Get memory via properties or mem_get_info fallback
                    mem_mb = 0
                    try:
                        props = torch.cuda.get_device_properties(i)
                        mem_mb = round(props.total_mem / (1024**2))
                    except (AttributeError, RuntimeError):
                        pass
                    if mem_mb == 0:
                        try:
                            _, total = torch.cuda.mem_get_info(i)
                            mem_mb = round(total / (1024**2))
                        except Exception:
                            pass
                    info["devices"].append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total_mb": mem_mb,
                    })
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error getting CUDA info: {e}")

        return info

    def get_status(self) -> dict:
        """
        Get comprehensive SGLang status information.

        Returns:
            Dict with availability, version, CUDA info.
        """
        available = self.is_available()
        status = {
            "available": available,
            "version": self.get_version() if available else None,
            "python_executable": self.get_python_executable(),
        }

        if available:
            status["cuda"] = self.get_cuda_info()
        else:
            status["cuda"] = {"cuda_available": False, "device_count": 0, "devices": []}

        return status

    def reset_cache(self) -> None:
        """Reset cached availability and version info."""
        self._available = None
        self._version = None
