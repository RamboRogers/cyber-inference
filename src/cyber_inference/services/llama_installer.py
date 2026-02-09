"""
Automatic llama.cpp installation and updates.

Handles:
- Platform detection (macOS, Linux, Jetson)
- GPU support detection (CUDA, Metal, CPU)
- Binary download from GitHub releases
- Version management and updates
"""

import asyncio
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)

# GitHub releases API
LLAMA_CPP_REPO = "ggerganov/llama.cpp"
GITHUB_API_URL = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"


class LlamaInstaller:
    """
    Manages llama.cpp installation and updates.

    Automatically detects platform and GPU support to download
    the appropriate binary.
    """

    def __init__(self, bin_dir: Optional[Path] = None):
        """
        Initialize the installer.

        Args:
            bin_dir: Directory for binaries (default from settings)
        """
        settings = get_settings()
        self.bin_dir = bin_dir or settings.bin_dir
        self.bin_dir.mkdir(parents=True, exist_ok=True)

        self._platform = platform.system().lower()
        self._arch = platform.machine().lower()
        self._gpu_backend: Optional[str] = None

        logger.info("[info]LlamaInstaller initialized[/info]")
        logger.debug(f"  Platform: {self._platform}")
        logger.debug(f"  Architecture: {self._arch}")
        logger.debug(f"  Binary directory: {self.bin_dir}")

    async def detect_gpu_backend(self) -> str:
        """
        Detect the best GPU backend for this system.

        Returns:
            Backend name: 'cuda', 'metal', 'vulkan', or 'cpu'
        """
        logger.info("[info]Detecting GPU backend...[/info]")

        # Check for NVIDIA CUDA
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            fallback = Path("/usr/bin/nvidia-smi")
            if fallback.exists():
                nvidia_smi = str(fallback)

        if nvidia_smi:
            try:
                result = subprocess.run(
                    [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    logger.info("[success]Detected NVIDIA GPU - using CUDA backend[/success]")
                    self._gpu_backend = "cuda"
                    return "cuda"
            except Exception as e:
                logger.debug(f"CUDA detection failed: {e}")
        elif Path("/proc/driver/nvidia/gpus").exists():
            logger.info("[success]Detected NVIDIA driver via /proc - using CUDA backend[/success]")
            self._gpu_backend = "cuda"
            return "cuda"

        # Check for Apple Metal (macOS)
        if self._platform == "darwin":
            # All modern macOS systems support Metal
            logger.info("[success]macOS detected - using Metal backend[/success]")
            self._gpu_backend = "metal"
            return "metal"

        # Check for Vulkan
        if shutil.which("vulkaninfo"):
            try:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    logger.info("[success]Vulkan support detected[/success]")
                    self._gpu_backend = "vulkan"
                    return "vulkan"
            except Exception as e:
                logger.debug(f"Vulkan detection failed: {e}")

        logger.info("[warning]No GPU acceleration detected - using CPU backend[/warning]")
        self._gpu_backend = "cpu"
        return "cpu"

    def _get_release_asset_name(self, backend: str) -> str:
        """
        Get the appropriate release asset name for this platform.

        Args:
            backend: GPU backend (cuda, metal, cpu)

        Returns:
            Asset filename pattern to match
        """
        if self._platform == "darwin":
            # macOS
            if self._arch in ("arm64", "aarch64"):
                return "llama-.*-macos-arm64"
            else:
                return "llama-.*-macos-x64"

        elif self._platform == "linux":
            # Linux
            if backend == "cuda":
                if self._arch in ("arm64", "aarch64"):
                    # Jetson or ARM with CUDA
                    return "llama-.*-linux-arm64-cuda"
                else:
                    return "llama-.*-linux-x64-cuda"
            elif self._arch in ("arm64", "aarch64"):
                return "llama-.*-linux-arm64"
            else:
                return "llama-.*-linux-x64"

        elif self._platform == "windows":
            if backend == "cuda":
                return "llama-.*-win-cuda-.*-x64"
            else:
                return "llama-.*-win-x64"

        raise ValueError(f"Unsupported platform: {self._platform}")

    async def get_latest_release(self) -> dict:
        """
        Get the latest release info from GitHub.

        Returns:
            Release information dictionary
        """
        logger.info("[info]Fetching latest llama.cpp release...[/info]")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                GITHUB_API_URL,
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=30,
            )
            response.raise_for_status()
            release = response.json()

            logger.info(f"[success]Latest release: {release['tag_name']}[/success]")
            logger.debug(f"  Published: {release['published_at']}")
            logger.debug(f"  Assets: {len(release['assets'])}")

            return release

    async def download_file(self, url: str, dest: Path, expected_size: Optional[int] = None) -> None:
        """
        Download a file with progress logging.

        Args:
            url: URL to download
            dest: Destination path
            expected_size: Expected file size in bytes
        """
        logger.info(f"[info]Downloading: {url}[/info]")
        logger.debug(f"  Destination: {dest}")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url, timeout=300) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                last_log_percent = 0

                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            if percent >= last_log_percent + 10:
                                logger.info(f"  Download progress: {percent}%")
                                last_log_percent = percent

        logger.info(f"[success]Download complete: {dest.name} ({downloaded / (1024*1024):.1f} MB)[/success]")

    async def extract_archive(self, archive_path: Path, dest_dir: Path) -> None:
        """
        Extract a downloaded archive.

        Args:
            archive_path: Path to the archive
            dest_dir: Destination directory
        """
        logger.info(f"[info]Extracting: {archive_path.name}[/info]")

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        elif archive_path.suffix in (".gz", ".tgz") or ".tar" in archive_path.name:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(dest_dir)
        else:
            raise ValueError(f"Unknown archive format: {archive_path.suffix}")

        logger.info("[success]Extraction complete[/success]")

    def _find_llama_server(self, search_dir: Path) -> Optional[Path]:
        """
        Find the llama-server binary in the extracted files.

        Args:
            search_dir: Directory to search

        Returns:
            Path to llama-server binary, or None
        """
        patterns = ["llama-server", "llama-server.exe", "server", "server.exe"]

        for pattern in patterns:
            for path in search_dir.rglob(pattern):
                if path.is_file():
                    logger.debug(f"Found llama-server: {path}")
                    return path

        return None

    async def install(
        self,
        platform: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Install or update llama.cpp.

        Args:
            platform: Override platform detection
            force: Force reinstall even if already installed

        Returns:
            Path to the llama-server binary
        """
        logger.info("[highlight]═══════════════════════════════════════════════════════════[/highlight]")
        logger.info("[highlight]           Installing llama.cpp                            [/highlight]")
        logger.info("[highlight]═══════════════════════════════════════════════════════════[/highlight]")

        # Check if already installed
        llama_server_path = self.bin_dir / "llama-server"
        if self._platform == "windows":
            llama_server_path = self.bin_dir / "llama-server.exe"

        if llama_server_path.exists() and not force:
            logger.info(f"[success]llama-server already installed: {llama_server_path}[/success]")

            # Check version
            try:
                result = subprocess.run(
                    [str(llama_server_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                logger.info(f"  Version: {result.stdout.strip() or result.stderr.strip()}")
            except Exception:
                pass

            return llama_server_path

        # Detect GPU backend
        backend = await self.detect_gpu_backend()
        logger.info(f"Selected backend: {backend}")

        # Get latest release
        release = await self.get_latest_release()

        # Find matching asset
        import re
        asset_pattern = self._get_release_asset_name(backend)
        logger.debug(f"Looking for asset matching: {asset_pattern}")

        matching_asset = None
        for asset in release["assets"]:
            if re.match(asset_pattern, asset["name"]):
                matching_asset = asset
                break

        if not matching_asset:
            # Fall back to generic build
            logger.warning(f"[warning]No specific build found, trying generic...[/warning]")
            for asset in release["assets"]:
                if self._platform in asset["name"].lower():
                    matching_asset = asset
                    break

        if not matching_asset:
            raise RuntimeError(
                f"No compatible llama.cpp build found for {self._platform} {self._arch}"
            )

        logger.info(f"[info]Selected asset: {matching_asset['name']}[/info]")

        # Download
        download_url = matching_asset["browser_download_url"]
        archive_name = matching_asset["name"]
        archive_path = self.bin_dir / archive_name

        await self.download_file(download_url, archive_path, matching_asset.get("size"))

        # Extract
        extract_dir = self.bin_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        await self.extract_archive(archive_path, extract_dir)

        # Find and copy llama-server
        server_binary = self._find_llama_server(extract_dir)
        if not server_binary:
            raise RuntimeError("llama-server binary not found in archive")

        # Copy to bin directory
        shutil.copy2(server_binary, llama_server_path)

        # Make executable
        if self._platform != "windows":
            os.chmod(llama_server_path, os.stat(llama_server_path).st_mode | stat.S_IEXEC)

        # Copy required dynamic libraries (same directory as the binary)
        source_dir = server_binary.parent
        lib_count = 0

        # Patterns for dynamic libraries
        lib_patterns = []
        if self._platform == "darwin":
            lib_patterns = ["*.dylib"]
        elif self._platform == "linux":
            lib_patterns = ["*.so", "*.so.*"]
        elif self._platform == "windows":
            lib_patterns = ["*.dll"]

        for pattern in lib_patterns:
            for lib_file in source_dir.glob(pattern):
                dest_lib = self.bin_dir / lib_file.name
                shutil.copy2(lib_file, dest_lib)
                logger.debug(f"  Copied library: {lib_file.name}")
                lib_count += 1

        if lib_count > 0:
            logger.info(f"[success]Copied {lib_count} library files[/success]")

        # Cleanup
        archive_path.unlink()
        shutil.rmtree(extract_dir)

        logger.info(f"[success]llama-server installed: {llama_server_path}[/success]")

        # Verify
        try:
            result = subprocess.run(
                [str(llama_server_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            logger.info(f"  Version: {result.stdout.strip() or result.stderr.strip()}")
        except Exception as e:
            logger.warning(f"Could not verify installation: {e}")

        return llama_server_path

    async def get_installed_version(self) -> Optional[str]:
        """
        Get the version of the installed llama-server.

        Returns:
            Version string, or None if not installed
        """
        llama_server_path = self.get_binary_path()

        if not llama_server_path.exists():
            return None

        try:
            result = subprocess.run(
                [str(llama_server_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            raw = result.stdout.strip() or result.stderr.strip()
            # Extract just the "version: NNNN (hash)" line from noisy output
            # llama-server --version also prints CUDA device info
            for line in raw.splitlines():
                if line.strip().startswith("version:"):
                    return line.strip()
            return raw.splitlines()[-1].strip() if raw else "unknown"
        except Exception:
            return "unknown"

    def _find_system_binary(self) -> Optional[Path]:
        """
        Check if llama-server exists in system PATH.

        Returns:
            Path to system binary if found, None otherwise
        """
        binary_name = "llama-server.exe" if self._platform == "windows" else "llama-server"
        system_path = shutil.which(binary_name)
        if system_path:
            return Path(system_path)
        return None

    def is_installed(self) -> bool:
        """
        Check if llama-server is installed.

        Checks system PATH first, then falls back to bin_dir.
        """
        # Check system PATH first
        if self._find_system_binary() is not None:
            return True

        # Fall back to bin_dir
        llama_server_path = self.bin_dir / "llama-server"
        if self._platform == "windows":
            llama_server_path = self.bin_dir / "llama-server.exe"

        return llama_server_path.exists()

    def get_binary_path(self) -> Path:
        """
        Get the path to the llama-server binary.

        Checks system PATH first, then falls back to bin_dir.
        """
        # Check system PATH first
        system_binary = self._find_system_binary()
        if system_binary is not None:
            return system_binary

        # Fall back to bin_dir
        llama_server_path = self.bin_dir / "llama-server"
        if self._platform == "windows":
            llama_server_path = self.bin_dir / "llama-server.exe"
        return llama_server_path
