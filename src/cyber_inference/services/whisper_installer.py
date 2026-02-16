"""
Automatic whisper.cpp installation and updates.

Handles:
- Platform detection (macOS, Linux, Windows)
- GPU support detection (CUDA, Metal, CPU)
- Binary download from GitHub releases
- Version management and updates
- System binary detection (uses PATH binaries if available)
"""

import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Optional, cast
from urllib.parse import urlparse

import httpx

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)

# GitHub releases API for whisper.cpp
WHISPER_CPP_REPO = "ggerganov/whisper.cpp"
GITHUB_API_URL = f"https://api.github.com/repos/{WHISPER_CPP_REPO}/releases/latest"
HOMEBREW_FORMULA_API_URL = "https://formulae.brew.sh/api/formula/whisper-cpp.json"
HOMEBREW_ARM64_LINUX_BOTTLE_KEY = "arm64_linux"


class WhisperInstaller:
    """
    Manages whisper.cpp installation and updates.

    Automatically detects platform and GPU support to download
    the appropriate binary. Checks system PATH first before
    downloading a new binary.
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

        logger.info("[info]WhisperInstaller initialized[/info]")
        logger.debug(f"  Platform: {self._platform}")
        logger.debug(f"  Architecture: {self._arch}")
        logger.debug(f"  Binary directory: {self.bin_dir}")

    async def detect_gpu_backend(self) -> str:
        """
        Detect the best GPU backend for this system.

        Returns:
            Backend name: 'cuda', 'metal', 'vulkan', or 'cpu'
        """
        logger.info("[info]Detecting GPU backend for whisper.cpp...[/info]")

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
        # whisper.cpp release naming convention
        if self._platform == "darwin":
            # macOS - whisper.cpp uses "macos" or similar
            if self._arch in ("arm64", "aarch64"):
                return r"whisper-.*-bin-macos-arm64"
            else:
                return r"whisper-.*-bin-macos-x64"

        elif self._platform == "linux":
            if backend == "cuda":
                if self._arch in ("arm64", "aarch64"):
                    return r"whisper-.*-bin-linux-arm64-cuda"
                else:
                    return r"whisper-.*-bin-linux-x64-cuda"
            elif self._arch in ("arm64", "aarch64"):
                return r"whisper-.*-bin-linux-arm64"
            else:
                return r"whisper-.*-bin-linux-x64"

        elif self._platform == "windows":
            if backend == "cuda":
                return r"whisper-.*-bin-win-cuda-.*-x64"
            else:
                return r"whisper-.*-bin-win-x64"

        raise ValueError(f"Unsupported platform: {self._platform}")

    async def get_latest_release(self) -> dict:
        """
        Get the latest release info from GitHub.

        Returns:
            Release information dictionary
        """
        logger.info("[info]Fetching latest whisper.cpp release...[/info]")

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

    def _is_linux_arm64(self) -> bool:
        """Return True when running on Linux ARM64/AArch64."""
        return self._platform == "linux" and self._arch in ("arm64", "aarch64")

    def _matches_arch(self, asset_name: str) -> bool:
        """Check whether an asset name is compatible with the current architecture."""
        name = asset_name.lower()
        if self._arch in ("arm64", "aarch64"):
            return "arm64" in name or "aarch64" in name
        return "x64" in name or "x86_64" in name or "amd64" in name

    def _select_github_asset(self, release: dict[str, Any], backend: str) -> Optional[dict[str, Any]]:
        """
        Select a compatible GitHub release asset for this platform/arch/backend.
        """
        assets = release.get("assets", [])
        if not isinstance(assets, list):
            return None

        asset_pattern = self._get_release_asset_name(backend)
        logger.debug(f"Looking for asset matching: {asset_pattern}")

        # Primary: strict backend/platform/arch pattern.
        for raw_asset in assets:
            if not isinstance(raw_asset, dict):
                continue
            asset = cast(dict[str, Any], raw_asset)
            name = asset.get("name", "")
            if isinstance(name, str) and re.match(asset_pattern, name, re.IGNORECASE):
                return asset

        # Secondary for Linux CUDA: allow same-arch CPU build.
        if self._platform == "linux" and backend == "cuda":
            cpu_pattern = self._get_release_asset_name("cpu")
            logger.warning(
                "[warning]No CUDA build found; trying same-arch CPU release asset.[/warning]"
            )
            for raw_asset in assets:
                if not isinstance(raw_asset, dict):
                    continue
                asset = cast(dict[str, Any], raw_asset)
                name = asset.get("name", "")
                if isinstance(name, str) and re.match(cpu_pattern, name, re.IGNORECASE):
                    return asset

        # Final fallback: generic but still platform + architecture constrained.
        logger.warning("[warning]No specific build found, trying generic arch-matched asset...[/warning]")
        for raw_asset in assets:
            if not isinstance(raw_asset, dict):
                continue
            asset = cast(dict[str, Any], raw_asset)
            name = asset.get("name", "")
            if not isinstance(name, str):
                continue
            name_lower = name.lower()
            if self._platform not in name_lower:
                continue
            if not self._matches_arch(name_lower):
                continue
            if "bin" not in name_lower:
                continue
            return asset

        return None

    async def _get_homebrew_arm64_linux_bottle(self) -> Optional[dict[str, str]]:
        """
        Resolve Homebrew bottle metadata for Linux ARM64 fallback.
        """
        logger.info("[info]Checking Homebrew bottle fallback for Linux ARM64...[/info]")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(HOMEBREW_FORMULA_API_URL, timeout=30)
            response.raise_for_status()
            formula_json = response.json()

        if not isinstance(formula_json, dict):
            raise RuntimeError("Unexpected Homebrew formula response format")
        formula = cast(dict[str, Any], formula_json)

        bottle_section = formula.get("bottle", {})
        if not isinstance(bottle_section, dict):
            return None
        stable_section = bottle_section.get("stable", {})
        if not isinstance(stable_section, dict):
            return None
        files = stable_section.get("files", {})
        if not isinstance(files, dict):
            return None
        bottle = files.get(HOMEBREW_ARM64_LINUX_BOTTLE_KEY)
        if not isinstance(bottle, dict):
            return None

        url = bottle.get("url")
        sha256 = bottle.get("sha256")
        if not isinstance(url, str) or not isinstance(sha256, str):
            return None

        stable_version = formula.get("versions", {}).get("stable")
        version_label = stable_version if isinstance(stable_version, str) else "stable"
        archive_name = f"whisper-cpp-{version_label}-arm64_linux.bottle.tar.gz"

        return {"name": archive_name, "url": url, "sha256": sha256}

    async def _get_ghcr_token_for_url(self, url: str) -> Optional[str]:
        """
        Fetch an anonymous GHCR token for a blob URL.
        """
        parsed = urlparse(url)
        if parsed.netloc != "ghcr.io":
            return None

        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 4 or path_parts[0] != "v2":
            return None

        if "blobs" in path_parts:
            marker_index = path_parts.index("blobs")
        elif "manifests" in path_parts:
            marker_index = path_parts.index("manifests")
        else:
            return None

        repository = "/".join(path_parts[1:marker_index]).strip("/")
        if not repository:
            return None

        params = {
            "service": "ghcr.io",
            "scope": f"repository:{repository}:pull",
        }
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get("https://ghcr.io/token", params=params, timeout=30)
            response.raise_for_status()
            token_data = response.json()

        token = token_data.get("token")
        if isinstance(token, str) and token:
            return token
        return None

    async def _get_download_headers(self, url: str) -> dict[str, str]:
        """
        Build optional auth headers for download URLs.
        """
        parsed = urlparse(url)
        if parsed.netloc != "ghcr.io":
            return {}
        token = await self._get_ghcr_token_for_url(url)
        if token is None:
            raise RuntimeError(f"Unable to acquire GHCR token for download URL: {url}")
        return {"Authorization": f"Bearer {token}"}

    def _verify_sha256(self, file_path: Path, expected_sha256: str) -> None:
        """
        Verify downloaded file SHA256 against expected digest.
        """
        digest = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest().lower()
        expected = expected_sha256.lower()
        if actual != expected:
            raise RuntimeError(
                f"SHA256 mismatch for {file_path.name}: expected {expected}, got {actual}"
            )
        logger.info(f"[success]Checksum verified for {file_path.name}[/success]")

    async def download_file(
        self,
        url: str,
        dest: Path,
        expected_size: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Download a file with progress logging.

        Args:
            url: URL to download
            dest: Destination path
            expected_size: Expected file size in bytes
            headers: Optional HTTP headers for authenticated downloads
        """
        logger.info(f"[info]Downloading: {url}[/info]")
        logger.debug(f"  Destination: {dest}")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers, timeout=300) as response:
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

        size_mb = downloaded / (1024 * 1024)
        logger.info(f"[success]Download complete: {dest.name} ({size_mb:.1f} MB)[/success]")

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

    def _find_whisper_server(self, search_dir: Path) -> Optional[Path]:
        """
        Find the whisper-server binary in the extracted files.

        Args:
            search_dir: Directory to search

        Returns:
            Path to whisper-server binary, or None
        """
        # whisper.cpp server binary names
        patterns = [
            "whisper-server",
            "whisper-server.exe",
            "server",
            "server.exe",
            "main",
            "main.exe",
        ]

        for pattern in patterns:
            for path in search_dir.rglob(pattern):
                if path.is_file() and path.stat().st_mode & stat.S_IXUSR:
                    logger.debug(f"Found whisper-server candidate: {path}")
                    # Verify it's actually whisper-server by checking help output
                    try:
                        result = subprocess.run(
                            [str(path), "--help"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        output = result.stdout + result.stderr
                        if "whisper" in output.lower() or "transcri" in output.lower():
                            logger.debug(f"Confirmed whisper-server: {path}")
                            return path
                    except Exception:
                        pass

        # Fallback: look for any executable with 'server' in name
        for path in search_dir.rglob("*server*"):
            if path.is_file():
                logger.debug(f"Found server binary (fallback): {path}")
                return path

        return None

    def _find_system_binary(self) -> Optional[Path]:
        """
        Check if whisper-server exists in system PATH.

        Returns:
            Path to system binary if found, None otherwise
        """
        # Check various possible names for whisper server binary
        binary_names = [
            "whisper-server",
            "whisper-cpp-server",
            "whisper.cpp-server",
        ]

        if self._platform == "windows":
            binary_names = [f"{name}.exe" for name in binary_names]

        for name in binary_names:
            system_path = shutil.which(name)
            if system_path:
                logger.debug(f"Found system whisper binary: {system_path}")
                return Path(system_path)

        return None

    def _copy_dynamic_libraries(self, search_dir: Path) -> int:
        """
        Copy dynamic libraries from extracted files into bin_dir.
        """
        lib_patterns = []
        if self._platform == "darwin":
            lib_patterns = ["*.dylib"]
        elif self._platform == "linux":
            lib_patterns = ["*.so", "*.so.*"]
        elif self._platform == "windows":
            lib_patterns = ["*.dll"]

        copied = 0
        copied_names: set[str] = set()
        for pattern in lib_patterns:
            for lib_file in search_dir.rglob(pattern):
                if not lib_file.is_file():
                    continue
                if lib_file.name in copied_names:
                    continue
                dest_lib = self.bin_dir / lib_file.name
                if dest_lib.exists():
                    # Don't overwrite llama.cpp or existing runtime libs.
                    continue
                shutil.copy2(lib_file, dest_lib)
                copied_names.add(lib_file.name)
                logger.debug(f"  Copied library: {lib_file.name}")
                copied += 1

        return copied

    def is_installed(self) -> bool:
        """
        Check if whisper-server is installed.

        Checks system PATH first, then falls back to bin_dir.
        """
        # Check system PATH first
        if self._find_system_binary() is not None:
            return True

        # Fall back to bin_dir
        whisper_server_path = self.bin_dir / "whisper-server"
        if self._platform == "windows":
            whisper_server_path = self.bin_dir / "whisper-server.exe"

        return whisper_server_path.exists()

    def get_binary_path(self) -> Path:
        """
        Get the path to the whisper-server binary.

        Checks system PATH first, then falls back to bin_dir.
        """
        # Check system PATH first
        system_binary = self._find_system_binary()
        if system_binary is not None:
            return system_binary

        # Fall back to bin_dir
        whisper_server_path = self.bin_dir / "whisper-server"
        if self._platform == "windows":
            whisper_server_path = self.bin_dir / "whisper-server.exe"
        return whisper_server_path

    async def install(
        self,
        platform_override: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Install or update whisper.cpp.

        Args:
            platform_override: Override platform detection
            force: Force reinstall even if already installed

        Returns:
            Path to the whisper-server binary
        """
        logger.info("[highlight]═════════════════════════════════════════════════[/highlight]")
        logger.info("[highlight]           Installing whisper.cpp                [/highlight]")
        logger.info("[highlight]═════════════════════════════════════════════════[/highlight]")

        # Check system PATH first
        system_binary = self._find_system_binary()
        if system_binary and not force:
            logger.info(f"[success]Using system whisper-server: {system_binary}[/success]")
            version = await self.get_installed_version()
            if version:
                logger.info(f"  Version: {version}")
            return system_binary

        # Check if already installed in bin_dir
        whisper_server_path = self.bin_dir / "whisper-server"
        if self._platform == "windows":
            whisper_server_path = self.bin_dir / "whisper-server.exe"

        if whisper_server_path.exists() and not force:
            logger.info(f"[success]whisper-server installed: {whisper_server_path}[/success]")
            version = await self.get_installed_version()
            if version:
                logger.info(f"  Version: {version}")
            return whisper_server_path

        # Detect GPU backend
        backend = await self.detect_gpu_backend()
        logger.info(f"Selected backend: {backend}")

        # Get latest release
        release = await self.get_latest_release()

        # Find matching release asset
        matching_asset = self._select_github_asset(release, backend)

        expected_sha256: Optional[str] = None
        download_headers: dict[str, str] = {}
        expected_size: Optional[int] = None

        if matching_asset:
            logger.info(f"[info]Selected GitHub asset: {matching_asset['name']}[/info]")
            download_url = matching_asset["browser_download_url"]
            archive_name = matching_asset["name"]
            expected_size = matching_asset.get("size")
        elif self._is_linux_arm64():
            # Linux ARM64 fallback: use Homebrew bottle metadata when GitHub has no compatible binary.
            bottle = await self._get_homebrew_arm64_linux_bottle()
            if not bottle:
                raise RuntimeError(
                    "No compatible whisper.cpp build found in GitHub release assets and "
                    "no Linux ARM64 Homebrew bottle is currently available."
                )
            download_url = bottle["url"]
            archive_name = bottle["name"]
            expected_sha256 = bottle["sha256"]
            download_headers = await self._get_download_headers(download_url)
            logger.warning(
                "[warning]Using Homebrew Linux ARM64 bottle fallback because no GitHub "
                "release asset matched this system.[/warning]"
            )
            logger.info(f"[info]Selected Homebrew bottle: {archive_name}[/info]")
        else:
            raise RuntimeError(
                f"No compatible whisper.cpp build found for {self._platform} {self._arch}. "
                "You may need to build from source or install whisper-server manually."
            )

        # Download
        archive_path = self.bin_dir / archive_name

        await self.download_file(
            download_url,
            archive_path,
            expected_size=expected_size,
            headers=download_headers or None,
        )

        if expected_sha256:
            self._verify_sha256(archive_path, expected_sha256)

        # Extract
        extract_dir = self.bin_dir / "whisper_extracted"
        extract_dir.mkdir(exist_ok=True)
        await self.extract_archive(archive_path, extract_dir)

        # Find and copy whisper-server
        server_binary = self._find_whisper_server(extract_dir)
        if not server_binary:
            raise RuntimeError("whisper-server binary not found in archive")

        # Copy to bin directory
        shutil.copy2(server_binary, whisper_server_path)

        # Make executable
        if self._platform != "windows":
            os.chmod(whisper_server_path, os.stat(whisper_server_path).st_mode | stat.S_IEXEC)

        # Copy required dynamic libraries (search full extracted tree for bottle compatibility)
        lib_count = self._copy_dynamic_libraries(extract_dir)

        if lib_count > 0:
            logger.info(f"[success]Copied {lib_count} library files[/success]")

        # Cleanup
        archive_path.unlink()
        shutil.rmtree(extract_dir)

        logger.info(f"[success]whisper-server installed: {whisper_server_path}[/success]")

        # Verify
        version = await self.get_installed_version()
        if version:
            logger.info(f"  Version: {version}")

        return whisper_server_path

    async def get_installed_version(self) -> Optional[str]:
        """
        Get the version of the installed whisper-server.

        Returns:
            Version string, or None if not installed
        """
        binary_path = self.get_binary_path()

        if not binary_path.exists():
            return None

        try:
            result = subprocess.run(
                [str(binary_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip() or result.stderr.strip()
            if version:
                return version

            # Some versions don't have --version, try --help
            result = subprocess.run(
                [str(binary_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Extract version from help output if present
            output = result.stdout + result.stderr
            for line in output.split("\n"):
                if "version" in line.lower() or "whisper" in line.lower():
                    return line.strip()[:100]  # Limit length

            return "installed (version unknown)"
        except Exception:
            return "installed (version check failed)"
