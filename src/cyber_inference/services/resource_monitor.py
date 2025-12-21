"""
System resource monitoring for Cyber-Inference.

Provides real-time monitoring of:
- CPU usage and core count
- Memory (RAM) usage
- GPU detection and memory (CUDA/Metal/CPU)
- Disk usage
- Process-specific resource tracking
"""

import asyncio
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """GPU information container."""
    name: str
    vendor: str  # nvidia, apple, amd, none
    total_memory_mb: float
    used_memory_mb: float
    temperature: Optional[float] = None
    utilization_percent: Optional[float] = None


@dataclass
class SystemResources:
    """Current system resource state."""
    timestamp: datetime

    # CPU
    cpu_count: int
    cpu_percent: float
    cpu_freq_mhz: Optional[float]

    # Memory
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float

    # Swap
    swap_total_mb: float
    swap_used_mb: float
    swap_percent: float

    # GPU
    gpu: Optional[GPUInfo]

    # Disk
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    disk_percent: float


class ResourceMonitor:
    """
    Monitors system resources with async updates.

    Provides real-time resource data for:
    - Dynamic model loading decisions
    - Resource limit enforcement
    - Dashboard display
    """

    def __init__(self, update_interval: float = 2.0):
        """
        Initialize the resource monitor.

        Args:
            update_interval: Seconds between resource updates
        """
        self.update_interval = update_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._current_resources: Optional[SystemResources] = None
        self._platform = platform.system().lower()
        self._gpu_vendor: Optional[str] = None
        self._thor_memory_override_logged = False

        logger.info(f"[info]ResourceMonitor initialized[/info]")
        logger.debug(f"  Platform: {self._platform}")
        logger.debug(f"  Update interval: {update_interval}s")

    async def start(self) -> None:
        """Start the background resource monitoring task."""
        if self._running:
            logger.warning("ResourceMonitor already running")
            return

        logger.info("[info]Starting resource monitor background task[/info]")

        # Detect GPU on startup
        await self._detect_gpu()

        # Initial resource collection
        self._current_resources = await self._collect_resources()

        # Start background task
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

        logger.info("[success]Resource monitor started[/success]")

    async def stop(self) -> None:
        """Stop the background monitoring task."""
        if not self._running:
            return

        logger.info("[info]Stopping resource monitor[/info]")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("[success]Resource monitor stopped[/success]")

    async def _monitor_loop(self) -> None:
        """Background loop for periodic resource collection."""
        logger.info("Resource monitor loop started")

        # Only log resource stats periodically (every 60 iterations = ~2 minutes at 2s interval)
        log_counter = 0
        log_interval = 60

        while self._running:
            try:
                self._current_resources = await self._collect_resources()

                # Only log resources periodically to avoid log spam
                log_counter += 1
                if log_counter >= log_interval:
                    logger.debug(
                        f"Resources: CPU={self._current_resources.cpu_percent:.1f}%, "
                        f"RAM={self._current_resources.memory_percent:.1f}%"
                    )
                    log_counter = 0

            except Exception as e:
                logger.error(f"Error collecting resources: {e}")

            await asyncio.sleep(self.update_interval)

    async def _detect_gpu(self) -> None:
        """Detect available GPU and its vendor."""
        logger.info("[info]Detecting GPU...[/info]")

        # Check for NVIDIA GPU (nvidia-smi)
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
                    self._gpu_vendor = "nvidia"
                    gpu_name = result.stdout.strip().split("\n")[0]
                    logger.info(f"[success]Detected NVIDIA GPU: {gpu_name}[/success]")
                    return
            except Exception as e:
                logger.debug(f"nvidia-smi check failed: {e}")

        nvidia_proc = Path("/proc/driver/nvidia/gpus")
        if nvidia_proc.exists():
            gpu_name = None
            try:
                info_files = list(nvidia_proc.glob("*/information"))
                if info_files:
                    with info_files[0].open("r") as handle:
                        for line in handle:
                            if line.lower().startswith("model:"):
                                gpu_name = line.split(":", 1)[1].strip()
                                break
            except Exception as e:
                logger.debug(f"NVIDIA /proc check failed: {e}")

            self._gpu_vendor = "nvidia"
            if gpu_name:
                logger.info(f"[success]Detected NVIDIA GPU: {gpu_name}[/success]")
            else:
                logger.info("[success]Detected NVIDIA GPU via /proc[/success]")
            return

        # Check for Apple Silicon (macOS with Metal)
        if self._platform == "darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if "Apple" in result.stdout or "M1" in result.stdout or "M2" in result.stdout or "M3" in result.stdout or "M4" in result.stdout:
                    self._gpu_vendor = "apple"
                    logger.info("[success]Detected Apple Silicon with Metal support[/success]")
                    return
            except Exception as e:
                logger.debug(f"Apple GPU check failed: {e}")

        # Check for AMD GPU (rocm-smi)
        if shutil.which("rocm-smi"):
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showproductname"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self._gpu_vendor = "amd"
                    logger.info("[success]Detected AMD GPU with ROCm[/success]")
                    return
            except Exception as e:
                logger.debug(f"rocm-smi check failed: {e}")

        self._gpu_vendor = None
        logger.info("[warning]No GPU detected - will use CPU inference[/warning]")

    async def _get_nvidia_gpu_info(self) -> Optional[GPUInfo]:
        """Get NVIDIA GPU information via nvidia-smi."""
        def _parse_float(value: str) -> Optional[float]:
            cleaned = value.strip().lower()
            if not cleaned or cleaned in {"n/a", "not supported", "unknown"}:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                total_memory = _parse_float(parts[1])
                used_memory = _parse_float(parts[2])
                temperature = _parse_float(parts[3]) if len(parts) > 3 else None
                utilization = _parse_float(parts[4]) if len(parts) > 4 else None
                return GPUInfo(
                    name=parts[0],
                    vendor="nvidia",
                    total_memory_mb=total_memory or 0.0,
                    used_memory_mb=used_memory or 0.0,
                    temperature=temperature,
                    utilization_percent=utilization,
                )
        except Exception as e:
            logger.debug(f"Failed to get NVIDIA GPU info: {e}")

        return None

    async def _get_apple_gpu_info(self) -> Optional[GPUInfo]:
        """Get Apple Silicon GPU information."""
        try:
            # On Apple Silicon, GPU memory is unified with system memory
            # We report a portion of system memory as "GPU memory"
            mem = psutil.virtual_memory()

            # Estimate GPU memory as ~75% of available for Metal
            # This is an approximation as macOS manages unified memory dynamically
            total_gb = mem.total / (1024 ** 3)

            # Determine chip name
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            chip_name = result.stdout.strip() if result.returncode == 0 else "Apple Silicon"

            return GPUInfo(
                name=chip_name,
                vendor="apple",
                total_memory_mb=total_gb * 1024 * 0.75,  # Estimate
                used_memory_mb=(mem.total - mem.available) / (1024 ** 2) * 0.5,  # Estimate
                temperature=None,  # Not easily available
                utilization_percent=None,
            )
        except Exception as e:
            logger.debug(f"Failed to get Apple GPU info: {e}")

        return None

    async def _collect_resources(self) -> SystemResources:
        """Collect current system resources."""
        # CPU info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else None

        # Memory info
        mem = psutil.virtual_memory()
        total_memory_mb = mem.total / (1024 ** 2)
        available_memory_mb = mem.available / (1024 ** 2)
        used_memory_mb = mem.used / (1024 ** 2)
        memory_percent = mem.percent

        # Swap info
        swap = psutil.swap_memory()

        # Disk info (use models directory or current directory)
        disk_path = Path.cwd()
        disk = shutil.disk_usage(disk_path)

        # GPU info
        gpu: Optional[GPUInfo] = None
        if self._gpu_vendor == "nvidia":
            gpu = await self._get_nvidia_gpu_info()
        elif self._gpu_vendor == "apple":
            gpu = await self._get_apple_gpu_info()

        if gpu and "thor" in gpu.name.lower():
            total_memory_mb = 128 * 1024
            available_memory_mb = max(total_memory_mb - used_memory_mb, 0.0)
            memory_percent = (used_memory_mb / total_memory_mb) * 100 if total_memory_mb else memory_percent
            if not self._thor_memory_override_logged:
                logger.info("[info]Applying Thor memory override: 128GB[/info]")
                self._thor_memory_override_logged = True

        return SystemResources(
            timestamp=datetime.now(),
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            cpu_freq_mhz=cpu_freq_mhz,
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            used_memory_mb=used_memory_mb,
            memory_percent=memory_percent,
            swap_total_mb=swap.total / (1024 ** 2),
            swap_used_mb=swap.used / (1024 ** 2),
            swap_percent=swap.percent,
            gpu=gpu,
            disk_total_gb=disk.total / (1024 ** 3),
            disk_used_gb=disk.used / (1024 ** 3),
            disk_free_gb=disk.free / (1024 ** 3),
            disk_percent=(disk.used / disk.total) * 100,
        )

    async def get_resources(self) -> SystemResources:
        """Get the current resource state."""
        if self._current_resources is None:
            self._current_resources = await self._collect_resources()
        return self._current_resources

    async def get_system_info(self) -> dict:
        """Get static system information for logging/display."""
        resources = await self.get_resources()

        gpu_info = None
        gpu_memory_total = None
        gpu_memory_used = None

        if resources.gpu:
            gpu_info = f"{resources.gpu.name} ({resources.gpu.vendor.upper()})"
            if resources.gpu.total_memory_mb > 0:
                gpu_memory_total = resources.gpu.total_memory_mb / 1024
                gpu_memory_used = (
                    resources.gpu.used_memory_mb / 1024
                    if resources.gpu.used_memory_mb is not None
                    else None
                )

        return {
            "platform": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "cpu_count": resources.cpu_count,
            "total_memory_gb": resources.total_memory_mb / 1024,
            "gpu_info": gpu_info,
            "gpu_vendor": self._gpu_vendor,
            "gpu_memory_total_gb": gpu_memory_total,
            "gpu_memory_used_gb": gpu_memory_used,
        }

    async def check_memory_available(self, required_mb: float) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_mb: Required memory in megabytes

        Returns:
            True if memory is available
        """
        resources = await self.get_resources()
        available = resources.available_memory_mb

        is_available = available >= required_mb

        logger.debug(
            f"Memory check: required={required_mb:.0f}MB, "
            f"available={available:.0f}MB, sufficient={is_available}"
        )

        return is_available

    async def get_process_memory(self, pid: int) -> Optional[float]:
        """
        Get memory usage for a specific process.

        Args:
            pid: Process ID

        Returns:
            Memory usage in MB, or None if process not found
        """
        try:
            process = psutil.Process(pid)
            mem_info = process.memory_info()
            return mem_info.rss / (1024 ** 2)
        except psutil.NoSuchProcess:
            return None

    def get_gpu_vendor(self) -> Optional[str]:
        """Get the detected GPU vendor."""
        return self._gpu_vendor

    def has_gpu(self) -> bool:
        """Check if a GPU is available."""
        return self._gpu_vendor is not None
