"""
Subprocess manager for llama.cpp server instances.

Manages:
- Starting/stopping llama-server processes
- Dynamic port allocation
- Health checking
- Resource tracking per process
- Automatic cleanup on shutdown
"""

import asyncio
import re
import signal
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import psutil

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger
from cyber_inference.services.llama_installer import LlamaInstaller

logger = get_logger(__name__)


@dataclass
class LlamaProcess:
    """Represents a running llama-server process."""
    model_name: str
    model_path: Path
    port: int
    mmproj_path: Optional[Path] = None
    pid: Optional[int] = None
    process: Optional[asyncio.subprocess.Process] = None
    status: str = "starting"  # starting, running, stopping, stopped, error
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    last_request_at: Optional[datetime] = None
    request_count: int = 0
    context_size: int = 4096
    gpu_layers: int = -1
    threads: Optional[int] = None

    # Resource tracking
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0


class ProcessManager:
    """
    Manages llama-server subprocess lifecycle.

    Handles:
    - Process spawning with proper arguments
    - Health monitoring
    - Graceful shutdown
    - Port management
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        bin_dir: Optional[Path] = None,
        base_port: int = 8338,
    ):
        """
        Initialize the process manager.

        Args:
            models_dir: Directory containing GGUF models
            bin_dir: Directory containing llama-server binary
            base_port: Starting port for llama servers
        """
        settings = get_settings()
        self.models_dir = models_dir or settings.models_dir
        self.bin_dir = bin_dir or settings.bin_dir
        self.base_port = base_port or settings.llama_server_base_port

        self._processes: dict[str, LlamaProcess] = {}
        self._port_allocations: set[int] = set()
        self._installer = LlamaInstaller(bin_dir=self.bin_dir)
        self._initialized = False
        self._shutdown_event = asyncio.Event()

        logger.info("[info]ProcessManager initialized[/info]")
        logger.debug(f"  Models dir: {self.models_dir}")
        logger.debug(f"  Binary dir: {self.bin_dir}")
        logger.debug(f"  Base port: {self.base_port}")

    async def initialize(self) -> None:
        """
        Initialize the process manager.

        Ensures llama-server is installed and ready.
        """
        logger.info("[info]Initializing ProcessManager...[/info]")

        # Check/install llama-server
        if not self._installer.is_installed():
            logger.info("[info]llama-server not found, attempting to install...[/info]")
            try:
                await self._installer.install()
            except Exception as e:
                logger.warning(f"[warning]Could not auto-install llama-server: {e}[/warning]")
                logger.warning("[warning]You can manually install it using: cyber-inference install-llama[/warning]")
        else:
            # Determine which location was used
            binary_path = self._installer.get_binary_path()
            # Check if it's from system PATH (not in bin_dir)
            try:
                # Resolve to absolute paths for comparison
                binary_abs = binary_path.resolve()
                bin_dir_abs = self.bin_dir.resolve()
                if not str(binary_abs).startswith(str(bin_dir_abs)):
                    location = "system PATH"
                else:
                    location = f"bin_dir ({self.bin_dir})"
            except Exception:
                # Fallback if path resolution fails
                if str(binary_path).startswith(str(self.bin_dir)):
                    location = f"bin_dir ({self.bin_dir})"
                else:
                    location = "system PATH"

            version = await self._installer.get_installed_version()
            logger.info(f"[success]llama-server found in {location}: {version}[/success]")
            logger.debug(f"  Binary path: {binary_path}")

        self._initialized = True
        logger.info("[success]ProcessManager initialized successfully[/success]")

    def _find_available_port(self) -> int:
        """
        Find an available port for a new server.

        Returns:
            Available port number
        """
        port = self.base_port

        while port < self.base_port + 100:
            if port not in self._port_allocations:
                # Also check if port is actually available
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("127.0.0.1", port))
                        self._port_allocations.add(port)
                        logger.debug(f"Allocated port: {port}")
                        return port
                except OSError:
                    pass
            port += 1

        raise RuntimeError("No available ports for llama-server")

    def _release_port(self, port: int) -> None:
        """Release a port allocation."""
        if port in self._port_allocations:
            self._port_allocations.discard(port)
            logger.debug(f"Released port: {port}")

    def _find_mmproj(self, model_path: Path) -> Optional[Path]:
        mmproj_files = sorted(
            [p for p in model_path.parent.glob("*.gguf") if "mmproj" in p.name.lower()]
        )
        if not mmproj_files:
            return None

        exact = model_path.parent / f"mmproj-{model_path.stem}.gguf"
        if exact.exists():
            return exact

        base_name = re.sub(r"(?i)-q\d+.*$", "", model_path.stem)
        prefix = f"mmproj-{base_name}".lower()

        prefixed = [p for p in mmproj_files if p.name.lower().startswith(prefix)]
        if len(prefixed) == 1:
            return prefixed[0]
        if prefixed:
            return sorted(prefixed, key=lambda p: len(p.name))[0]

        if len(mmproj_files) == 1:
            return mmproj_files[0]

        contains = [p for p in mmproj_files if base_name.lower() in p.name.lower()]
        if len(contains) == 1:
            return contains[0]
        if contains:
            return sorted(contains, key=lambda p: len(p.name))[0]

        logger.warning(
            "[warning]Multiple mmproj files found but none matched model %s[/warning]",
            model_path.stem,
        )
        return None

    async def start_server(
        self,
        model_name: str,
        model_path: Path,
        context_size: Optional[int] = None,
        gpu_layers: Optional[int] = None,
        threads: Optional[int] = None,
        embedding: bool = False,
    ) -> LlamaProcess:
        """
        Start a new llama-server process for a model.

        Args:
            model_name: Name identifier for the model
            model_path: Path to the GGUF model file
            context_size: Context window size (default from settings)
            gpu_layers: Number of GPU layers (-1 for auto)
            threads: Number of CPU threads

        Returns:
            LlamaProcess instance
        """
        logger.info(f"[highlight]Starting llama-server for model: {model_name}[/highlight]")

        if model_name in self._processes:
            existing = self._processes[model_name]
            if existing.status == "running":
                logger.warning(f"Model {model_name} already running on port {existing.port}")
                return existing

        settings = get_settings()

        # Allocate port
        port = self._find_available_port()
        logger.info(f"  Allocated port: {port}")

        # Build command
        llama_server = self._installer.get_binary_path()

        ctx_size = context_size or settings.default_context_size
        n_gpu_layers = gpu_layers if gpu_layers is not None else settings.llama_gpu_layers
        n_threads = threads or settings.llama_threads
        mmproj_path = self._find_mmproj(model_path)

        cmd = [
            str(llama_server),
            "--model", str(model_path),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(n_gpu_layers),
        ]

        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])
            logger.info(f"  Using mmproj: {mmproj_path.name}")

        if n_threads:
            cmd.extend(["--threads", str(n_threads)])

        # Enable embedding endpoint for embedding models
        if embedding:
            cmd.append("--embedding")
            logger.info("  Embedding mode enabled")

        logger.debug(f"  Command: {' '.join(cmd)}")

        # Create process entry
        llama_proc = LlamaProcess(
            model_name=model_name,
            model_path=model_path,
            mmproj_path=mmproj_path,
            port=port,
            context_size=ctx_size,
            gpu_layers=n_gpu_layers,
            threads=n_threads,
        )

        try:
            # Start the process
            # Redirect stderr to stdout so all output goes through one pipe
            # This prevents buffer deadlock when stderr fills up
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            )

            llama_proc.process = process
            llama_proc.pid = process.pid
            llama_proc.status = "starting"

            logger.info(f"  Process started with PID: {process.pid}")

            # Store in tracking dict
            self._processes[model_name] = llama_proc

            # Start output monitoring (reads both stdout and stderr since they're merged)
            asyncio.create_task(self._monitor_output(model_name, process))

            # Wait for server to be ready
            await self._wait_for_ready(model_name, port)

            llama_proc.status = "running"
            logger.info(f"[success]llama-server ready for {model_name} on port {port}[/success]")

            return llama_proc

        except Exception as e:
            logger.error(f"[error]Failed to start llama-server: {e}[/error]")
            llama_proc.status = "error"
            llama_proc.error_message = str(e)
            self._release_port(port)
            raise

    async def _monitor_output(
        self,
        model_name: str,
        process: asyncio.subprocess.Process,
    ) -> None:
        """Monitor and log process output."""
        try:
            async for line in process.stdout:
                decoded = line.decode().strip()
                if decoded:
                    logger.debug(f"[{model_name}] {decoded}")
        except Exception as e:
            logger.debug(f"Output monitoring ended for {model_name}: {e}")

    async def _wait_for_ready(
        self,
        model_name: str,
        port: int,
        timeout: float = 120.0,
        check_interval: float = 1.0,
    ) -> None:
        """
        Wait for the llama-server to be ready.

        Args:
            model_name: Model name for logging
            port: Server port
            timeout: Maximum wait time in seconds
            check_interval: Time between health checks
        """
        logger.info(f"  Waiting for server to be ready (timeout: {timeout}s)...")

        health_url = f"http://127.0.0.1:{port}/health"
        slots_url = f"http://127.0.0.1:{port}/slots"
        start_time = asyncio.get_event_loop().time()

        health_ok = False
        slots_ok = False

        async with httpx.AsyncClient() as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time

                if elapsed > timeout:
                    raise TimeoutError(f"Server failed to start within {timeout}s")

                try:
                    # First check health endpoint
                    if not health_ok:
                        response = await client.get(health_url, timeout=2.0)
                        if response.status_code == 200:
                            health_ok = True
                            logger.debug(f"  Health check passed after {elapsed:.1f}s")

                    # Then check slots endpoint to ensure model is actually loaded
                    if health_ok and not slots_ok:
                        response = await client.get(slots_url, timeout=2.0)
                        if response.status_code == 200:
                            slots_data = response.json()
                            # Check if at least one slot is ready
                            if isinstance(slots_data, list) and len(slots_data) > 0:
                                slots_ok = True
                                logger.debug(f"  Slots ready after {elapsed:.1f}s")

                    # Both checks passed - server is ready
                    if health_ok and slots_ok:
                        # Give a small buffer for the server to fully stabilize
                        await asyncio.sleep(0.5)
                        logger.info(f"  Server fully ready after {elapsed:.1f}s")
                        return

                except Exception as e:
                    logger.debug(f"  Readiness check failed: {e}")

                # Check if process died
                proc = self._processes.get(model_name)
                if proc and proc.process and proc.process.returncode is not None:
                    raise RuntimeError(f"Server process exited with code {proc.process.returncode}")

                logger.debug(f"  Waiting for server... ({elapsed:.1f}s)")
                await asyncio.sleep(check_interval)

    async def stop_server(self, model_name: str, timeout: float = 10.0) -> None:
        """
        Stop a running llama-server.

        Args:
            model_name: Name of the model to stop
            timeout: Seconds to wait for graceful shutdown
        """
        if model_name not in self._processes:
            logger.warning(f"No process found for model: {model_name}")
            return

        llama_proc = self._processes[model_name]

        if llama_proc.status in ("stopped", "error"):
            logger.debug(f"Model {model_name} already stopped")
            return

        logger.info(f"[info]Stopping llama-server for {model_name}...[/info]")
        llama_proc.status = "stopping"

        if llama_proc.process:
            try:
                # Try graceful termination first
                llama_proc.process.terminate()

                try:
                    await asyncio.wait_for(
                        llama_proc.process.wait(),
                        timeout=timeout,
                    )
                    logger.info(f"[success]Server stopped gracefully: {model_name}[/success]")
                except asyncio.TimeoutError:
                    # Force kill
                    logger.warning(f"[warning]Force killing server: {model_name}[/warning]")
                    llama_proc.process.kill()
                    await llama_proc.process.wait()

            except Exception as e:
                logger.error(f"Error stopping server: {e}")

        # Cleanup
        self._release_port(llama_proc.port)
        llama_proc.status = "stopped"
        del self._processes[model_name]

        logger.info(f"[success]Server stopped and cleaned up: {model_name}[/success]")

    async def restart_server(self, model_name: str) -> LlamaProcess:
        """Restart a running server."""
        if model_name not in self._processes:
            raise ValueError(f"No process found for model: {model_name}")

        proc = self._processes[model_name]
        model_path = proc.model_path
        context_size = proc.context_size
        gpu_layers = proc.gpu_layers
        threads = proc.threads

        await self.stop_server(model_name)
        return await self.start_server(
            model_name,
            model_path,
            context_size=context_size,
            gpu_layers=gpu_layers,
            threads=threads,
        )

    async def shutdown(self) -> None:
        """Shutdown all running servers."""
        logger.info("[warning]Shutting down all llama-server instances...[/warning]")

        self._shutdown_event.set()

        # Stop all processes
        for model_name in list(self._processes.keys()):
            try:
                await self.stop_server(model_name)
            except Exception as e:
                logger.error(f"Error stopping {model_name}: {e}")

        logger.info("[success]All servers shut down[/success]")

    def get_process(self, model_name: str) -> Optional[LlamaProcess]:
        """Get a process by model name."""
        return self._processes.get(model_name)

    def get_running_models(self) -> list[str]:
        """Get list of currently running model names."""
        return [
            name for name, proc in self._processes.items()
            if proc.status == "running"
        ]

    def get_all_processes(self) -> list[LlamaProcess]:
        """Get all tracked processes."""
        return list(self._processes.values())

    async def get_server_url(self, model_name: str) -> Optional[str]:
        """Get the URL for a running model server."""
        proc = self._processes.get(model_name)
        if proc and proc.status == "running":
            return f"http://127.0.0.1:{proc.port}"
        return None

    async def update_request_stats(self, model_name: str, tokens: int = 0) -> None:
        """Update request statistics for a model."""
        if model_name in self._processes:
            proc = self._processes[model_name]
            proc.last_request_at = datetime.now()
            proc.request_count += 1

            # Update memory usage
            if proc.pid:
                try:
                    ps = psutil.Process(proc.pid)
                    proc.memory_mb = ps.memory_info().rss / (1024 ** 2)
                except psutil.NoSuchProcess:
                    pass

    async def check_health(self, model_name: str) -> bool:
        """Check if a model server is healthy."""
        proc = self._processes.get(model_name)
        if not proc or proc.status != "running":
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{proc.port}/health",
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception:
            return False
