"""
Subprocess manager for inference server instances.

Manages:
- Starting/stopping llama-server, whisper-server, and transformers server processes
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
from cyber_inference.services.whisper_installer import WhisperInstaller

logger = get_logger(__name__)


@dataclass
class LlamaProcess:
    """Represents a running inference server process (llama, whisper, or transformers)."""
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

    # Server type: 'llama', 'whisper', or 'transformers'
    server_type: str = "llama"

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
        self._whisper_installer = WhisperInstaller(bin_dir=self.bin_dir)
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
            logger.info(f"[success]llama-server binary: {version} ({location})[/success]")
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
        """
        Find the mmproj file for a model by scanning the models directory.

        This is a fallback when mmproj_path is not stored in the database.
        Prefers the standardized naming: mmproj-{model_stem}.gguf
        """
        mmproj_files = sorted(
            [p for p in model_path.parent.glob("*.gguf") if "mmproj" in p.name.lower()]
        )
        if not mmproj_files:
            return None

        # Strategy 1: Exact match - mmproj-{model_stem}.gguf
        exact = model_path.parent / f"mmproj-{model_path.stem}.gguf"
        if exact.exists():
            return exact

        # Strategy 2: Match by base name (without quant suffix)
        # Remove quantization patterns: -Q4_K_M, .BF16, etc.
        patterns = [
            r"(?i)[._-]q\d+[_a-z0-9]*$",
            r"(?i)[._-](?:bf16|f16|f32|fp16|fp32)$",
        ]
        base_name = model_path.stem
        for pattern in patterns:
            base_name = re.sub(pattern, "", base_name)

        prefix = f"mmproj-{base_name}".lower()
        prefixed = [p for p in mmproj_files if p.name.lower().startswith(prefix)]
        if len(prefixed) == 1:
            return prefixed[0]
        if prefixed:
            return sorted(prefixed, key=lambda p: len(p.name))[0]

        # Strategy 3: Base name appears anywhere in mmproj filename
        base_lower = base_name.lower()
        contains = [p for p in mmproj_files if base_lower in p.name.lower()]
        if len(contains) == 1:
            return contains[0]
        if contains:
            return sorted(contains, key=lambda p: len(p.name))[0]

        logger.info(
            "[info]No mmproj match found for model %s[/info]",
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
        mmproj_path: Optional[Path] = None,
    ) -> LlamaProcess:
        """
        Start a new llama-server process for a model.

        Args:
            model_name: Name identifier for the model
            model_path: Path to the GGUF model file
            context_size: Context window size (default from settings)
            gpu_layers: Number of GPU layers (-1 for auto)
            threads: Number of CPU threads
            embedding: Enable embedding mode
            mmproj_path: Path to mmproj file for vision models (auto-detect if None)

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

        # Use provided mmproj_path or try to find one
        if mmproj_path is None:
            mmproj_path = self._find_mmproj(model_path)
        elif not mmproj_path.exists():
            logger.warning(f"[warning]Specified mmproj not found: {mmproj_path}[/warning]")
            mmproj_path = self._find_mmproj(model_path)

        cmd = [
            str(llama_server),
            "--model", str(model_path),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(n_gpu_layers),
        ]

        if mmproj_path and mmproj_path.exists():
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

    async def start_whisper_server(
        self,
        model_name: str,
        model_path: Path,
        gpu_layers: Optional[int] = None,
        threads: Optional[int] = None,
    ) -> LlamaProcess:
        """
        Start a new whisper-server process for a transcription model.

        Args:
            model_name: Name identifier for the model
            model_path: Path to the GGUF Whisper model file
            gpu_layers: Number of GPU layers (-1 for auto)
            threads: Number of CPU threads

        Returns:
            LlamaProcess instance (with server_type='whisper')
        """
        logger.info(f"[highlight]Starting whisper-server for model: {model_name}[/highlight]")

        if model_name in self._processes:
            existing = self._processes[model_name]
            if existing.status == "running":
                logger.warning(f"Model {model_name} already running on port {existing.port}")
                return existing

        settings = get_settings()

        # Check if whisper-server is installed
        if not self._whisper_installer.is_installed():
            logger.info("[info]whisper-server not found, attempting to install...[/info]")
            try:
                await self._whisper_installer.install()
            except Exception as e:
                logger.warning(f"[warning]Could not auto-install whisper-server: {e}[/warning]")
                logger.warning("[warning]Install manually: cyber-inference install-whisper[/warning]")
                raise RuntimeError(f"whisper-server not installed: {e}")

        # Allocate port
        port = self._find_available_port()
        logger.info(f"  Allocated port: {port}")

        # Build command
        whisper_server = self._whisper_installer.get_binary_path()
        n_threads = threads or settings.llama_threads

        cmd = [
            str(whisper_server),
            "-m", str(model_path),
            "--port", str(port),
            "--host", "127.0.0.1",
        ]

        # Add threads
        if n_threads:
            cmd.extend(["-t", str(n_threads)])

        # Enable flash attention for better performance
        cmd.append("-fa")

        logger.debug(f"  Command: {' '.join(cmd)}")

        # Create process entry
        whisper_proc = LlamaProcess(
            model_name=model_name,
            model_path=model_path,
            port=port,
            threads=n_threads,
            server_type="whisper",
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            whisper_proc.process = process
            whisper_proc.pid = process.pid
            whisper_proc.status = "starting"

            logger.info(f"  Process started with PID: {process.pid}")

            # Store in tracking dict
            self._processes[model_name] = whisper_proc

            # Start output monitoring
            asyncio.create_task(self._monitor_output(model_name, process))

            # Wait for whisper server to be ready
            await self._wait_for_whisper_ready(model_name, port)

            whisper_proc.status = "running"
            logger.info(f"[success]whisper-server ready for {model_name} on port {port}[/success]")

            return whisper_proc

        except Exception as e:
            logger.error(f"[error]Failed to start whisper-server: {e}[/error]")
            whisper_proc.status = "error"
            whisper_proc.error_message = str(e)
            self._release_port(port)
            raise

    async def _wait_for_whisper_ready(
        self,
        model_name: str,
        port: int,
        timeout: float = 60.0,
        check_interval: float = 1.0,
    ) -> None:
        """
        Wait for the whisper-server to be ready.

        Args:
            model_name: Model name for logging
            port: Server port
            timeout: Maximum wait time in seconds
            check_interval: Time between health checks
        """
        logger.info(f"  Waiting for whisper-server to be ready (timeout: {timeout}s)...")

        # whisper.cpp server typically responds on /inference or root
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient() as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time

                if elapsed > timeout:
                    raise TimeoutError(f"Whisper server failed to start within {timeout}s")

                try:
                    # Try root endpoint first (common for whisper.cpp server)
                    response = await client.get(
                        f"http://127.0.0.1:{port}/",
                        timeout=2.0,
                    )
                    # Any response (even 404) means server is up
                    logger.info(f"  Whisper server ready after {elapsed:.1f}s")
                    return
                except Exception:
                    pass

                # Check if process died
                proc = self._processes.get(model_name)
                if proc and proc.process and proc.process.returncode is not None:
                    raise RuntimeError(f"Whisper server exited with code {proc.process.returncode}")

                logger.debug(f"  Waiting for whisper server... ({elapsed:.1f}s)")
                await asyncio.sleep(check_interval)

    async def start_transformers_server(
        self,
        model_name: str,
        model_path: Path,
        embedding: bool = False,
    ) -> LlamaProcess:
        """
        Start a new transformers inference server process for a model.

        Uses HuggingFace transformers AutoModelForCausalLM + model.generate()
        via a lightweight subprocess server suitable for edge/SoC hardware.

        Args:
            model_name: Name identifier for the model
            model_path: Path to the HuggingFace model directory
            embedding: Enable embedding mode

        Returns:
            LlamaProcess instance (with server_type='transformers')
        """
        logger.info(f"[highlight]Starting transformers server for model: {model_name}[/highlight]")

        if model_name in self._processes:
            existing = self._processes[model_name]
            if existing.status == "running":
                logger.warning(f"Model {model_name} already running on port {existing.port}")
                return existing

        # Allocate port
        port = self._find_available_port()
        logger.info(f"  Allocated port: {port}")

        # Build command
        import sys
        python_exe = sys.executable

        cmd = [
            python_exe, "-m", "cyber_inference.services.transformers_server",
            "--model-path", str(model_path),
            "--port", str(port),
            "--host", "127.0.0.1",
        ]

        logger.debug(f"  Command: {' '.join(cmd)}")

        # Create process entry
        tf_proc = LlamaProcess(
            model_name=model_name,
            model_path=model_path,
            port=port,
            server_type="transformers",
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            tf_proc.process = process
            tf_proc.pid = process.pid
            tf_proc.status = "starting"

            logger.info(f"  Process started with PID: {process.pid}")

            # Store in tracking dict
            self._processes[model_name] = tf_proc

            # Start output monitoring
            asyncio.create_task(self._monitor_output(model_name, process))

            # Wait for server to be ready (uses same /health + /v1/models check)
            await self._wait_for_server_ready(model_name, port, timeout=300.0, server_label="Transformers")

            tf_proc.status = "running"
            logger.info(
                f"[success]Transformers server ready for {model_name} on port {port}[/success]"
            )

            return tf_proc

        except Exception as e:
            logger.error(f"[error]Failed to start transformers server: {e}[/error]")
            tf_proc.status = "error"
            tf_proc.error_message = str(e)
            self._release_port(port)
            raise

    async def _wait_for_server_ready(
        self,
        model_name: str,
        port: int,
        timeout: float = 600.0,
        check_interval: float = 2.0,
        server_label: str = "Server",
    ) -> None:
        """
        Wait for a subprocess server (transformers) to be ready.

        Polls /health and /v1/models until both succeed or timeout.

        Args:
            model_name: Model name for logging
            port: Server port
            timeout: Maximum wait time in seconds
            check_interval: Time between health checks
            server_label: Label for log messages (e.g. "Transformers")
        """
        logger.info(f"  Waiting for {server_label} server to be ready (timeout: {timeout}s)...")

        health_url = f"http://127.0.0.1:{port}/health"
        models_url = f"http://127.0.0.1:{port}/v1/models"
        start_time = asyncio.get_event_loop().time()

        health_ok = False
        models_ok = False

        async with httpx.AsyncClient() as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time

                if elapsed > timeout:
                    raise TimeoutError(
                        f"{server_label} server failed to start within {timeout}s"
                    )

                try:
                    # Check health endpoint
                    if not health_ok:
                        response = await client.get(health_url, timeout=3.0)
                        if response.status_code == 200:
                            health_ok = True
                            logger.debug(f"  {server_label} health check passed after {elapsed:.1f}s")

                    # Then check /v1/models to confirm model is loaded
                    if health_ok and not models_ok:
                        response = await client.get(models_url, timeout=3.0)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("data") and len(data["data"]) > 0:
                                models_ok = True
                                logger.debug(
                                    f"  {server_label} model loaded after {elapsed:.1f}s"
                                )

                    # Both checks passed
                    if health_ok and models_ok:
                        await asyncio.sleep(0.5)
                        logger.info(
                            f"  {server_label} server fully ready after {elapsed:.1f}s"
                        )
                        return

                except Exception as e:
                    logger.debug(f"  {server_label} readiness check failed: {e}")

                # Check if process died
                proc = self._processes.get(model_name)
                if proc and proc.process and proc.process.returncode is not None:
                    raise RuntimeError(
                        f"{server_label} server exited with code {proc.process.returncode}"
                    )

                logger.debug(f"  Waiting for {server_label} server... ({elapsed:.1f}s)")
                await asyncio.sleep(check_interval)

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
                    # Log errors/tracebacks at INFO level so they're visible
                    lower = decoded.lower()
                    if any(kw in lower for kw in ("error", "traceback", "exception", "failed")):
                        logger.info(f"[{model_name}] {decoded}")
                    else:
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
        logger.info("[warning]Shutting down all inference server instances...[/warning]")

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

    async def update_request_stats(
        self,
        model_name: str,
        tokens: int = 0,
        increment_count: bool = True,
    ) -> None:
        """Update request activity/statistics for a model."""
        if model_name in self._processes:
            proc = self._processes[model_name]
            proc.last_request_at = datetime.now()
            if increment_count:
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
