"""
Automatic model loading and unloading service.

Handles:
- On-demand model loading when API requests come in
- Idle model unloading after configurable timeout
- Memory-based unloading when resources are constrained
- Model prioritization based on usage patterns
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger
from cyber_inference.services.model_manager import ModelManager
from cyber_inference.services.process_manager import ProcessManager, LlamaProcess
from cyber_inference.services.resource_monitor import ResourceMonitor

logger = get_logger(__name__)


class AutoLoader:
    """
    Manages automatic model loading and unloading.

    Features:
    - Lazy loading: Models load when first requested
    - Idle unloading: Models unload after idle timeout
    - Resource management: Unload models when memory is low
    - Queue management: Only load up to max_loaded_models at once
    """

    def __init__(
        self,
        process_manager: Optional[ProcessManager] = None,
        model_manager: Optional[ModelManager] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
    ):
        """
        Initialize the auto-loader.

        Args:
            process_manager: Process manager instance
            model_manager: Model manager instance
            resource_monitor: Resource monitor instance
        """
        settings = get_settings()

        self._process_manager = process_manager
        self._model_manager = model_manager
        self._resource_monitor = resource_monitor

        self._idle_timeout = settings.model_idle_timeout
        self._max_loaded = settings.max_loaded_models
        self._max_memory_percent = settings.max_memory_percent

        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._locks: dict[str, asyncio.Lock] = {}

        logger.info("[info]AutoLoader initialized[/info]")
        logger.debug(f"  Idle timeout: {self._idle_timeout}s")
        logger.debug(f"  Max loaded models: {self._max_loaded}")
        logger.debug(f"  Max memory percent: {self._max_memory_percent}%")

    def _get_process_manager(self) -> ProcessManager:
        """Get or create process manager."""
        if self._process_manager is None:
            from cyber_inference.main import get_process_manager
            self._process_manager = get_process_manager()
        return self._process_manager

    def _get_model_manager(self) -> ModelManager:
        """Get or create model manager."""
        if self._model_manager is None:
            self._model_manager = ModelManager()
        return self._model_manager

    def _get_resource_monitor(self) -> ResourceMonitor:
        """Get or create resource monitor."""
        if self._resource_monitor is None:
            from cyber_inference.main import get_resource_monitor
            self._resource_monitor = get_resource_monitor()
        return self._resource_monitor

    def _get_lock(self, model_name: str) -> asyncio.Lock:
        """Get or create a lock for a model."""
        if model_name not in self._locks:
            self._locks[model_name] = asyncio.Lock()
        return self._locks[model_name]

    async def start(self) -> None:
        """Start the auto-loader background tasks."""
        if self._running:
            return

        logger.info("[info]Starting AutoLoader background tasks[/info]")
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("[success]AutoLoader started[/success]")

    async def stop(self) -> None:
        """Stop the auto-loader."""
        if not self._running:
            return

        logger.info("[info]Stopping AutoLoader[/info]")
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("[success]AutoLoader stopped[/success]")

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up idle models."""
        logger.debug("Cleanup loop started")

        while self._running:
            try:
                await self._check_idle_models()
                await self._check_memory_pressure()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _check_idle_models(self) -> None:
        """Check for and unload idle models."""
        pm = self._get_process_manager()
        now = datetime.now()
        idle_threshold = timedelta(seconds=self._idle_timeout)

        for proc in pm.get_all_processes():
            if proc.status != "running":
                continue

            last_request = proc.last_request_at or proc.started_at
            idle_time = now - last_request

            if idle_time > idle_threshold:
                logger.info(
                    f"[warning]Unloading idle model: {proc.model_name} "
                    f"(idle for {idle_time.total_seconds():.0f}s)[/warning]"
                )
                await self.unload_model(proc.model_name)

    async def _check_memory_pressure(self) -> None:
        """Check memory and unload models if necessary."""
        rm = self._get_resource_monitor()
        resources = await rm.get_resources()

        if resources.memory_percent > self._max_memory_percent:
            logger.warning(
                f"[warning]Memory pressure: {resources.memory_percent:.1f}% "
                f"(threshold: {self._max_memory_percent}%)[/warning]"
            )

            # Unload least recently used model
            pm = self._get_process_manager()
            processes = pm.get_all_processes()

            if processes:
                # Skip models that are still starting up (e.g. SGLang loading weights)
                candidates = [p for p in processes if p.status == "running"]
                if not candidates:
                    logger.debug("No running models to unload (all still starting)")
                    return

                # Sort by last request time (oldest first)
                candidates.sort(
                    key=lambda p: p.last_request_at or p.started_at
                )

                oldest = candidates[0]
                logger.info(f"[warning]Unloading LRU model: {oldest.model_name}[/warning]")
                await self.unload_model(oldest.model_name)

    async def ensure_model_loaded(self, model_name: str) -> str:
        """
        Ensure a model is loaded and return its server URL.

        If the model is not loaded, it will be loaded automatically.
        If max models are loaded, the least recently used will be unloaded.

        Args:
            model_name: Name of the model to load

        Returns:
            URL of the model's server (e.g., "http://127.0.0.1:8338")
        """
        lock = self._get_lock(model_name)

        async with lock:
            logger.info(f"[info]Ensuring model is loaded: {model_name}[/info]")

            pm = self._get_process_manager()

            # Check if already loaded
            url = await pm.get_server_url(model_name)
            if url:
                logger.debug(f"Model already loaded: {model_name} at {url}")
                return url

            # Check if we need to unload something first
            running = pm.get_running_models()
            if len(running) >= self._max_loaded:
                logger.info(
                    f"[warning]Max models loaded ({self._max_loaded}), "
                    f"unloading oldest[/warning]"
                )

                # Find oldest model
                oldest_name = None
                oldest_time = datetime.now()

                for name in running:
                    proc = pm.get_process(name)
                    if proc:
                        last_used = proc.last_request_at or proc.started_at
                        if last_used < oldest_time:
                            oldest_time = last_used
                            oldest_name = name

                if oldest_name:
                    await self.unload_model(oldest_name)

            # Load the model
            return await self.load_model(model_name)

    async def load_model(self, model_name: str) -> str:
        """
        Load a model and return its server URL.

        Routes to the appropriate engine (llama.cpp, whisper.cpp, or SGLang)
        based on the model's engine_type.

        Args:
            model_name: Name of the model to load

        Returns:
            URL of the model's server
        """
        logger.info(f"[highlight]Loading model: {model_name}[/highlight]")

        mm = self._get_model_manager()
        pm = self._get_process_manager()

        # Get model info (includes path, type, engine_type, and mmproj_path)
        model_info = await mm.get_model(model_name)
        if not model_info:
            raise ValueError(f"Model not found: {model_name}")

        model_path = await mm.get_model_path(model_name)
        if not model_path:
            raise ValueError(f"Model path not found: {model_name}")

        logger.debug(f"Model path: {model_path}")

        # Check engine type
        engine_type = model_info.get("engine_type", "llama")

        # Get mmproj_path if this is a multimodal model
        mmproj_path = None
        if model_info.get("mmproj_path"):
            mmproj_path = Path(model_info["mmproj_path"])
            logger.debug(f"mmproj path from DB: {mmproj_path}")

        # Check model type
        model_type = model_info.get("model_type")
        is_embedding = model_type == "embedding"
        is_transcription = model_type == "transcription"

        # Auto-detect model types by name AND repo ID if type not set
        if not is_embedding and not is_transcription:
            name_lower = model_name.lower()
            repo_id = model_info.get("hf_repo_id") or ""
            repo_lower = repo_id.lower()
            check_string = f"{name_lower} {repo_lower}"

            embedding_patterns = ["embed", "bge", "e5-", "gte-", "stella", "nomic"]
            transcription_patterns = ["whisper", "distil-whisper", "faster-whisper"]

            is_embedding = any(pattern in check_string for pattern in embedding_patterns)
            is_transcription = any(pattern in check_string for pattern in transcription_patterns)

        if is_embedding:
            logger.info("  Model type: embedding")
        elif is_transcription:
            logger.info("  Model type: transcription (whisper)")

        logger.info(f"  Engine type: {engine_type}")

        # Start the appropriate server based on engine_type
        if engine_type == "sglang":
            # Use SGLang server
            proc = await pm.start_sglang_server(
                model_name,
                model_path,
                embedding=is_embedding,
            )
        elif engine_type == "transformers":
            # Use lightweight transformers server
            proc = await pm.start_transformers_server(
                model_name,
                model_path,
                embedding=is_embedding,
            )
        elif is_transcription:
            # Use whisper-server for transcription models
            proc = await pm.start_whisper_server(model_name, model_path)
        else:
            # Use llama-server for chat/embedding models
            proc = await pm.start_server(
                model_name,
                model_path,
                embedding=is_embedding,
                mmproj_path=mmproj_path,
            )

        if proc.status != "running":
            raise RuntimeError(f"Failed to start server: {proc.error_message}")

        # Update last used
        await mm.update_last_used(model_name)

        url = f"http://127.0.0.1:{proc.port}"
        logger.info(f"[success]Model loaded: {model_name} at {url} [{engine_type}][/success]")

        return url

    async def unload_model(self, model_name: str) -> None:
        """
        Unload a model.

        Args:
            model_name: Name of the model to unload
        """
        logger.info(f"[warning]Unloading model: {model_name}[/warning]")

        pm = self._get_process_manager()
        await pm.stop_server(model_name)

        logger.info(f"[success]Model unloaded: {model_name}[/success]")

    async def record_request(self, model_name: str) -> None:
        """Record that a request was made to a model."""
        pm = self._get_process_manager()
        await pm.update_request_stats(model_name)

    async def list_available_models(self) -> list[dict]:
        """List all available models (downloaded and enabled)."""
        mm = self._get_model_manager()
        models = await mm.list_models()

        return [
            m for m in models
            if m["is_downloaded"] and m["is_enabled"]
        ]

    async def get_model_info(self, model_name: str) -> Optional[dict]:
        """Get information about a specific model."""
        mm = self._get_model_manager()
        return await mm.get_model(model_name)

    async def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded models."""
        pm = self._get_process_manager()
        return pm.get_running_models()

    async def get_model_status(self, model_name: str) -> dict:
        """Get detailed status of a model."""
        pm = self._get_process_manager()
        mm = self._get_model_manager()

        model = await mm.get_model(model_name)
        if not model:
            return {"status": "not_found"}

        proc = pm.get_process(model_name)

        return {
            "name": model_name,
            "is_downloaded": model["is_downloaded"],
            "is_enabled": model["is_enabled"],
            "is_loaded": proc is not None and proc.status == "running",
            "port": proc.port if proc else None,
            "status": proc.status if proc else "not_loaded",
            "memory_mb": proc.memory_mb if proc else 0,
            "request_count": proc.request_count if proc else 0,
            "last_request_at": proc.last_request_at.isoformat() if proc and proc.last_request_at else None,
        }

