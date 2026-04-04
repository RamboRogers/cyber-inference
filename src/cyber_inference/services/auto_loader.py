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
from typing import Any

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger
from cyber_inference.services.model_manager import ModelManager
from cyber_inference.services.process_manager import LlamaProcess, ProcessManager
from cyber_inference.services.resource_monitor import ResourceMonitor

logger = get_logger(__name__)


REQUEST_DEFAULT_SUPPORT: dict[str, set[str]] = {
    "llama": {"temperature", "top_p", "top_k", "max_tokens", "repeat_penalty"},
    "transformers": {"temperature", "top_p", "max_tokens"},
    "whisper": set(),
}

GLOBAL_RUNTIME_REFRESH_KEYS = {
    "default_context_size",
    "max_context_size",
    "model_idle_unload_enabled",
    "model_idle_timeout",
    "max_loaded_models",
    "max_memory_percent",
    "llama_gpu_layers",
}

GLOBAL_RELOAD_KEYS = {"default_context_size", "llama_gpu_layers"}


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
        process_manager: ProcessManager | None = None,
        model_manager: ModelManager | None = None,
        resource_monitor: ResourceMonitor | None = None,
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
        self._idle_unload_enabled = settings.model_idle_unload_enabled

        self._running = False
        self._cleanup_task: asyncio.Task | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._model_events: dict[str, dict[str, Any]] = {}

        logger.info("[info]AutoLoader initialized[/info]")
        logger.debug(f"  Idle timeout configured: {self._idle_timeout}s")
        logger.debug(f"  Idle unload enabled: {self._idle_unload_enabled}")
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
                if self._idle_unload_enabled:
                    await self._check_idle_models()
                await self._check_memory_pressure()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _check_idle_models(self) -> None:
        """Check for and unload idle models."""
        if not self._idle_unload_enabled:
            return

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
                await self.unload_model(proc.model_name, reason="idle_timeout")

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
                # Skip models that are still starting up (e.g. loading large weights)
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
                await self.unload_model(oldest.model_name, reason="memory_pressure")

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
                    await self.unload_model(
                        oldest_name,
                        reason=f"capacity_eviction:{model_name}",
                    )

            # Load the model
            return await self.load_model(model_name, reason="on_demand_load")

    def _record_event(
        self,
        model_name: str,
        event: str,
        reason: str | None = None,
        **details: Any,
    ) -> None:
        """Record the latest runtime event for a model."""
        self._model_events[model_name] = {
            "event": event,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            **details,
        }

    def refresh_runtime_settings(self) -> dict[str, Any]:
        """Refresh runtime policy values from cached settings."""
        settings = get_settings()
        self._idle_timeout = settings.model_idle_timeout
        self._max_loaded = settings.max_loaded_models
        self._max_memory_percent = settings.max_memory_percent
        self._idle_unload_enabled = settings.model_idle_unload_enabled
        return {
            "idle_timeout": self._idle_timeout,
            "idle_unload_enabled": self._idle_unload_enabled,
            "max_loaded_models": self._max_loaded,
            "max_memory_percent": self._max_memory_percent,
        }

    def _get_saved_generation_defaults(self, model_info: dict[str, Any]) -> dict[str, Any]:
        """Return saved generation defaults in request-field shape."""
        return {
            "temperature": model_info.get("default_temperature"),
            "top_p": model_info.get("default_top_p"),
            "top_k": model_info.get("default_top_k"),
            "max_tokens": model_info.get("default_max_tokens"),
            "repeat_penalty": model_info.get("default_repeat_penalty"),
        }

    def _build_effective_runtime_config(
        self,
        model_info: dict[str, Any],
        proc: LlamaProcess | None,
    ) -> dict[str, Any]:
        """Build an operator-facing view of the effective runtime config."""
        settings = get_settings()
        server_type = proc.server_type if proc else model_info.get("engine_type", "llama")
        supported_fields = REQUEST_DEFAULT_SUPPORT.get(server_type, set())
        saved_defaults = self._get_saved_generation_defaults(model_info)
        effective_request_defaults = {
            key: value
            for key, value in saved_defaults.items()
            if key in supported_fields and value is not None
        }
        unsupported_saved_defaults = [
            key for key, value in saved_defaults.items()
            if key not in supported_fields and value is not None
        ]
        configured_context_size = model_info.get("default_context_size")
        native_context_size = model_info.get("context_length")
        if proc:
            launch_context_size = proc.context_size
            launch_config = proc.effective_config.get("launch_config", {})
            if isinstance(launch_config, dict):
                context_source = str(launch_config.get("context_source", "running"))
            else:
                context_source = "running"
        elif configured_context_size:
            launch_context_size = configured_context_size
            context_source = "configured_default"
        elif native_context_size:
            launch_context_size = native_context_size
            context_source = "model_native_max"
        else:
            launch_context_size = settings.default_context_size
            context_source = "global_default"
        launch_config = {
            "context_size": launch_context_size,
            "configured_context_size": configured_context_size,
            "native_context_size": native_context_size,
            "context_source": context_source,
            "gpu_layers": proc.gpu_layers if proc else (settings.llama_gpu_layers if server_type == "llama" else None),
        }
        return {
            "server_type": server_type,
            "launch_config": launch_config,
            "request_defaults": effective_request_defaults,
            "unsupported_saved_defaults": unsupported_saved_defaults,
            "supports_request_defaults": sorted(supported_fields),
        }

    async def get_request_defaults(
        self,
        model_name: str,
        server_type: str | None = None,
    ) -> dict[str, Any]:
        """Get supported saved request defaults for a model/server type."""
        model_info = await self.get_model_info(model_name)
        if not model_info:
            return {}

        resolved_server_type = server_type or model_info.get("engine_type", "llama")
        saved_defaults = self._get_saved_generation_defaults(model_info)
        supported_fields = REQUEST_DEFAULT_SUPPORT.get(resolved_server_type, set())
        return {
            key: value
            for key, value in saved_defaults.items()
            if key in supported_fields and value is not None
        }

    async def load_model(
        self,
        model_name: str,
        reason: str = "manual_load",
        reload_count: int = 0,
        context_size_override: int | None = None,
        gpu_layers_override: int | None = None,
    ) -> str:
        """
        Load a model and return its server URL.

        Routes to the appropriate engine (llama.cpp, whisper.cpp, or transformers)
        based on the model's engine_type.

        Args:
            model_name: Name of the model to load

        Returns:
            URL of the model's server
        """
        logger.info(f"[highlight]Loading model: {model_name}[/highlight]")
        self._record_event(model_name, "loading", reason=reason)

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

        # Determine context size: per-model override > model native > global default
        context_size: int | None
        if context_size_override is not None:
            context_size = context_size_override
        else:
            configured_context = model_info.get("default_context_size")
            native_context = model_info.get("context_length")
            context_size = (
                int(configured_context)
                if configured_context is not None
                else int(native_context)
                if native_context is not None
                else None  # let start_server() fall back to global default
            )
        if context_size:
            logger.info(f"  Context size: {context_size}")

        effective_config = self._build_effective_runtime_config(model_info, proc=None)
        effective_config["launch_config"]["context_size"] = context_size or effective_config["launch_config"]["context_size"]
        effective_config["launch_config"]["context_source"] = (
            "load_override"
            if context_size_override is not None
            else effective_config["launch_config"]["context_source"]
        )
        if gpu_layers_override is not None:
            effective_config["launch_config"]["gpu_layers"] = gpu_layers_override

        # Start the appropriate server based on engine_type
        if engine_type == "transformers":
            # Use lightweight transformers server
            proc = await pm.start_transformers_server(
                model_name,
                model_path,
                embedding=is_embedding,
            )
            proc.last_transition_reason = reason
            proc.reload_count = reload_count
        elif is_transcription:
            # Use whisper-server for transcription models
            proc = await pm.start_whisper_server(model_name, model_path)
            proc.last_transition_reason = reason
            proc.reload_count = reload_count
        else:
            # Use llama-server for chat/embedding models
            proc = await pm.start_server(
                model_name,
                model_path,
                embedding=is_embedding,
                mmproj_path=mmproj_path,
                context_size=context_size,
                gpu_layers=gpu_layers_override,
                effective_config=effective_config,
                transition_reason=reason,
                reload_count=reload_count,
            )

        if proc.status != "running":
            raise RuntimeError(f"Failed to start server: {proc.error_message}")

        # Update last used
        await mm.update_last_used(model_name)

        url = f"http://127.0.0.1:{proc.port}"
        proc.effective_config = effective_config
        logger.info(f"[success]Model loaded: {model_name} at {url} [{engine_type}][/success]")
        self._record_event(
            model_name,
            "loaded",
            reason=reason,
            port=proc.port,
            server_type=proc.server_type,
        )

        return url

    async def unload_model(self, model_name: str, reason: str = "manual_unload") -> None:
        """
        Unload a model.

        Args:
            model_name: Name of the model to unload
        """
        logger.info(f"[warning]Unloading model: {model_name}[/warning]")
        self._record_event(model_name, "unloading", reason=reason)

        pm = self._get_process_manager()
        await pm.stop_server(model_name)

        logger.info(f"[success]Model unloaded: {model_name}[/success]")
        self._record_event(model_name, "unloaded", reason=reason)

    async def reload_model(self, model_name: str, reason: str) -> dict[str, Any]:
        """Reload a running model so saved config becomes authoritative."""
        try:
            pm = self._get_process_manager()
        except RuntimeError:
            status = await self.get_model_status(model_name)
            status["reload_triggered"] = False
            status["message"] = "Runtime not initialized; changes apply on next load."
            return status
        proc = pm.get_process(model_name)
        if not proc or proc.status != "running":
            status = await self.get_model_status(model_name)
            status["reload_triggered"] = False
            status["message"] = "Model is not currently loaded; changes apply on next load."
            return status

        next_reload_count = proc.reload_count + 1
        self._record_event(model_name, "reloading", reason=reason)
        await self.unload_model(model_name, reason=f"reload:{reason}")
        await self.load_model(
            model_name,
            reason=f"reload:{reason}",
            reload_count=next_reload_count,
        )
        status = await self.get_model_status(model_name)
        status["reload_triggered"] = True
        status["message"] = "Running model reloaded with the updated settings."
        return status

    async def reconcile_model_config_change(self, model_name: str) -> dict[str, Any]:
        """Apply model config changes to a running model when needed."""
        return await self.reload_model(model_name, reason="model_config_updated")

    async def reconcile_global_config_change(self, key: str) -> dict[str, Any]:
        """Apply a global config change to live runtime state."""
        runtime_state = self.refresh_runtime_settings()
        reloaded_models: list[str] = []

        try:
            pm = self._get_process_manager()
        except RuntimeError:
            pm = None

        if key in GLOBAL_RELOAD_KEYS and pm is not None:
            for proc in list(pm.get_all_processes()):
                if proc.server_type != "llama":
                    continue
                await self.reload_model(proc.model_name, reason=f"global_config:{key}")
                reloaded_models.append(proc.model_name)

        return {
            "applied_live": key in GLOBAL_RUNTIME_REFRESH_KEYS,
            "reload_triggered": bool(reloaded_models),
            "reloaded_models": reloaded_models,
            "runtime_policy": runtime_state,
            "restart_required": key not in GLOBAL_RUNTIME_REFRESH_KEYS,
        }

    async def record_request(self, model_name: str) -> None:
        """Record that a request was made to a model."""
        pm = self._get_process_manager()
        await pm.update_request_stats(model_name)

    async def touch_request(self, model_name: str) -> None:
        """Update request activity timestamp without incrementing request count."""
        pm = self._get_process_manager()
        await pm.update_request_stats(model_name, increment_count=False)

    async def list_available_models(self) -> list[dict]:
        """List all available models (downloaded and enabled)."""
        mm = self._get_model_manager()
        models = await mm.list_models()

        return [
            m for m in models
            if m["is_downloaded"] and m["is_enabled"]
        ]

    async def get_model_info(self, model_name: str) -> dict | None:
        """Get information about a specific model."""
        mm = self._get_model_manager()
        return await mm.get_model(model_name)

    async def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded models."""
        try:
            pm = self._get_process_manager()
        except RuntimeError:
            return []
        return pm.get_running_models()

    async def get_model_status(self, model_name: str) -> dict:
        """Get detailed status of a model."""
        mm = self._get_model_manager()

        model = await mm.get_model(model_name)
        if not model:
            return {"status": "not_found"}

        try:
            pm = self._get_process_manager()
            proc = pm.get_process(model_name)
        except RuntimeError:
            proc = None
        effective_config = self._build_effective_runtime_config(model, proc)
        last_event = self._model_events.get(model_name, {})

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
            "server_type": proc.server_type if proc else model.get("engine_type", "llama"),
            "last_transition_reason": proc.last_transition_reason if proc else last_event.get("reason"),
            "reload_count": proc.reload_count if proc else 0,
            "effective_config": proc.effective_config if proc and proc.effective_config else effective_config,
            "supports_request_defaults": effective_config["supports_request_defaults"],
            "unsupported_saved_defaults": effective_config["unsupported_saved_defaults"],
            "last_event": last_event,
        }
