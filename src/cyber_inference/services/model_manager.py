"""
Model management for Cyber-Inference.

Handles:
- HuggingFace model discovery and download
- Local model registration and tracking
- Model metadata extraction
- Download progress tracking
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cyber_inference.core.config import get_settings
from cyber_inference.core.database import get_db_session
from cyber_inference.core.logging import get_logger
from cyber_inference.models.db_models import Model

logger = get_logger(__name__)


class ModelManager:
    """
    Manages model discovery, download, and registration.

    Integrates with HuggingFace Hub for model downloads
    and maintains local database of available models.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the model manager.

        Args:
            models_dir: Directory for storing models
        """
        settings = get_settings()
        self.models_dir = models_dir or settings.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._hf_token = settings.hf_token
        self._hf_api = HfApi(token=self._hf_token)

        logger.info("[info]ModelManager initialized[/info]")
        logger.debug(f"  Models directory: {self.models_dir}")
        logger.debug(f"  HuggingFace token: {'configured' if self._hf_token else 'not set'}")

    async def search_models(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search HuggingFace for GGUF models.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of model information dicts
        """
        logger.info(f"[info]Searching HuggingFace for: {query}[/info]")

        try:
            # Search for GGUF models
            models = list(self._hf_api.list_models(
                search=query,
                filter="gguf",
                limit=limit,
                sort="downloads",
                direction=-1,
            ))

            results = []
            for model in models:
                results.append({
                    "id": model.id,
                    "author": model.author,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "last_modified": model.last_modified,
                })

            logger.info(f"[success]Found {len(results)} models[/success]")
            return results

        except Exception as e:
            logger.error(f"[error]HuggingFace search failed: {e}[/error]")
            raise

    async def list_repo_files(self, repo_id: str) -> list[dict]:
        """
        List GGUF files in a HuggingFace repository.

        Args:
            repo_id: HuggingFace repository ID

        Returns:
            List of file information dicts
        """
        logger.info(f"[info]Listing files in repo: {repo_id}[/info]")

        try:
            files = list_repo_files(repo_id, token=self._hf_token)

            gguf_files = []
            for filename in files:
                if filename.endswith(".gguf"):
                    # Get file info
                    try:
                        info = self._hf_api.get_hf_file_metadata(
                            f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                        )
                        size = info.size if hasattr(info, 'size') else 0
                    except Exception:
                        size = 0

                    # Extract quantization from filename
                    quant_match = re.search(r'[qQ](\d+)_?([kKmM])?', filename)
                    quantization = quant_match.group(0) if quant_match else None

                    gguf_files.append({
                        "filename": filename,
                        "size_bytes": size,
                        "quantization": quantization,
                    })

            logger.info(f"[success]Found {len(gguf_files)} GGUF files[/success]")
            return gguf_files

        except Exception as e:
            logger.error(f"[error]Failed to list repo files: {e}[/error]")
            raise

    async def download_model(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Download a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID
            filename: Specific file to download (auto-detect if None)
            force: Force redownload even if exists

        Returns:
            Path to downloaded model file
        """
        logger.info(f"[highlight]Downloading model from: {repo_id}[/highlight]")

        # If no filename specified, find the best GGUF file
        if filename is None:
            files = await self.list_repo_files(repo_id)
            if not files:
                raise ValueError(f"No GGUF files found in {repo_id}")

            # Prefer Q4_K_M or similar balanced quantization
            preferred_quants = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0"]
            filename = files[0]["filename"]

            for quant in preferred_quants:
                for f in files:
                    if quant.lower() in f["filename"].lower():
                        filename = f["filename"]
                        break
                else:
                    continue
                break

            logger.info(f"  Auto-selected file: {filename}")

        # Notify download starting
        await self._notify_progress(repo_id, filename, 0, "starting")

        # Check if already downloaded
        local_path = self.models_dir / filename
        if local_path.exists() and not force:
            logger.info(f"[success]Model already exists: {local_path}[/success]")

            # Ensure it's registered in DB
            await self._register_model(repo_id, filename, local_path)

            # Notify complete
            await self._notify_progress(repo_id, filename, 100, "complete")

            return local_path

        # Download with progress
        logger.info(f"  Downloading to: {local_path}")

        try:
            # Create a progress tracker
            last_progress = [0]  # Use list to allow modification in nested function

            def progress_callback(current: int, total: int) -> None:
                if total > 0:
                    progress = int((current / total) * 100)
                    # Only log every 10%
                    if progress >= last_progress[0] + 10:
                        last_progress[0] = progress
                        logger.info(f"  Download progress: {progress}%")
                        # Schedule async notification
                        try:
                            asyncio.get_event_loop().create_task(
                                self._notify_progress(repo_id, filename, progress, "downloading")
                            )
                        except RuntimeError:
                            pass  # No event loop available

            # Notify downloading
            await self._notify_progress(repo_id, filename, 5, "downloading")

            # Use huggingface_hub for download with progress
            downloaded_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id,
                filename=filename,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False,
                token=self._hf_token,
            )

            # Move to expected location if needed
            downloaded_path = Path(downloaded_path)
            if downloaded_path != local_path:
                downloaded_path.rename(local_path)

            logger.info(f"[success]Download complete: {local_path}[/success]")

            # Register in database
            await self._register_model(repo_id, filename, local_path)

            # Notify complete
            await self._notify_progress(repo_id, filename, 100, "complete")

            return local_path

        except Exception as e:
            logger.error(f"[error]Download failed: {e}[/error]")
            # Notify error
            await self._notify_progress(repo_id, filename, 0, "error", str(e))
            raise

    async def _notify_progress(
        self,
        repo_id: str,
        filename: str,
        progress: float,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Send download progress notification via WebSocket."""
        try:
            from cyber_inference.api.websocket import notify_download_progress
            await notify_download_progress(repo_id, filename, progress, status, error)
        except Exception as e:
            logger.debug(f"Could not send progress notification: {e}")

    async def _register_model(
        self,
        repo_id: str,
        filename: str,
        file_path: Path,
    ) -> Model:
        """Register a model in the database."""
        logger.debug(f"Registering model: {filename}")

        # Extract info from filename
        quant_match = re.search(r'[qQ](\d+)_?([kKmM])?', filename)
        quantization = quant_match.group(0) if quant_match else None

        # Get file size
        size_bytes = file_path.stat().st_size if file_path.exists() else 0

        # Determine model name (use filename without extension)
        model_name = filename.replace(".gguf", "")

        async with get_db_session() as session:
            # Check if already registered
            result = await session.execute(
                select(Model).where(Model.name == model_name)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing record
                existing.file_path = str(file_path)
                existing.size_bytes = size_bytes
                existing.is_downloaded = True
                existing.download_progress = 100.0
                await session.commit()
                logger.debug(f"Updated existing model record: {model_name}")
                return existing

            # Create new record
            model = Model(
                name=model_name,
                filename=filename,
                file_path=str(file_path),
                hf_repo_id=repo_id,
                hf_filename=filename,
                size_bytes=size_bytes,
                quantization=quantization,
                is_downloaded=True,
                download_progress=100.0,
            )
            session.add(model)
            await session.commit()

            logger.info(f"[success]Model registered: {model_name}[/success]")
            return model

    async def list_models(self) -> list[dict]:
        """
        List all models (registered and local files).

        Returns:
            List of model information dicts
        """
        logger.debug("Listing all models")

        models = []

        # Get models from database
        async with get_db_session() as session:
            result = await session.execute(select(Model))
            db_models = result.scalars().all()

            for model in db_models:
                models.append({
                    "id": model.id,
                    "name": model.name,
                    "filename": model.filename,
                    "path": model.file_path,
                    "hf_repo_id": model.hf_repo_id,
                    "size_bytes": model.size_bytes,
                    "quantization": model.quantization,
                    "context_length": model.context_length,
                    "model_type": model.model_type,
                    "is_downloaded": model.is_downloaded,
                    "is_enabled": model.is_enabled,
                    "last_used_at": model.last_used_at,
                    "registered": True,
                })

        # Scan for unregistered local files
        registered_files = {m["filename"] for m in models}

        for file_path in self.models_dir.glob("*.gguf"):
            if file_path.name not in registered_files:
                models.append({
                    "id": None,
                    "name": file_path.stem,
                    "filename": file_path.name,
                    "path": str(file_path),
                    "hf_repo_id": None,
                    "size_bytes": file_path.stat().st_size,
                    "quantization": None,
                    "context_length": 4096,
                    "model_type": None,
                    "is_downloaded": True,
                    "is_enabled": True,
                    "last_used_at": None,
                    "registered": False,
                })

        logger.debug(f"Found {len(models)} models")
        return models

    async def get_model(self, name: str) -> Optional[dict]:
        """
        Get model by name.

        Args:
            name: Model name

        Returns:
            Model information dict, or None if not found
        """
        models = await self.list_models()
        for model in models:
            if model["name"] == name:
                return model
        return None

    async def get_model_path(self, name: str) -> Optional[Path]:
        """
        Get the file path for a model.

        Args:
            name: Model name

        Returns:
            Path to model file, or None if not found
        """
        model = await self.get_model(name)
        if model and model["is_downloaded"]:
            return Path(model["path"])
        return None

    async def delete_model(self, name: str) -> bool:
        """
        Delete a model (file and database record).

        Args:
            name: Model name

        Returns:
            True if deleted, False if not found
        """
        logger.info(f"[warning]Deleting model: {name}[/warning]")

        model = await self.get_model(name)
        if not model:
            logger.warning(f"Model not found: {name}")
            return False

        # Delete file
        file_path = Path(model["path"])
        if file_path.exists():
            file_path.unlink()
            logger.info(f"  Deleted file: {file_path}")

        # Delete from database
        if model["id"]:
            async with get_db_session() as session:
                result = await session.execute(
                    select(Model).where(Model.id == model["id"])
                )
                db_model = result.scalar_one_or_none()
                if db_model:
                    await session.delete(db_model)
                    await session.commit()
                    logger.info(f"  Deleted database record")

        logger.info(f"[success]Model deleted: {name}[/success]")
        return True

    async def update_last_used(self, name: str) -> None:
        """Update the last_used_at timestamp for a model."""
        async with get_db_session() as session:
            result = await session.execute(
                select(Model).where(Model.name == name)
            )
            model = result.scalar_one_or_none()
            if model:
                model.last_used_at = datetime.now()
                await session.commit()

    async def register_local_model(
        self,
        file_path: Path,
        name: Optional[str] = None,
    ) -> Model:
        """
        Register a local GGUF file that was manually added.

        Args:
            file_path: Path to the GGUF file
            name: Custom name (default: filename without extension)

        Returns:
            Created Model record
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        if not file_path.suffix == ".gguf":
            raise ValueError("Model file must be a .gguf file")

        model_name = name or file_path.stem

        return await self._register_model(
            repo_id="local",
            filename=file_path.name,
            file_path=file_path,
        )

