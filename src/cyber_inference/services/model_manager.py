"""
Model management for Cyber-Inference.

Handles:
- HuggingFace model discovery and download (GGUF and transformers formats)
- Local model registration and tracking
- Model metadata extraction
- Download progress tracking
"""

import asyncio
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
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

    @staticmethod
    def _read_gguf_context_length(file_path: Path) -> Optional[int]:
        def read_exact(handle, size: int) -> bytes:
            data = handle.read(size)
            if len(data) != size:
                raise ValueError("Unexpected end of file")
            return data

        def read_u8(handle) -> int:
            return int.from_bytes(read_exact(handle, 1), "little", signed=False)

        def read_i8(handle) -> int:
            return int.from_bytes(read_exact(handle, 1), "little", signed=True)

        def read_u16(handle) -> int:
            return int.from_bytes(read_exact(handle, 2), "little", signed=False)

        def read_i16(handle) -> int:
            return int.from_bytes(read_exact(handle, 2), "little", signed=True)

        def read_u32(handle) -> int:
            return int.from_bytes(read_exact(handle, 4), "little", signed=False)

        def read_u64(handle) -> int:
            return int.from_bytes(read_exact(handle, 8), "little", signed=False)

        def read_i32(handle) -> int:
            return int.from_bytes(read_exact(handle, 4), "little", signed=True)

        def read_i64(handle) -> int:
            return int.from_bytes(read_exact(handle, 8), "little", signed=True)

        def read_string(handle) -> str:
            length = read_u64(handle)
            if length == 0:
                return ""
            return read_exact(handle, length).decode("utf-8", errors="ignore")

        def skip_string(handle) -> None:
            length = read_u64(handle)
            if length:
                read_exact(handle, length)

        type_sizes = {
            0: 1,  # UINT8
            1: 1,  # INT8
            2: 2,  # UINT16
            3: 2,  # INT16
            4: 4,  # UINT32
            5: 4,  # INT32
            6: 4,  # FLOAT32
            7: 1,  # BOOL
            10: 8,  # UINT64
            11: 8,  # INT64
            12: 8,  # FLOAT64
        }
        GGUF_STRING = 8
        GGUF_ARRAY = 9

        context_keys = {
            "llama.context_length",
            "context_length",
            "n_ctx",
            "llama.n_ctx",
            "n_ctx_train",
            "llama.n_ctx_train",
        }

        def is_context_key(key: str) -> bool:
            if key in context_keys:
                return True
            if key.endswith((".context_length", ".n_ctx", ".n_ctx_train")):
                return True
            return False

        def skip_array(handle, elem_type: int, length: int) -> None:
            if elem_type == GGUF_STRING:
                for _ in range(length):
                    skip_string(handle)
                return
            elem_size = type_sizes.get(elem_type)
            if elem_size is None:
                raise ValueError("Unknown GGUF array element type")
            read_exact(handle, elem_size * length)

        def skip_value(handle, value_type: int) -> None:
            if value_type == GGUF_STRING:
                skip_string(handle)
                return
            if value_type == GGUF_ARRAY:
                elem_type = read_u32(handle)
                length = read_u64(handle)
                skip_array(handle, elem_type, length)
                return
            size = type_sizes.get(value_type)
            if size is None:
                raise ValueError("Unknown GGUF value type")
            read_exact(handle, size)

        def read_numeric_value(handle, value_type: int) -> Optional[int]:
            if value_type == 0:
                return read_u8(handle)
            if value_type == 1:
                return read_i8(handle)
            if value_type == 2:
                return read_u16(handle)
            if value_type == 3:
                return read_i16(handle)
            if value_type == 4:
                return read_u32(handle)
            if value_type == 5:
                return read_i32(handle)
            if value_type == 7:
                return read_u8(handle)
            if value_type == 10:
                return read_u64(handle)
            if value_type == 11:
                return read_i64(handle)
            if value_type == GGUF_STRING:
                value = read_string(handle)
                if value.isdigit():
                    return int(value)
                return None
            if value_type == GGUF_ARRAY:
                elem_type = read_u32(handle)
                length = read_u64(handle)
                if length == 1:
                    return read_numeric_value(handle, elem_type)
                skip_array(handle, elem_type, length)
                return None
            skip_value(handle, value_type)
            return None

        def select_context_length(
            architecture: Optional[str],
            candidates: dict[str, int],
        ) -> Optional[int]:
            if not candidates:
                return None

            if architecture:
                arch_variants = {
                    architecture,
                    architecture.replace("-", "_"),
                    architecture.replace("_", "-"),
                }
                for arch in arch_variants:
                    for suffix in ("context_length", "n_ctx", "n_ctx_train"):
                        key = f"{arch}.{suffix}"
                        if key in candidates:
                            return candidates[key]

            for key in (
                "llama.context_length",
                "llama.n_ctx",
                "llama.n_ctx_train",
                "context_length",
                "n_ctx",
                "n_ctx_train",
            ):
                if key in candidates:
                    return candidates[key]

            for key, value in candidates.items():
                if key.startswith(("clip.", "vision.", "mmproj.")):
                    continue
                return value

            return None

        try:
            with file_path.open("rb") as handle:
                magic = read_exact(handle, 4)
                if magic != b"GGUF":
                    return None

                _version = read_u32(handle)
                _tensor_count = read_u64(handle)
                kv_count = read_u64(handle)

                architecture = None
                candidates: dict[str, int] = {}

                for _ in range(kv_count):
                    key = read_string(handle)
                    key_lower = key.lower()
                    value_type = read_u32(handle)

                    if key_lower == "general.architecture":
                        if value_type == GGUF_STRING:
                            architecture = read_string(handle).lower()
                        else:
                            skip_value(handle, value_type)
                        continue

                    if is_context_key(key_lower):
                        value = read_numeric_value(handle, value_type)
                        if value is not None:
                            candidates[key_lower] = int(value)
                        continue

                    skip_value(handle, value_type)

                return select_context_length(architecture, candidates)
        except Exception as e:
            logger.debug(f"Failed to read GGUF metadata from {file_path.name}: {e}")
            return None

        return None

    @staticmethod
    def _read_transformers_context_length(model_dir: Path) -> Optional[int]:
        """Read context length from a HuggingFace transformers model's config.json."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None
        try:
            import json
            config = json.loads(config_path.read_text())
            # Priority order of config keys that indicate context length
            for key in (
                "max_position_embeddings",
                "n_positions",
                "max_sequence_length",
                "seq_length",
                "sliding_window",
            ):
                value = config.get(key)
                if isinstance(value, int) and value > 0:
                    logger.debug(f"Transformers context length from {key}: {value}")
                    return value
        except Exception as e:
            logger.debug(f"Failed to read config.json from {model_dir}: {e}")
        return None

    @staticmethod
    def _is_mmproj_file(filename: str) -> bool:
        name = filename.lower()
        return name.endswith(".gguf") and "mmproj" in name

    @staticmethod
    def _split_repo_and_filename(
        repo_id: str,
        filename: Optional[str],
    ) -> tuple[str, Optional[str]]:
        if filename:
            return repo_id, filename

        cleaned = repo_id.strip()
        if cleaned.startswith("https://huggingface.co/"):
            cleaned = cleaned[len("https://huggingface.co/"):]
        cleaned = cleaned.lstrip("/").split("?", 1)[0]

        if not cleaned.endswith(".gguf"):
            return repo_id, filename

        parts = cleaned.split("/")
        if len(parts) < 3:
            return repo_id, filename

        for marker in ("blob", "resolve", "tree"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    parts = parts[:idx] + parts[idx + 2:]
                else:
                    parts = parts[:idx]
                break

        if len(parts) < 3 or not parts[-1].endswith(".gguf"):
            return repo_id, filename

        new_repo_id = "/".join(parts[:2])
        new_filename = "/".join(parts[2:])
        return new_repo_id, new_filename

    @staticmethod
    def _extract_model_base_name(filename: str) -> str:
        """
        Extract the base model name by removing quantization suffixes.

        Handles patterns like:
        - Model-Name-Q4_K_M.gguf -> Model-Name
        - Model-Name.Q8_0.gguf -> Model-Name
        - Model-Name.BF16.gguf -> Model-Name
        - Model-Name-F16.gguf -> Model-Name
        """
        stem = Path(filename).stem
        # Remove common quantization patterns (case-insensitive)
        # Pattern 1: -Q4_K_M, -Q8_0, etc. (hyphen prefix)
        # Pattern 2: .Q4_K_M, .Q8_0, .BF16, .F16, .F32 (dot prefix)
        # Pattern 3: _Q4_K_M (underscore prefix)
        patterns = [
            r"(?i)[._-]q\d+[_a-z0-9]*$",  # Q4_K_M, Q8_0, etc.
            r"(?i)[._-](?:bf16|f16|f32)$",  # BF16, F16, F32
            r"(?i)[._-](?:fp16|fp32)$",  # FP16, FP32
        ]
        for pattern in patterns:
            stem = re.sub(pattern, "", stem)
        return stem

    @staticmethod
    def _extract_quant_suffix(filename: str) -> Optional[str]:
        """
        Extract the quantization suffix from a filename.

        Returns lowercase suffix like 'bf16', 'f16', 'q8_0', 'q4_k_m', etc.
        """
        stem = Path(filename).stem.lower()
        # Match quantization patterns at end of filename
        patterns = [
            r"[._-](q\d+[_a-z0-9]*)$",  # Q4_K_M, Q8_0, etc.
            r"[._-](bf16|f16|f32|fp16|fp32)$",  # BF16, F16, etc.
        ]
        for pattern in patterns:
            match = re.search(pattern, stem, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    @staticmethod
    def _select_mmproj_file(files: list[str], model_filename: str) -> Optional[str]:
        """
        Select the appropriate mmproj file for a model.

        Handles various naming patterns:
        - mmproj-{model}.gguf
        - {Model}.mmproj-{quant}.gguf (e.g., Cosmos-Reason2-8B.mmproj-bf16.gguf)
        - mmproj-{quant}.gguf
        """
        mmproj_files = sorted([f for f in files if ModelManager._is_mmproj_file(f)])
        if not mmproj_files:
            return None

        def basename(path: str) -> str:
            return Path(path).name

        def pick_by_quant_preference(candidates: list[str], model_quant: Optional[str]) -> Optional[str]:
            """Pick from candidates preferring matching quant or default order."""
            if not candidates:
                return None

            # If we have a model quantization, try to match it exactly first
            if model_quant:
                model_quant_lower = model_quant.lower()
                for candidate in candidates:
                    cand_lower = basename(candidate).lower()
                    # Check for exact quantization match (e.g., bf16 matches bf16, not f16)
                    if f"mmproj-{model_quant_lower}" in cand_lower or f".mmproj-{model_quant_lower}" in cand_lower:
                        return candidate

            # Preference order: f16 > bf16 > f32 > q8_0 > others
            preferred_suffixes = ("f16", "bf16", "f32", "q8_0", "q4_0")
            for suffix in preferred_suffixes:
                for candidate in candidates:
                    cand_lower = basename(candidate).lower()
                    if f"mmproj-{suffix}" in cand_lower or f".mmproj-{suffix}" in cand_lower:
                        return candidate

            # Return shortest name as fallback
            return sorted(candidates, key=lambda x: len(basename(x)))[0]

        model_stem = Path(model_filename).stem
        model_base = ModelManager._extract_model_base_name(model_filename)
        model_quant = ModelManager._extract_quant_suffix(model_filename)
        model_base_lower = model_base.lower()

        # Strategy 1: Exact match - mmproj-{model_stem}.gguf
        exact = f"mmproj-{model_stem}.gguf".lower()
        for candidate in mmproj_files:
            if basename(candidate).lower() == exact:
                return candidate

        # Strategy 2: Pattern {ModelBase}.mmproj-{quant}.gguf
        # e.g., Cosmos-Reason2-8B.mmproj-bf16.gguf for Cosmos-Reason2-8B.BF16.gguf
        pattern_matches = []
        for candidate in mmproj_files:
            cand_lower = basename(candidate).lower()
            # Check if filename starts with model base name and contains .mmproj
            if cand_lower.startswith(model_base_lower) and ".mmproj" in cand_lower:
                pattern_matches.append(candidate)

        if pattern_matches:
            return pick_by_quant_preference(pattern_matches, model_quant)

        # Strategy 3: Prefix match - mmproj-{model_base}*.gguf
        prefix = f"mmproj-{model_base_lower}"
        prefixed = [f for f in mmproj_files if basename(f).lower().startswith(prefix)]
        if prefixed:
            return pick_by_quant_preference(prefixed, model_quant)

        # Strategy 4: Single mmproj file in repo
        if len(mmproj_files) == 1:
            return mmproj_files[0]

        # Strategy 5: Generic mmproj files (no model name prefix)
        generic_patterns = [
            "mmproj-f16.gguf", "mmproj-bf16.gguf", "mmproj-f32.gguf",
            "mmproj-q8_0.gguf", "mmproj-q4_0.gguf", "mmproj.gguf"
        ]
        lowered_map = {basename(f).lower(): f for f in mmproj_files}
        for pattern in generic_patterns:
            if pattern in lowered_map:
                return lowered_map[pattern]

        # Strategy 6: Model base name appears anywhere in mmproj filename
        contains = [f for f in mmproj_files if model_base_lower in basename(f).lower()]
        if contains:
            return pick_by_quant_preference(contains, model_quant)

        logger.warning(
            "[warning]Multiple mmproj files found but none matched model %s[/warning]",
            model_stem,
        )
        return None

    async def _download_mmproj(
        self,
        repo_id: str,
        model_path: Path,
        repo_files: Optional[list[str]] = None,
        mmproj_filename: Optional[str] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Download the mmproj file for a multimodal model.

        Args:
            repo_id: HuggingFace repository ID
            model_path: Path to the main model file
            repo_files: Optional list of repo files (to avoid re-fetching)
            mmproj_filename: Explicit mmproj filename to download (auto-select if None)
            force: Force redownload even if exists

        Returns:
            Path to downloaded mmproj file, or None if not applicable
        """
        try:
            files = repo_files or list_repo_files(repo_id, token=self._hf_token)
        except Exception as e:
            logger.warning(f"[warning]Could not list repo files for mmproj: {e}[/warning]")
            return None

        # Use explicit filename or auto-select
        if not mmproj_filename:
            mmproj_filename = self._select_mmproj_file(files, model_path.name)

        if not mmproj_filename:
            return None

        # Verify the file exists in repo
        if mmproj_filename not in files:
            logger.warning(f"[warning]Specified mmproj file not found in repo: {mmproj_filename}[/warning]")
            return None

        local_path = model_path.parent / f"mmproj-{model_path.stem}.gguf"
        if local_path.exists() and not force:
            logger.info(f"  mmproj already present: {local_path}")
            return local_path

        logger.info(f"  Downloading mmproj: {mmproj_filename}")
        try:
            downloaded_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id,
                filename=mmproj_filename,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False,
                token=self._hf_token,
            )
            downloaded_path = Path(downloaded_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if downloaded_path != local_path:
                if local_path.exists():
                    local_path.unlink()
                downloaded_path.rename(local_path)
                logger.info(f"  Aligned mmproj filename to: {local_path.name}")
            logger.info(f"[success]mmproj download complete: {local_path}[/success]")
            return local_path
        except Exception as e:
            logger.warning(f"[warning]mmproj download failed: {e}[/warning]")
            return None

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

    async def list_repo_files(self, repo_id: str, files: Optional[list[str]] = None) -> list[dict]:
        """
        List GGUF files in a HuggingFace repository.

        Args:
            repo_id: HuggingFace repository ID
            files: Optional pre-fetched list of filenames (sizes will be fetched separately)

        Returns:
            List of file information dicts
        """
        logger.info(f"[info]Listing files in repo: {repo_id}[/info]")

        try:
            # If files list provided, we still need sizes - fetch repo tree
            repo_tree = await asyncio.to_thread(
                self._hf_api.list_repo_tree,
                repo_id,
                recursive=True,
            )

            # Build a map of filename -> size
            file_sizes: dict[str, int] = {}
            all_files: list[str] = []
            for item in repo_tree:
                if hasattr(item, 'path') and hasattr(item, 'size'):
                    file_sizes[item.path] = item.size or 0
                    all_files.append(item.path)

            # Use provided files list or all files from tree
            filenames_to_check = files if files else all_files

            model_files = []
            for filename in filenames_to_check:
                is_gguf = filename.endswith(".gguf") and not self._is_mmproj_file(filename)
                is_whisper_bin = filename.startswith("ggml-") and filename.endswith(".bin")

                if is_gguf or is_whisper_bin:
                    size = file_sizes.get(filename, 0)
                    quantization = self._extract_quant_suffix(filename)

                    model_files.append({
                        "filename": filename,
                        "size_bytes": size,
                        "quantization": quantization,
                    })

            logger.info(f"[success]Found {len(model_files)} model files[/success]")
            return model_files

        except Exception as e:
            logger.error(f"[error]Failed to list repo files: {e}[/error]")
            raise

    async def list_repo_files_detailed(self, repo_id: str) -> dict:
        """
        List all GGUF files in a repository with model/mmproj pairing.

        Args:
            repo_id: HuggingFace repository ID

        Returns:
            Dict with model_files, mmproj_files, is_multimodal, and suggestions
        """
        logger.info(f"[info]Listing detailed files in repo: {repo_id}[/info]")

        try:
            # Use list_repo_tree to get file info including sizes in one API call
            repo_tree = await asyncio.to_thread(
                self._hf_api.list_repo_tree,
                repo_id,
                recursive=True,
            )

            # Build a map of filename -> size from repo tree
            file_sizes: dict[str, int] = {}
            for item in repo_tree:
                # RepoFile objects have path and size attributes
                if hasattr(item, 'path') and hasattr(item, 'size'):
                    file_sizes[item.path] = item.size or 0

            model_files = []
            mmproj_files = []

            # Check if this is a whisper.cpp repo (contains ggml-*.bin files)
            is_whisper_repo = any(
                f.startswith("ggml-") and f.endswith(".bin")
                for f in file_sizes.keys()
            )

            for filename, size in file_sizes.items():
                # Support both GGUF files and whisper.cpp bin files
                is_gguf = filename.endswith(".gguf")
                is_whisper_bin = filename.startswith("ggml-") and filename.endswith(".bin")

                if not (is_gguf or is_whisper_bin):
                    continue

                quantization = self._extract_quant_suffix(filename)

                file_info = {
                    "filename": filename,
                    "size_bytes": size,
                    "quantization": quantization,
                    "is_mmproj": self._is_mmproj_file(filename) if is_gguf else False,
                }

                if is_gguf and self._is_mmproj_file(filename):
                    mmproj_files.append(file_info)
                else:
                    model_files.append(file_info)

            is_multimodal = len(mmproj_files) > 0

            # Auto-suggest best model file (prefer Q4_K_M, then others)
            suggested_model = None
            if model_files:
                preferred_quants = ["q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q4_0", "q8_0"]
                for quant in preferred_quants:
                    for f in model_files:
                        if f["quantization"] and quant == f["quantization"].lower():
                            suggested_model = f["filename"]
                            break
                    if suggested_model:
                        break
                if not suggested_model:
                    suggested_model = model_files[0]["filename"]

            # Auto-suggest mmproj for the suggested model
            suggested_mmproj = None
            if suggested_model and mmproj_files:
                suggested_mmproj = self._select_mmproj_file(
                    [f["filename"] for f in mmproj_files],
                    suggested_model
                )

            result = {
                "repo_id": repo_id,
                "model_files": model_files,
                "mmproj_files": mmproj_files,
                "is_multimodal": is_multimodal,
                "suggested_model": suggested_model,
                "suggested_mmproj": suggested_mmproj,
            }

            logger.info(f"[success]Found {len(model_files)} model files, {len(mmproj_files)} mmproj files[/success]")
            return result

        except Exception as e:
            logger.error(f"[error]Failed to list repo files: {e}[/error]")
            raise

    def get_suggested_mmproj(self, model_filename: str, mmproj_files: list[str]) -> Optional[str]:
        """
        Get the suggested mmproj file for a given model filename.

        Args:
            model_filename: The model filename to match
            mmproj_files: List of available mmproj filenames

        Returns:
            Suggested mmproj filename or None
        """
        return self._select_mmproj_file(mmproj_files, model_filename)

    async def download_model(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        mmproj_filename: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Download a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID
            filename: Specific file to download (auto-detect if None)
            mmproj_filename: Specific mmproj file to download for vision models (auto-detect if None)
            force: Force redownload even if exists

        Returns:
            Path to downloaded model file
        """
        repo_id = repo_id.strip()
        filename = filename.strip() if filename else None
        mmproj_filename = mmproj_filename.strip() if mmproj_filename else None
        parsed = self._split_repo_and_filename(repo_id, filename)
        repo_id, filename = parsed

        logger.info(f"[highlight]Downloading model from: {repo_id}[/highlight]")
        repo_files = list_repo_files(repo_id, token=self._hf_token)

        # If no filename specified, find the best GGUF file
        if filename is None:
            files = await self.list_repo_files(repo_id, files=repo_files)
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

            # Download mmproj and get path
            mmproj_path = await self._download_mmproj(
                repo_id, local_path,
                repo_files=repo_files,
                mmproj_filename=mmproj_filename,
                force=force
            )

            # Ensure it's registered in DB with mmproj_path
            await self._register_model(repo_id, filename, local_path, mmproj_path=mmproj_path)

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

            # Download mmproj and get path
            mmproj_path = await self._download_mmproj(
                repo_id, local_path,
                repo_files=repo_files,
                mmproj_filename=mmproj_filename,
                force=force
            )

            # Register in database with mmproj_path
            await self._register_model(repo_id, filename, local_path, mmproj_path=mmproj_path)

            # Notify complete
            await self._notify_progress(repo_id, filename, 100, "complete")

            return local_path

        except Exception as e:
            logger.error(f"[error]Download failed: {e}[/error]")
            # Clean up partial download
            if local_path.exists():
                try:
                    local_path.unlink()
                    logger.info(f"  Cleaned up partial file: {local_path}")
                except Exception as cleanup_err:
                    logger.warning(f"  Could not clean up partial file: {cleanup_err}")
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
        mmproj_path: Optional[Path] = None,
        engine_type: Optional[str] = None,
        model_name_override: Optional[str] = None,
    ) -> Model:
        """Register a model in the database.

        Args:
            repo_id: HuggingFace repository ID
            filename: Model filename
            file_path: Local file path
            mmproj_path: Optional mmproj file path for vision models
            engine_type: Engine type ('llama', 'whisper', 'transformers')
            model_name_override: Override the auto-generated model name
        """
        logger.debug(f"Registering model: {filename}")

        # Extract quantization from filename using our improved method
        quantization = self._extract_quant_suffix(filename)

        # Get file size (for directory-based models, sum all files)
        if file_path.is_dir():
            size_bytes = sum(
                f.stat().st_size for f in file_path.rglob("*") if f.is_file()
            )
        else:
            size_bytes = file_path.stat().st_size if file_path.exists() else 0

        # Determine model name
        if model_name_override:
            model_name = model_name_override
        elif filename.endswith(".gguf"):
            model_name = filename.replace(".gguf", "")
        elif filename.endswith(".bin"):
            model_name = filename.replace(".bin", "")
        else:
            model_name = Path(filename).stem

        # Read context length from model metadata
        context_length = None
        if filename.endswith(".gguf") and file_path.is_file():
            context_length = self._read_gguf_context_length(file_path)
        elif file_path.is_dir():
            context_length = self._read_transformers_context_length(file_path)

        # Convert mmproj_path to string if provided
        mmproj_path_str = str(mmproj_path) if mmproj_path else None

        # Determine engine_type if not provided
        if not engine_type:
            engine_type = "llama"  # default

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
                existing.engine_type = engine_type
                if context_length:
                    existing.context_length = context_length

                # Update mmproj_path
                if mmproj_path_str:
                    existing.mmproj_path = mmproj_path_str
                    logger.info(f"  Updated mmproj path: {mmproj_path_str}")

                # Auto-detect model type if not set
                if not existing.model_type:
                    name_lower = model_name.lower()
                    repo_lower = (repo_id or existing.hf_repo_id or "").lower()
                    check_string = f"{name_lower} {repo_lower}"

                    embedding_patterns = ["embed", "bge", "e5-", "gte-", "stella", "nomic"]
                    transcription_patterns = ["whisper", "distil-whisper", "faster-whisper"]

                    if any(pattern in check_string for pattern in embedding_patterns):
                        existing.model_type = "embedding"
                        logger.info("  Auto-detected model type: embedding")
                    elif any(pattern in check_string for pattern in transcription_patterns):
                        existing.model_type = "transcription"
                        logger.info("  Auto-detected model type: transcription")

                await session.commit()
                logger.debug(f"Updated existing model record: {model_name}")
                return existing

            # Auto-detect model type from name AND repo ID
            model_type = None
            name_lower = model_name.lower()
            repo_lower = (repo_id or "").lower()
            # Check both filename and repo ID for patterns
            check_string = f"{name_lower} {repo_lower}"

            embedding_patterns = ["embed", "bge", "e5-", "gte-", "stella", "nomic"]
            transcription_patterns = ["whisper", "distil-whisper", "faster-whisper"]

            if any(pattern in check_string for pattern in embedding_patterns):
                model_type = "embedding"
                logger.info("  Auto-detected model type: embedding")
            elif any(pattern in check_string for pattern in transcription_patterns):
                model_type = "transcription"
                logger.info("  Auto-detected model type: transcription")

            # Create new record
            model = Model(
                name=model_name,
                filename=filename,
                file_path=str(file_path),
                hf_repo_id=repo_id,
                hf_filename=filename,
                size_bytes=size_bytes,
                quantization=quantization,
                context_length=context_length or 4096,
                model_type=model_type,
                engine_type=engine_type,
                mmproj_path=mmproj_path_str,
                is_downloaded=True,
                download_progress=100.0,
            )
            session.add(model)
            await session.commit()

            engine_label = f" [{engine_type}]" if engine_type != "llama" else ""
            if mmproj_path_str:
                logger.info(
                    f"[success]Model registered: {model_name}{engine_label} (with mmproj)[/success]"
                )
            else:
                logger.info(f"[success]Model registered: {model_name}{engine_label}[/success]")
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

            updated = False
            for model in db_models:
                # Backfill context_length for models that still have the 4096 default
                if model.file_path and (model.context_length is None or model.context_length == 4096):
                    file_path = Path(model.file_path)
                    context_length = None
                    if file_path.exists() and file_path.suffix == ".gguf":
                        context_length = self._read_gguf_context_length(file_path)
                    elif file_path.is_dir():
                        context_length = self._read_transformers_context_length(file_path)
                    if context_length and context_length != model.context_length:
                        model.context_length = context_length
                        updated = True

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
                    "engine_type": model.engine_type or "llama",
                    "mmproj_path": model.mmproj_path,
                    "is_downloaded": model.is_downloaded,
                    "is_enabled": model.is_enabled,
                    "last_used_at": model.last_used_at,
                    "registered": True,
                    "default_context_size": model.default_context_size,
                    "default_temperature": model.default_temperature,
                    "default_top_p": model.default_top_p,
                    "default_top_k": model.default_top_k,
                    "default_max_tokens": model.default_max_tokens,
                    "default_repeat_penalty": model.default_repeat_penalty,
                })

            if updated:
                await session.commit()

        # Scan for unregistered local files
        registered_files = {m["filename"] for m in models}
        registered_names = {m["name"] for m in models}

        # Scan for both GGUF and whisper.cpp bin files
        for file_path in list(self.models_dir.glob("*.gguf")) + list(self.models_dir.glob("ggml-*.bin")):
            if file_path.suffix == ".gguf" and self._is_mmproj_file(file_path.name):
                continue
            if file_path.name not in registered_files:
                context_length = self._read_gguf_context_length(file_path)
                # Try to find associated mmproj file
                mmproj_path = None
                potential_mmproj = file_path.parent / f"mmproj-{file_path.stem}.gguf"
                if potential_mmproj.exists():
                    mmproj_path = str(potential_mmproj)

                models.append({
                    "id": None,
                    "name": file_path.stem,
                    "filename": file_path.name,
                    "path": str(file_path),
                    "hf_repo_id": None,
                    "size_bytes": file_path.stat().st_size,
                    "quantization": None,
                    "context_length": context_length or 4096,
                    "model_type": None,
                    "engine_type": "llama",
                    "mmproj_path": mmproj_path,
                    "is_downloaded": True,
                    "is_enabled": True,
                    "last_used_at": None,
                    "registered": False,
                    "default_context_size": None,
                    "default_temperature": None,
                    "default_top_p": None,
                    "default_top_k": None,
                    "default_max_tokens": None,
                    "default_repeat_penalty": None,
                })

        # Scan for unregistered transformers models (directories in models/transformers/)
        settings = get_settings()
        transformers_dir = settings.transformers_models_dir
        if transformers_dir.exists():
            for model_dir in transformers_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                # Check for config.json to identify valid HuggingFace model dirs
                if not (model_dir / "config.json").exists():
                    continue
                dir_name = model_dir.name
                if dir_name not in registered_names:
                    # Calculate total size
                    total_size = sum(
                        f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                    )
                    ctx_len = self._read_transformers_context_length(model_dir) or 4096
                    models.append({
                        "id": None,
                        "name": dir_name,
                        "filename": dir_name,
                        "path": str(model_dir),
                        "hf_repo_id": None,
                        "size_bytes": total_size,
                        "quantization": None,
                        "context_length": ctx_len,
                        "model_type": None,
                        "engine_type": "transformers",
                        "mmproj_path": None,
                        "is_downloaded": True,
                        "is_enabled": True,
                        "last_used_at": None,
                        "registered": False,
                        "default_context_size": None,
                        "default_temperature": None,
                        "default_top_p": None,
                        "default_top_k": None,
                        "default_max_tokens": None,
                        "default_repeat_penalty": None,
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
        Delete a model (file/directory and database record).

        Handles both single GGUF files and directory-based model formats.

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

        # Delete file or directory (ignore if already gone)
        file_path = Path(model["path"])
        try:
            if file_path.exists():
                if file_path.is_dir():
                    # Transformers/HF models are directories
                    shutil.rmtree(file_path)
                    logger.info(f"  Deleted directory: {file_path}")
                else:
                    file_path.unlink()
                    logger.info(f"  Deleted file: {file_path}")
            else:
                logger.info(f"  File already gone: {file_path}")
        except Exception as e:
            logger.warning(f"  Could not delete file: {e}")

        # Always try to delete database record
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

        if file_path.suffix not in (".gguf", ".bin"):
            raise ValueError("Model file must be a .gguf or .bin file")

        model_name = name or file_path.stem

        return await self._register_model(
            repo_id="local",
            filename=file_path.name,
            file_path=file_path,
        )

    # ── Transformers Model Management ─────────────────────────────────

    @staticmethod
    def _sanitize_repo_name(repo_id: str) -> str:
        """
        Convert a HuggingFace repo_id to a safe directory name.

        'meta-llama/Meta-Llama-3-8B-Instruct' -> 'Meta-Llama-3-8B-Instruct'
        """
        # Use the model name part (after the slash)
        parts = repo_id.strip().rstrip("/").split("/")
        return parts[-1] if len(parts) > 1 else parts[0]

    async def download_transformers_model(
        self,
        repo_id: str,
        force: bool = False,
    ) -> Path:
        """
        Download a HuggingFace model for use with the transformers engine.

        Uses snapshot_download to download the full model repository
        to models/transformers/{model_name}/.

        Args:
            repo_id: HuggingFace repository ID (e.g., 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8')
            force: Force redownload even if exists

        Returns:
            Path to the downloaded model directory
        """
        repo_id = repo_id.strip()
        model_name = self._sanitize_repo_name(repo_id)

        settings = get_settings()
        tf_dir = settings.transformers_models_dir
        tf_dir.mkdir(parents=True, exist_ok=True)

        local_dir = tf_dir / model_name

        logger.info(f"[highlight]Downloading transformers model: {repo_id}[/highlight]")
        logger.info(f"  Target directory: {local_dir}")

        # Notify download starting
        await self._notify_progress(repo_id, model_name, 0, "starting")

        # Check if already downloaded
        if local_dir.exists() and (local_dir / "config.json").exists() and not force:
            logger.info(f"[success]Transformers model already exists: {local_dir}[/success]")
            await self._register_model(
                repo_id=repo_id,
                filename=model_name,
                file_path=local_dir,
                engine_type="transformers",
                model_name_override=model_name,
            )
            await self._notify_progress(repo_id, model_name, 100, "complete")
            return local_dir

        # Download with progress tracking
        await self._notify_progress(repo_id, model_name, 5, "downloading")

        try:
            def _do_download() -> str:
                """Run snapshot_download in a thread."""
                return snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    token=self._hf_token,
                )

            # Run the download in a thread to avoid blocking
            await asyncio.to_thread(_do_download)

            logger.info(f"[success]Transformers model download complete: {local_dir}[/success]")

            # Register in database
            await self._register_model(
                repo_id=repo_id,
                filename=model_name,
                file_path=local_dir,
                engine_type="transformers",
                model_name_override=model_name,
            )

            # Notify complete
            await self._notify_progress(repo_id, model_name, 100, "complete")

            return local_dir

        except Exception as e:
            logger.error(f"[error]Transformers model download failed: {e}[/error]")
            # Clean up partial download
            if local_dir.exists():
                try:
                    shutil.rmtree(local_dir)
                    logger.info(f"  Cleaned up partial download: {local_dir}")
                except Exception as cleanup_err:
                    logger.warning(f"  Could not clean up: {cleanup_err}")
            await self._notify_progress(repo_id, model_name, 0, "error", str(e))
            raise
