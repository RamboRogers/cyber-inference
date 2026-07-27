"""
Model management for Cyber-Inference.

Handles:
- HuggingFace model discovery and download (GGUF and transformers formats)
- Local model registration and tracking
- Model metadata extraction
- Download progress tracking
"""

import asyncio
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from huggingface_hub import HfApi, hf_hub_url, list_repo_files, snapshot_download
from sqlalchemy import select
from tqdm.auto import tqdm as base_tqdm

from cyber_inference.core.config import LEGACY_MTP_DRAFT_N_MAX, get_settings
from cyber_inference.core.database import get_db_session
from cyber_inference.core.logging import get_logger
from cyber_inference.models.db_models import Model

logger = get_logger(__name__)

GGUF_STRING_TYPE = 8
GGUF_ARRAY_TYPE = 9
HUGGINGFACE_HOSTS = {"huggingface.co", "www.huggingface.co"}


def normalize_huggingface_reference(
    repo_id: str,
    filename: str | None = None,
) -> tuple[str, str | None]:
    """Normalize a HuggingFace repo ID or model URL into canonical download inputs."""
    cleaned = repo_id.strip()
    normalized_filename = filename.strip() if filename else None
    if not cleaned:
        raise ValueError("HuggingFace repository ID or URL is required")

    parsed = urlparse(cleaned)
    if parsed.scheme or parsed.netloc:
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or hostname not in HUGGINGFACE_HOSTS:
            raise ValueError(
                "Expected a HuggingFace repository ID or "
                "https://huggingface.co/<owner>/<repository> URL"
            )
        parts = [part for part in parsed.path.split("/") if part]
    else:
        without_fragment = cleaned.split("#", 1)[0]
        without_query = without_fragment.split("?", 1)[0]
        parts = [part for part in without_query.split("/") if part]

    if len(parts) < 2:
        raise ValueError(
            "HuggingFace repository reference must include both an owner and repository"
        )

    normalized_repo_id = "/".join(parts[:2])
    remainder = parts[2:]
    if not remainder:
        return normalized_repo_id, normalized_filename

    if remainder[0] in {"blob", "resolve", "tree"}:
        if len(remainder) < 3:
            if remainder[0] == "tree" and len(remainder) == 2:
                return normalized_repo_id, normalized_filename
            raise ValueError("Direct HuggingFace model URL must include a filename")
        remainder = remainder[2:]

    direct_filename = "/".join(remainder)
    if direct_filename.lower().endswith((".gguf", ".bin")):
        return normalized_repo_id, normalized_filename or direct_filename

    raise ValueError(
        "HuggingFace URL must point to a repository or a direct .gguf/.bin model file"
    )


@dataclass(frozen=True)
class GgufShardInfo:
    """Parsed terminal GGUF shard suffix metadata."""

    prefix: str
    separator: str
    index: int
    total: int
    index_width: int
    total_width: int

    @property
    def group_key(self) -> str:
        return f"{self.prefix}{self.separator}of-{self.total:0{self.total_width}d}.gguf"

    @property
    def is_primary(self) -> bool:
        return self.index == 1

    def shard_filename(self, index: int) -> str:
        return (
            f"{self.prefix}{self.separator}"
            f"{index:0{self.index_width}d}-of-{self.total:0{self.total_width}d}.gguf"
        )


@dataclass(frozen=True)
class ModelDownloadResult:
    """Result returned by GGUF downloads after registration."""

    path: Path
    model_name: str
    filename: str
    size_bytes: int
    is_split_gguf: bool = False
    shard_filenames: list[str] = field(default_factory=list)


class ModelManager:
    """
    Manages model discovery, download, and registration.

    Integrates with HuggingFace Hub for model downloads
    and maintains local database of available models.
    """

    _gguf_metadata_cache: dict[str, tuple[int, int, dict[str, Any] | None]] = {}

    def __init__(self, models_dir: Path | None = None):
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

    def _path_for_storage(self, path: Path | None) -> str | None:
        """Return the DB storage form for a path."""
        if path is None:
            return None

        path_obj = Path(path)
        if not path_obj.is_absolute():
            return path_obj.as_posix()

        try:
            rel = path_obj.resolve(strict=False).relative_to(self.models_dir.resolve(strict=False))
            return rel.as_posix()
        except ValueError:
            return str(path_obj)

    def _resolve_stored_path(self, path_str: str | None) -> Path | None:
        """Resolve a stored DB path string to a usable filesystem path."""
        if not path_str:
            return None

        path_obj = Path(path_str)
        if path_obj.is_absolute():
            return path_obj
        return (self.models_dir / path_obj).resolve(strict=False)

    @staticmethod
    def _read_gguf_metadata_summary(file_path: Path) -> dict[str, Any] | None:
        def read_exact(handle, size: int) -> bytes:
            data = handle.read(size)
            if len(data) != size:
                raise ValueError("Unexpected end of file")
            return cast(bytes, data)

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

        def is_mtp_key(key: str) -> bool:
            if key == "nextn_predict_layers":
                return True
            return key.endswith(".nextn_predict_layers")

        def skip_array(handle, elem_type: int, length: int) -> None:
            if elem_type == GGUF_STRING_TYPE:
                for _ in range(length):
                    skip_string(handle)
                return
            elem_size = type_sizes.get(elem_type)
            if elem_size is None:
                raise ValueError("Unknown GGUF array element type")
            read_exact(handle, elem_size * length)

        def skip_value(handle, value_type: int) -> None:
            if value_type == GGUF_STRING_TYPE:
                skip_string(handle)
                return
            if value_type == GGUF_ARRAY_TYPE:
                elem_type = read_u32(handle)
                length = read_u64(handle)
                skip_array(handle, elem_type, length)
                return
            size = type_sizes.get(value_type)
            if size is None:
                raise ValueError("Unknown GGUF value type")
            read_exact(handle, size)

        def read_numeric_value(handle, value_type: int) -> int | None:
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
            if value_type == GGUF_STRING_TYPE:
                string_value = read_string(handle)
                if string_value.isdigit():
                    return int(string_value)
                return None
            if value_type == GGUF_ARRAY_TYPE:
                elem_type = read_u32(handle)
                length = read_u64(handle)
                if length == 1:
                    return read_numeric_value(handle, elem_type)
                skip_array(handle, elem_type, length)
                return None
            skip_value(handle, value_type)
            return None

        def select_context_length(
            architecture: str | None,
            candidates: dict[str, int],
        ) -> int | None:
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

        def summarize_tool_metadata(
            architecture: str | None,
            chat_template: str | None,
            response_schema: str | None,
            mtp_nextn_predict_layers: int | None,
        ) -> dict[str, Any]:
            chat_template_lower = chat_template.lower() if chat_template else ""
            response_schema_lower = response_schema.lower() if response_schema else ""
            mtp_capable = (
                mtp_nextn_predict_layers is not None and mtp_nextn_predict_layers > 0
            )
            return {
                "architecture": architecture,
                "context_length": select_context_length(architecture, candidates),
                "mtp_capable": mtp_capable,
                "mtp_detection_source": (
                    "metadata_nextn_predict_layers" if mtp_capable else None
                ),
                "mtp_nextn_predict_layers": mtp_nextn_predict_layers,
                "has_chat_template": bool(chat_template),
                "has_tool_call_tokens": any(
                    marker in chat_template_lower
                    for marker in ("<tool_call", "<|tool_call", "</tool_call>", "</|tool_call|>")
                ),
                "has_tool_response_tokens": any(
                    marker in chat_template_lower
                    for marker in ("<tool_response", "<|tool_response", "</tool_response>", "</|tool_response|>")
                ),
                "has_response_schema_tool_calls": "tool_calls" in response_schema_lower,
                "has_gemma4_tool_parser": "gemma4-tool-call" in (
                    f"{chat_template_lower}\n{response_schema_lower}"
                ),
            }

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
                chat_template: str | None = None
                response_schema: str | None = None
                mtp_nextn_predict_layers: int | None = None

                for _ in range(kv_count):
                    key = read_string(handle)
                    key_lower = key.lower()
                    value_type = read_u32(handle)

                    if key_lower == "general.architecture":
                        if value_type == GGUF_STRING_TYPE:
                            architecture = read_string(handle).lower()
                        else:
                            skip_value(handle, value_type)
                        continue

                    if is_context_key(key_lower):
                        value = read_numeric_value(handle, value_type)
                        if value is not None:
                            candidates[key_lower] = int(value)
                        continue

                    if is_mtp_key(key_lower):
                        value = read_numeric_value(handle, value_type)
                        if value is not None:
                            mtp_nextn_predict_layers = int(value)
                        continue

                    if key_lower.endswith("chat_template") and value_type == GGUF_STRING_TYPE:
                        string_value = read_string(handle)
                        if string_value and chat_template is None:
                            chat_template = string_value
                        continue

                    if key_lower.endswith("response_schema") and value_type == GGUF_STRING_TYPE:
                        string_value = read_string(handle)
                        if string_value and response_schema is None:
                            response_schema = string_value
                        continue

                    skip_value(handle, value_type)

                return summarize_tool_metadata(
                    architecture,
                    chat_template,
                    response_schema,
                    mtp_nextn_predict_layers,
                )
        except Exception as e:
            logger.debug(f"Failed to read GGUF metadata from {file_path.name}: {e}")
            return None

        return None

    @classmethod
    def _read_gguf_metadata_summary_cached(cls, file_path: Path) -> dict[str, Any] | None:
        """Read GGUF metadata with a file-metadata cache."""
        try:
            stat = file_path.stat()
        except OSError:
            return None

        cache_key = str(file_path)
        cached = cls._gguf_metadata_cache.get(cache_key)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            cached_summary = cached[2]
            return dict(cached_summary) if isinstance(cached_summary, dict) else None

        summary = cls._read_gguf_metadata_summary(file_path)
        cls._gguf_metadata_cache[cache_key] = (
            stat.st_mtime_ns,
            stat.st_size,
            dict(summary) if isinstance(summary, dict) else None,
        )
        return summary

    @classmethod
    def _read_gguf_context_length(cls, file_path: Path) -> int | None:
        summary = cls._read_gguf_metadata_summary_cached(file_path)
        context_length = summary.get("context_length") if summary else None
        return int(context_length) if isinstance(context_length, int) else None

    @staticmethod
    def _has_mtp_identity(identity: str) -> bool:
        lowered = identity.lower()
        return (
            "-mtp" in lowered
            or "_mtp" in lowered
            or "/mtp" in lowered
            or "mtp-gguf" in lowered
            or "nextn" in lowered
        )

    @classmethod
    def _is_mtp_candidate(
        cls,
        repo_id: str | None = None,
        filename: str | None = None,
        metadata_summary: dict[str, Any] | None = None,
    ) -> bool:
        if metadata_summary and metadata_summary.get("mtp_capable") is True:
            return True
        identity = " ".join(part for part in (repo_id, filename) if part)
        return cls._has_mtp_identity(identity)

    @classmethod
    def _resolve_mtp_metadata(
        cls,
        repo_id: str | None,
        filename: str | None,
        metadata_summary: dict[str, Any] | None,
        mtp_draft_path: Path | None = None,
        mtp_draft_metadata_summary: dict[str, Any] | None = None,
    ) -> dict[str, object | None]:
        metadata_summary = metadata_summary or {}
        mtp_draft_metadata_summary = mtp_draft_metadata_summary or {}
        if mtp_draft_path is not None:
            nextn_layers = mtp_draft_metadata_summary.get("mtp_nextn_predict_layers")
            return {
                "mtp_capable": True,
                "mtp_detection_source": "separate",
                "mtp_nextn_predict_layers": (
                    int(nextn_layers)
                    if isinstance(nextn_layers, int) and nextn_layers > 0
                    else None
                ),
            }
        metadata_capable = metadata_summary.get("mtp_capable") is True
        nextn_layers = metadata_summary.get("mtp_nextn_predict_layers")
        mtp_nextn_predict_layers = (
            int(nextn_layers) if isinstance(nextn_layers, int) and nextn_layers > 0 else None
        )
        if metadata_capable:
            return {
                "mtp_capable": True,
                "mtp_detection_source": "embedded",
                "mtp_nextn_predict_layers": mtp_nextn_predict_layers,
            }
        if cls._is_mtp_candidate(repo_id, filename):
            return {
                "mtp_capable": True,
                "mtp_detection_source": "repo_or_filename",
                "mtp_nextn_predict_layers": None,
            }
        return {
            "mtp_capable": False,
            "mtp_detection_source": None,
            "mtp_nextn_predict_layers": None,
        }

    @staticmethod
    def _read_transformers_context_length(model_dir: Path) -> int | None:
        """Read context length from a HuggingFace transformers model's config.json."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None
        try:
            import json
            config = json.loads(config_path.read_text())
            context_keys = (
                "max_position_embeddings",
                "n_positions",
                "max_sequence_length",
                "seq_length",
                "sliding_window",
            )
            # Check top-level config first
            for key in context_keys:
                value = config.get(key)
                if isinstance(value, int) and value > 0:
                    logger.debug(f"Transformers context length from {key}: {value}")
                    return value
            # VLM models (Qwen3-VL etc.) nest context length under text_config
            for nested_key in ("text_config", "language_config", "llm_config"):
                nested = config.get(nested_key, {})
                if not isinstance(nested, dict):
                    continue
                for key in context_keys:
                    value = nested.get(key)
                    if isinstance(value, int) and value > 0:
                        logger.debug(f"Transformers context length from {nested_key}.{key}: {value}")
                        return value
        except Exception as e:
            logger.debug(f"Failed to read config.json from {model_dir}: {e}")
        return None

    @staticmethod
    def _detect_vlm_from_config(model_dir: Path) -> str | None:
        """Detect VLM model type from config.json architectures/vision_config."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return None
        try:
            import json
            config = json.loads(config_path.read_text())
            if "vision_config" in config:
                return "vlm"
            for arch in config.get("architectures", []):
                if "VL" in arch or "Vision" in arch:
                    return "vlm"
        except Exception:
            pass
        return None

    @staticmethod
    def _is_mmproj_file(filename: str) -> bool:
        name = filename.lower()
        return name.endswith(".gguf") and "mmproj" in name

    @staticmethod
    def _is_mtp_draft_file(filename: str) -> bool:
        """Return whether a GGUF is a separate MTP draft-model artifact."""
        name = Path(filename).name.lower()
        return name.endswith(".gguf") and bool(re.match(r"^mtp(?:[-_.]|$)", name))

    @staticmethod
    def _is_dflash_draft_file(filename: str) -> bool:
        """Return whether a GGUF is a separate DFlash draft-model artifact."""
        name = Path(filename).name.lower()
        return name.endswith(".gguf") and bool(re.match(r"^dflash(?:[-_.]|$)", name))

    @classmethod
    def _artifact_type(cls, filename: str) -> str:
        """Classify repository/local GGUF artifacts by runtime role."""
        if cls._is_mmproj_file(filename):
            return "mmproj"
        if cls._is_mtp_draft_file(filename):
            return "mtp"
        if cls._is_dflash_draft_file(filename):
            return "dflash"
        return "model"

    @staticmethod
    def _parse_gguf_shard_filename(filename: str) -> GgufShardInfo | None:
        """Parse terminal split-GGUF suffixes like Model-00001-of-00003.gguf."""
        if ModelManager._artifact_type(filename) != "model":
            return None

        match = re.match(
            r"^(?P<prefix>.+?)(?P<separator>[._-])"
            r"(?P<index>\d+)-of-(?P<total>\d+)\.gguf$",
            filename,
            re.IGNORECASE,
        )
        if not match:
            return None

        index_text = match.group("index")
        total_text = match.group("total")
        index = int(index_text)
        total = int(total_text)
        if index < 1 or total < 2 or index > total:
            return None

        return GgufShardInfo(
            prefix=match.group("prefix"),
            separator=match.group("separator"),
            index=index,
            total=total,
            index_width=len(index_text),
            total_width=len(total_text),
        )

    @staticmethod
    def _canonical_split_gguf_name(filename: str) -> str:
        """Return the canonical model name for a split GGUF filename."""
        shard = ModelManager._parse_gguf_shard_filename(filename)
        if shard:
            return shard.prefix
        return Path(filename).stem

    @staticmethod
    def _is_complete_local_file(path: Path, expected_size: int | None) -> bool:
        if not path.exists() or not path.is_file():
            return False
        if expected_size is None:
            return True
        return path.stat().st_size == expected_size

    @staticmethod
    def _group_gguf_model_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group terminal split-GGUF shards into one logical model option."""
        singles: list[dict[str, Any]] = []
        groups: dict[str, dict[str, Any]] = {}

        for file_info in files:
            filename = str(file_info["filename"])
            shard = ModelManager._parse_gguf_shard_filename(filename)
            if not shard:
                item = dict(file_info)
                item.setdefault("is_split", False)
                item.setdefault("shard_count", None)
                item.setdefault("shard_total_size_bytes", None)
                item.setdefault("shard_filenames", [])
                item.setdefault("primary_filename", None)
                item.setdefault("is_complete", True)
                item.setdefault("missing_shard_filenames", [])
                singles.append(item)
                continue

            group = groups.setdefault(
                shard.group_key,
                {
                    "shard": shard,
                    "items": {},
                },
            )
            group["items"][shard.index] = dict(file_info)

        grouped: list[dict[str, Any]] = []
        for group in groups.values():
            group_shard = group["shard"]
            assert isinstance(group_shard, GgufShardInfo)
            items = group["items"]
            known_indexes = sorted(items)
            primary_filename = group_shard.shard_filename(1)
            primary_item = items.get(1) or items[known_indexes[0]]
            shard_filenames = [items[index]["filename"] for index in known_indexes]
            missing = [
                group_shard.shard_filename(index)
                for index in range(1, group_shard.total + 1)
                if index not in items
            ]
            total_size = sum(int(items[index].get("size_bytes") or 0) for index in known_indexes)

            grouped_item = dict(primary_item)
            grouped_item.update(
                {
                    "filename": primary_filename,
                    "size_bytes": total_size,
                    "quantization": ModelManager._extract_quant_suffix(primary_filename),
                    "is_mmproj": False,
                    "is_split": True,
                    "shard_count": group_shard.total,
                    "shard_total_size_bytes": total_size,
                    "shard_filenames": shard_filenames,
                    "primary_filename": primary_filename,
                    "is_complete": not missing,
                    "missing_shard_filenames": missing,
                    "shard_sizes": {
                        items[index]["filename"]: int(items[index].get("size_bytes") or 0)
                        for index in known_indexes
                    },
                }
            )
            grouped.append(grouped_item)

        return sorted(singles + grouped, key=lambda item: str(item["filename"]).lower())

    @staticmethod
    def _local_split_gguf_metadata(file_path: Path) -> dict[str, Any] | None:
        """Return local split-GGUF metadata for a shard path when a primary shard exists."""
        shard = ModelManager._parse_gguf_shard_filename(file_path.name)
        if not shard:
            return None

        primary_filename = shard.shard_filename(1)
        primary_path = file_path.parent / primary_filename
        if not primary_path.exists():
            return None

        shard_paths = [
            file_path.parent / shard.shard_filename(index)
            for index in range(1, shard.total + 1)
        ]
        existing_paths = [path for path in shard_paths if path.exists()]
        return {
            "primary_filename": primary_filename,
            "primary_path": primary_path,
            "shard_filenames": [path.name for path in existing_paths],
            "shard_count": shard.total,
            "size_bytes": sum(path.stat().st_size for path in existing_paths if path.is_file()),
            "is_complete": len(existing_paths) == shard.total,
        }

    @staticmethod
    def _split_repo_and_filename(
        repo_id: str,
        filename: str | None,
    ) -> tuple[str, str | None]:
        return normalize_huggingface_reference(repo_id, filename)

    @staticmethod
    def _new_download_id() -> str:
        """Create a unique download session identifier."""
        return f"dl-{uuid.uuid4().hex}"

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
    def _extract_quant_suffix(filename: str) -> str | None:
        """
        Extract the quantization suffix from a filename.

        Returns lowercase suffix like 'bf16', 'f16', 'q8_0', 'q4_k_m', etc.
        """
        shard = ModelManager._parse_gguf_shard_filename(filename)
        stem = (shard.prefix if shard else Path(filename).stem).lower()
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

    @classmethod
    def _normalized_artifact_model_name(cls, filename: str) -> str:
        """Normalize an artifact filename to the target model identity it belongs to."""
        artifact_type = cls._artifact_type(filename)
        basename = Path(filename).name
        if artifact_type in {"mtp", "dflash"}:
            basename = re.sub(
                rf"^{artifact_type}(?:[-_.]+)",
                "",
                basename,
                flags=re.IGNORECASE,
            )
        shard = cls._parse_gguf_shard_filename(basename)
        if shard is not None:
            basename = f"{shard.prefix}.gguf"
        base = cls._extract_model_base_name(basename)
        return re.sub(r"[^a-z0-9]+", "", base.lower())

    @classmethod
    def _select_mtp_file(cls, files: list[str], model_filename: str) -> str | None:
        """Select the separate MTP draft model that best matches a target GGUF."""
        mtp_files = sorted(f for f in files if cls._is_mtp_draft_file(f))
        if not mtp_files:
            return None

        model_identity = cls._normalized_artifact_model_name(model_filename)
        matching = [
            candidate
            for candidate in mtp_files
            if cls._normalized_artifact_model_name(candidate) == model_identity
        ]
        if not matching:
            return None

        model_quant = cls._extract_quant_suffix(model_filename)
        if model_quant:
            for candidate in matching:
                if cls._extract_quant_suffix(candidate) == model_quant:
                    return candidate

        for preferred_quant in ("q4_0", "q8_0", "bf16", "f16"):
            for candidate in matching:
                if cls._extract_quant_suffix(candidate) == preferred_quant:
                    return candidate

        return sorted(matching, key=lambda value: (len(Path(value).name), value.lower()))[0]

    def get_suggested_mtp(self, model_filename: str, mtp_files: list[str]) -> str | None:
        """Return the best matching separate MTP draft model."""
        return self._select_mtp_file(mtp_files, model_filename)

    @classmethod
    def _mtp_draft_matches_target(cls, model_path: Path, draft_path: Path) -> bool:
        """Return whether a separate MTP head belongs to the target model family."""
        return cls._normalized_artifact_model_name(
            model_path.name
        ) == cls._normalized_artifact_model_name(draft_path.name)

    def _find_local_mtp_draft(self, model_path: Path) -> Path | None:
        """Find a validated sibling MTP draft model for a local target GGUF."""
        candidates = [
            path
            for path in model_path.parent.glob("*.gguf")
            if self._is_mtp_draft_file(path.name)
        ]
        selected = self._select_mtp_file(
            [path.name for path in candidates],
            model_path.name,
        )
        if not selected:
            return None
        draft_path = model_path.parent / selected
        metadata = self._read_gguf_metadata_summary_cached(draft_path) or {}
        return draft_path if metadata.get("mtp_capable") is True else None

    @staticmethod
    def _select_mmproj_file(files: list[str], model_filename: str) -> str | None:
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

        def pick_by_quant_preference(candidates: list[str], model_quant: str | None) -> str | None:
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
        repo_files: list[str] | None = None,
        mmproj_filename: str | None = None,
        force: bool = False,
        download_id: str | None = None,
    ) -> Path | None:
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
            expected_size = None
            try:
                repo_tree = await asyncio.to_thread(
                    self._hf_api.list_repo_tree,
                    repo_id,
                    recursive=True,
                )
                for item in repo_tree:
                    if getattr(item, "path", None) == mmproj_filename:
                        expected_size = getattr(item, "size", None)
                        break
            except Exception:
                pass

            downloaded_path = await self._download_file_with_progress(
                repo_id=repo_id,
                filename=mmproj_filename,
                local_path=local_path,
                expected_size=expected_size,
                status_label="Downloading mmproj",
                download_id=download_id,
                phase="downloading_mmproj",
            )
            logger.info(f"[success]mmproj download complete: {local_path}[/success]")
            return Path(downloaded_path)
        except Exception as e:
            logger.warning(f"[warning]mmproj download failed: {e}[/warning]")
            return None

    def _validate_mtp_draft_path(self, model_path: Path, draft_path: Path) -> dict[str, Any]:
        """Validate that a separate draft file is a distinct MTP-capable GGUF."""
        if draft_path.resolve(strict=False) == model_path.resolve(strict=False):
            raise ValueError("MTP draft model must be different from the target model")
        if not draft_path.exists() or not draft_path.is_file():
            raise ValueError(f"MTP draft model file is missing: {draft_path}")
        if not self._mtp_draft_matches_target(model_path, draft_path):
            raise ValueError(
                f"MTP draft model {draft_path.name} does not match target model "
                f"{model_path.name}"
            )
        metadata = self._read_gguf_metadata_summary_cached(draft_path) or {}
        if metadata.get("mtp_capable") is not True:
            raise ValueError(
                "Selected MTP draft file does not contain nextn_predict_layers metadata: "
                f"{draft_path.name}"
            )
        return metadata

    async def _download_mtp_draft(
        self,
        repo_id: str,
        model_path: Path,
        mtp_filename: str,
        repo_files: list[str],
        force: bool = False,
        download_id: str | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """Download and validate a required separate MTP draft model."""
        if not self._is_mtp_draft_file(mtp_filename):
            raise ValueError(f"Selected file is not an MTP draft GGUF: {mtp_filename}")
        if mtp_filename not in repo_files:
            raise ValueError(f"MTP draft file not found in repo: {mtp_filename}")

        local_path = self.models_dir / mtp_filename
        if self._is_complete_local_file(local_path, None) and not force:
            logger.info(f"  MTP draft model already present: {local_path}")
            return local_path, self._validate_mtp_draft_path(model_path, local_path)

        expected_size: int | None = None
        try:
            repo_tree = await asyncio.to_thread(
                self._hf_api.list_repo_tree,
                repo_id,
                recursive=True,
            )
            for item in repo_tree:
                if getattr(item, "path", None) == mtp_filename:
                    size = getattr(item, "size", None)
                    expected_size = int(size) if isinstance(size, int) else None
                    break
        except Exception:
            pass

        logger.info(f"  Downloading MTP draft model: {mtp_filename}")
        try:
            downloaded_path = await self._download_file_with_progress(
                repo_id=repo_id,
                filename=mtp_filename,
                local_path=local_path,
                expected_size=expected_size,
                status_label="Downloading MTP draft model",
                download_id=download_id,
                phase="downloading_mtp",
            )
            metadata = self._validate_mtp_draft_path(model_path, Path(downloaded_path))
            logger.info(f"[success]MTP draft model download complete: {local_path}[/success]")
            return Path(downloaded_path), metadata
        except Exception:
            partial_path = local_path.with_suffix(f"{local_path.suffix}.part")
            if partial_path.exists():
                partial_path.unlink()
            raise

    async def _download_file_with_progress(
        self,
        repo_id: str,
        filename: str,
        local_path: Path,
        expected_size: int | None = None,
        status_label: str = "Downloading file",
        download_id: str | None = None,
        phase: str = "downloading",
        items_complete: int | None = None,
        items_total: int | None = None,
    ) -> Path:
        """Stream a repo file download while emitting smooth GUI progress updates."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = local_path.with_suffix(f"{local_path.suffix}.part")
        if temp_path.exists():
            temp_path.unlink()

        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            endpoint=self._hf_api.endpoint,
        )
        headers = {"user-agent": "cyber-inference/0.1.0"}
        if self._hf_token:
            headers["authorization"] = f"Bearer {self._hf_token}"

        last_notified_progress = -1.0
        last_notify_ts = 0.0
        started_at = time.monotonic()

        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                total_bytes = expected_size or int(response.headers.get("content-length") or 0) or None
                downloaded_bytes = 0
                await self._notify_progress(
                    repo_id,
                    filename,
                    0,
                    "downloading",
                    message=f"{status_label}…",
                    downloaded_bytes=0,
                    total_bytes=total_bytes,
                    download_id=download_id,
                    phase=phase,
                    items_complete=items_complete,
                    items_total=items_total,
                )
                with temp_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded_bytes += len(chunk)
                        now = time.monotonic()
                        progress = (
                            min(100.0, (downloaded_bytes / total_bytes) * 100.0)
                            if total_bytes
                            else 0.0
                        )
                        if (
                            total_bytes is None
                            or progress - last_notified_progress >= 1.0
                            or now - last_notify_ts >= 0.5
                        ):
                            elapsed = max(now - started_at, 0.001)
                            bytes_per_second = downloaded_bytes / elapsed
                            await self._notify_progress(
                                repo_id,
                                filename,
                                progress,
                                "downloading",
                                message=status_label,
                                downloaded_bytes=downloaded_bytes,
                                total_bytes=total_bytes,
                                bytes_per_second=bytes_per_second,
                                download_id=download_id,
                                phase=phase,
                                items_complete=items_complete,
                                items_total=items_total,
                            )
                            last_notified_progress = progress
                            last_notify_ts = now

        if local_path.exists():
            local_path.unlink()
        temp_path.rename(local_path)
        return local_path

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

    async def list_repo_files(self, repo_id: str, files: list[str] | None = None) -> list[dict]:
        """
        List GGUF files in a HuggingFace repository.

        Args:
            repo_id: HuggingFace repository ID
            files: Optional pre-fetched list of filenames (sizes will be fetched separately)

        Returns:
            List of file information dicts
        """
        repo_id, _ = self._split_repo_and_filename(repo_id, None)
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
                is_gguf = (
                    filename.endswith(".gguf")
                    and self._artifact_type(filename) == "model"
                )
                is_whisper_bin = filename.startswith("ggml-") and filename.endswith(".bin")

                if is_gguf or is_whisper_bin:
                    size = file_sizes.get(filename, 0)
                    quantization = self._extract_quant_suffix(filename)

                    model_files.append({
                        "filename": filename,
                        "size_bytes": size,
                        "quantization": quantization,
                        "artifact_type": "model",
                    })

            model_files = self._group_gguf_model_files(model_files)
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
        repo_id, _ = self._split_repo_and_filename(repo_id, None)
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
            mtp_files = []

            for filename, size in file_sizes.items():
                # Support both GGUF files and whisper.cpp bin files
                is_gguf = filename.endswith(".gguf")
                is_whisper_bin = filename.startswith("ggml-") and filename.endswith(".bin")

                if not (is_gguf or is_whisper_bin):
                    continue

                quantization = self._extract_quant_suffix(filename)
                artifact_type = self._artifact_type(filename) if is_gguf else "model"

                file_info = {
                    "filename": filename,
                    "size_bytes": size,
                    "quantization": quantization,
                    "artifact_type": artifact_type,
                    "is_mmproj": artifact_type == "mmproj",
                }

                if artifact_type == "mmproj":
                    mmproj_files.append(file_info)
                elif artifact_type == "mtp":
                    mtp_files.append(file_info)
                elif artifact_type == "model":
                    model_files.append(file_info)

            model_files = self._group_gguf_model_files(model_files)

            is_multimodal = len(mmproj_files) > 0
            embedded_mtp_candidate = self._is_mtp_candidate(
                repo_id,
                " ".join(str(f["filename"]) for f in model_files),
            )
            is_mtp_candidate = bool(mtp_files) or embedded_mtp_candidate

            # Auto-suggest best model file (prefer Q4_K_M, then others)
            suggested_model: str | None = None
            if model_files:
                preferred_quants = (
                    ["q4_k_xl", "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q4_0", "q8_0"]
                    if embedded_mtp_candidate
                    else ["q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q4_0", "q8_0"]
                )
                complete_files = [f for f in model_files if f.get("is_complete", True)]
                for quant in preferred_quants:
                    for f in complete_files:
                        quantization_value = f.get("quantization")
                        if (
                            isinstance(quantization_value, str)
                            and quant == quantization_value.lower()
                        ):
                            suggested_model = str(f["filename"])
                            break
                    if suggested_model:
                        break
                if not suggested_model and complete_files:
                    suggested_model = str(complete_files[0]["filename"])

            # Auto-suggest mmproj for the suggested model
            suggested_mmproj = None
            if suggested_model and mmproj_files and not is_mtp_candidate:
                suggested_mmproj = self._select_mmproj_file(
                    [str(f["filename"]) for f in mmproj_files],
                    suggested_model
                )
            suggested_mtp = (
                self._select_mtp_file(
                    [str(f["filename"]) for f in mtp_files],
                    suggested_model,
                )
                if suggested_model
                else None
            )

            result = {
                "repo_id": repo_id,
                "model_files": model_files,
                "mmproj_files": mmproj_files,
                "mtp_files": mtp_files,
                "is_multimodal": is_multimodal,
                "suggested_model": suggested_model,
                "suggested_mmproj": suggested_mmproj,
                "suggested_mtp": suggested_mtp,
                "is_mtp_candidate": is_mtp_candidate,
                "mtp_default_enabled": is_mtp_candidate,
                "suggested_mtp_mode": "auto" if is_mtp_candidate else None,
                "suggested_spec_draft_n_max": (
                    get_settings().llama_mtp_default_draft_n_max
                    if is_mtp_candidate
                    else None
                ),
            }

            logger.info(
                "[success]Found "
                f"{len(model_files)} model files, {len(mmproj_files)} mmproj files, "
                f"and {len(mtp_files)} MTP draft files[/success]"
            )
            return result

        except Exception as e:
            logger.error(f"[error]Failed to list repo files: {e}[/error]")
            raise

    def get_suggested_mmproj(self, model_filename: str, mmproj_files: list[str]) -> str | None:
        """
        Get the suggested mmproj file for a given model filename.

        Args:
            model_filename: The model filename to match
            mmproj_files: List of available mmproj filenames

        Returns:
            Suggested mmproj filename or None
        """
        return self._select_mmproj_file(mmproj_files, model_filename)

    def _resolve_repo_model_file(
        self,
        files: list[dict[str, Any]],
        filename: str,
    ) -> dict[str, Any] | None:
        """Find a logical repo model option by primary or shard filename."""
        for file_info in files:
            if file_info["filename"] == filename:
                return file_info
            if file_info.get("primary_filename") == filename:
                return file_info
            if filename in file_info.get("shard_filenames", []):
                return file_info
        return None

    async def download_model(
        self,
        repo_id: str,
        filename: str | None = None,
        mmproj_filename: str | None = None,
        mtp_filename: str | None = None,
        force: bool = False,
        download_id: str | None = None,
    ) -> ModelDownloadResult:
        """
        Download a model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID
            filename: Specific file to download (auto-detect if None)
            mmproj_filename: Specific mmproj file to download for vision models (auto-detect if None)
            mtp_filename: Specific separate MTP draft model (auto-detect if available)
            force: Force redownload even if exists

        Returns:
            Download result with canonical model identity
        """
        download_id = download_id or self._new_download_id()
        filename = filename.strip() if filename else None
        mmproj_filename = mmproj_filename.strip() if mmproj_filename else None
        mtp_filename = mtp_filename.strip() if mtp_filename else None
        parsed = self._split_repo_and_filename(repo_id, filename)
        repo_id, filename = parsed

        logger.info(f"[highlight]Downloading model from: {repo_id}[/highlight]")
        repo_files = list_repo_files(repo_id, token=self._hf_token)

        files = await self.list_repo_files(repo_id, files=repo_files)
        repo_mtp_candidate = self._is_mtp_candidate(
            repo_id,
            " ".join(str(file.get("filename") or "") for file in files),
        )

        # If no filename specified, find the best GGUF file
        if filename is None:
            if not files:
                raise ValueError(f"No GGUF files found in {repo_id}")

            # Prefer Q4_K_M or similar balanced quantization
            preferred_quants = (
                ["Q4_K_XL", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0"]
                if repo_mtp_candidate
                else ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q4_0"]
            )
            complete_files = [file for file in files if file.get("is_complete", True)]
            if not complete_files:
                raise ValueError(f"No complete GGUF files found in {repo_id}")

            filename = complete_files[0]["filename"]

            for quant in preferred_quants:
                for f in complete_files:
                    if quant.lower() in f["filename"].lower():
                        filename = f["filename"]
                        break
                else:
                    continue
                break

            logger.info(f"  Auto-selected file: {filename}")

        selected_file = self._resolve_repo_model_file(files, filename)
        if not selected_file:
            raise ValueError(f"GGUF file not found in {repo_id}: {filename}")
        if not selected_file.get("is_complete", True):
            missing = ", ".join(selected_file.get("missing_shard_filenames", []))
            detail = f" Missing shards: {missing}" if missing else ""
            raise ValueError(f"Split GGUF shard set is incomplete for {filename}.{detail}")

        is_split = bool(selected_file.get("is_split"))
        primary_filename = str(selected_file.get("primary_filename") or selected_file["filename"])
        filename = primary_filename
        selected_mtp_filename = mtp_filename or self._select_mtp_file(repo_files, filename)
        if mtp_filename:
            if not self._is_mtp_draft_file(mtp_filename):
                raise ValueError(f"Selected file is not an MTP draft GGUF: {mtp_filename}")
            if self._normalized_artifact_model_name(mtp_filename) != (
                self._normalized_artifact_model_name(filename)
            ):
                raise ValueError(
                    f"MTP draft file {mtp_filename} does not match target model {filename}"
                )
        is_mtp_candidate = (
            selected_mtp_filename is not None
            or self._is_mtp_candidate(repo_id, filename)
        )
        should_download_mmproj = mmproj_filename is not None or not is_mtp_candidate
        expected_size = int(selected_file["size_bytes"]) if selected_file else None
        shard_filenames = list(selected_file.get("shard_filenames") or [filename])
        shard_sizes = {
            str(name): int(size or 0)
            for name, size in (selected_file.get("shard_sizes") or {}).items()
        }
        model_name = (
            self._canonical_split_gguf_name(filename)
            if is_split
            else Path(filename).stem
        )

        # Notify download starting
        await self._notify_progress(
            repo_id,
            filename,
            0,
            "starting",
            message="Preparing download…",
            total_bytes=expected_size,
            download_id=download_id,
            phase="preparing",
        )

        # Check if already downloaded
        local_path = self.models_dir / filename
        if is_split:
            for index, shard_filename in enumerate(shard_filenames, start=1):
                shard_path = self.models_dir / shard_filename
                shard_expected_size = shard_sizes.get(shard_filename) or None
                if self._is_complete_local_file(shard_path, shard_expected_size) and not force:
                    logger.info(f"  Shard already present: {shard_path}")
                    await self._notify_progress(
                        repo_id,
                        shard_filename,
                        (index / len(shard_filenames)) * 95,
                        "skipped",
                        message=f"Shard already downloaded ({index}/{len(shard_filenames)})",
                        downloaded_bytes=shard_expected_size,
                        total_bytes=shard_expected_size,
                        download_id=download_id,
                        phase="downloading_model",
                        items_complete=index,
                        items_total=len(shard_filenames),
                    )
                    continue

                await self._download_file_with_progress(
                    repo_id=repo_id,
                    filename=shard_filename,
                    local_path=shard_path,
                    expected_size=shard_expected_size,
                    status_label=f"Downloading shard {index}/{len(shard_filenames)}",
                    download_id=download_id,
                    phase="downloading_model",
                    items_complete=index - 1,
                    items_total=len(shard_filenames),
                )

            local_path = self.models_dir / primary_filename
            mtp_draft_path = None
            if selected_mtp_filename:
                mtp_draft_path, _ = await self._download_mtp_draft(
                    repo_id=repo_id,
                    model_path=local_path,
                    mtp_filename=selected_mtp_filename,
                    repo_files=repo_files,
                    force=force,
                    download_id=download_id,
                )
            mmproj_path = (
                await self._download_mmproj(
                    repo_id, local_path,
                    repo_files=repo_files,
                    mmproj_filename=mmproj_filename,
                    force=force,
                    download_id=download_id,
                )
                if should_download_mmproj
                else None
            )
            await self._notify_progress(
                repo_id,
                filename,
                100,
                "registering",
                message="Finalizing model registration…",
                downloaded_bytes=expected_size,
                total_bytes=expected_size,
                download_id=download_id,
                phase="registering",
                items_complete=len(shard_filenames),
                items_total=len(shard_filenames),
            )
            await self._register_model(
                repo_id,
                filename,
                local_path,
                mmproj_path=mmproj_path,
                mtp_draft_path=mtp_draft_path,
                model_name_override=model_name,
                size_bytes_override=expected_size,
                is_split_gguf=True,
                gguf_shard_filenames=shard_filenames,
            )
            await self._notify_progress(
                repo_id,
                filename,
                100,
                "complete",
                message="Download complete.",
                downloaded_bytes=expected_size,
                total_bytes=expected_size,
                download_id=download_id,
                phase="complete",
                items_complete=len(shard_filenames),
                items_total=len(shard_filenames),
            )
            return ModelDownloadResult(
                path=local_path,
                model_name=model_name,
                filename=filename,
                size_bytes=expected_size or 0,
                is_split_gguf=True,
                shard_filenames=shard_filenames,
            )

        if self._is_complete_local_file(local_path, expected_size) and not force:
            logger.info(f"[success]Model already exists: {local_path}[/success]")

            mtp_draft_path = None
            if selected_mtp_filename:
                mtp_draft_path, _ = await self._download_mtp_draft(
                    repo_id=repo_id,
                    model_path=local_path,
                    mtp_filename=selected_mtp_filename,
                    repo_files=repo_files,
                    force=force,
                    download_id=download_id,
                )

            # Download mmproj and get path
            mmproj_path = (
                await self._download_mmproj(
                    repo_id, local_path,
                    repo_files=repo_files,
                    mmproj_filename=mmproj_filename,
                    force=force,
                    download_id=download_id,
                )
                if should_download_mmproj
                else None
            )

            # Ensure it's registered in DB with mmproj_path
            await self._register_model(
                repo_id,
                filename,
                local_path,
                mmproj_path=mmproj_path,
                mtp_draft_path=mtp_draft_path,
            )

            # Notify complete
            await self._notify_progress(
                repo_id,
                filename,
                100,
                "complete",
                message="Model already downloaded — registration complete.",
                downloaded_bytes=expected_size,
                total_bytes=expected_size,
                download_id=download_id,
                phase="complete",
            )

            return ModelDownloadResult(
                path=local_path,
                model_name=model_name,
                filename=filename,
                size_bytes=expected_size or (local_path.stat().st_size if local_path.exists() else 0),
                is_split_gguf=False,
                shard_filenames=[],
            )

        # Download with progress
        logger.info(f"  Downloading to: {local_path}")

        try:
            await self._download_file_with_progress(
                repo_id=repo_id,
                filename=filename,
                local_path=local_path,
                expected_size=expected_size,
                status_label="Downloading model",
                download_id=download_id,
                phase="downloading_model",
            )

            logger.info(f"[success]Download complete: {local_path}[/success]")

            mtp_draft_path = None
            if selected_mtp_filename:
                mtp_draft_path, _ = await self._download_mtp_draft(
                    repo_id=repo_id,
                    model_path=local_path,
                    mtp_filename=selected_mtp_filename,
                    repo_files=repo_files,
                    force=force,
                    download_id=download_id,
                )

            # Download mmproj and get path
            mmproj_path = (
                await self._download_mmproj(
                    repo_id, local_path,
                    repo_files=repo_files,
                    mmproj_filename=mmproj_filename,
                    force=force,
                    download_id=download_id,
                )
                if should_download_mmproj
                else None
            )

            # Register in database with mmproj_path
            await self._notify_progress(
                repo_id,
                filename,
                100,
                "registering",
                message="Finalizing model registration…",
                downloaded_bytes=expected_size,
                total_bytes=expected_size,
                download_id=download_id,
                phase="registering",
            )
            await self._register_model(
                repo_id,
                filename,
                local_path,
                mmproj_path=mmproj_path,
                mtp_draft_path=mtp_draft_path,
            )

            # Notify complete
            await self._notify_progress(
                repo_id,
                filename,
                100,
                "complete",
                message="Download complete.",
                downloaded_bytes=expected_size,
                total_bytes=expected_size,
                download_id=download_id,
                phase="complete",
            )

            return ModelDownloadResult(
                path=local_path,
                model_name=model_name,
                filename=filename,
                size_bytes=expected_size or (local_path.stat().st_size if local_path.exists() else 0),
                is_split_gguf=False,
                shard_filenames=[],
            )

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
            await self._notify_progress(
                repo_id,
                filename,
                0,
                "error",
                str(e),
                download_id=download_id,
                phase="error",
            )
            raise

    async def _notify_progress(
        self,
        repo_id: str,
        filename: str,
        progress: float,
        status: str,
        error: str | None = None,
        message: str | None = None,
        downloaded_bytes: int | None = None,
        total_bytes: int | None = None,
        bytes_per_second: float | None = None,
        download_id: str | None = None,
        phase: str | None = None,
        items_complete: int | None = None,
        items_total: int | None = None,
    ) -> None:
        """Send download progress notification via WebSocket."""
        try:
            from cyber_inference.api.websocket import notify_download_progress
            await notify_download_progress(
                repo_id,
                filename,
                progress,
                status,
                error,
                message=message,
                downloaded_bytes=downloaded_bytes,
                total_bytes=total_bytes,
                bytes_per_second=bytes_per_second,
                download_id=download_id,
                phase=phase,
                items_complete=items_complete,
                items_total=items_total,
            )
        except Exception as e:
            logger.debug(f"Could not send progress notification: {e}")

    async def _reconcile_split_model_identity(
        self,
        session: Any,
        canonical_name: str,
        filename: str,
        file_path: Path,
    ) -> str:
        """Upgrade legacy shard-named rows to the canonical split-GGUF name when safe."""
        canonical_result = await session.execute(
            select(Model).where(Model.name == canonical_name)
        )
        if canonical_result.scalar_one_or_none():
            return canonical_name

        primary_shard = self._parse_gguf_shard_filename(filename)
        if not primary_shard:
            return canonical_name

        result = await session.execute(select(Model))
        candidates = []
        for model in result.scalars().all():
            names_to_check = [
                model.filename,
                Path(model.file_path).name if model.file_path else "",
                f"{model.name}.gguf" if model.name else "",
            ]
            if any(
                (shard := self._parse_gguf_shard_filename(name))
                and shard.group_key == primary_shard.group_key
                for name in names_to_check
            ):
                candidates.append(model)

        if not candidates:
            return canonical_name

        def candidate_sort_key(model: Model) -> tuple[int, int]:
            shard = self._parse_gguf_shard_filename(model.filename or "")
            return (0 if shard and shard.is_primary else 1, model.id or 0)

        candidates.sort(key=candidate_sort_key)
        winner = candidates[0]
        winner.name = canonical_name
        winner.filename = filename
        winner.file_path = self._path_for_storage(file_path) or str(file_path)
        logger.info(f"  Reconciled split GGUF model identity: {canonical_name}")
        return canonical_name

    async def _register_model(
        self,
        repo_id: str,
        filename: str,
        file_path: Path,
        mmproj_path: Path | None = None,
        mtp_draft_path: Path | None = None,
        engine_type: str | None = None,
        model_name_override: str | None = None,
        size_bytes_override: int | None = None,
        is_split_gguf: bool = False,
        gguf_shard_filenames: list[str] | None = None,
    ) -> Model:
        """Register a model in the database.

        Args:
            repo_id: HuggingFace repository ID
            filename: Model filename
            file_path: Local file path
            mmproj_path: Optional mmproj file path for vision models
            mtp_draft_path: Optional separate MTP draft-model path
            engine_type: Engine type ('llama', 'whisper', 'transformers')
            model_name_override: Override the auto-generated model name
        """
        logger.debug(f"Registering model: {filename}")

        # Extract quantization from filename using our improved method
        quantization = self._extract_quant_suffix(filename)

        # Get file size (for directory-based models, sum all files)
        if size_bytes_override is not None:
            size_bytes = size_bytes_override
        elif file_path.is_dir():
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

        # Read context length and MTP support from model metadata
        context_length = None
        metadata_summary: dict[str, Any] = {}
        if filename.endswith(".gguf") and file_path.is_file():
            metadata_summary = self._read_gguf_metadata_summary_cached(file_path) or {}
            context_value = metadata_summary.get("context_length")
            context_length = int(context_value) if isinstance(context_value, int) else None
        elif file_path.is_dir():
            context_length = self._read_transformers_context_length(file_path)
        mtp_draft_metadata: dict[str, Any] = {}
        if mtp_draft_path is not None:
            mtp_draft_metadata = self._validate_mtp_draft_path(
                file_path,
                mtp_draft_path,
            )
        mtp_metadata = self._resolve_mtp_metadata(
            repo_id,
            filename,
            metadata_summary,
            mtp_draft_path=mtp_draft_path,
            mtp_draft_metadata_summary=mtp_draft_metadata,
        )
        mtp_capable = bool(mtp_metadata["mtp_capable"])
        mtp_detection_source = cast(str | None, mtp_metadata["mtp_detection_source"])
        mtp_nextn_predict_layers = cast(int | None, mtp_metadata["mtp_nextn_predict_layers"])

        # Convert mmproj_path to string if provided
        mmproj_path_str = self._path_for_storage(mmproj_path)
        mtp_draft_path_str = self._path_for_storage(mtp_draft_path)

        # Determine engine_type if not provided
        if not engine_type:
            engine_type = "llama"  # default

        async with get_db_session() as session:
            if is_split_gguf:
                model_name = await self._reconcile_split_model_identity(
                    session=session,
                    canonical_name=model_name,
                    filename=filename,
                    file_path=file_path,
                )

            # Check if already registered
            result = await session.execute(
                select(Model).where(Model.name == model_name)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing record
                existing.file_path = self._path_for_storage(file_path) or str(file_path)
                existing.size_bytes = size_bytes
                existing.is_downloaded = True
                existing.download_progress = 100.0
                existing.engine_type = engine_type
                existing.is_split_gguf = is_split_gguf
                existing.gguf_shard_count = (
                    len(gguf_shard_filenames or []) if is_split_gguf else None
                )
                existing.gguf_shard_filenames = gguf_shard_filenames if is_split_gguf else None
                if context_length:
                    existing.context_length = context_length

                existing.mtp_capable = mtp_capable
                existing.mtp_detection_source = mtp_detection_source
                existing.mtp_nextn_predict_layers = mtp_nextn_predict_layers
                existing.mtp_draft_path = mtp_draft_path_str
                if mtp_capable:
                    if not existing.mtp_mode:
                        existing.mtp_mode = "auto"
                    if existing.mtp_spec_draft_n_max in (None, LEGACY_MTP_DRAFT_N_MAX):
                        existing.mtp_spec_draft_n_max = (
                            get_settings().llama_mtp_default_draft_n_max
                        )
                elif existing.mtp_mode not in {"enabled", "disabled"}:
                    existing.mtp_mode = None
                    existing.mtp_spec_draft_n_max = None

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
                    vlm_patterns = ["vlm", "-vl-", "-vl ", "vision", "llava", "cosmos-reason"]

                    if any(pattern in check_string for pattern in embedding_patterns):
                        existing.model_type = "embedding"
                        logger.info("  Auto-detected model type: embedding")
                    elif any(pattern in check_string for pattern in transcription_patterns):
                        existing.model_type = "transcription"
                        logger.info("  Auto-detected model type: transcription")
                    elif any(pattern in check_string for pattern in vlm_patterns):
                        existing.model_type = "vlm"
                        logger.info("  Auto-detected model type: vlm")
                    elif file_path.is_dir():
                        existing.model_type = self._detect_vlm_from_config(file_path)
                        if existing.model_type:
                            logger.info("  Auto-detected model type from config.json: vlm")

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
            vlm_patterns = ["vlm", "-vl-", "-vl ", "vision", "llava", "cosmos-reason"]

            if any(pattern in check_string for pattern in embedding_patterns):
                model_type = "embedding"
                logger.info("  Auto-detected model type: embedding")
            elif any(pattern in check_string for pattern in transcription_patterns):
                model_type = "transcription"
                logger.info("  Auto-detected model type: transcription")
            elif any(pattern in check_string for pattern in vlm_patterns):
                model_type = "vlm"
                logger.info("  Auto-detected model type: vlm")
            elif file_path.is_dir():
                model_type = self._detect_vlm_from_config(file_path)
                if model_type:
                    logger.info("  Auto-detected model type from config.json: vlm")

            # Create new record
            model = Model(
                name=model_name,
                filename=filename,
                file_path=self._path_for_storage(file_path) or str(file_path),
                hf_repo_id=repo_id,
                hf_filename=filename,
                size_bytes=size_bytes,
                quantization=quantization,
                context_length=context_length or 4096,
                model_type=model_type,
                engine_type=engine_type,
                mmproj_path=mmproj_path_str,
                mtp_draft_path=mtp_draft_path_str,
                mtp_capable=mtp_capable,
                mtp_mode="auto" if mtp_capable else None,
                mtp_detection_source=mtp_detection_source,
                mtp_nextn_predict_layers=mtp_nextn_predict_layers,
                mtp_spec_draft_n_max=(
                    get_settings().llama_mtp_default_draft_n_max if mtp_capable else None
                ),
                is_split_gguf=is_split_gguf,
                gguf_shard_count=len(gguf_shard_filenames or []) if is_split_gguf else None,
                gguf_shard_filenames=gguf_shard_filenames if is_split_gguf else None,
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

    async def list_models(self, *, include_file_metadata: bool = True) -> list[dict]:
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
            db_model_names = {model.name for model in db_models}

            updated = False
            for model in db_models:
                metadata_summary: dict[str, Any] = {}
                file_path = self._resolve_stored_path(model.file_path)
                split_metadata = (
                    self._local_split_gguf_metadata(file_path)
                    if file_path and file_path.suffix == ".gguf"
                    else None
                )
                if file_path and file_path.suffix == ".gguf" and not split_metadata:
                    shard = self._parse_gguf_shard_filename(file_path.name)
                    if shard:
                        continue
                if split_metadata:
                    canonical_name = self._canonical_split_gguf_name(
                        split_metadata["primary_filename"]
                    )
                    if model.name != canonical_name and canonical_name in db_model_names:
                        continue
                    if model.name != canonical_name and canonical_name not in db_model_names:
                        model.name = canonical_name
                        model.filename = split_metadata["primary_filename"]
                        model.file_path = (
                            self._path_for_storage(split_metadata["primary_path"])
                            or str(split_metadata["primary_path"])
                        )
                        db_model_names.add(canonical_name)
                        updated = True
                    model.is_split_gguf = True
                    model.gguf_shard_count = split_metadata["shard_count"]
                    model.gguf_shard_filenames = split_metadata["shard_filenames"]
                    model.size_bytes = split_metadata["size_bytes"]
                    updated = True

                # Backfill context_length for models that still have the 4096 default
                if model.file_path:
                    file_path = self._resolve_stored_path(model.file_path)
                    if file_path is None:
                        continue
                    if include_file_metadata and file_path.exists() and file_path.suffix == ".gguf":
                        metadata_summary = self._read_gguf_metadata_summary_cached(file_path) or {}
                    elif file_path.is_dir():
                        context_length = self._read_transformers_context_length(file_path)
                        metadata_summary = {"context_length": context_length}
                    else:
                        metadata_summary = {}

                    context_length = metadata_summary.get("context_length")
                    if isinstance(context_length, int) and context_length != model.context_length:
                        model.context_length = context_length
                        updated = True
                    mtp_draft_path = self._resolve_stored_path(model.mtp_draft_path)
                    mtp_draft_metadata: dict[str, Any] = {}
                    if include_file_metadata and mtp_draft_path is not None:
                        try:
                            mtp_draft_metadata = self._validate_mtp_draft_path(
                                file_path,
                                mtp_draft_path,
                            )
                        except ValueError as exc:
                            logger.warning(
                                f"[warning]Clearing invalid MTP draft association for "
                                f"{model.name}: {exc}[/warning]"
                            )
                            model.mtp_draft_path = None
                            mtp_draft_path = None
                            updated = True
                    if (
                        mtp_draft_path is None
                        and file_path.suffix == ".gguf"
                        and file_path.exists()
                    ):
                        mtp_draft_path = self._find_local_mtp_draft(file_path)
                        if mtp_draft_path is not None:
                            model.mtp_draft_path = self._path_for_storage(mtp_draft_path)
                            mtp_draft_metadata = (
                                self._read_gguf_metadata_summary_cached(mtp_draft_path)
                                or {}
                            )
                            updated = True
                    if not include_file_metadata and model.mtp_capable:
                        mtp_metadata: dict[str, object | None] = {
                            "mtp_capable": True,
                            "mtp_detection_source": (
                                "separate"
                                if mtp_draft_path is not None
                                else model.mtp_detection_source
                            ),
                            "mtp_nextn_predict_layers": model.mtp_nextn_predict_layers,
                        }
                    else:
                        mtp_metadata = self._resolve_mtp_metadata(
                            model.hf_repo_id,
                            model.filename,
                            metadata_summary,
                            mtp_draft_path=mtp_draft_path,
                            mtp_draft_metadata_summary=mtp_draft_metadata,
                        )
                    mtp_capable = bool(mtp_metadata["mtp_capable"])
                    mtp_detection_source = cast(str | None, mtp_metadata["mtp_detection_source"])
                    mtp_nextn_predict_layers = cast(
                        int | None,
                        mtp_metadata["mtp_nextn_predict_layers"],
                    )
                    if model.mtp_capable != mtp_capable:
                        model.mtp_capable = mtp_capable
                        updated = True
                    if model.mtp_detection_source != mtp_detection_source:
                        model.mtp_detection_source = mtp_detection_source
                        updated = True
                    if model.mtp_nextn_predict_layers != mtp_nextn_predict_layers:
                        model.mtp_nextn_predict_layers = mtp_nextn_predict_layers
                        updated = True
                    if mtp_capable and not model.mtp_mode:
                        model.mtp_mode = "auto"
                        updated = True
                    if mtp_capable and model.mtp_spec_draft_n_max in (
                        None,
                        LEGACY_MTP_DRAFT_N_MAX,
                    ):
                        model.mtp_spec_draft_n_max = (
                            get_settings().llama_mtp_default_draft_n_max
                        )
                        updated = True

                engine_type = model.engine_type or "llama"
                is_vlm = bool(model.mmproj_path)
                if engine_type == "transformers":
                    resolved_path = self._resolve_stored_path(model.file_path)
                    is_vlm = bool(resolved_path and self._detect_vlm_from_config(resolved_path) == "vlm")

                resolved_file_path = self._resolve_stored_path(model.file_path)
                resolved_mmproj_path = self._resolve_stored_path(model.mmproj_path)
                resolved_mtp_draft_path = self._resolve_stored_path(model.mtp_draft_path)

                models.append({
                    "id": model.id,
                    "name": model.name,
                    "filename": model.filename,
                    "path": str(resolved_file_path) if resolved_file_path else model.file_path,
                    "hf_repo_id": model.hf_repo_id,
                    "size_bytes": model.size_bytes,
                    "quantization": model.quantization,
                    "context_length": model.context_length,
                    "model_type": model.model_type,
                    "engine_type": engine_type,
                    "mmproj_path": (
                        str(resolved_mmproj_path) if resolved_mmproj_path else model.mmproj_path
                    ),
                    "mtp_draft_path": (
                        str(resolved_mtp_draft_path)
                        if resolved_mtp_draft_path
                        else model.mtp_draft_path
                    ),
                    "mtp_capable": model.mtp_capable,
                    "mtp_mode": model.mtp_mode,
                    "mtp_detection_source": model.mtp_detection_source,
                    "mtp_nextn_predict_layers": model.mtp_nextn_predict_layers,
                    "mtp_spec_draft_n_max": model.mtp_spec_draft_n_max,
                    "is_split_gguf": bool(model.is_split_gguf),
                    "gguf_shard_count": model.gguf_shard_count,
                    "gguf_shard_filenames": model.gguf_shard_filenames,
                    "is_vlm": is_vlm,
                    "is_downloaded": model.is_downloaded,
                    "is_enabled": model.is_enabled,
                    "download_progress": model.download_progress,
                    "created_at": model.created_at,
                    "last_used_at": model.last_used_at,
                    "registered": True,
                    "default_context_size": model.default_context_size,
                    "default_temperature": model.default_temperature,
                    "default_top_p": model.default_top_p,
                    "default_top_k": model.default_top_k,
                    "default_max_tokens": model.default_max_tokens,
                    "default_repeat_penalty": model.default_repeat_penalty,
                    "tool_template_mode": model.tool_template_mode,
                    "tool_template_name": model.tool_template_name,
                    "tool_template_path": model.tool_template_path,
                    "tool_jinja_enabled": model.tool_jinja_enabled,
                    "gguf_has_chat_template": bool(metadata_summary.get("has_chat_template")),
                    "gguf_has_tool_call_tokens": bool(metadata_summary.get("has_tool_call_tokens")),
                    "gguf_has_tool_response_tokens": bool(metadata_summary.get("has_tool_response_tokens")),
                    "gguf_has_response_schema_tool_calls": bool(metadata_summary.get("has_response_schema_tool_calls")),
                    "gguf_has_gemma4_tool_parser": bool(metadata_summary.get("has_gemma4_tool_parser")),
                    "gguf_architecture": metadata_summary.get("architecture"),
                })

            if updated:
                await session.commit()

        # Scan for unregistered local files
        registered_files = {m["filename"] for m in models}
        registered_names = {m["name"] for m in models}

        # Scan for both GGUF and whisper.cpp bin files
        for file_path in list(self.models_dir.glob("*.gguf")) + list(self.models_dir.glob("ggml-*.bin")):
            if (
                file_path.suffix == ".gguf"
                and self._artifact_type(file_path.name) != "model"
            ):
                continue
            split_metadata = (
                self._local_split_gguf_metadata(file_path)
                if file_path.suffix == ".gguf"
                else None
            )
            if file_path.suffix == ".gguf" and not split_metadata:
                shard = self._parse_gguf_shard_filename(file_path.name)
                if shard:
                    continue
            if split_metadata:
                if file_path.name != split_metadata["primary_filename"]:
                    continue
                model_name = self._canonical_split_gguf_name(file_path.name)
                if model_name in registered_names:
                    continue
            else:
                model_name = file_path.stem

            if file_path.name not in registered_files:
                metadata_summary = (
                    self._read_gguf_metadata_summary_cached(file_path) if include_file_metadata else {}
                ) or {}
                context_length = metadata_summary.get("context_length")
                mtp_draft_path = (
                    self._find_local_mtp_draft(file_path)
                    if file_path.suffix == ".gguf"
                    else None
                )
                mtp_draft_metadata = (
                    self._read_gguf_metadata_summary_cached(mtp_draft_path) or {}
                    if mtp_draft_path is not None
                    else {}
                )
                mtp_metadata = self._resolve_mtp_metadata(
                    None,
                    file_path.name,
                    metadata_summary,
                    mtp_draft_path=mtp_draft_path,
                    mtp_draft_metadata_summary=mtp_draft_metadata,
                )
                mtp_capable = bool(mtp_metadata["mtp_capable"])
                # Try to find associated mmproj file
                mmproj_path = None
                potential_mmproj = file_path.parent / f"mmproj-{file_path.stem}.gguf"
                if potential_mmproj.exists():
                    mmproj_path = str(potential_mmproj)

                models.append({
                    "id": None,
                    "name": model_name,
                    "filename": file_path.name,
                    "path": str(file_path),
                    "hf_repo_id": None,
                    "size_bytes": (
                        split_metadata["size_bytes"] if split_metadata else file_path.stat().st_size
                    ),
                    "quantization": None,
                    "context_length": context_length or 4096,
                    "model_type": None,
                    "engine_type": "llama",
                    "mmproj_path": mmproj_path,
                    "mtp_draft_path": (
                        str(mtp_draft_path) if mtp_draft_path is not None else None
                    ),
                    "mtp_capable": mtp_capable,
                    "mtp_mode": "auto" if mtp_capable else None,
                    "mtp_detection_source": mtp_metadata["mtp_detection_source"],
                    "mtp_nextn_predict_layers": mtp_metadata["mtp_nextn_predict_layers"],
                    "mtp_spec_draft_n_max": (
                        get_settings().llama_mtp_default_draft_n_max
                        if mtp_capable
                        else None
                    ),
                    "is_split_gguf": bool(split_metadata),
                    "gguf_shard_count": split_metadata["shard_count"] if split_metadata else None,
                    "gguf_shard_filenames": (
                        split_metadata["shard_filenames"] if split_metadata else None
                    ),
                    "is_vlm": bool(mmproj_path),
                    "is_downloaded": True,
                    "is_enabled": True,
                    "download_progress": 100.0,
                    "created_at": None,
                    "last_used_at": None,
                    "registered": False,
                    "default_context_size": None,
                    "default_temperature": None,
                    "default_top_p": None,
                    "default_top_k": None,
                    "default_max_tokens": None,
                    "default_repeat_penalty": None,
                    "tool_template_mode": None,
                    "tool_template_name": None,
                    "tool_template_path": None,
                    "tool_jinja_enabled": None,
                    "gguf_has_chat_template": bool(metadata_summary.get("has_chat_template")),
                    "gguf_has_tool_call_tokens": bool(metadata_summary.get("has_tool_call_tokens")),
                    "gguf_has_tool_response_tokens": bool(metadata_summary.get("has_tool_response_tokens")),
                    "gguf_has_response_schema_tool_calls": bool(metadata_summary.get("has_response_schema_tool_calls")),
                    "gguf_has_gemma4_tool_parser": bool(metadata_summary.get("has_gemma4_tool_parser")),
                    "gguf_architecture": metadata_summary.get("architecture"),
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
                    is_vlm = self._detect_vlm_from_config(model_dir) == "vlm"
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
                        "mtp_draft_path": None,
                        "mtp_capable": False,
                        "mtp_mode": None,
                        "mtp_detection_source": None,
                        "mtp_nextn_predict_layers": None,
                        "mtp_spec_draft_n_max": None,
                        "is_split_gguf": False,
                        "gguf_shard_count": None,
                        "gguf_shard_filenames": None,
                        "is_vlm": is_vlm,
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
                        "tool_template_mode": None,
                        "tool_template_name": None,
                        "tool_template_path": None,
                        "tool_jinja_enabled": None,
                        "gguf_has_chat_template": False,
                        "gguf_has_tool_call_tokens": False,
                        "gguf_has_tool_response_tokens": False,
                        "gguf_has_response_schema_tool_calls": False,
                        "gguf_has_gemma4_tool_parser": False,
                        "gguf_architecture": None,
                    })

        logger.debug(f"Found {len(models)} models")
        return models

    async def get_model(self, name: str) -> dict | None:
        """
        Get model by name.

        Args:
            name: Model name

        Returns:
            Model information dict, or None if not found
        """
        models = await self.list_models(include_file_metadata=True)
        for model in models:
            if model["name"] == name:
                return model
        return None

    async def get_model_path(self, name: str) -> Path | None:
        """
        Get the file path for a model.

        Args:
            name: Model name

        Returns:
            Path to model file, or None if not found
        """
        model = await self.get_model(name)
        if model and model["is_downloaded"]:
            return self._resolve_stored_path(model["path"])
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
        file_path = self._resolve_stored_path(model["path"])
        if file_path is None:
            logger.warning(f"Model path missing for {name}")
            return False
        mtp_draft_path = self._resolve_stored_path(model.get("mtp_draft_path"))
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
        draft_is_referenced = False
        if model["id"]:
            async with get_db_session() as session:
                draft_references = await session.execute(
                    select(Model.id, Model.mtp_draft_path).where(
                        Model.id != model["id"],
                        Model.mtp_draft_path.is_not(None),
                    )
                )
                if mtp_draft_path is not None:
                    draft_is_referenced = any(
                        (other_path := self._resolve_stored_path(row[1])) is not None
                        and other_path.resolve(strict=False)
                        == mtp_draft_path.resolve(strict=False)
                        for row in draft_references.all()
                    )
                result = await session.execute(
                    select(Model).where(Model.id == model["id"])
                )
                db_model = result.scalar_one_or_none()
                if db_model:
                    await session.delete(db_model)
                    await session.commit()
                    logger.info("  Deleted database record")

        if (
            mtp_draft_path is not None
            and not draft_is_referenced
            and not file_path.exists()
        ):
            draft_resolved = mtp_draft_path.resolve(strict=False)
            models_root = self.models_dir.resolve(strict=False)
            managed_draft = draft_resolved.is_relative_to(models_root)
            other_local_target = any(
                candidate.resolve(strict=False) != file_path.resolve(strict=False)
                and self._artifact_type(candidate.name) == "model"
                and self._mtp_draft_matches_target(candidate, draft_resolved)
                for candidate in draft_resolved.parent.glob("*.gguf")
            )
            if (
                managed_draft
                and not other_local_target
                and draft_resolved.is_file()
            ):
                try:
                    draft_resolved.unlink()
                    logger.info(f"  Deleted unreferenced MTP draft model: {draft_resolved}")
                except OSError as exc:
                    logger.warning(f"  Could not delete MTP draft model: {exc}")

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
        name: str | None = None,
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
        if file_path.suffix == ".gguf" and self._artifact_type(file_path.name) != "model":
            raise ValueError("Auxiliary GGUF files cannot be registered as primary models")

        mtp_draft_path = (
            self._find_local_mtp_draft(file_path)
            if file_path.suffix == ".gguf"
            else None
        )
        return await self._register_model(
            repo_id="local",
            filename=file_path.name,
            file_path=file_path,
            mtp_draft_path=mtp_draft_path,
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
        download_id: str | None = None,
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
        repo_id, direct_filename = self._split_repo_and_filename(repo_id, None)
        if direct_filename:
            raise ValueError(
                "Transformers downloads require a HuggingFace repository URL, "
                "not a direct model file URL"
            )
        download_id = download_id or self._new_download_id()
        model_name = self._sanitize_repo_name(repo_id)

        settings = get_settings()
        tf_dir = settings.transformers_models_dir
        tf_dir.mkdir(parents=True, exist_ok=True)

        local_dir = tf_dir / model_name

        logger.info(f"[highlight]Downloading transformers model: {repo_id}[/highlight]")
        logger.info(f"  Target directory: {local_dir}")

        # Notify download starting
        await self._notify_progress(
            repo_id,
            model_name,
            0,
            "starting",
            message="Preparing transformers download…",
            download_id=download_id,
            phase="preparing",
        )

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
            await self._notify_progress(
                repo_id,
                model_name,
                100,
                "complete",
                message="Transformers model already downloaded — registration complete.",
                download_id=download_id,
                phase="complete",
            )
            return local_dir

        # Download with progress tracking
        await self._notify_progress(
            repo_id,
            model_name,
            5,
            "downloading",
            message="Resolving repository snapshot…",
            download_id=download_id,
            phase="resolving_repository",
        )

        try:
            loop = asyncio.get_running_loop()

            class SnapshotProgress(base_tqdm):
                def update(self, n=1):
                    super().update(n)
                    total = int(self.total or 0)
                    completed = int(self.n)
                    progress = min(95.0, 10.0 + ((completed / total) * 80.0)) if total else 10.0
                    message = (
                        f"Downloading repository files ({completed}/{total})…"
                        if total
                        else "Downloading repository files…"
                    )
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(
                            self_owner._notify_progress(
                                repo_id,
                                model_name,
                                progress,
                                "downloading",
                                message=message,
                                download_id=download_id,
                                phase="downloading_repository",
                                items_complete=completed if total else None,
                                items_total=total if total else None,
                            )
                        )
                    )

            self_owner = self

            def _do_download() -> str:
                """Run snapshot_download in a thread."""
                return snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    token=self._hf_token,
                    tqdm_class=SnapshotProgress,
                )

            # Run the download in a thread to avoid blocking
            await asyncio.to_thread(_do_download)

            logger.info(f"[success]Transformers model download complete: {local_dir}[/success]")

            await self._notify_progress(
                repo_id,
                model_name,
                97,
                "registering",
                message="Finalizing transformers model registration…",
                download_id=download_id,
                phase="registering",
            )
            # Register in database
            await self._register_model(
                repo_id=repo_id,
                filename=model_name,
                file_path=local_dir,
                engine_type="transformers",
                model_name_override=model_name,
            )

            # Notify complete
            await self._notify_progress(
                repo_id,
                model_name,
                100,
                "complete",
                message="Transformers model download complete.",
                download_id=download_id,
                phase="complete",
            )

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
            await self._notify_progress(
                repo_id,
                model_name,
                0,
                "error",
                str(e),
                download_id=download_id,
                phase="error",
            )
            raise
