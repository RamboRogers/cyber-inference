"""
OpenAI-compatible V1 API endpoints.

Implements:
- POST /v1/chat/completions
- POST /v1/completions
- POST /v1/embeddings
- GET /v1/models

Proxies requests to the appropriate llama-server instance,
handling automatic model loading and streaming responses.
"""

import asyncio
import json
import re
import tempfile
import time
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from cyber_inference.core.database import get_db
from cyber_inference.core.logging import get_logger
from cyber_inference.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelsResponse,
    ModelInfo,
    TranscriptionResponse,
    TranscriptionSegment,
)
from cyber_inference.services.auto_loader import AutoLoader

logger = get_logger(__name__)

router = APIRouter()

# Global auto-loader instance (initialized in main.py lifespan)
_auto_loader: Optional[AutoLoader] = None


def get_auto_loader() -> AutoLoader:
    """Get the auto-loader instance."""
    global _auto_loader
    if _auto_loader is None:
        # Lazy initialization
        from cyber_inference.services.auto_loader import AutoLoader
        _auto_loader = AutoLoader()
    return _auto_loader


def _guess_image_mime(base64_data: str) -> str:
    if base64_data.startswith("/9j/"):
        return "image/jpeg"
    if base64_data.startswith("iVBORw0KGgo"):
        return "image/png"
    if base64_data.startswith("R0lGOD"):
        return "image/gif"
    if base64_data.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def _normalize_image_url_value(value):
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return value
        if trimmed.startswith("data:") or trimmed.startswith("http://") or trimmed.startswith("https://"):
            return trimmed
        mime = _guess_image_mime(trimmed)
        return f"data:{mime};base64,{trimmed}"
    if isinstance(value, dict):
        url_value = value.get("url")
        if isinstance(url_value, str):
            value["url"] = _normalize_image_url_value(url_value)
        return value
    return value


def _normalize_content_part(part):
    if hasattr(part, "model_dump"):
        data = part.model_dump(exclude_none=True)
    elif isinstance(part, dict):
        data = dict(part)
    else:
        return part

    if data.get("type") == "image_url":
        image_value = data.get("image_url")
        image_value = _normalize_image_url_value(image_value)
        if isinstance(image_value, str):
            data["image_url"] = {"url": image_value}
        else:
            data["image_url"] = image_value
    return data


def _serialize_message_content(content):
    if isinstance(content, list):
        return [_normalize_content_part(part) for part in content]

    if hasattr(content, "model_dump"):
        data = _normalize_content_part(content)
        if isinstance(data, dict) and "type" in data:
            return [data]
        return data

    if isinstance(content, dict):
        if "type" in content:
            return [_normalize_content_part(content)]
        return content

    return content


async def init_auto_loader(auto_loader: AutoLoader) -> None:
    """Initialize the auto-loader (called from main.py)."""
    global _auto_loader
    _auto_loader = auto_loader


def _get_server_type(model_name: str) -> str:
    """Get the server type for a loaded model."""
    try:
        from cyber_inference.main import get_process_manager
        pm = get_process_manager()
        proc = pm.get_process(model_name)
        return proc.server_type if proc else "llama"
    except Exception:
        return "llama"


# ── Channel marker normalization ─────────────────────────────────
# GPT-OSS and similar models emit channel tokens for thinking/final output.
# llama-server may partially strip special tokens, producing inconsistent
# output: e.g. bare "analysis ..." at the start with raw
# "assistant<|channel|>final<|message|>" mid-text.  We handle all variants.

_CHANNEL_MARKERS = {
    "<|start|>assistant<|channel|>analysis<|message|>": "<think>",
    "assistant<|channel|>analysis<|message|>": "<think>",
    "<|channel|>analysis<|message|>": "<think>",
    "analysis<|message|>": "<think>",
    "<|start|>assistant<|channel|>final<|message|>": "</think>",
    "assistant<|channel|>final<|message|>": "</think>",
    "<|channel|>final<|message|>": "</think>",
    "final<|message|>": "</think>",
}
_SORTED_MARKERS = sorted(_CHANNEL_MARKERS.items(), key=lambda kv: len(kv[0]), reverse=True)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")

# Separate marker lists for the stream normalizer state machine
_ANALYSIS_MARKERS = [m for m, r in _SORTED_MARKERS if r == "<think>"]
_FINAL_MARKERS = [m for m, r in _SORTED_MARKERS if r == "</think>"]
_SAFE_MARGIN = 64  # buffer margin to avoid splitting partial markers


def _normalize_channel_markers(text: str) -> str:
    """Normalize model-specific channel tokens into <think>/</think> tags."""
    for marker, replacement in _SORTED_MARKERS:
        text = text.replace(marker, replacement)
    text = text.replace("<|return|>", "\n")
    text = _SPECIAL_TOKEN_RE.sub("", text)

    # Handle partially-stripped markers: llama-server may strip <|channel|>
    # and <|message|> at the start (leaving bare "analysis ...") but keep
    # them mid-text (producing </think> via the marker dict above).
    # If we got </think> but no <think>, wrap the leading content.
    if "</think>" in text and "<think>" not in text:
        parts = text.split("</think>", 1)
        analysis = parts[0].strip()
        # Strip the bare "analysis" channel identifier word
        if analysis.lower().startswith("analysis"):
            analysis = analysis[len("analysis"):].strip()
        text = f"<think>{analysis}</think>{parts[1]}"

    return text


class _StreamNormalizer:
    """Scanning state-machine normalizer for channel markers in streaming text.

    The old carry-buffer approach could split markers across process boundaries.
    This version accumulates text and scans for complete markers, emitting only
    the "safe" prefix (text that can't be part of a marker) progressively.

    States:
      - Accumulating: scanning buffer for analysis/final markers
      - Done: past the final marker, emit text directly with cleanup
    """

    def __init__(self):
        self._buf = ""
        self._think_open = False
        self._done = False

    def feed(self, text: str) -> str:
        """Feed a streaming text chunk. Returns normalized text to emit."""
        if self._done:
            return self._cleanup(text)
        self._buf += text
        return self._scan()

    def flush(self) -> str:
        """Flush remaining buffered text at end of stream."""
        if self._done:
            text = self._cleanup(self._buf)
            self._buf = ""
            return text
        text = self._buf
        self._buf = ""
        if not text:
            return "</think>" if self._think_open else ""
        # Use the full (non-streaming) normalizer for correctness
        result = _normalize_channel_markers(text)
        if self._think_open:
            if result.startswith("<think>"):
                result = result[len("<think>"):]
            if "</think>" not in result:
                result += "</think>"
        return result

    def _scan(self) -> str:
        out = ""

        # ── Detect thinking-phase start ──────────────────────────
        if not self._think_open:
            for marker in _ANALYSIS_MARKERS:
                idx = self._buf.find(marker)
                if idx >= 0:
                    out += self._cleanup(self._buf[:idx])
                    self._buf = self._buf[idx + len(marker):]
                    self._think_open = True
                    out += "<think>"
                    return out + self._scan()

            # Bare "analysis" at stream start (all special tokens stripped)
            stripped = self._buf.lstrip()
            if stripped.lower().startswith("analysis ") or stripped.lower().startswith("analysis\n"):
                self._buf = stripped[len("analysis"):]
                self._think_open = True
                out += "<think>"
                return out + self._scan()

        # ── Detect final-phase transition ────────────────────────
        for marker in _FINAL_MARKERS:
            idx = self._buf.find(marker)
            if idx >= 0:
                before = self._cleanup(self._buf[:idx])
                self._buf = self._buf[idx + len(marker):]
                self._done = True
                if self._think_open:
                    out += before + "</think>"
                else:
                    out += before
                if self._buf:
                    out += self._cleanup(self._buf)
                    self._buf = ""
                return out

        # ── Emit safe prefix ─────────────────────────────────────
        if len(self._buf) > _SAFE_MARGIN:
            safe_end = len(self._buf) - _SAFE_MARGIN
            safe = self._buf[:safe_end]
            # Don't split incomplete <|...|> tokens at the boundary
            lp = safe.rfind("<|")
            if lp >= 0 and safe.find("|>", lp) < 0:
                safe_end = lp
                safe = safe[:safe_end]
            if safe_end > 0:
                out += self._cleanup(safe)
                self._buf = self._buf[safe_end:]

        return out

    @staticmethod
    def _cleanup(text: str) -> str:
        """Remove remaining special tokens."""
        text = text.replace("<|return|>", "\n")
        return _SPECIAL_TOKEN_RE.sub("", text)


async def _apply_model_defaults(request, model_name: str) -> None:
    """Apply per-model inference defaults for fields not explicitly set in the request.

    Uses Pydantic v2's model_fields_set to distinguish between explicit values
    and schema defaults, so user-provided values are never overridden.
    """
    auto_loader = get_auto_loader()
    model_info = await auto_loader.get_model_info(model_name)
    if not model_info:
        return

    field_map = {
        "temperature": "default_temperature",
        "top_p": "default_top_p",
        "max_tokens": "default_max_tokens",
    }
    # These fields only exist on ChatCompletionRequest
    if hasattr(request, "frequency_penalty"):
        field_map["frequency_penalty"] = "default_repeat_penalty"

    explicitly_set = request.model_fields_set
    for request_field, model_key in field_map.items():
        if request_field not in explicitly_set:
            value = model_info.get(model_key)
            if value is not None:
                setattr(request, request_field, value)


@router.get("/models")
async def list_models(db: AsyncSession = Depends(get_db)) -> ModelsResponse:
    """
    List available models.

    Returns models that are downloaded and enabled.
    """
    logger.info("[info]GET /v1/models - Listing available models[/info]")

    auto_loader = get_auto_loader()
    models = await auto_loader.list_available_models()

    model_list = []
    for model in models:
        model_list.append(ModelInfo(
            id=model["name"],
            created=int(datetime.now().timestamp()),
            owned_by="cyber-inference",
        ))

    logger.info(f"[success]Returning {len(model_list)} models[/success]")

    return ModelsResponse(data=model_list)


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    logger.info(f"[info]GET /v1/models/{model_id}[/info]")

    auto_loader = get_auto_loader()
    model = await auto_loader.get_model_info(model_id)

    if not model:
        logger.warning(f"[warning]Model not found: {model_id}[/warning]")
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return ModelInfo(
        id=model["name"],
        created=int(datetime.now().timestamp()),
        owned_by="cyber-inference",
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
):
    """
    Create a chat completion.

    Proxies to the appropriate llama-server instance,
    auto-loading the model if necessary.
    """
    logger.info("[highlight]POST /v1/chat/completions[/highlight]")
    logger.info(f"  Model: {request.model}")
    logger.info(f"  Messages: {len(request.messages)}")
    logger.info(f"  Stream: {request.stream}")
    logger.debug(f"  Temperature: {request.temperature}")
    logger.debug(f"  Max tokens: {request.max_tokens}")

    auto_loader = get_auto_loader()

    # Ensure model is loaded
    try:
        server_url = await auto_loader.ensure_model_loaded(request.model)
    except Exception as e:
        logger.error(f"[error]Failed to load model: {e}[/error]")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")

    logger.debug(f"  Server URL: {server_url}")

    # Apply per-model inference defaults for fields not explicitly set
    await _apply_model_defaults(request, request.model)

    # Determine server type for engine-specific behavior
    server_type = _get_server_type(request.model)

    # Prepare request for the inference server
    token_limit = request.max_tokens or 512
    llama_request = {
        "messages": [
            {
                "role": m.role,
                "content": _serialize_message_content(m.content),
                **({"name": m.name} if m.name else {}),
            }
            for m in request.messages
        ],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": token_limit,
        "stream": request.stream,
    }

    # n_predict is llama.cpp-specific, not used by transformers
    if server_type != "transformers":
        llama_request["n_predict"] = token_limit

    if request.stop:
        llama_request["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

    if request.stream:
        return EventSourceResponse(
            _stream_chat_completion(server_url, llama_request, request.model)
        )

    # Non-streaming request
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{server_url}/v1/chat/completions",
                json=llama_request,
            )
            if response.status_code >= 400:
                logger.error(
                    "[error]llama-server error %s: %s[/error]",
                    response.status_code,
                    response.text.strip() or "no response body",
                )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as e:
        logger.error(f"[error]Proxy request failed: {e}[/error]")
        raise HTTPException(status_code=502, detail=f"Inference server error: {e}")

    # Update stats
    await auto_loader.record_request(request.model)

    # Transform response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    choices = []
    for i, choice in enumerate(result.get("choices", [])):
        raw_content = choice.get("message", {}).get("content", "")
        choices.append(ChatCompletionChoice(
            index=i,
            message=ChatMessage(
                role=choice.get("message", {}).get("role", "assistant"),
                content=_normalize_channel_markers(raw_content),
            ),
            finish_reason=choice.get("finish_reason"),
        ))

    usage = result.get("usage", {})

    completion_response = ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )

    total_tok = usage.get('total_tokens', 0)
    logger.info(f"[success]Chat completion successful: {total_tok} tokens[/success]")

    return completion_response


async def _stream_chat_completion(
    server_url: str,
    request: dict,
    model: str,
) -> AsyncGenerator[dict, None]:
    """Stream chat completion chunks from llama-server."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    auto_loader = get_auto_loader()
    # Mark model active immediately to prevent idle unload during long stream startup.
    await auto_loader.touch_request(model)
    keepalive_task = asyncio.create_task(_stream_activity_keepalive(model))
    normalizer = _StreamNormalizer()

    logger.debug(f"Starting streaming response for {model}")

    def _make_chunk(content: str, finish_reason=None) -> dict:
        return {
            "data": json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": content} if content else {},
                    "finish_reason": finish_reason,
                }],
            })
        }

    try:
        # Streaming can legitimately exceed several minutes for large models.
        stream_timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)
        async with httpx.AsyncClient(timeout=stream_timeout) as client:
            async with client.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json=request,
            ) as response:
                if response.status_code >= 400:
                    error_body = (await response.aread()).decode("utf-8", "ignore")
                    logger.error(
                        "[error]llama-server error %s: %s[/error]",
                        response.status_code,
                        error_body.strip() or "no response body",
                    )
                    response.raise_for_status()

                sent_role = False
                async for line in response.aiter_lines():
                    # Keep model activity fresh while stream is in progress.
                    await auto_loader.touch_request(model)
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # Forward the role chunk as-is
                        if "role" in delta and not sent_role:
                            sent_role = True
                            yield {
                                "data": json.dumps({
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"role": delta["role"]},
                                        "finish_reason": None,
                                    }],
                                })
                            }
                            # If the role chunk also has content, fall through
                            if "content" not in delta:
                                continue

                        content = delta.get("content", "")
                        if content:
                            normalized = normalizer.feed(content)
                            if normalized:
                                yield _make_chunk(normalized)

                # Flush remaining buffered text
                remaining = normalizer.flush()
                if remaining:
                    yield _make_chunk(remaining)

                yield _make_chunk("", finish_reason="stop")
                yield {"data": "[DONE]"}

    except Exception as e:
        logger.error(
            f"[error]Streaming error ({type(e).__name__}): {e}[/error]"
        )
        yield {"data": json.dumps({"error": str(e)})}
    finally:
        keepalive_task.cancel()
        with suppress(asyncio.CancelledError):
            await keepalive_task

    logger.debug(f"Streaming complete for {model}")

    # Count the completed streamed request once.
    await auto_loader.record_request(model)


@router.post("/completions", response_model=None)
async def completions(
    request: CompletionRequest,
    http_request: Request,
):
    """
    Create a text completion.

    Legacy endpoint for non-chat completions.
    """
    logger.info("[highlight]POST /v1/completions[/highlight]")
    logger.info(f"  Model: {request.model}")
    logger.info(f"  Stream: {request.stream}")

    auto_loader = get_auto_loader()

    # Ensure model is loaded
    try:
        server_url = await auto_loader.ensure_model_loaded(request.model)
    except Exception as e:
        logger.error(f"[error]Failed to load model: {e}[/error]")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")

    # Apply per-model inference defaults for fields not explicitly set
    await _apply_model_defaults(request, request.model)

    # Determine server type for engine-specific behavior
    server_type = _get_server_type(request.model)

    # Prepare request
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    token_limit = request.max_tokens or 512

    if server_type == "transformers":
        # Transformers uses OpenAI-compatible /v1/completions
        llama_request = {
            "prompt": prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": token_limit,
            "stream": request.stream,
            "model": request.model,
        }
    else:
        # llama.cpp uses its native /completion endpoint
        llama_request = {
            "prompt": prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n_predict": token_limit,
            "stream": request.stream,
        }

    if request.stop:
        llama_request["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

    if request.stream:
        return EventSourceResponse(
            _stream_completion(server_url, llama_request, request.model, server_type)
        )

    # Non-streaming
    completion_url = (
        f"{server_url}/v1/completions"
        if server_type == "transformers"
        else f"{server_url}/completion"
    )

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(completion_url, json=llama_request)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as e:
        logger.error(f"[error]Proxy request failed: {e}[/error]")
        raise HTTPException(status_code=502, detail=f"Inference server error: {e}")

    await auto_loader.record_request(request.model)

    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    if server_type == "transformers":
        # Transformers returns OpenAI-compatible format
        choices = result.get("choices", [{}])
        choice = choices[0] if choices else {}
        usage = result.get("usage", {})
        return CompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[CompletionChoice(
                text=choice.get("text", ""),
                index=0,
                finish_reason=choice.get("finish_reason", "stop"),
            )],
            usage=CompletionUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    # llama.cpp native format
    return CompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=request.model,
        choices=[CompletionChoice(
            text=result.get("content", ""),
            index=0,
            finish_reason=result.get("stop_type", "stop"),
        )],
        usage=CompletionUsage(
            prompt_tokens=result.get("tokens_evaluated", 0),
            completion_tokens=result.get("tokens_predicted", 0),
            total_tokens=result.get("tokens_evaluated", 0) + result.get("tokens_predicted", 0),
        ),
    )


async def _stream_completion(
    server_url: str,
    request: dict,
    model: str,
    server_type: str = "llama",
) -> AsyncGenerator[dict, None]:
    """Stream completion chunks."""
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    auto_loader = get_auto_loader()
    await auto_loader.touch_request(model)
    keepalive_task = asyncio.create_task(_stream_activity_keepalive(model))

    completion_url = (
        f"{server_url}/v1/completions"
        if server_type == "transformers"
        else f"{server_url}/completion"
    )

    try:
        stream_timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)
        async with httpx.AsyncClient(timeout=stream_timeout) as client:
            async with client.stream(
                "POST",
                completion_url,
                json=request,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    await auto_loader.touch_request(model)
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield {"data": "[DONE]"}
                            break

                        try:
                            chunk = json.loads(data)

                            if server_type == "transformers":
                                # Transformers returns OpenAI-compatible format
                                yield {
                                    "data": json.dumps({
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": created,
                                        "model": model,
                                        "choices": chunk.get("choices", []),
                                    })
                                }
                            else:
                                yield {
                                    "data": json.dumps({
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": created,
                                        "model": model,
                                        "choices": [{
                                            "text": chunk.get("content", ""),
                                            "index": 0,
                                            "finish_reason": None if not chunk.get("stop") else "stop",
                                        }],
                                    })
                                }
                        except json.JSONDecodeError:
                            continue

    except Exception as e:
        logger.error(
            f"[error]Streaming error ({type(e).__name__}): {e}[/error]"
        )
        yield {"data": json.dumps({"error": str(e)})}
    finally:
        keepalive_task.cancel()
        with suppress(asyncio.CancelledError):
            await keepalive_task

    # Count the completed streamed request once.
    await auto_loader.record_request(model)


async def _stream_activity_keepalive(model: str, interval_seconds: float = 15.0) -> None:
    """Keep last-request activity fresh while a stream is open."""
    auto_loader = get_auto_loader()
    while True:
        await auto_loader.touch_request(model)
        await asyncio.sleep(interval_seconds)


@router.post("/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    http_request: Request,
) -> EmbeddingResponse:
    """
    Create embeddings for text.
    """
    logger.info("[highlight]POST /v1/embeddings[/highlight]")
    logger.info(f"  Model: {request.model}")

    auto_loader = get_auto_loader()

    # Ensure model is loaded
    try:
        server_url = await auto_loader.ensure_model_loaded(request.model)
    except Exception as e:
        logger.error(f"[error]Failed to load model: {e}[/error]")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")

    # Determine server type for engine-specific behavior
    server_type = _get_server_type(request.model)

    # Prepare input - OpenAI API accepts string or array of strings
    inputs = request.input if isinstance(request.input, list) else [request.input]
    logger.info(f"  Inputs: {len(inputs)} text(s)")
    logger.debug(f"  Server type: {server_type}")

    embeddings_data = []
    total_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            if server_type == "transformers":
                # Transformers supports OpenAI-compatible /v1/embeddings
                response = await client.post(
                    f"{server_url}/v1/embeddings",
                    json={
                        "input": inputs,
                        "model": request.model,
                    },
                )
                response.raise_for_status()
                result = response.json()

                embeddings_data = result.get("data", [])
                usage = result.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
            else:
                # llama.cpp: process each input one at a time
                for idx, text in enumerate(inputs):
                    response = await client.post(
                        f"{server_url}/embedding",
                        json={"content": text},
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Parse llama.cpp response format: [{"index": 0, "embedding": [[...]]}]
                    if isinstance(result, list) and len(result) > 0:
                        item = result[0]
                        embedding = item.get("embedding", [])
                        # embedding might be nested [[...]] or flat [...]
                        if isinstance(embedding, list) and len(embedding) > 0:
                            if isinstance(embedding[0], list):
                                embedding = embedding[0]  # Unnest
                    else:
                        # Fallback for simple format {"embedding": [...]}
                        embedding = result.get("embedding", [])

                    embeddings_data.append({
                        "object": "embedding",
                        "index": idx,
                        "embedding": embedding,
                    })

                    # Rough token count (approximation)
                    total_tokens += len(text.split())

    except httpx.HTTPError as e:
        logger.error(f"[error]Embedding request failed: {e}[/error]")
        raise HTTPException(status_code=502, detail=f"Inference server error: {e}")

    await auto_loader.record_request(request.model)

    return EmbeddingResponse(
        data=embeddings_data,
        model=request.model,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    )


@router.post("/audio/transcriptions", response_model=None)
async def transcriptions(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(..., description="Model to use for transcription"),
    language: Optional[str] = Form(None, description="Language of the audio (ISO-639-1)"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide transcription"),
    response_format: str = Form("json", description="Format: json, text, verbose_json, srt, vtt"),
    temperature: float = Form(0.0, ge=0.0, le=1.0, description="Sampling temperature"),
):
    """
    Transcribe audio to text.

    OpenAI-compatible transcription endpoint. Accepts audio files and returns
    transcribed text using whisper.cpp.

    Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg
    """
    logger.info("[highlight]POST /v1/audio/transcriptions[/highlight]")
    logger.info(f"  Model: {model}")
    logger.info(f"  File: {file.filename} ({file.content_type})")
    logger.info(f"  Language: {language or 'auto-detect'}")
    logger.info(f"  Response format: {response_format}")

    auto_loader = get_auto_loader()

    # Validate file type
    allowed_types = [
        "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a",
        "audio/wav", "audio/x-wav", "audio/webm", "audio/flac",
        "audio/ogg", "video/mp4", "video/webm",
    ]

    # Also check by extension
    allowed_extensions = [
        ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".flac", ".ogg"
    ]

    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    content_type = file.content_type or ""

    if content_type not in allowed_types and file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported: {', '.join(allowed_extensions)}"
        )

    # Ensure model is loaded
    try:
        server_url = await auto_loader.ensure_model_loaded(model)
    except Exception as e:
        logger.error(f"[error]Failed to load model: {e}[/error]")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")

    logger.debug(f"  Server URL: {server_url}")

    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / (file.filename or "audio.wav")

    try:
        # Write uploaded content to temp file
        content = await file.read()
        temp_path.write_bytes(content)
        logger.debug(f"  Saved to temp file: {temp_path} ({len(content)} bytes)")

        # Call whisper-server inference endpoint
        async with httpx.AsyncClient(timeout=300) as client:
            # whisper.cpp server accepts POST /inference with multipart form
            with open(temp_path, "rb") as audio_file:
                files = {"file": (file.filename, audio_file, content_type or "audio/wav")}
                data = {
                    "temperature": str(temperature),
                    "response_format": response_format,
                }

                if language:
                    data["language"] = language
                if prompt:
                    data["prompt"] = prompt

                response = await client.post(
                    f"{server_url}/inference",
                    files=files,
                    data=data,
                )

                if response.status_code >= 400:
                    logger.error(
                        "[error]whisper-server error %s: %s[/error]",
                        response.status_code,
                        response.text.strip()[:500],
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"Transcription failed: {response.text[:200]}"
                    )

                is_json = response_format in ("json", "verbose_json")
                result = response.json() if is_json else response.text

    except httpx.HTTPError as e:
        logger.error(f"[error]Transcription request failed: {e}[/error]")
        raise HTTPException(status_code=502, detail=f"Inference server error: {e}")
    finally:
        # Cleanup temp file
        try:
            temp_path.unlink()
            Path(temp_dir).rmdir()
        except Exception:
            pass

    await auto_loader.record_request(model)

    # Return response based on format
    if response_format == "text":
        if isinstance(result, dict):
            return PlainTextResponse(result.get("text", ""))
        return PlainTextResponse(str(result))

    if response_format in ("srt", "vtt"):
        return PlainTextResponse(str(result), media_type="text/plain")

    # JSON formats
    if isinstance(result, dict):
        # Parse whisper.cpp response into OpenAI format
        segments = None
        if response_format == "verbose_json" and "segments" in result:
            segments = [
                TranscriptionSegment(
                    id=i,
                    seek=seg.get("seek", 0),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    tokens=seg.get("tokens", []),
                    temperature=seg.get("temperature", 0.0),
                    avg_logprob=seg.get("avg_logprob", 0.0),
                    compression_ratio=seg.get("compression_ratio", 0.0),
                    no_speech_prob=seg.get("no_speech_prob", 0.0),
                )
                for i, seg in enumerate(result.get("segments", []))
            ]

        response_obj = TranscriptionResponse(
            text=result.get("text", ""),
            language=result.get("language", language),
            duration=result.get("duration"),
            segments=segments,
        )

        logger.info(f"[success]Transcription complete: {len(response_obj.text)} chars[/success]")
        return response_obj

    # Fallback: return raw result
    return {"text": str(result)}


@router.post("/audio/translations", response_model=None)
async def translations(
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form(..., description="Model to use for translation"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide translation"),
    response_format: str = Form("json", description="Format: json, text, verbose_json, srt, vtt"),
    temperature: float = Form(0.0, ge=0.0, le=1.0, description="Sampling temperature"),
):
    """
    Translate audio to English text.

    OpenAI-compatible translation endpoint. Translates audio in any supported
    language to English text.
    """
    logger.info("[highlight]POST /v1/audio/translations[/highlight]")
    logger.info(f"  Model: {model}")
    logger.info(f"  File: {file.filename}")

    # For translation, we call transcription with task=translate and language=en
    # whisper.cpp handles this via the translate task

    auto_loader = get_auto_loader()

    # Ensure model is loaded
    try:
        server_url = await auto_loader.ensure_model_loaded(model)
    except Exception as e:
        logger.error(f"[error]Failed to load model: {e}[/error]")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")

    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / (file.filename or "audio.wav")

    try:
        content = await file.read()
        temp_path.write_bytes(content)

        async with httpx.AsyncClient(timeout=300) as client:
            with open(temp_path, "rb") as audio_file:
                files = {"file": (file.filename, audio_file, file.content_type or "audio/wav")}
                data = {
                    "temperature": str(temperature),
                    "response_format": response_format,
                    "translate": "true",  # Enable translation mode
                }

                if prompt:
                    data["prompt"] = prompt

                response = await client.post(
                    f"{server_url}/inference",
                    files=files,
                    data=data,
                )

                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Translation failed: {response.text[:200]}"
                    )

                is_json = response_format in ("json", "verbose_json")
                result = response.json() if is_json else response.text

    except httpx.HTTPError as e:
        logger.error(f"[error]Translation request failed: {e}[/error]")
        raise HTTPException(status_code=502, detail=f"Inference server error: {e}")
    finally:
        try:
            temp_path.unlink()
            Path(temp_dir).rmdir()
        except Exception:
            pass

    await auto_loader.record_request(model)

    if response_format == "text":
        if isinstance(result, dict):
            return PlainTextResponse(result.get("text", ""))
        return PlainTextResponse(str(result))

    if isinstance(result, dict):
        from cyber_inference.models.schemas import TranslationResponse
        return TranslationResponse(
            text=result.get("text", ""),
            language=result.get("language"),
            duration=result.get("duration"),
        )

    return {"text": str(result)}
