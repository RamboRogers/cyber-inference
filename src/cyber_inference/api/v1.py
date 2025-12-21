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
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
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


def _serialize_message_content(content):
    if isinstance(content, list):
        serialized = []
        for part in content:
            if hasattr(part, "model_dump"):
                serialized.append(part.model_dump(exclude_none=True))
            else:
                serialized.append(part)
        return serialized
    if hasattr(content, "model_dump"):
        return content.model_dump(exclude_none=True)
    return content


async def init_auto_loader(auto_loader: AutoLoader) -> None:
    """Initialize the auto-loader (called from main.py)."""
    global _auto_loader
    _auto_loader = auto_loader


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
    logger.info(f"[highlight]POST /v1/chat/completions[/highlight]")
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

    # Prepare request for llama-server
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
        "n_predict": request.max_tokens or 512,
        "stream": request.stream,
    }

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
        choices.append(ChatCompletionChoice(
            index=i,
            message=ChatMessage(
                role=choice.get("message", {}).get("role", "assistant"),
                content=choice.get("message", {}).get("content", ""),
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

    logger.info(f"[success]Chat completion successful: {usage.get('total_tokens', 0)} tokens[/success]")

    return completion_response


async def _stream_chat_completion(
    server_url: str,
    request: dict,
    model: str,
) -> AsyncGenerator[dict, None]:
    """Stream chat completion chunks from llama-server."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    logger.debug(f"Starting streaming response for {model}")

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json=request,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield {"data": "[DONE]"}
                            break

                        try:
                            chunk = json.loads(data)
                            # Transform chunk to match OpenAI format
                            yield {
                                "data": json.dumps({
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": chunk.get("choices", []),
                                })
                            }
                        except json.JSONDecodeError:
                            continue

    except Exception as e:
        logger.error(f"[error]Streaming error: {e}[/error]")
        yield {"data": json.dumps({"error": str(e)})}

    logger.debug(f"Streaming complete for {model}")

    # Update stats
    auto_loader = get_auto_loader()
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
    logger.info(f"[highlight]POST /v1/completions[/highlight]")
    logger.info(f"  Model: {request.model}")
    logger.info(f"  Stream: {request.stream}")

    auto_loader = get_auto_loader()

    # Ensure model is loaded
    try:
        server_url = await auto_loader.ensure_model_loaded(request.model)
    except Exception as e:
        logger.error(f"[error]Failed to load model: {e}[/error]")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")

    # Prepare request
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    llama_request = {
        "prompt": prompt,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "n_predict": request.max_tokens or 512,
        "stream": request.stream,
    }

    if request.stop:
        llama_request["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

    if request.stream:
        return EventSourceResponse(
            _stream_completion(server_url, llama_request, request.model)
        )

    # Non-streaming
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{server_url}/completion",
                json=llama_request,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as e:
        logger.error(f"[error]Proxy request failed: {e}[/error]")
        raise HTTPException(status_code=502, detail=f"Inference server error: {e}")

    await auto_loader.record_request(request.model)

    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"

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
) -> AsyncGenerator[dict, None]:
    """Stream completion chunks."""
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{server_url}/completion",
                json=request,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield {"data": "[DONE]"}
                            break

                        try:
                            chunk = json.loads(data)
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
        logger.error(f"[error]Streaming error: {e}[/error]")
        yield {"data": json.dumps({"error": str(e)})}

    auto_loader = get_auto_loader()
    await auto_loader.record_request(model)


@router.post("/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    http_request: Request,
) -> EmbeddingResponse:
    """
    Create embeddings for text.
    """
    logger.info(f"[highlight]POST /v1/embeddings[/highlight]")
    logger.info(f"  Model: {request.model}")

    auto_loader = get_auto_loader()

    # Ensure model is loaded
    try:
        server_url = await auto_loader.ensure_model_loaded(request.model)
    except Exception as e:
        logger.error(f"[error]Failed to load model: {e}[/error]")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")

    # Prepare input - OpenAI API accepts string or array of strings
    inputs = request.input if isinstance(request.input, list) else [request.input]
    logger.info(f"  Inputs: {len(inputs)} text(s)")

    embeddings_data = []
    total_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Process each input (llama.cpp takes one at a time)
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
