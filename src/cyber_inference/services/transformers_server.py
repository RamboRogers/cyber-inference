"""
Lightweight HuggingFace Transformers inference server.

A minimal OpenAI-compatible API server that loads models directly
with transformers AutoModelForCausalLM + model.generate().

Designed for edge/SoC hardware with a lightweight native transformers path.
Launched as a subprocess by ProcessManager.

Usage:
    python -m cyber_inference.services.transformers_server \
        --model-path /path/to/model \
        --port 8338
"""

import argparse
import asyncio
import json
import re
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="cyber-inference transformers server")

# Global model state
_model = None
_processor = None  # AutoProcessor for VLM models
_tokenizer = None  # Tokenizer used for encode/decode/streaming across all model types
_model_name = ""
_device = None
_is_vlm = False

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
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")


# ── Request / Response schemas ──────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str | list = ""  # str for text, list[dict] for multimodal VLM input


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, alias="max_tokens")
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | None = None


class EmbeddingRequest(BaseModel):
    model: str = ""
    input: str | list[str] = ""


# ── Endpoints ───────────────────────────────────────────────────


@app.get("/health")
async def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "transformers",
            }
        ],
    }


def _normalize_generated_text(text: str) -> str:
    """Normalize model-specific control tokens into readable text."""
    for marker, replacement in sorted(_CHANNEL_MARKERS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(marker, replacement)
    text = text.replace("<|return|>", "\n")
    text = _SPECIAL_TOKEN_RE.sub("", text)

    # Handle partially-stripped markers (bare "analysis" at start)
    if "</think>" in text and "<think>" not in text:
        parts = text.split("</think>", 1)
        analysis = parts[0].strip()
        if analysis.lower().startswith("analysis"):
            analysis = analysis[len("analysis"):].strip()
        text = f"<think>{analysis}</think>{parts[1]}"

    return text


def _get_tokenizer():
    """Return the active tokenizer or raise a clear error if uninitialized."""
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not initialized")
    return _tokenizer


def _get_pad_token_id() -> int | None:
    """Resolve pad token ID for generation with robust fallbacks."""
    tokenizer = _get_tokenizer()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if eos_token_id is not None:
        return eos_token_id

    model_config = getattr(_model, "config", None)
    model_eos_token_id = getattr(model_config, "eos_token_id", None)
    if isinstance(model_eos_token_id, (list, tuple)):
        model_eos_token_id = model_eos_token_id[0] if model_eos_token_id else None
    return model_eos_token_id


def _generate(inputs, request) -> torch.Tensor:
    """Run model.generate() with request parameters (blocking, run in thread).

    Args:
        inputs: Either an input_ids tensor (text models) or a dict of
                BatchFeature tensors (VLM models with pixel_values etc.).
        request: The request object with generation parameters.
    """
    if isinstance(inputs, Mapping):
        # VLM: unpack the full BatchFeature dict
        generate_kwargs = dict(inputs)
    else:
        generate_kwargs = {"input_ids": inputs}

    generate_kwargs["max_new_tokens"] = request.max_tokens
    generate_kwargs["do_sample"] = request.temperature > 0

    if request.temperature > 0:
        generate_kwargs["temperature"] = request.temperature
        generate_kwargs["top_p"] = request.top_p

    tokenizer = _get_tokenizer()
    if request.stop:
        from transformers import StoppingCriteria, StoppingCriteriaList

        stop_ids = []
        for stop_seq in request.stop:
            encoded = tokenizer.encode(stop_seq, add_special_tokens=False)
            if encoded:
                stop_ids.append(encoded)

        if stop_ids:
            class StopOnTokens(StoppingCriteria):
                def __call__(self, ids, scores, **kwargs):
                    for stop_seq in stop_ids:
                        if ids[0][-len(stop_seq):].tolist() == stop_seq:
                            return True
                    return False

            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens()])

    pad_token_id = _get_pad_token_id()
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        output = _model.generate(**generate_kwargs)
    return output


def _normalize_vlm_content_part(part) -> dict:
    """Normalize a single VLM content part into processor-compatible format."""
    if isinstance(part, dict):
        data = dict(part)
        part_type = data.get("type")

        # OpenAI-style image parts need conversion for HF ProcessorMixin.
        # Processor chat template expects {"type":"image", "url|image|path|base64": ...}
        # while OpenAI clients send {"type":"image_url", "image_url": {"url": ...}}.
        if part_type in {"image_url", "input_image"}:
            image_value = data.get("image_url")
            if isinstance(image_value, dict):
                image_value = (
                    image_value.get("url")
                    or image_value.get("image")
                    or image_value.get("path")
                    or image_value.get("base64")
                )
            if image_value is None:
                image_value = (
                    data.get("url")
                    or data.get("image")
                    or data.get("path")
                    or data.get("base64")
                )

            normalized = {"type": "image"}
            if image_value is not None:
                normalized["url"] = image_value if isinstance(image_value, str) else str(image_value)
            return normalized

        if part_type == "image":
            if "image_url" in data and "url" not in data:
                image_value = data["image_url"]
                if isinstance(image_value, dict):
                    image_value = image_value.get("url")
                if isinstance(image_value, str):
                    data["url"] = image_value
            return data

        if part_type == "text" and "text" not in data:
            data["text"] = ""
        return data

    if isinstance(part, str):
        return {"type": "text", "text": part}

    return {"type": "text", "text": str(part)}


def _normalize_vlm_messages(messages: list[dict]) -> list[dict]:
    """Normalize OpenAI-style chat messages for VLM processor chat templates."""
    normalized_messages: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, list):
            normalized_content = [_normalize_vlm_content_part(part) for part in content]
        elif isinstance(content, dict):
            if "type" in content:
                normalized_content = [_normalize_vlm_content_part(content)]
            elif "image_url" in content:
                normalized_content = [
                    _normalize_vlm_content_part({"type": "image_url", "image_url": content["image_url"]})
                ]
            else:
                normalized_content = [{"type": "text", "text": str(content)}]
        else:
            normalized_content = [{"type": "text", "text": str(content)}]

        normalized_messages.append({"role": role, "content": normalized_content})

    return normalized_messages


def _prepare_inputs(messages: list[dict]) -> tuple:
    """Prepare model inputs from chat messages.

    Returns:
        (inputs, prompt_len) where inputs is either an input_ids tensor
        (text models) or a dict of BatchFeature tensors (VLM models).
    """
    if _is_vlm:
        if _processor is None:
            raise RuntimeError("Processor not initialized for VLM model")

        # VLM: processor.apply_chat_template returns a BatchFeature dict
        # with input_ids, attention_mask, pixel_values, image_grid_thw, etc.
        normalized_messages = _normalize_vlm_messages(messages)
        inputs = _processor.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(_model.device)
        prompt_len = inputs["input_ids"].shape[1]
        return dict(inputs), prompt_len

    # Text model: standard tokenizer path
    tokenizer = _get_tokenizer()
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    except Exception:
        # Fallback: manual formatting
        text = ""
        for m in messages:
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            text += f"<|{m['role']}|>\n{content}\n"
        text += "<|assistant|>\n"
        input_ids = tokenizer.encode(text, return_tensors="pt")

    input_ids = input_ids.to(_model.device)
    prompt_len = input_ids.shape[1]
    return input_ids, prompt_len


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    inputs, prompt_len = _prepare_inputs(messages)

    if request.stream:
        return StreamingResponse(
            _stream_chat(inputs, prompt_len, request),
            media_type="text/event-stream",
        )

    # Run generation in thread pool to avoid blocking the event loop
    output = await asyncio.to_thread(_generate, inputs, request)
    new_tokens = output[0][prompt_len:]
    tokenizer = _get_tokenizer()
    text = _normalize_generated_text(
        tokenizer.decode(new_tokens, skip_special_tokens=False)
    ).strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(new_tokens),
            "total_tokens": prompt_len + len(new_tokens),
        },
    }


async def _stream_chat(inputs, prompt_len, request):
    """Stream chat completions using TextIteratorStreamer.

    Generation runs in a background thread.  The streamer's blocking
    ``__next__`` calls are dispatched via ``run_in_executor`` so that
    the asyncio event loop stays responsive for health checks and
    concurrent requests.

    Args:
        inputs: Either an input_ids tensor (text) or BatchFeature dict (VLM).
    """
    from transformers import TextIteratorStreamer

    tokenizer = _get_tokenizer()
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if isinstance(inputs, Mapping):
        # VLM: unpack the full BatchFeature dict
        generate_kwargs = dict(inputs)
    else:
        generate_kwargs = {"input_ids": inputs}

    generate_kwargs["max_new_tokens"] = request.max_tokens
    generate_kwargs["do_sample"] = request.temperature > 0
    generate_kwargs["streamer"] = streamer

    if request.temperature > 0:
        generate_kwargs["temperature"] = request.temperature
        generate_kwargs["top_p"] = request.top_p

    pad_token_id = _get_pad_token_id()
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id

    generation_error: list[BaseException] = []

    def run_generate() -> None:
        try:
            with torch.inference_mode():
                _model.generate(**generate_kwargs)
        except BaseException as exc:
            generation_error.append(exc)
            streamer.on_finalized_text("", stream_end=True)

    thread = threading.Thread(target=run_generate, daemon=True)
    thread.start()

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    # Read from the blocking TextIteratorStreamer via the thread-pool
    # executor so we don't block the asyncio event loop.
    # Raw chunks are emitted without normalization here; the v1.py proxy
    # layer applies its own streaming normalizer which handles channel
    # markers correctly across chunk boundaries.
    loop = asyncio.get_running_loop()
    iter_stream = iter(streamer)
    _sentinel = object()

    def _next_chunk():
        return next(iter_stream, _sentinel)

    while True:
        text_chunk = await loop.run_in_executor(None, _next_chunk)
        if text_chunk is _sentinel:
            break
        if not text_chunk:
            continue
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text_chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join(timeout=5.0)
    if generation_error:
        print(
            f"[transformers-server] generation error: {generation_error[0]}",
            file=sys.stderr,
            flush=True,
        )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tokenizer = _get_tokenizer()
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(_model.device)
    prompt_len = input_ids.shape[1]

    output = await asyncio.to_thread(_generate, input_ids, request)
    new_tokens = output[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(new_tokens),
            "total_tokens": prompt_len + len(new_tokens),
        },
    }


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = request.input if isinstance(request.input, list) else [request.input]
    all_embeddings = []
    tokenizer = _get_tokenizer()

    for i, text in enumerate(inputs):
        tokens = tokenizer(text, return_tensors="pt", truncation=True).to(_model.device)
        with torch.inference_mode():
            output = _model(**tokens, output_hidden_states=True)
        # Use last hidden state mean pooling
        hidden = output.hidden_states[-1]
        embedding = hidden.mean(dim=1).squeeze().cpu().tolist()
        all_embeddings.append({"object": "embedding", "index": i, "embedding": embedding})

    return {
        "object": "list",
        "data": all_embeddings,
        "model": _model_name,
        "usage": {"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": 0},
    }


# ── Model loading & main ────────────────────────────────────────


def _detect_unified_memory() -> bool:
    """Detect if CUDA device uses unified memory (e.g. NVIDIA Thor SoC, Jetson).

    On unified memory systems, device_map="auto" may incorrectly offload layers
    to CPU because CUDA reports only a fraction of the total shared memory.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    # Unified memory devices: NVIDIA Thor (cc 11.0), Jetson Orin (cc 8.7+)
    # These have is_integrated=True or share system memory with GPU
    if getattr(props, "is_integrated", False):
        return True
    # Heuristic: if total CUDA memory < 50GB but device name suggests SoC
    soc_names = ["thor", "orin", "jetson", "tegra"]
    if any(name in props.name.lower() for name in soc_names):
        return True
    return False


def _detect_vlm(model_path: str) -> bool:
    """Detect if model is a Vision-Language Model from its config.json."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
        # Check for vision_config section (definitive VLM indicator)
        if "vision_config" in config:
            return True
        # Check architectures for VL/Vision patterns
        for arch in config.get("architectures", []):
            if "VL" in arch or "Vision" in arch:
                return True
    except Exception:
        pass
    return False


def load_model(model_path: str, device: str = "auto"):
    """Load model and tokenizer/processor."""
    global _model, _processor, _tokenizer, _model_name, _device, _is_vlm

    _model_name = Path(model_path).name
    _is_vlm = _detect_vlm(model_path)
    _processor = None
    _tokenizer = None

    print(f"[transformers-server] Loading model: {model_path}", flush=True)
    print(f"[transformers-server] VLM detected: {_is_vlm}", flush=True)
    print(f"[transformers-server] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"[transformers-server] CUDA device: {props.name}, "
            f"compute capability: {props.major}.{props.minor}, "
            f"VRAM: {props.total_memory / 1024**3:.1f} GB",
            flush=True,
        )
        unified = _detect_unified_memory()
        if unified:
            print("[transformers-server] Unified memory detected, forcing device_map='cuda:0'", flush=True)
            if device == "auto":
                device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[transformers-server] MPS (Apple Metal) available", flush=True)

    # Check for MXFP4 kernel support
    try:
        from transformers.quantizers.quantizer_mxfp4 import is_kernels_available
        kernels_ok = is_kernels_available()
        print(f"[transformers-server] MXFP4 kernels available: {kernels_ok}", flush=True)
        if not kernels_ok:
            print(
                "[transformers-server] WARNING: kernels package not installed, "
                "MXFP4 models will be dequantized to BF16 (slow)",
                flush=True,
            )
    except ImportError:
        pass

    # Load tokenizer/processor
    if _is_vlm:
        from transformers import AutoProcessor
        print("[transformers-server] Using AutoProcessor (VLM)", flush=True)
        _processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        _tokenizer = getattr(_processor, "tokenizer", None)
        if _tokenizer is None:
            raise RuntimeError(
                "[transformers-server] VLM processor did not provide a tokenizer attribute"
            )
    else:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

    processor_name = _processor.__class__.__name__ if _processor is not None else "None"
    tokenizer_name = _tokenizer.__class__.__name__ if _tokenizer is not None else "None"
    print(
        f"[transformers-server] Processor: {processor_name}, Tokenizer: {tokenizer_name}",
        flush=True,
    )

    load_kwargs = {
        "device_map": device,
        "trust_remote_code": True,
        "dtype": "auto",
    }

    # Load model with appropriate auto class
    if _is_vlm:
        from transformers import AutoModelForVision2Seq
        print("[transformers-server] Using AutoModelForVision2Seq (VLM)", flush=True)
        try:
            _model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        except ImportError as e:
            print(
                f"[transformers-server] FATAL: Missing required package: {e}\n"
                f"[transformers-server] Install it and restart.",
                file=sys.stderr, flush=True,
            )
            raise
    else:
        from transformers import AutoModelForCausalLM
        try:
            _model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except (ValueError, KeyError) as e:
            print(f"[transformers-server] AutoModelForCausalLM failed: {e}", flush=True)
            try:
                from transformers import AutoModelForVision2Seq
                print("[transformers-server] Trying AutoModelForVision2Seq...", flush=True)
                _model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
            except (ValueError, KeyError, ImportError):
                from transformers import AutoModel
                print("[transformers-server] Trying AutoModel...", flush=True)
                _model = AutoModel.from_pretrained(model_path, **load_kwargs)
        except ImportError as e:
            print(
                f"[transformers-server] FATAL: Missing required package: {e}\n"
                f"[transformers-server] Install it and restart.",
                file=sys.stderr, flush=True,
            )
            raise

    _model.eval()

    _device = next(_model.parameters()).device
    print(
        f"[transformers-server] Model loaded on {_device}, "
        f"dtype={next(_model.parameters()).dtype}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Transformers inference server")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--port", type=int, default=8338)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    load_model(args.model_path, args.device)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
