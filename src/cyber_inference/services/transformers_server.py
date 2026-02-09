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
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="cyber-inference transformers server")

# Global model state
_model = None
_tokenizer = None
_model_name = ""
_device = None

_CHANNEL_MARKERS = {
    "<|start|>assistant<|channel|>analysis<|message|>": "<think>",
    "assistant<|channel|>analysis<|message|>": "<think>",
    "<|channel|>analysis<|message|>": "<think>",
    "<|start|>assistant<|channel|>final<|message|>": "</think>",
    "assistant<|channel|>final<|message|>": "</think>",
    "<|channel|>final<|message|>": "</think>",
}
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")
_STREAM_CARRY_SIZE = 512


# ── Request / Response schemas ──────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


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
    return _SPECIAL_TOKEN_RE.sub("", text)


def _generate(input_ids: torch.Tensor, request) -> torch.Tensor:
    """Run model.generate() with request parameters."""
    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": request.max_tokens,
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        generate_kwargs["temperature"] = request.temperature
        generate_kwargs["top_p"] = request.top_p
    if request.stop:
        from transformers import StoppingCriteria, StoppingCriteriaList

        stop_ids = [_tokenizer.encode(s, add_special_tokens=False) for s in request.stop]

        class StopOnTokens(StoppingCriteria):
            def __call__(self, ids, scores, **kwargs):
                for stop_seq in stop_ids:
                    if ids[0][-len(stop_seq):].tolist() == stop_seq:
                        return True
                return False

        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens()])

    if _tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = _tokenizer.eos_token_id

    with torch.no_grad():
        output = _model.generate(**generate_kwargs)
    return output


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        input_ids = _tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    except Exception:
        # Fallback: manual formatting
        text = ""
        for m in messages:
            text += f"<|{m['role']}|>\n{m['content']}\n"
        text += "<|assistant|>\n"
        input_ids = _tokenizer.encode(text, return_tensors="pt")

    input_ids = input_ids.to(_model.device)
    prompt_len = input_ids.shape[1]

    if request.stream:
        return StreamingResponse(
            _stream_chat(input_ids, prompt_len, request),
            media_type="text/event-stream",
        )

    output = _generate(input_ids, request)
    new_tokens = output[0][prompt_len:]
    text = _normalize_generated_text(
        _tokenizer.decode(new_tokens, skip_special_tokens=False)
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


async def _stream_chat(input_ids, prompt_len, request):
    """Stream chat completions using TextIteratorStreamer."""
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": request.max_tokens,
        "do_sample": request.temperature > 0,
        "streamer": streamer,
    }
    if request.temperature > 0:
        generate_kwargs["temperature"] = request.temperature
        generate_kwargs["top_p"] = request.top_p

    if _tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = _tokenizer.eos_token_id

    generation_error: list[BaseException] = []

    def run_generate() -> None:
        try:
            _model.generate(**generate_kwargs)
        except BaseException as exc:  # pragma: no cover - defensive runtime guard
            generation_error.append(exc)
            # Ensure stream consumers are unblocked if generation fails in the worker thread.
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

    stream_carry = ""

    def emit_text_chunk(text: str) -> str | None:
        normalized = _normalize_generated_text(text)
        if not normalized:
            return None
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": normalized},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    for text_chunk in streamer:
        if not text_chunk:
            continue
        stream_carry += text_chunk
        if len(stream_carry) <= _STREAM_CARRY_SIZE:
            continue

        # Preserve a suffix window so channel markers split across streamer
        # chunks still map cleanly to <think> / </think>.
        process_text = stream_carry[:-_STREAM_CARRY_SIZE]
        stream_carry = stream_carry[-_STREAM_CARRY_SIZE:]
        encoded_chunk = emit_text_chunk(process_text)
        if encoded_chunk:
            yield encoded_chunk

    encoded_chunk = emit_text_chunk(stream_carry)
    if encoded_chunk:
        yield encoded_chunk

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

    input_ids = _tokenizer.encode(request.prompt, return_tensors="pt").to(_model.device)
    prompt_len = input_ids.shape[1]

    output = _generate(input_ids, request)
    new_tokens = output[0][prompt_len:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

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

    for i, text in enumerate(inputs):
        tokens = _tokenizer(text, return_tensors="pt", truncation=True).to(_model.device)
        with torch.no_grad():
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


def load_model(model_path: str, device: str = "auto"):
    """Load model and tokenizer."""
    global _model, _tokenizer, _model_name, _device

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _model_name = Path(model_path).name
    print(f"[transformers-server] Loading model: {model_path}", flush=True)

    _tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
        dtype="auto",
    )
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
