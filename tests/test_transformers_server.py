"""Regression tests for transformers server tokenizer/processor handling."""

import json
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

import cyber_inference.services.transformers_server as ts


@pytest.fixture(autouse=True)
def _restore_transformers_server_globals():
    state = (
        ts._model,
        ts._processor,
        ts._tokenizer,
        ts._model_name,
        ts._device,
        ts._is_vlm,
    )
    yield
    (
        ts._model,
        ts._processor,
        ts._tokenizer,
        ts._model_name,
        ts._device,
        ts._is_vlm,
    ) = state


def test_generate_uses_tokenizer_for_vlm_processor_case():
    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 42

        def __init__(self):
            self.encoded: list[str] = []

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            self.encoded.append(text)
            if text == "":
                return []
            if text == "END":
                return [9, 10]
            return [1]

    class FakeModel:
        def __init__(self):
            self.kwargs = {}

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[1, 2, 3]])

    ts._is_vlm = True
    ts._processor = object()
    ts._tokenizer = FakeTokenizer()
    ts._model = FakeModel()

    request = SimpleNamespace(
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        stop=["", "END"],
    )
    output = ts._generate(torch.tensor([[1, 2]]), request)

    assert output.tolist() == [[1, 2, 3]]
    assert ts._tokenizer.encoded == ["", "END"]
    assert ts._model.kwargs["pad_token_id"] == 42
    assert "stopping_criteria" in ts._model.kwargs


def test_generate_accepts_mapping_inputs_for_vlm_batch_features():
    class FakeBatchFeature(Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    class FakeTokenizer:
        pad_token_id = 5
        eos_token_id = None

    class FakeModel:
        def __init__(self):
            self.kwargs = {}

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([[1, 2, 3]])

    ts._is_vlm = True
    ts._processor = object()
    ts._tokenizer = FakeTokenizer()
    ts._model = FakeModel()

    batch = FakeBatchFeature(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
    )
    request = SimpleNamespace(max_tokens=8, temperature=0.0, top_p=1.0, stop=None)
    _ = ts._generate(batch, request)

    assert isinstance(ts._model.kwargs["input_ids"], torch.Tensor)
    assert ts._model.kwargs["pad_token_id"] == 5


@pytest.mark.asyncio
async def test_stream_chat_first_chunk_no_processor_eos_crash(monkeypatch):
    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 7

    class FakeStreamer:
        def __init__(self, *args, **kwargs):
            self._chunks: list[str] = []
            self._ended = False

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if text:
                self._chunks.append(text)
            if stream_end:
                self._ended = True

        def __iter__(self):
            return self

        def __next__(self):
            if self._chunks:
                return self._chunks.pop(0)
            if self._ended:
                raise StopIteration
            raise StopIteration

    class FakeModel:
        def generate(self, **kwargs):
            kwargs["streamer"].on_finalized_text("", stream_end=True)
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr("transformers.TextIteratorStreamer", FakeStreamer)

    ts._is_vlm = True
    ts._processor = object()
    ts._tokenizer = FakeTokenizer()
    ts._model = FakeModel()
    ts._model_name = "Qwen3-VL-8B-Instruct"

    request = SimpleNamespace(max_tokens=8, temperature=0.0, top_p=1.0)
    stream = ts._stream_chat(torch.tensor([[1, 2]]), 2, request)
    first_chunk = await stream.__anext__()
    await stream.aclose()

    assert first_chunk.startswith("data: ")
    payload = json.loads(first_chunk[len("data: "):].strip())
    assert payload["choices"][0]["delta"]["role"] == "assistant"


def test_prepare_inputs_vlm_uses_processor_apply_chat_template():
    class FakeBatch(dict):
        moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    class FakeProcessor:
        def __init__(self):
            self.called = False
            self.messages = None
            self.batch = None

        def apply_chat_template(self, messages, *args, **kwargs):
            self.called = True
            self.messages = messages
            self.batch = FakeBatch(
                {
                    "input_ids": torch.tensor([[10, 11, 12]]),
                    "token_type_ids": torch.tensor([[0, 0, 0]]),
                }
            )
            return self.batch

    class GuardTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("Tokenizer path should not run for VLM input prep")

    class FakeModel:
        device = torch.device("cpu")

    processor = FakeProcessor()
    ts._is_vlm = True
    ts._processor = processor
    ts._tokenizer = GuardTokenizer()
    ts._model = FakeModel()

    inputs, prompt_len = ts._prepare_inputs([{"role": "user", "content": "hi"}])

    assert processor.called is True
    assert processor.messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert "token_type_ids" not in inputs
    assert prompt_len == 3
    assert processor.batch.moved_to == torch.device("cpu")


def test_normalize_vlm_messages_handles_text_and_image_variants():
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "caption"},
                {"type": "image_url", "image_url": "https://example.com/img.png"},
            ],
        },
        {
            "role": "user",
            "content": {"image_url": {"url": "data:image/png;base64,AAAABBBB"}},
        },
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": {"url": "ZmFrZWJhc2U2NA=="}}],
        },
        {"role": "assistant", "content": {"type": "text"}},
    ]

    normalized = ts._normalize_vlm_messages(messages)

    assert normalized[0]["content"] == [{"type": "text", "text": "hello"}]
    assert normalized[1]["content"][1] == {"type": "image", "url": "https://example.com/img.png"}
    assert normalized[2]["content"] == [{"type": "image", "url": "data:image/png;base64,AAAABBBB"}]
    assert normalized[3]["content"] == [{"type": "image", "url": "ZmFrZWJhc2U2NA=="}]
    assert normalized[4]["content"] == [{"type": "text", "text": ""}]
