# Cyber-Inference

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-00ff9f?style=for-the-badge&logo=python&logoColor=00ff9f" alt="Python">
  <img src="https://img.shields.io/badge/License-GPLv3-00ff9f?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/llama.cpp-Powered-00ff9f?style=for-the-badge" alt="llama.cpp">
  <img src="https://img.shields.io/badge/whisper.cpp-Powered-00ff9f?style=for-the-badge" alt="whisper.cpp">
  <img src="https://img.shields.io/badge/SGLang-Powered-00ff9f?style=for-the-badge" alt="SGLang">
  <img src="https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Jetson">
</p>

<p align="center">
<img src="cyber-inference.png">
  <strong>Edge Inference Server Management with OpenAI-Compatible API</strong>
</p>

---

Cyber-Inference is a web GUI management tool for running OpenAI-compatible inference servers on the edge. It provides three inference engines -- **llama.cpp** for GGUF models, **SGLang** for full-precision HuggingFace models on NVIDIA GPUs, and **whisper.cpp** for audio transcription. All engines share a unified API, automatic model management, dynamic resource allocation, and a cyberpunk-themed web interface.

## Why Cyber-Inference?

I made this project because I need to deploy inference servers on the edge, for labs and for personal use. I want an easy way to copy paste models from HuggingFace and have them automatically downloaded and loaded. I don't want to manage memory, and I just want to have standby inference that is ready to go when I need it (auto model loading and unloading).

Specifically, I made this project for MacOS and NVIDIA Jetson devices. On NVIDIA hardware, SGLang is automatically installed and configured for GPU-accelerated inference with full HuggingFace models. On machines without NVIDIA GPUs, llama.cpp handles inference with GGUF-quantized models.

## Features

- **Cyberpunk Web GUI** - Dark-mode interface with neon green accents, real-time updates, and responsive design
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI's `/v1/` endpoints
- **Three Inference Engines** - llama.cpp (GGUF), SGLang (HuggingFace/NVIDIA), and whisper.cpp (audio)
- **Automatic Model Management** - Download models from HuggingFace with one click
- **Dynamic Loading** - Models load on-demand and unload when idle
- **GPU Acceleration** - NVIDIA CUDA (auto-detected), Apple Metal, and CPU fallback
- **Jetson Ready** - Native support for NVIDIA Jetson Thor, Orin, and AGX platforms
- **Resource Monitoring** - Real-time CPU, RAM, and GPU usage tracking
- **Optional Security** - Admin password protection with JWT authentication
- **Docker Ready** - Full Docker and docker-compose support with NVIDIA runtime
- **Audio Transcription** - Speech-to-text with Whisper models via whisper.cpp
- **Vision Models** - Multimodal support with automatic mmproj file handling
- **Embeddings** - Text embedding models for RAG and semantic search

## Inference Engines

| Engine | Format | Hardware | Models | Use Case |
|--------|--------|----------|--------|----------|
| **llama.cpp** | GGUF | CPU, Metal, CUDA | Llama, Qwen, Mistral, etc. | Quantized models, edge devices, Mac |
| **SGLang** | HuggingFace (safetensors) | NVIDIA GPU | Any HF transformer model | Full-precision GPU inference |
| **whisper.cpp** | GGUF | CPU, Metal, CUDA | Whisper variants | Audio transcription |

### Supported Model Types

| Type | Engine | Models | API Endpoint |
|------|--------|--------|--------------|
| **Chat/LLM** | llama.cpp or SGLang | Llama, Qwen, Mistral, etc. | `/v1/chat/completions` |
| **Vision (VLM)** | llama.cpp | Qwen-VL, GLM-4V, etc. | `/v1/chat/completions` |
| **Completions** | llama.cpp or SGLang | Any text model | `/v1/completions` |
| **Embeddings** | llama.cpp or SGLang | BGE, Qwen-Embed, E5, etc. | `/v1/embeddings` |
| **Transcription** | whisper.cpp | Whisper Large V3, etc. | `/v1/audio/transcriptions` |
| **Translation** | whisper.cpp | Whisper Large V3, etc. | `/v1/audio/translations` |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### One-Shot Script

The easiest way to get started. On NVIDIA hardware, SGLang and CUDA PyTorch are installed automatically. On other hardware, llama.cpp is used.

```bash
# Clone the repository
git clone https://github.com/ramborogers/cyber-inference.git
cd cyber-inference

# Run the one shot script
./start.sh
```

What `start.sh` does:
1. Installs `uv` if missing
2. Verifies Python 3.12+
3. Detects NVIDIA GPU and CUDA version
4. Syncs dependencies (`uv sync`)
5. If NVIDIA: installs SGLang + CUDA PyTorch + sgl-kernel (versions detected dynamically)
6. Starts the server with auto-restart and exponential backoff

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/ramborogers/cyber-inference.git
cd cyber-inference

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Running

```bash
# Start the server
uv run cyber-inference serve

# Or with custom options
uv run cyber-inference serve --port 8337 --log-level info
```

Visit **http://localhost:8337** to access the web interface.

### First Steps

<img src="download.png">

1. Open the web GUI at http://localhost:8337
2. Navigate to **Models** and download a model
   - For **GGUF** (llama.cpp): paste a repo like `ggml-org/Qwen3-4B-GGUF`
   - For **SGLang** (NVIDIA): paste a repo like `Qwen/Qwen2.5-7B-Instruct` and select the SGLang engine
3. The model will automatically load when you make an API request
4. Use the OpenAI-compatible API at http://localhost:8337/v1/

## API Usage

All endpoints are OpenAI-compatible. Use the official OpenAI Python SDK or any compatible client.

### Chat Completion

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8337/v1",
    api_key="not-needed"  # No API key required by default
)

response = client.chat.completions.create(
    model="Qwen3-4B-Q4_K_M",  # GGUF model name (llama.cpp)
    # model="Qwen2.5-7B-Instruct",  # Or a full HF model (SGLang)
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Embeddings

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/v1", api_key="not-needed")

response = client.embeddings.create(
    model="Qwen3-Embedding-0.6B-Q8_0",
    input="Hello, world!"
)

print(response.data[0].embedding[:5])  # First 5 dimensions
```

### Audio Transcription

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/v1", api_key="not-needed")

with open("audio.mp3", "rb") as audio_file:
    response = client.audio.transcriptions.create(
        model="ggml-large-v3-turbo",  # Your whisper model
        file=audio_file
    )

print(response.text)
```

### cURL Examples

```bash
# Chat completion
curl http://localhost:8337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Q4_K_M",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Embeddings
curl http://localhost:8337/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Embedding-0.6B-Q8_0",
    "input": "Hello, world!"
  }'

# Audio transcription (multipart form)
curl http://localhost:8337/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=ggml-large-v3-turbo"
```

## Docker Deployment

### Basic (CPU)

```bash
docker-compose up -d
```

### NVIDIA GPU (Desktop/Server)

```bash
docker-compose -f docker-compose.nvidia.yml up -d
```

### NVIDIA Jetson (Thor/Orin/AGX)

For Jetson devices, use the Jetson-optimized compose file:

```bash
# Recommended: Use Jetson-optimized configuration
docker-compose -f docker-compose.jetson.yml up -d

# Or run directly with nvidia runtime
docker run -d \
  --name cyber-inference \
  --runtime nvidia \
  -p 8337:8337 \
  -v cyber-models:/app/models \
  -v cyber-data:/app/data \
  -e CYBER_INFERENCE_LLAMA_GPU_LAYERS=-1 \
  cyber-inference:jetson
```

**Jetson Prerequisites:**
- JetPack 6.0+ installed
- NVIDIA Container Runtime configured (`/etc/docker/daemon.json`):
  ```json
  {
    "runtimes": {
      "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
      }
    },
    "default-runtime": "nvidia"
  }
  ```
- Restart Docker after configuration: `sudo systemctl restart docker`

**Recommended Models for Jetson:**

| Device | Memory | Recommended Models |
|--------|--------|-------------------|
| Jetson Thor | 128GB+ | GPT-OSS 20B, Nemotron Nano 3 30B, Whisper Large V3 |
| Jetson Orin | 32-64GB | Qwen3 4B, GLM 4.6V Flash, Whisper Large V3 Turbo |
| Jetson AGX | 16-32GB | Qwen3 4B, GLM 4.6V Flash, Whisper Medium |
| Jetson Nano | 4-8GB | Qwen3 Embedding 0.6B, BGE M3, Whisper Small |

### Build from Source

```bash
docker build -t cyber-inference .
docker run -d --name cyber-inference -p 8337:8337 \
  -v cyber-models:/app/models \
  -v cyber-data:/app/data \
  cyber-inference
```

### Build for Jetson (ARM64)

```bash
# On Jetson device or with buildx for ARM64
docker build -f Dockerfile.nvidia -t cyber-inference:jetson .
docker run --name cyber-inference --runtime nvidia -d --gpus all -p 8337:8337 \
  -v cyber-models:/app/models \
  -v cyber-data:/app/data \
  cyber-inference:jetson
```

## Configuration

Configure via environment variables (prefixed with `CYBER_INFERENCE_`) or the web GUI settings page:

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `CYBER_INFERENCE_PORT` | 8337 | Server port |
| `CYBER_INFERENCE_HOST` | 0.0.0.0 | Bind address |
| `CYBER_INFERENCE_LOG_LEVEL` | INFO | Log level |
| `CYBER_INFERENCE_MODELS_DIR` | ./models | Model storage directory |
| `CYBER_INFERENCE_DATA_DIR` | ./data | Data/logs/config directory |
| `CYBER_INFERENCE_MAX_LOADED_MODELS` | 1 | Maximum concurrent models |
| `CYBER_INFERENCE_MODEL_IDLE_TIMEOUT` | 300 | Seconds before unloading idle model |
| `CYBER_INFERENCE_MAX_MEMORY_PERCENT` | 80.0 | Maximum memory usage percentage |
| `CYBER_INFERENCE_ADMIN_PASSWORD` | None | Optional admin password |
| `CYBER_INFERENCE_HF_TOKEN` | None | HuggingFace token for private models |

### llama.cpp

| Variable | Default | Description |
|----------|---------|-------------|
| `CYBER_INFERENCE_LLAMA_GPU_LAYERS` | -1 | GPU layers (-1 = auto) |
| `CYBER_INFERENCE_LLAMA_SERVER_BASE_PORT` | 8338 | Base port for llama.cpp servers |
| `CYBER_INFERENCE_LLAMA_THREADS` | None | CPU threads (auto if unset) |

### SGLang

| Variable | Default | Description |
|----------|---------|-------------|
| `CYBER_INFERENCE_NO_SGLANG` | 0 | Set to 1 to disable SGLang even on NVIDIA |
| `CYBER_INFERENCE_SGLANG_MEM_FRACTION` | 0.85 | KV cache pool memory fraction |
| `CYBER_INFERENCE_SGLANG_TP_SIZE` | 1 | Tensor parallelism degree (number of GPUs) |
| `CYBER_INFERENCE_SGLANG_BASE_PORT` | 8350 | Base port for SGLang servers |

### Startup Script

| Variable | Default | Description |
|----------|---------|-------------|
| `CYBER_INFERENCE_RESTART_DELAY` | 2 | Base delay between restarts (seconds) |
| `CYBER_INFERENCE_MAX_RESTARTS` | 10 | Max consecutive crash restarts before giving up |

## CLI Commands

```bash
# Start the server
cyber-inference serve [OPTIONS]

# Initialize directories and database
cyber-inference init

# Install/update llama.cpp server binary
cyber-inference install-llama

# Install/update whisper.cpp server binary
cyber-inference install-whisper

# Install SGLang with CUDA support
cyber-inference install-sglang [--cuda cu130]

# Download a GGUF model (llama.cpp)
cyber-inference download-model ggml-org/gpt-oss-20b-GGUF

# Download a HuggingFace model (SGLang)
cyber-inference download-model Qwen/Qwen2.5-7B-Instruct --engine sglang

# List downloaded models
cyber-inference list-models

# Show version
cyber-inference version
```

## API Endpoints

### V1 Endpoints (OpenAI-Compatible)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{model_id}` | Get model info |
| POST | `/v1/chat/completions` | Chat completion (streaming supported) |
| POST | `/v1/completions` | Text completion (streaming supported) |
| POST | `/v1/embeddings` | Generate embeddings |
| POST | `/v1/audio/transcriptions` | Transcribe audio to text (Whisper) |
| POST | `/v1/audio/translations` | Translate audio to English (Whisper) |

All V1 endpoints automatically route to the correct engine (llama.cpp, SGLang, or whisper.cpp) based on the model's `engine_type`.

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/admin/login` | Authenticate and receive JWT |
| GET | `/admin/status` | Server status and loaded models |
| GET | `/admin/resources` | System resource usage (CPU, RAM, GPU) |
| GET | `/admin/models` | List all registered models (includes `engine_type`) |
| GET | `/admin/models/repo-files` | List GGUF files in a HuggingFace repo |
| GET | `/admin/models/suggest-mmproj` | Get suggested mmproj for a vision model |
| POST | `/admin/models/download` | Download a GGUF model |
| POST | `/admin/models/download-sglang` | Download a HuggingFace model for SGLang |
| POST | `/admin/models/{name}/load` | Load a model into memory |
| POST | `/admin/models/{name}/unload` | Unload a model from memory |
| DELETE | `/admin/models/{name}` | Delete a model (files + database) |
| GET | `/admin/sessions` | List active model sessions |
| GET | `/admin/config` | Get current configuration |
| PUT | `/admin/config/{key}` | Update a configuration value |
| GET | `/admin/sglang/status` | SGLang engine status and CUDA info |
| GET | `/admin/sglang/repo-info` | HuggingFace repo info for SGLang |
| POST | `/admin/shutdown` | Graceful server shutdown |

All `/admin/*` endpoints require a Bearer token when `CYBER_INFERENCE_ADMIN_PASSWORD` is set.

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI (interactive API docs) |
| GET | `/redoc` | ReDoc documentation |
| WS | `/ws/logs` | Real-time log streaming |
| WS | `/ws/status` | Real-time status updates |

## Architecture

```
+-------------------------------------------------------------+
|                    FastAPI Application                       |
|  +----------+  +----------+  +----------+  +----------+    |
|  | Web GUI  |  |  V1 API  |  |Admin API |  |WebSocket |    |
|  +----------+  +----------+  +----------+  +----------+    |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                      Core Services                          |
|  +----------+  +----------+  +----------+  +----------+    |
|  | Process  |  |  Model   |  | Resource |  |  Auto    |    |
|  | Manager  |  | Manager  |  | Monitor  |  | Loader   |    |
|  +----------+  +----------+  +----------+  +----------+    |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                   Inference Engines                          |
|  +----------------+  +----------------+  +----------------+ |
|  |  llama-server  |  | SGLang server  |  | whisper-server | |
|  |  (GGUF models) |  | (HF models)   |  | (transcription)| |
|  |  :8338, 8339.. |  | :8350, 8351.. |  | :834X, ...     | |
|  +----------------+  +----------------+  +----------------+ |
+-------------------------------------------------------------+
```

**Engine selection is automatic.** When you download a model, it is tagged with an `engine_type` (`llama`, `sglang`, or `whisper`). The V1 API routes each request to the correct backend based on this tag. Models load on first request and unload after the idle timeout.

## SGLang Engine

SGLang provides high-performance GPU inference for full-precision HuggingFace models. It is automatically installed when `start.sh` detects an NVIDIA GPU.

**How it works:**
- `start.sh` detects NVIDIA GPU via `nvidia-smi`
- Installs `sglang[all]` and its dependencies
- Reads the resolved torch and sgl-kernel versions dynamically
- Verifies CUDA wheel availability before installing
- Replaces CPU torch/sgl-kernel with CUDA-enabled versions
- Runs a smoke test to confirm CUDA is working

**Upgrading SGLang:** Delete `.venv` and re-run `./start.sh`. The script dynamically detects versions -- no hardcoded pins.

**Disabling SGLang:** `CYBER_INFERENCE_NO_SGLANG=1 ./start.sh`

## Whisper Transcription

Cyber-Inference supports speech-to-text transcription via whisper.cpp. Download Whisper models from the **ggerganov/whisper.cpp** repository.

**Recommended Whisper Models:**

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| `ggml-large-v3-turbo` | 809M | Excellent | Fast | **Recommended** - Best balance |
| `ggml-large-v3` | 1.5B | Best | Slower | Maximum accuracy |
| `ggml-medium` | 769M | Good | Fast | General use |
| `ggml-small` | 244M | Fair | Very Fast | Low resource/real-time |

**Supported Audio Formats:** mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg

## Security

- **Admin Password**: Set `CYBER_INFERENCE_ADMIN_PASSWORD` to protect admin endpoints
- **JWT Authentication**: Secure token-based auth for admin operations
- **Local Binding**: By default, binds to `0.0.0.0` -- restrict for production

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

GPU GPLv3 Licensed.

(c) Matthew Rogers 2025. All rights reserved.

*Free Software*

---

### Connect With Me

[![GitHub](https://img.shields.io/badge/GitHub-ramborogers-181717?style=for-the-badge&logo=github)](https://github.com/ramborogers)
[![Twitter](https://img.shields.io/badge/Twitter-@matthewrogers-1DA1F2?style=for-the-badge&logo=twitter)](https://x.com/matthewrogers)
[![Website](https://img.shields.io/badge/Web-matthewrogers.org-00ADD8?style=for-the-badge&logo=google-chrome)](https://matthewrogers.org)
