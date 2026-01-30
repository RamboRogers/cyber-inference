# Cyber-Inference

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-00ff9f?style=for-the-badge&logo=python&logoColor=00ff9f" alt="Python">
  <img src="https://img.shields.io/badge/License-GPLv3-00ff9f?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/llama.cpp-Powered-00ff9f?style=for-the-badge" alt="llama.cpp">
  <img src="https://img.shields.io/badge/whisper.cpp-Powered-00ff9f?style=for-the-badge" alt="whisper.cpp">
  <img src="https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Jetson">
</p>

<p align="center">
<img src="cyber-inference.png">
  <strong>🌐 Edge Inference Server Management with OpenAI-Compatible API</strong>
</p>

---

Cyber-Inference is a web GUI management tool for running OpenAI-compatible inference servers. Built on llama.cpp and whisper.cpp, it provides automatic model management, dynamic resource allocation, and a beautiful cyberpunk-themed interface designed for edge deployment. Supports LLMs, Vision Language Models (VLM), Embeddings, and Audio Transcription (Whisper).

## Why Cyber-Inference?

I made this project because I need to deploy inference servers on the edge, for labs and for personal use. I want an easy way to copy paste models from huggingface and have them automatically downloaded and loaded into the inference server llama.cpp. I don't want to manage memmory, and I just want to have standby inference that is ready to go when I need it (auto model loading and unloading).

The new llama.cpp is amazing, but the models linger and suck up memory and resources and it doesn't support model downloads from the webui. I wrote this project to solve these problems.

Specifically, I made this project for MacOS and NVIDIA Jetson devices.

## ✨ Features

- **🖥️ Cyberpunk Web GUI** - Beautiful dark-mode interface with neon green accents, real-time updates, and responsive design
- **🔌 OpenAI-Compatible API** - Drop-in replacement for OpenAI's `/v1/` endpoints
- **📦 Automatic Model Management** - Download models from HuggingFace with one click
- **⚡ Dynamic Loading** - Models load on-demand and unload when idle
- **🎮 GPU Acceleration** - Automatic detection and support for NVIDIA CUDA, Apple Metal, and CPU
- **🤖 Jetson Ready** - Native support for NVIDIA Jetson Thor, Orin, and AGX platforms
- **📊 Resource Monitoring** - Real-time CPU, RAM, and GPU usage tracking
- **🔒 Optional Security** - Admin password protection with JWT authentication
- **🐳 Docker Ready** - Full Docker and docker-compose support with NVIDIA runtime
- **🎙️ Audio Transcription** - Speech-to-text with Whisper models via whisper.cpp
- **👁️ Vision Models** - Multimodal support with automatic mmproj file handling
- **📊 Embeddings** - Text embedding models for RAG and semantic search

## 🎯 Supported Model Types

| Type | Server | Models | API Endpoint |
|------|--------|--------|--------------|
| **Chat/LLM** | llama-server | Llama, Qwen, Mistral, etc. | `/v1/chat/completions` |
| **Vision (VLM)** | llama-server | Qwen-VL, GLM-4V, etc. | `/v1/chat/completions` |
| **Embeddings** | llama-server | BGE, Qwen-Embed, E5, etc. | `/v1/embeddings` |
| **Transcription** | whisper-server | Whisper Large V3, etc. | `/v1/audio/transcriptions` |

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## One shot Script

```bash
# Clone the repository
git clone https://github.com/ramborogers/cyber-inference.git
cd cyber-inference

# Run the one shot script
./start.sh
```

### Installation

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
2. Navigate to **Models** and download a model (e.g., `ggml-org/Qwen3-4B-GGUF`)
3. The model will automatically load when you make an API request
4. Use the OpenAI-compatible API at http://localhost:8337/v1/

## 📖 API Usage

### Chat Completion (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8337/v1",
    api_key="not-needed"  # No API key required by default
)

response = client.chat.completions.create(
    model="Qwen3-4B-Q4_K_M",  # Your downloaded model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Embeddings (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/v1", api_key="not-needed")

response = client.embeddings.create(
    model="Qwen3-Embedding-0.6B-Q8_0",
    input="Hello, world!"
)

print(response.data[0].embedding[:5])  # First 5 dimensions
```

### Audio Transcription (Python)

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

# Audio transcription
curl http://localhost:8337/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=ggml-large-v3-turbo"
```

## 🐳 Docker Deployment

### Basic (CPU)

```bash
docker-compose up -d
```

### NVIDIA GPU (Desktop/Server)

```bash
docker-compose -f docker-compose.nvidia.yml up -d
```

### 🤖 NVIDIA Jetson (Thor/Orin/AGX)

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

## ⚙️ Configuration

Configure via environment variables or command-line arguments:

| Variable | Default | Description |
|----------|---------|-------------|
| `CYBER_INFERENCE_PORT` | 8337 | Server port |
| `CYBER_INFERENCE_HOST` | 0.0.0.0 | Bind address |
| `CYBER_INFERENCE_LOG_LEVEL` | INFO | Log level |
| `CYBER_INFERENCE_MODELS_DIR` | ./models | Model storage directory |
| `CYBER_INFERENCE_MAX_LOADED_MODELS` | 3 | Maximum concurrent models |
| `CYBER_INFERENCE_MODEL_IDLE_TIMEOUT` | 300 | Seconds before unloading idle model |
| `CYBER_INFERENCE_ADMIN_PASSWORD` | None | Optional admin password |
| `CYBER_INFERENCE_LLAMA_GPU_LAYERS` | -1 | GPU layers (-1 = auto) |
| `CYBER_INFERENCE_HF_TOKEN` | None | HuggingFace token for private models |

## 🔧 CLI Commands

```bash
# Start the server
cyber-inference serve [OPTIONS]

# Initialize directories and database
cyber-inference init

# Install/update llama.cpp server binary
cyber-inference install-llama

# Install/update whisper.cpp server binary (for transcription)
cyber-inference install-whisper

# Download a model
cyber-inference download-model ggml-org/gpt-oss-20b-GGUF

# List downloaded models
cyber-inference list-models

# Show version
cyber-inference version
```

## 📡 API Endpoints

### V1 Endpoints (OpenAI-Compatible)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (streaming supported) |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Generate embeddings |
| POST | `/v1/audio/transcriptions` | Transcribe audio to text (Whisper) |
| POST | `/v1/audio/translations` | Translate audio to English (Whisper) |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/admin/login` | Authenticate admin and receive JWT |
| GET | `/admin/status` | Server status |
| GET | `/admin/resources` | System resources |
| GET | `/admin/models` | List all models |
| POST | `/admin/models/download` | Download a model |
| POST | `/admin/models/{name:path}/load` | Load a model |
| POST | `/admin/models/{name:path}/unload` | Unload a model |
| DELETE | `/admin/models/{name:path}` | Delete a model |
| GET | `/admin/sessions` | List active model sessions |
| GET | `/admin/config` | List configuration values |
| PUT | `/admin/config/{key}` | Update a configuration value |
| POST | `/admin/shutdown` | Graceful shutdown (for restart) |

All `/admin/*` endpoints require a Bearer token when `CYBER_INFERENCE_ADMIN_PASSWORD` is set.

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc documentation |
| WS | `/ws/logs` | Real-time log streaming |
| WS | `/ws/status` | Real-time status updates |

## 🖼️ Screenshots

The web interface features a cyberpunk aesthetic with:
- Real-time dashboard with resource monitoring
- Model management with download progress
- Live log streaming
- Settings configuration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Web GUI  │  │  V1 API  │  │Admin API │  │WebSocket │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Core Services                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Process  │  │  Model   │  │ Resource │  │  Auto    │    │
│  │ Manager  │  │ Manager  │  │ Monitor  │  │ Loader   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Inference Servers                          │
│  ┌────────────────────────┐  ┌────────────────────────┐    │
│  │     llama-server       │  │    whisper-server      │    │
│  │  (LLM/VLM/Embedding)   │  │   (Transcription)      │    │
│  │  :8338, :8339, ...     │  │   :834X, ...           │    │
│  └────────────────────────┘  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🎙️ Whisper Transcription

Cyber-Inference supports speech-to-text transcription via whisper.cpp. Download Whisper models from the **ggerganov/whisper.cpp** repository.

**Recommended Whisper Models:**

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| `ggml-large-v3-turbo` | 809M | Excellent | Fast | **Recommended** - Best balance |
| `ggml-large-v3` | 1.5B | Best | Slower | Maximum accuracy |
| `ggml-medium` | 769M | Good | Fast | General use |
| `ggml-small` | 244M | Fair | Very Fast | Low resource/real-time |

**Supported Audio Formats:** mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg

## 🔒 Security

- **Admin Password**: Set `CYBER_INFERENCE_ADMIN_PASSWORD` to protect admin endpoints
- **JWT Authentication**: Secure token-based auth for admin operations
- **Local Binding**: By default, binds to `0.0.0.0` - restrict for production

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚖️ License

GPU GPLv3 Licensed.

(c) Matthew Rogers 2025. All rights reserved.

*Free Software*

---

### Connect With Me 🤝

[![GitHub](https://img.shields.io/badge/GitHub-ramborogers-181717?style=for-the-badge&logo=github)](https://github.com/ramborogers)
[![Twitter](https://img.shields.io/badge/Twitter-@matthewrogers-1DA1F2?style=for-the-badge&logo=twitter)](https://x.com/matthewrogers)
[![Website](https://img.shields.io/badge/Web-matthewrogers.org-00ADD8?style=for-the-badge&logo=google-chrome)](https://matthewrogers.org)
