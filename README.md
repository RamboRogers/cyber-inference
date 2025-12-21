# Cyber-Inference

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-00ff9f?style=for-the-badge&logo=python&logoColor=00ff9f" alt="Python">
  <img src="https://img.shields.io/badge/License-GPLv3-00ff9f?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/llama.cpp-Powered-00ff9f?style=for-the-badge" alt="llama.cpp">
  <img src="https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Jetson">
</p>

<p align="center">
<img src="cyber-inference.png">
  <strong>🌐 Edge Inference Server Management with OpenAI-Compatible API</strong>
</p>

---

Cyber-Inference is a web GUI management tool for running OpenAI-compatible inference servers. Built on llama.cpp, it provides automatic model management, dynamic resource allocation, and a beautiful cyberpunk-themed interface designed for edge deployment.

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

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

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
2. Navigate to **Models** and download a model (e.g., `Qwen/Qwen3-VL-4B-Instruct-GGUF`)
3. The model will automatically load when you make an API request
4. Use the OpenAI-compatible API at http://localhost:8337/v1/

## 📖 API Usage

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8337/v1",
    api_key="not-needed"  # No API key required by default
)

response = client.chat.completions.create(
    model="Llama-3.2-3B-Instruct-Q4_K_M",  # Your downloaded model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl http://localhost:8337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-Q4_K_M",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
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
| Jetson Thor | 128GB+ | GPT-OSS 20B, Nemotron Nano 3 30B |
| Jetson Orin | 32-64GB | Qwen3 4B, GLM 4.6V Flash, Nemotron Nano 3 |
| Jetson AGX | 16-32GB | Qwen3 4B, GLM 4.6V Flash |
| Jetson Nano | 4-8GB | Qwen3 Embedding 0.6B, BGE M3 |

### Build from Source

```bash
docker build -t cyber-inference .
docker run -p 8337:8337 -v cyber-models:/app/models cyber-inference
```

### Build for Jetson (ARM64)

```bash
# On Jetson device or with buildx for ARM64
docker build -f Dockerfile.nvidia -t cyber-inference:jetson .
docker run --name cyber-inference --runtime nvidia -d --gpus all -p 8337:8337 -v cyber-models:/app/models cyber-inference:jetson
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

# Install/update llama.cpp
cyber-inference install-llama

# Download a model
cyber-inference download-model TheBloke/Llama-2-7B-GGUF

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

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/status` | Server status |
| GET | `/admin/resources` | System resources |
| GET | `/admin/models` | List all models |
| POST | `/admin/models/download` | Download a model |
| POST | `/admin/models/{name}/load` | Load a model |
| POST | `/admin/models/{name}/unload` | Unload a model |
| DELETE | `/admin/models/{name}` | Delete a model |

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
│  │ Process  │  │  Model   │  │ Resource │  │  Config  │    │
│  │ Manager  │  │ Manager  │  │ Monitor  │  │ Manager  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   llama.cpp Servers                          │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│     │ :8338      │  │ :8339      │  │ :834X      │          │
│     │ Model A    │  │ Model B    │  │ Model N    │          │
│     └────────────┘  └────────────┘  └────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

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

