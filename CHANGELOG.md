# Changelog

All notable changes to this project will be documented in this file.

The release log is maintained by the release automation and mirrored into GitHub Releases.

## [0.2.1] - 2026-04-17

### Release Notes

Changes since `v0.2.0`:

- Make GitHub releases materialize from pushes to master
- Automate release-tagged container publishing

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.2] - 2026-04-17

### Release Notes

Changes since `v0.2.1`:

- Dispatch container publishing from the release workflow

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.3] - 2026-04-17

### Release Notes

Changes since `v0.2.2`:

- Cleanup Readme

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.4] - 2026-04-17

### Release Notes

Changes since `v0.2.3`:

- README Update

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.5] - 2026-04-18

### Release Notes

Changes since `v0.2.4`:

- Bundle whisper.cpp runtime in Thor container builds

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.6] - 2026-04-18

### Release Notes

Changes since `v0.2.5`:

- Fix Thor smoke quoting so CUDA verification runs to completion

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.8] - 2026-05-18

### Release Notes

Changes since `v0.2.6`:

- Enable default MTP GGUF support with healthy release probes

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic MTP speculative decoding for detected GGUF models, with managed llama.cpp upgrade when needed
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.9] - 2026-05-18

### Release Notes

Changes since `v0.2.8`:

- Retune default MTP GGUF launches for llama.cpp's merged MTP path

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic MTP speculative decoding with `--parallel 1`, `--flash-attn on`, tuned draft tokens, and Qwen3.6 `preserve_thinking` chat-template kwargs
- Graceful startup migration for legacy auto-populated MTP draft-token settings
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.10] - 2026-05-18

### Release Notes

Changes since `v0.2.9`:

- No user-facing changes were recorded for this release.

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic MTP speculative decoding for detected GGUF models, with managed llama.cpp upgrade when needed
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development

## [0.2.11] - 2026-05-18

### Release Notes

Changes since `v0.2.10`:

- Keep release lockfile in sync

### Core Functions

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/audio/*`)
- Model download + registration from HuggingFace, including split GGUF shard sets
- Automatic MTP speculative decoding for detected GGUF models, with managed llama.cpp upgrade when needed
- Automatic lazy loading and idle unloading
- Web dashboard for model and resource management
- Optional admin auth (JWT)
- NVIDIA-only published container images for Linux AMD64 and Thor ARM64
- Native local startup paths for macOS Apple Silicon and non-container development
