# Docker

Cyber-Inference ships two NVIDIA-only container targets:

| Target | Image |
| --- | --- |
| Linux AMD64 NVIDIA hosts | `ghcr.io/ramborogers/cyber-inference:linux-amd64` |
| Thor / DGX Spark ARM64 NVIDIA hosts | `ghcr.io/ramborogers/cyber-inference:thor-arm64` |

## What each image contains

### Linux AMD64

- CUDA 13 runtime base
- Cyber-Inference app and Python environment
- empty `/app/bin` for runtime-installed `llama-server`

This image does **not** bundle `llama.cpp`. On startup, Cyber-Inference attempts to install a
compatible `llama-server` into `/app/bin`.

### Thor / DGX Spark ARM64

- CUDA 13 runtime base
- Cyber-Inference app and Python environment
- `llama-server` built natively on `thor.lab` and baked into `/app/bin`

The Thor publish workflow clones the current `llama.cpp` default branch, builds it on Thor hardware,
stages the runtime files into `docker/build/llama-bin`, and then builds the image. The GitHub
Actions job is routed to the self-hosted runner registered with the labels
`self-hosted`, `Linux`, `ARM64`, `NVIDIA`, and `Thor`.

## Required host mounts

Both images expect persistent host directories:

- `./data` -> `/app/data`
- `./models` -> `/app/models`

## Linux AMD64 run

```bash
mkdir -p data models

docker pull ghcr.io/ramborogers/cyber-inference:linux-amd64

docker run -d --name cyber-inference \
  --gpus all \
  -p 8337:8337 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  ghcr.io/ramborogers/cyber-inference:linux-amd64
```

## Thor / DGX Spark ARM64 run

```bash
mkdir -p data models

docker pull ghcr.io/ramborogers/cyber-inference:thor-arm64

docker run -d --name cyber-inference \
  --runtime nvidia \
  -p 8337:8337 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  ghcr.io/ramborogers/cyber-inference:thor-arm64
```

## Upgrade

Pick the correct tag for the host and restart with the same mounts:

```bash
TARGET_TAG=linux-amd64  # or thor-arm64

docker stop cyber-inference
docker rm cyber-inference
docker pull "ghcr.io/ramborogers/cyber-inference:${TARGET_TAG}"

docker run -d --name cyber-inference \
  --gpus all \
  -p 8337:8337 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  "ghcr.io/ramborogers/cyber-inference:${TARGET_TAG}"
```

For Thor hosts that require the NVIDIA runtime flag instead of `--gpus all`, replace that line with:

```bash
  --runtime nvidia \
```

## macOS

Docker is not the right path for macOS MPS. Use the native local install/startup flow instead.

## Publish workflow

The publish workflow lives at `.github/workflows/publish-containers.yml`.

- Linux AMD64 builds on a GitHub-hosted Linux runner.
- Thor ARM64 builds on the self-hosted GitHub runner labeled `self-hosted`, `Linux`, `ARM64`,
  `NVIDIA`, `Thor`.
- Both publish to GHCR under the tags above.
