# Cyber-Inference Docker assets

This directory contains the supported NVIDIA-only container lanes for Cyber-Inference.
The operator-facing images are explicit target tags rather than a hidden multi-arch tag:

| Lane | Target platform | Image tag | Build host | `llama.cpp` source |
| --- | --- | --- | --- | --- |
| Linux AMD64 NVIDIA | `linux/amd64` | `ghcr.io/ramborogers/cyber-inference:linux-amd64` | GitHub-hosted Linux runner | Pinned CUDA source build inside `docker/Dockerfile.linux-amd64` |
| Thor ARM64 NVIDIA | `linux/arm64` on Thor/DGX Spark hardware | `ghcr.io/ramborogers/cyber-inference:thor-arm64` | Self-hosted `thor.lab` runner | Native CUDA build staged by `.github/workflows/publish-containers.yml` |

Unsupported Docker paths were removed from the repository. Keep Docker documentation and automation
aligned to these two tags, host bind mounts for `/app/data` and `/app/models`, and NVIDIA runtime
flags.

## Files

- `Dockerfile.linux-amd64` — builds a pinned `llama.cpp` CUDA server binary and copies the staged
  runtime into `/app/bin` before publishing the Linux AMD64 image.
- `Dockerfile.thor-arm64` — consumes `docker/build/llama-bin/` artifacts produced on the Thor runner
  and copies them into `/app/bin`.
- `scripts/stage-llama-runtime.sh` — small helper used by both lanes to flatten a `llama.cpp` build
  tree into the runtime files required by the images.

## Runtime contract

Both images set the same application-facing paths and defaults:

- `CYBER_INFERENCE_BIN_DIR=/app/bin`
- `CYBER_INFERENCE_DATA_DIR=/app/data`
- `CYBER_INFERENCE_MODELS_DIR=/app/models`
- `/app/bin` first on `PATH`
- `LD_LIBRARY_PATH=/app/bin:...`
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- `CYBER_INFERENCE_LLAMA_GPU_LAYERS=-1`
- `EXPOSE 8337`
- `VOLUME ["/app/data", "/app/models"]`
- health check against `http://localhost:8337/health`

The images run `/app/bin/llama-server --version` during build so missing binaries or shared
libraries fail before publish.
