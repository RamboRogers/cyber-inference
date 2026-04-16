# Cyber-Inference Docker assets

This directory contains the supported NVIDIA-only container lanes for Cyber-Inference.
The operator-facing images are explicit target tags rather than a hidden multi-arch tag:

| Lane | Target platform | Image tag | Build host | `llama.cpp` source |
| --- | --- | --- | --- | --- |
| Linux AMD64 NVIDIA | `linux/amd64` | `ghcr.io/ramborogers/cyber-inference:linux-amd64` | GitHub-hosted Linux runner | Installed during Cyber-Inference startup into `/app/bin` |
| Thor ARM64 NVIDIA | `linux/arm64` on Thor/DGX Spark hardware | `ghcr.io/ramborogers/cyber-inference:thor-arm64` | Self-hosted GitHub runner labeled `self-hosted`, `Linux`, `ARM64`, `NVIDIA`, `Thor` | Native CUDA build from the current `llama.cpp` default branch staged by `.github/workflows/publish-containers.yml` |

Unsupported Docker paths were removed from the repository. Keep Docker documentation and automation
aligned to these two tags, host bind mounts for `/app/data` and `/app/models`, and NVIDIA runtime
flags.

## Files

- `Dockerfile.linux-amd64` — publishes the Cyber-Inference app image only and leaves `llama.cpp`
  installation to the app's normal runtime installer.
- `Dockerfile.thor-arm64` — consumes `docker/build/llama-bin/` artifacts produced on the Thor runner
  and copies them into `/app/bin`.
- `scripts/stage-llama-runtime.sh` — small helper used by the Thor workflow to flatten a
  `llama.cpp` build tree into the runtime files required by the image.

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

The Thor image verifies that `/app/bin/llama-server` is present during image build, then the GitHub
Actions workflow runs `/app/bin/llama-server --version` with the NVIDIA runtime enabled after the
image is built. The Linux AMD64 lane instead verifies that the image can run
`cyber-inference install-llama` successfully and produce `/app/bin/llama-server`.
