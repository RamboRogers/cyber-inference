# Docker Container Cleanup Design

This directory is the target home for the NVIDIA-only container assets described in
`.omx/plans/prd-docker-container-cleanup-20260416.md`. The current root-level
`Dockerfile`, `Dockerfile.nvidia`, and `docker-compose*.yml` files are retained only until the
implementation pass replaces them; they should not remain part of the supported container surface.

## Supported image lanes

Cyber-Inference should ship two explicit GHCR images rather than a hidden multi-arch tag:

| Lane | Target platform | Image tag | Build host | `llama.cpp` source |
| --- | --- | --- | --- | --- |
| Linux NVIDIA | `linux/amd64` | `ghcr.io/<owner>/cyber-inference:linux-amd64` | GitHub-hosted Linux runner | Deterministic x64 CUDA release asset fetched during image build |
| Thor NVIDIA | `linux/arm64` on Thor/DGX Spark hardware | `ghcr.io/<owner>/cyber-inference:thor-arm64` | `thor.lab` self-hosted runner | Native `llama.cpp` CUDA build performed by the workflow |

CPU-only containers, generic ARM64 images, Jetson compose aliases, and compose-first operator flows
are intentionally outside the supported Docker path.

## Target file layout

```text
docker/
  README.md                    # this design and implementation contract
  Dockerfile.linux-amd64       # Linux x64 NVIDIA image with baked /app/bin/llama-server
  Dockerfile.thor-arm64        # Thor ARM64 image consuming workflow-built llama.cpp artifacts
  common/                      # optional; add only if duplication becomes materially noisy
    README.md                  # documents any shared helper contract before helper scripts appear
```

Root-level Docker assets should be deleted during implementation once replacement lanes exist:

```text
Dockerfile                     # remove: CPU path is out of scope
Dockerfile.nvidia              # replace with target-specific Dockerfiles under docker/
docker-compose.yml             # remove: CPU/compose-first path is out of scope
docker-compose.nvidia.yml      # remove: README docker run is the canonical operator surface
docker-compose.jetson.yml      # remove: generic Jetson support is not the Thor lane
```

## Shared image contract

Both target Dockerfiles should keep these runtime conventions aligned so the application behavior is
identical after the target-specific binary acquisition step:

- `WORKDIR /app`
- `CYBER_INFERENCE_DATA_DIR=/app/data`
- `CYBER_INFERENCE_MODELS_DIR=/app/models`
- `CYBER_INFERENCE_BIN_DIR=/app/bin`
- `CYBER_INFERENCE_HOST=0.0.0.0`
- `CYBER_INFERENCE_PORT=8337`
- `CYBER_INFERENCE_LLAMA_GPU_LAYERS=-1`
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- `VOLUME ["/app/data", "/app/models"]`
- `EXPOSE 8337`
- health check against `http://localhost:8337/health`
- final command remains `./start.sh` unless the application startup contract changes separately

The image build must create `/app/bin`, `/app/data`, `/app/models`, and `/app/data/logs`. The shipped
`llama-server` must be executable from `/app/bin/llama-server` so the existing application lookup path
continues to work through `CYBER_INFERENCE_BIN_DIR` without runtime first-boot installation.

## `Dockerfile.linux-amd64` design

Purpose: build a deterministic NVIDIA Linux x64 image without relying on hardware autodetection.

Recommended structure:

1. Start from an NVIDIA CUDA Ubuntu runtime/devel base pinned in the Dockerfile or by a narrowly
   documented build arg.
2. Install only runtime/build utilities needed for Cyber-Inference startup and archive extraction:
   Python 3.12 tooling, `curl`, `ca-certificates`, `tar`/`unzip`, `libgomp1`, and required CUDA runtime
   libraries from the base image.
3. Install `uv` and copy the Python project files exactly as the existing NVIDIA Dockerfile does.
4. Run `uv sync` during build for faster startup.
5. Fetch an explicit `llama.cpp` Linux x64 CUDA release asset using build args such as:
   - `LLAMA_CPP_VERSION`
   - `LLAMA_CPP_LINUX_AMD64_CUDA_ASSET_URL` or a checked-in URL mapping
6. Extract `llama-server` and required bundled shared libraries into `/app/bin`.
7. Run a build-time smoke check: `/app/bin/llama-server --version`.
8. Keep the final image command as `./start.sh`.

Rejected for this lane:

- `cyber-inference install-llama` during image build, because the installer chooses assets from live
  platform/backend detection and build hosts may not expose NVIDIA hardware.
- Runtime download of `llama-server`, because published images should be operational without a first
  boot binary install.

## `Dockerfile.thor-arm64` design

Purpose: assemble a Thor ARM64 image from binaries produced natively on the Thor self-hosted runner.

Recommended structure:

1. Start from the Thor-compatible NVIDIA/L4T CUDA base selected and validated on `thor.lab`.
2. Install the same Cyber-Inference runtime dependencies as the Linux x64 lane.
3. Copy a workflow-provided artifact directory into the image, for example:
   - build context path: `.build/llama-cpp-thor/`
   - image path: `/app/bin/`
4. Ensure `/app/bin/llama-server` is executable.
5. Run `/app/bin/llama-server --version` during image build so missing runtime libraries fail before
   publish.
6. Keep the same environment, volume, port, health check, and startup command contract as the Linux
   x64 lane.

The Thor Dockerfile should not clone or compile `llama.cpp` itself. The native compile belongs in the
GitHub Actions job so compiler logs, cache behavior, and runner routing are visible at the workflow
layer.

## Optional shared helpers

Do not add helper scripts preemptively. Add `docker/common/` only if the implementation pass finds
meaningful duplication between the two Dockerfiles. Acceptable shared helpers are small shell fragments
or documented artifact-copy checks; target-specific binary acquisition must remain explicit in the lane
that owns it.

## Handoff contracts

- Workflow lane: build and push `ghcr.io/<owner>/cyber-inference:linux-amd64` from
  `docker/Dockerfile.linux-amd64`; build `llama.cpp` on `thor.lab`, stage its artifacts into the Docker
  build context, then build and push `ghcr.io/<owner>/cyber-inference:thor-arm64` from
  `docker/Dockerfile.thor-arm64`.
- README lane: document only the two supported tags above with bind-mounted host `data` and `models`
  directories, `--gpus all`, port `8337`, and an upgrade flow that preserves those host directories.
- Verification lane: confirm removed root Docker/compose files are not referenced as supported paths,
  both Dockerfiles contain `/app/bin/llama-server --version` smoke checks, and docs/workflows/Dockerfiles
  describe the same two-image matrix.
