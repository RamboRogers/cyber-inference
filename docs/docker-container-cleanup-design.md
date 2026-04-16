# Docker Container Cleanup Design

Date: 2026-04-16
Source of truth: `.omx/plans/prd-docker-container-cleanup-20260416.md`
Verification source: `.omx/plans/test-spec-docker-container-cleanup-20260416.md`

## Decision

Adopt the PRD's separate-lane design and implement Docker support as two explicit NVIDIA-only image lanes:

| Lane | Published tag | Build host | Binary source | Runtime contract |
| --- | --- | --- | --- | --- |
| Linux AMD64 | `ghcr.io/ramborogers/cyber-inference:linux-amd64` | GitHub-hosted Linux AMD64 runner | Deterministic `llama.cpp` Linux x64 CUDA release asset fetched during image build | `/app/bin/llama-server`, `/app/data`, `/app/models`, port `8337`, NVIDIA runtime |
| Thor ARM64 | `ghcr.io/ramborogers/cyber-inference:thor-arm64` | Self-hosted `thor.lab` runner | Native `llama.cpp` CUDA build produced in the workflow workspace on Thor hardware | `/app/bin/llama-server`, `/app/data`, `/app/models`, port `8337`, NVIDIA runtime |

The supported operator interface is copy-paste `docker run` with host bind mounts for `data` and `models`. CPU containers, compose-first deployment, and generic Jetson container examples should be removed from the supported repo surface.

## Current-State Inventory

Current Docker-facing files found in the repo:

- `Dockerfile` — CPU image; conflicts with the accepted NVIDIA-only support matrix.
- `Dockerfile.nvidia` — generic NVIDIA image; mixes Linux AMD64 and Jetson/Thor concerns and still relies on first-run `llama.cpp` installation.
- `docker-compose.yml` — CPU compose path; should be deleted.
- `docker-compose.nvidia.yml` — compose-first NVIDIA path; should be deleted unless a later approval explicitly keeps compose as non-canonical.
- `docker-compose.jetson.yml` — generic Jetson compose path; should be deleted because Thor is not generic hosted ARM64/Jetson.
- `README.md` — currently documents CPU, NVIDIA, and Jetson compose flows; must be rewritten around direct `docker run`.
- No `.github/workflows/` directory currently exists.

Runtime facts that constrain the design:

- `CYBER_INFERENCE_BIN_DIR=/app/bin` is already the container convention.
- `LlamaInstaller` resolves `llama-server` from `PATH` first, then the managed binary path from `CYBER_INFERENCE_BIN_DIR`.
- Container images should set `/app/bin` on `PATH` and place baked binaries there, so the existing lookup path finds the image-provided `llama-server` without runtime downloads.
- `start.sh` still runs `uv sync`; the Dockerfiles can continue pre-syncing dependencies for faster startup, but image correctness must not depend on first-boot `llama.cpp` installation.

## Concrete File Plan

### Delete obsolete root Docker assets

Delete these files after replacement artifacts and README docs exist:

- `Dockerfile`
- `Dockerfile.nvidia`
- `docker-compose.yml`
- `docker-compose.nvidia.yml`
- `docker-compose.jetson.yml`

Rationale: leaving them in place preserves contradictory CPU, compose, and generic Jetson stories that the PRD explicitly rejects.

### Add explicit Docker lane files

Create a `docker/` directory with first-class lane definitions:

- `docker/Dockerfile.linux-amd64`
  - Use an NVIDIA CUDA Ubuntu base suitable for `linux/amd64`.
  - Install Python/uv/runtime dependencies.
  - Copy project files needed by `uv sync` and runtime.
  - Fetch a pinned or parameterized `llama.cpp` Linux x64 CUDA release asset at build time.
  - Extract `llama-server` and required `.so*` runtime libraries into `/app/bin`.
  - Set `CYBER_INFERENCE_BIN_DIR=/app/bin` and `PATH=/app/bin:...`.
  - Keep `/app/data` and `/app/models` as bind-mount targets and volumes.
  - Verify `/app/bin/llama-server --version` during build.

- `docker/Dockerfile.thor-arm64`
  - Use a Thor-compatible NVIDIA/L4T CUDA base image appropriate for the self-hosted runner.
  - Do not clone or compile `llama.cpp` inside the Dockerfile.
  - Copy prebuilt workflow artifacts from a build-context directory such as `docker/build/llama-bin/` into `/app/bin`.
  - Set the same `CYBER_INFERENCE_BIN_DIR`, `PATH`, data/model directories, healthcheck, and server command as the Linux lane.
  - Verify `/app/bin/llama-server --version` during build.

- Optional helper scripts only if they reduce risk rather than add abstraction:
  - `docker/scripts/install-llama-linux-amd64.sh` for deterministic asset fetch/extract.
  - `docker/scripts/copy-llama-runtime-libs.sh` for Thor workflow packaging.

Keep the first implementation direct; add helpers only when Dockerfile duplication becomes error-prone.

### Add publish workflow

Create `.github/workflows/publish-containers.yml` with two visible jobs:

- `linux-amd64`
  - Runs on a GitHub-hosted Linux AMD64 runner.
  - Logs in to GHCR with `GITHUB_TOKEN` and `packages: write` permissions.
  - Builds `docker/Dockerfile.linux-amd64` for `linux/amd64`.
  - Pushes `ghcr.io/ramborogers/cyber-inference:linux-amd64`.
  - Emits or runs a post-build `llama-server --version` smoke check.

- `thor-arm64`
  - Runs only on labels that route to Thor, e.g. `[self-hosted, linux, ARM64, thor.lab]` adjusted to match the actual runner registration.
  - Checks out this repository.
  - Clones `https://github.com/ggml-org/llama.cpp`.
  - Builds with the PRD-required commands:

    ```bash
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
    cmake --build build --config Release -j$(nproc)
    ```

  - Copies `llama-server` plus required runtime libraries into `docker/build/llama-bin/`.
  - Runs `docker/build/llama-bin/llama-server --version` before image build.
  - Builds `docker/Dockerfile.thor-arm64` for `linux/arm64` using the prepared artifact directory.
  - Pushes `ghcr.io/ramborogers/cyber-inference:thor-arm64`.

Use explicit target tags first. Convenience tags such as `latest` should be deferred until both lanes are proven stable.

### Rewrite README operator surface

Update `README.md` to make this support matrix unambiguous:

- Feature list: replace `Docker and docker-compose support (CPU + NVIDIA)` with NVIDIA-only published container support.
- Docker section:
  - State that Docker support is NVIDIA-only.
  - State macOS/MPS users should use native local install, not Docker.
  - Include Linux AMD64 command:

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

  - Include Thor command using the same bind mounts and the Thor tag:

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

  - Include upgrade flow that preserves mounts:

    ```bash
    docker stop cyber-inference
    docker rm cyber-inference
    docker pull ghcr.io/ramborogers/cyber-inference:<target-tag>
    docker run -d --name cyber-inference ...same mounts and target image...
    ```

  - Remove all supported-path `docker-compose` examples.

### Optional follow-up docs

If implementation needs more operator detail than fits the README, add `docs/docker.md` and link it from `README.md`. Do not keep duplicate command sets unless both are tested together.

## Implementation Order

1. Add the new `docker/` lane files and workflow in one branch slice.
2. Update `README.md` to reference only the new files/tags and bind mounts.
3. Delete obsolete Docker/compose files.
4. Run static consistency checks and local syntax checks.
5. Run full Python quality gates because the container entrypoint depends on the Python package surface, even if no Python files changed.
6. Treat Thor runtime verification as blocked until the self-hosted runner executes the workflow.

## Risks and Mitigations

| Risk | Impact | Mitigation / evidence required |
| --- | --- | --- |
| Thor runner labels do not match `[self-hosted, linux, ARM64, thor.lab]` | Thor job never runs or runs on the wrong machine | Confirm labels before merge or keep a workflow comment naming the required labels; first CI run must show Thor job on `thor.lab`. |
| Missing Thor shared libraries | Image builds but `llama-server` fails at runtime | Copy `ldd`-reported local runtime libraries into `/app/bin`; run `llama-server --version` before and after image build. |
| Linux AMD64 release asset naming changes upstream | Build breaks or fetches wrong artifact | Parameterize the llama.cpp version/asset name and fail closed when the expected CUDA x64 asset is absent. |
| `/app/bin` binary is bypassed by PATH precedence | Runtime downloads or uses an unintended binary | Set `CYBER_INFERENCE_BIN_DIR=/app/bin`; put `/app/bin` first on `PATH`; verify `which llama-server` and `/app/bin/llama-server --version`. |
| GHCR image name casing | Push fails because GHCR repository names are lower-case | Use `ramborogers` lower-case in docs, and lower-case `${{ github.repository_owner }}` in workflow if dynamic owner values are used. |
| Deleting compose surprises existing users | Operator churn | README should explicitly state the supported Docker path is direct `docker run`; compose wrappers can be user-authored from the shown commands. |
| Docker build cannot be fully validated locally on a non-NVIDIA/non-Thor workstation | Incomplete evidence before CI | Separate static/local checks from hardware-gated checks and require workflow evidence before final release. |

## Verification Plan

Minimum local/static evidence before asking for implementation approval:

```bash
find . -maxdepth 3 \( -name 'Dockerfile*' -o -name 'docker-compose*.yml' -o -path './docker/*' -o -path './.github/workflows/*' \) -print | sort
rg -n "docker-compose|CPU container|Jetson compose|Dockerfile \(CPU\)|compose-first" README.md docs docker .github Dockerfile* docker-compose* || true
rg -n "ghcr.io/ramborogers/cyber-inference:(linux-amd64|thor-arm64)|/app/data|/app/models|CYBER_INFERENCE_BIN_DIR|/app/bin" README.md docker .github
uv run ruff check .
uv run mypy src/
uv run pytest
```

Additional Linux AMD64 CI evidence:

```bash
docker build -f docker/Dockerfile.linux-amd64 -t cyber-inference:linux-amd64 .
docker run --rm --entrypoint /app/bin/llama-server cyber-inference:linux-amd64 --version
docker run --rm --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -p 8337:8337 \
  cyber-inference:linux-amd64
```

Additional Thor self-hosted evidence:

- Workflow job runs on `thor.lab` self-hosted runner.
- `llama.cpp` clone and CMake/Ninja CUDA build succeed.
- `docker/build/llama-bin/llama-server --version` succeeds before image build.
- Built image contains `/app/bin/llama-server` and required libraries.
- Thor container starts with NVIDIA runtime and healthcheck passes.
- GHCR push logs show both `linux-amd64` and `thor-arm64` tags published.

## Approval Checklist

Before broad implementation starts, approve these choices:

- Use `docker/Dockerfile.linux-amd64` and `docker/Dockerfile.thor-arm64` as final file names.
- Delete all existing root Dockerfiles and compose files rather than keeping compatibility aliases.
- Use `ghcr.io/ramborogers/cyber-inference:linux-amd64` and `ghcr.io/ramborogers/cyber-inference:thor-arm64` as initial tags.
- Use Thor runner labels `[self-hosted, linux, ARM64, thor.lab]` unless the registered labels differ.
- Defer convenience tags such as `latest` until both target-specific tags have successful runtime evidence.
