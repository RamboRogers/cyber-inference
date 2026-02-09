#!/bin/bash
#
# Cyber-Inference Start Script
#
# This script verifies prerequisites and starts the Cyber-Inference server.
# It will auto-restart the server if it exits (use Ctrl+C to stop).
#
# When NVIDIA hardware is detected, SGLang + PyTorch CUDA wheels are
# installed automatically. No manual flags needed.
#
# Usage:
#     ./start.sh
#     CYBER_INFERENCE_NO_SGLANG=1 ./start.sh   # Force disable SGLang
#
# Requirements:
#     - uv (https://github.com/astral-sh/uv)
#     - python3 (3.12 or higher)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

error()   { echo -e "${RED}❌ Error: $1${NC}" >&2; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

check_command() { command -v "$1" >/dev/null 2>&1; }

echo "═══════════════════════════════════════════════════════════"
echo "  Cyber-Inference Startup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ──────────────────────────────────────────────────────────────
# 1. uv
# ──────────────────────────────────────────────────────────────
info "Checking for uv..."
if ! check_command uv; then
    warning "uv not found – installing..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        check_command uv || { error "uv installed but not in PATH"; exit 1; }
        success "uv installed"
    else
        error "Failed to install uv – see https://github.com/astral-sh/uv"
        exit 1
    fi
fi
success "Found uv: $(uv --version 2>&1 | head -1)"

# ──────────────────────────────────────────────────────────────
# 2. Python 3.12+
# ──────────────────────────────────────────────────────────────
info "Checking for python3..."
check_command python3 || { error "python3 not found"; exit 1; }

PYTHON_VERSION=$(python3 --version 2>&1)
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    error "Python 3.12+ required (found $PYTHON_VERSION)"
    exit 1
fi
success "Found $PYTHON_VERSION"

# ──────────────────────────────────────────────────────────────
# 3. NVIDIA GPU / CUDA detection
# ──────────────────────────────────────────────────────────────
CUDA_AVAILABLE=0
CUDA_VER_TAG=""   # e.g. cu130

if check_command nvidia-smi; then
    CUDA_AVAILABLE=1
    CUDA_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
    success "NVIDIA GPU: $CUDA_INFO"

    # Detect CUDA toolkit version from nvidia-smi
    CUDA_FULL=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    if [ -n "$CUDA_FULL" ]; then
        CUDA_MAJOR=$(echo "$CUDA_FULL" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_FULL" | cut -d. -f2)
        CUDA_VER_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"
        success "CUDA version: $CUDA_FULL ($CUDA_VER_TAG)"
    fi

    # Set CUDA_HOME for Triton/SGLang JIT (ptxas etc.)
    if [ -z "$CUDA_HOME" ]; then
        for d in /usr/local/cuda /usr/local/cuda-${CUDA_FULL} /usr/local/cuda-${CUDA_MAJOR}; do
            [ -d "$d" ] && { export CUDA_HOME="$d"; break; }
        done
    fi
    if [ -n "$CUDA_HOME" ]; then
        success "CUDA_HOME: $CUDA_HOME"
        [ -f "$CUDA_HOME/bin/ptxas" ] && export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
    fi
else
    info "No NVIDIA GPU detected"
fi

# ──────────────────────────────────────────────────────────────
# 4. Project root
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ──────────────────────────────────────────────────────────────
# 5. Sync base dependencies
# ──────────────────────────────────────────────────────────────
info "Syncing base dependencies..."
uv sync || { error "uv sync failed"; exit 1; }
success "Base dependencies OK"

# ──────────────────────────────────────────────────────────────
# 6. NVIDIA detected → install SGLang + CUDA PyTorch automatically
# ──────────────────────────────────────────────────────────────
NO_SGLANG="${CYBER_INFERENCE_NO_SGLANG:-0}"

if [ "$CUDA_AVAILABLE" -eq 1 ] && [ "$NO_SGLANG" != "1" ]; then
    echo ""
    info "NVIDIA GPU found – setting up SGLang engine..."

    # Determine CUDA wheel tag (default cu130)
    CUDA_WHL="${CUDA_VER_TAG:-cu130}"

    # 6a. Install sglang[all] via uv extras
    info "Installing sglang[all]..."
    if uv sync --extra sglang; then
        success "sglang[all] installed"
    else
        warning "sglang[all] sync failed – continuing without SGLang"
        CUDA_AVAILABLE=0
    fi

    if [ "$CUDA_AVAILABLE" -eq 1 ]; then
        # 6b. Read the versions that sglang's dependency resolver chose
        TORCH_VER=$(uv pip show torch 2>/dev/null | grep '^Version:' | awk '{print $2}' | sed 's/+.*//')
        KERNEL_VER=$(uv pip show sgl-kernel 2>/dev/null | grep '^Version:' | awk '{print $2}' | sed 's/+.*//')
        ARCH=$(uname -m)  # aarch64 or x86_64

        info "Detected torch ${TORCH_VER}, sgl-kernel ${KERNEL_VER}, arch ${ARCH}"

        # 6c. Replace PyTorch with CUDA wheels (--reinstall needed: CPU has same version number)
        TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_WHL}"
        TORCH_CHECK_URL="${TORCH_INDEX}/torch/"

        # Verify the CUDA index has this torch version before pinning
        if curl -sfL "$TORCH_CHECK_URL" 2>/dev/null | grep -q "torch-${TORCH_VER}"; then
            info "Verified: PyTorch ${TORCH_VER} exists on ${CUDA_WHL} index"
            if uv pip install --reinstall "torch==${TORCH_VER}" torchvision torchaudio --index-url "$TORCH_INDEX"; then
                success "PyTorch ${TORCH_VER}+${CUDA_WHL} installed"
            else
                warning "PyTorch CUDA install failed – SGLang may not work"
            fi
        else
            warning "PyTorch ${TORCH_VER} not found on ${CUDA_WHL} index – trying latest available"
            if uv pip install --reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"; then
                success "PyTorch (latest)+${CUDA_WHL} installed"
            else
                warning "PyTorch CUDA install failed – SGLang may not work"
            fi
        fi

        # 6d. Install sgl-kernel CUDA wheel (version detected dynamically)
        if [ -n "$KERNEL_VER" ]; then
            SGL_WHL_URL="https://github.com/sgl-project/whl/releases/download/v${KERNEL_VER}/sgl_kernel-${KERNEL_VER}+${CUDA_WHL}-cp310-abi3-manylinux2014_${ARCH}.whl"

            # Verify the wheel exists before attempting install
            if curl -sfIL -o /dev/null "$SGL_WHL_URL" 2>/dev/null; then
                info "Verified: sgl-kernel ${KERNEL_VER}+${CUDA_WHL} wheel exists"
                if uv pip install --reinstall "$SGL_WHL_URL"; then
                    success "sgl-kernel ${KERNEL_VER}+${CUDA_WHL} installed"
                else
                    warning "sgl-kernel direct wheel install failed"
                fi
            else
                warning "sgl-kernel ${KERNEL_VER}+${CUDA_WHL} wheel not found at GitHub releases"
                info "Falling back to sglang CUDA wheel index..."
                uv pip install --reinstall sgl-kernel \
                    --extra-index-url "https://docs.sglang.io/whl/${CUDA_WHL}/sgl-kernel/" || \
                    warning "sgl-kernel CUDA install failed – SGLang may not work with GPU"
            fi
        else
            warning "sgl-kernel not found in environment – skipping CUDA kernel install"
        fi

        # 6e. Ensure CuDNN is new enough (PyTorch 2.9.x + CuDNN <9.15 has a known bug)
        CUDNN_VER=$(uv run python -c "import torch; print(torch.backends.cudnn.version())" 2>/dev/null)
        if [ -n "$CUDNN_VER" ] && [ "$CUDNN_VER" -lt 91500 ] 2>/dev/null; then
            info "CuDNN ${CUDNN_VER} < 9.15 detected – upgrading for PyTorch compatibility..."
            uv pip install "nvidia-cudnn-cu12>=9.15" || warning "CuDNN upgrade failed"
        fi

        # 6f. Patch SGLang Blackwell detection to include cc 11.x (NVIDIA Thor)
        #     SGLang checks device_capability_majors=[10, 12] but Thor is cc 11.0
        SGLANG_COMMON=$(uv run python -c "import sglang.srt.utils.common as m; print(m.__file__)" 2>/dev/null)
        if [ -n "$SGLANG_COMMON" ] && [ -f "$SGLANG_COMMON" ]; then
            if grep -q 'device_capability_majors=\[10, 12\]' "$SGLANG_COMMON"; then
                sed -i 's/device_capability_majors=\[10, 12\]/device_capability_majors=[10, 11, 12]/' "$SGLANG_COMMON"
                success "Patched SGLang Blackwell check to include cc 11.x (Thor)"
            fi
        fi

        # 6g. Quick smoke test
        if uv run python -c "import sglang; import torch; assert torch.cuda.is_available(); print(f'SGLang {sglang.__version__} + PyTorch {torch.__version__} CUDA OK')" 2>/dev/null; then
            success "SGLang + CUDA verified"
        else
            warning "SGLang smoke test failed – server will still start (SGLang features may be unavailable)"
        fi
    fi
elif [ "$NO_SGLANG" = "1" ]; then
    info "SGLang disabled by CYBER_INFERENCE_NO_SGLANG=1"
fi

echo ""
info "Starting Cyber-Inference server..."
echo ""

# ──────────────────────────────────────────────────────────────
# 7. Run with auto-restart (exponential backoff, max 10 restarts)
# ──────────────────────────────────────────────────────────────
RESTART_DELAY="${CYBER_INFERENCE_RESTART_DELAY:-2}"
MAX_RESTARTS="${CYBER_INFERENCE_MAX_RESTARTS:-10}"
RESTART_COUNT=0

while true; do
    exit_code=0
    uv run cyber-inference serve || exit_code=$?

    RESTART_COUNT=$((RESTART_COUNT + 1))

    if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]; then
        error "Server failed ${MAX_RESTARTS} times in a row. Giving up."
        exit 1
    fi

    # Exponential backoff: base_delay * restart_count, capped at 30s
    DELAY=$((RESTART_DELAY * RESTART_COUNT))
    [ "$DELAY" -gt 30 ] && DELAY=30

    if [ "$exit_code" -eq 0 ]; then
        warning "Server exited cleanly. Restart ${RESTART_COUNT}/${MAX_RESTARTS} in ${DELAY}s..."
        # Clean exit resets the counter (likely intentional restart, not a crash loop)
        RESTART_COUNT=0
        DELAY="$RESTART_DELAY"
    else
        warning "Server exited with code ${exit_code}. Restart ${RESTART_COUNT}/${MAX_RESTARTS} in ${DELAY}s..."
    fi
    sleep "$DELAY"
done
