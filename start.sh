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
        # 6b. Replace PyTorch with proper CUDA wheels
        TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_WHL}"
        info "Installing PyTorch with ${CUDA_WHL} from ${TORCH_INDEX}..."
        if uv pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"; then
            success "PyTorch ${CUDA_WHL} installed"
        else
            warning "PyTorch ${CUDA_WHL} install failed – trying reinstall..."
            uv pip install --reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX" || \
                warning "PyTorch CUDA install failed"
        fi

        # 6c. Install sgl-kernel from CUDA-specific wheel
        ARCH=$(uname -m)  # aarch64 or x86_64
        SGL_KERNEL_VER="0.3.21"
        SGL_WHL="https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VER}/sgl_kernel-${SGL_KERNEL_VER}+${CUDA_WHL}-cp310-abi3-manylinux2014_${ARCH}.whl"
        info "Installing sgl-kernel ${SGL_KERNEL_VER}+${CUDA_WHL} (${ARCH})..."
        if uv pip install --reinstall "$SGL_WHL"; then
            success "sgl-kernel ${CUDA_WHL} installed"
        else
            warning "Direct wheel failed – trying index fallback..."
            uv pip install --reinstall sgl-kernel \
                --extra-index-url "https://docs.sglang.io/whl/${CUDA_WHL}/sgl-kernel/" || \
                warning "sgl-kernel install failed"
        fi

        # 6d. Quick smoke test
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
# 7. Run with auto-restart
# ──────────────────────────────────────────────────────────────
RESTART_DELAY="${CYBER_INFERENCE_RESTART_DELAY:-2}"

while true; do
    exit_code=0
    uv run cyber-inference serve || exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        warning "Server exited cleanly. Restarting in ${RESTART_DELAY}s..."
    else
        warning "Server exited with code ${exit_code}. Restarting in ${RESTART_DELAY}s..."
    fi
    sleep "$RESTART_DELAY"
done
