#!/bin/bash
#
# Cyber-Inference Start Script
#
# This script verifies prerequisites and starts the Cyber-Inference server.
# It will auto-restart the server if it exits (use Ctrl+C to stop).
#
# Usage:
#     ./start.sh
#
# Requirements:
#     - uv (https://github.com/astral-sh/uv)
#     - python3 (3.10 or higher)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
error() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to check if a command exists
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get command version
get_version() {
    if command -v "$1" >/dev/null 2>&1; then
        "$1" --version 2>&1 | head -n 1
    else
        echo "not found"
    fi
}

echo "═══════════════════════════════════════════════════════════"
echo "  Cyber-Inference Startup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check for uv (install automatically if not found)
info "Checking for uv..."
if ! check_command uv; then
    warning "uv is not installed. Installing automatically..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add uv to PATH for current session
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        if ! check_command uv; then
            error "uv was installed but not found in PATH"
            echo ""
            echo "Please add ~/.local/bin or ~/.cargo/bin to your PATH and restart."
            exit 1
        fi
        success "uv installed successfully"
    else
        error "Failed to install uv automatically"
        echo ""
        echo "Please install uv manually:"
        echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  Or visit: https://github.com/astral-sh/uv"
        echo ""
        exit 1
    fi
fi

UV_VERSION=$(get_version uv)
success "Found uv: $UV_VERSION"

# Check for python3
info "Checking for python3..."
if ! check_command python3; then
    error "python3 is not installed or not in PATH"
    echo ""
    echo "Please install Python 3.10 or higher:"
    echo "  macOS: brew install python@3.12"
    echo "  Linux: sudo apt-get install python3 python3-venv"
    echo ""
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
success "Found python3: $PYTHON_VERSION"

# Check Python version (3.10+)
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    error "Python 3.10 or higher is required (found $PYTHON_VERSION)"
    echo ""
    echo "Please upgrade Python to version 3.10 or higher."
    exit 1
fi

echo ""
info "All prerequisites met!"
echo ""

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run uv sync
info "Synchronizing dependencies with uv sync..."
if ! uv sync; then
    error "Failed to sync dependencies"
    exit 1
fi
success "Dependencies synchronized"

echo ""
info "Starting Cyber-Inference server..."
echo ""

# Auto-restart loop
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
