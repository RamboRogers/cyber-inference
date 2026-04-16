#!/usr/bin/env bash
# Stage a llama.cpp build into a compact runtime directory for Docker image assembly.
# Usage: stage-llama-runtime.sh <llama.cpp build dir> <output dir>
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <llama.cpp build dir> <output dir>" >&2
  exit 64
fi

build_dir="$1"
out_dir="$2"
server=""

for candidate in \
  "$build_dir/bin/llama-server" \
  "$build_dir/examples/server/llama-server" \
  "$build_dir/llama-server"; do
  if [[ -x "$candidate" ]]; then
    server="$candidate"
    break
  fi
done

if [[ -z "$server" ]]; then
  echo "llama-server not found under $build_dir" >&2
  find "$build_dir" -maxdepth 4 -type f -name 'llama-server*' -print >&2 || true
  exit 1
fi

rm -rf "$out_dir"
mkdir -p "$out_dir"
cp -a "$server" "$out_dir/llama-server"
chmod +x "$out_dir/llama-server"

# Copy llama.cpp-built shared objects next to llama-server. System and CUDA runtime
# libraries are supplied by the target NVIDIA base images; keeping only build-tree
# libraries avoids accidentally shadowing glibc or host driver libraries.
while IFS= read -r -d '' lib; do
  cp -a "$lib" "$out_dir/"
done < <(find "$build_dir" -type f \( -name 'libggml*.so*' -o -name 'libllama*.so*' -o -name 'libmtmd*.so*' \) -print0)

# Also copy symlink entries when the build generated soname symlinks.
while IFS= read -r -d '' lib; do
  cp -a "$lib" "$out_dir/"
done < <(find "$build_dir" -type l \( -name 'libggml*.so*' -o -name 'libllama*.so*' -o -name 'libmtmd*.so*' \) -print0)

if command -v ldd >/dev/null 2>&1; then
  echo "llama-server dynamic dependencies:"
  ldd "$out_dir/llama-server" || true
fi

LD_LIBRARY_PATH="$out_dir:${LD_LIBRARY_PATH:-}" "$out_dir/llama-server" --version
