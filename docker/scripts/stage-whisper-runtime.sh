#!/usr/bin/env bash
# Stage a whisper.cpp build into an isolated runtime directory for Docker image assembly.
# Usage: stage-whisper-runtime.sh <whisper.cpp build dir> <output dir>
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <whisper.cpp build dir> <output dir>" >&2
  exit 64
fi

build_dir="$1"
out_dir="$2"
payload_dir="$out_dir/whisper"
server=""

for candidate in \
  "$build_dir/bin/whisper-server" \
  "$build_dir/examples/server/whisper-server" \
  "$build_dir/whisper-server"; do
  if [[ -x "$candidate" ]]; then
    server="$candidate"
    break
  fi
done

if [[ -z "$server" ]]; then
  echo "whisper-server not found under $build_dir" >&2
  find "$build_dir" -maxdepth 4 -type f -name 'whisper-server*' -print >&2 || true
  exit 1
fi

rm -rf "$out_dir"
mkdir -p "$payload_dir"
cp -a "$server" "$payload_dir/whisper-server"
chmod +x "$payload_dir/whisper-server"

# Keep whisper runtime libraries isolated from /app/bin so they cannot collide
# with the separately built llama.cpp ggml runtime.
while IFS= read -r -d '' lib; do
  cp -a "$lib" "$payload_dir/"
done < <(
  find "$build_dir" -type f \( -name 'libggml*.so*' -o -name 'libwhisper*.so*' \) -print0
)

while IFS= read -r -d '' lib; do
  cp -a "$lib" "$payload_dir/"
done < <(
  find "$build_dir" -type l \( -name 'libggml*.so*' -o -name 'libwhisper*.so*' \) -print0
)

printf '%s\n' \
  '#!/bin/sh' \
  'set -eu' \
  'wrapper_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)' \
  'export LD_LIBRARY_PATH="$wrapper_dir/whisper:${LD_LIBRARY_PATH:-}"' \
  'exec "$wrapper_dir/whisper/whisper-server" "$@"' \
  > "$out_dir/whisper-server"
chmod +x "$out_dir/whisper-server"

if command -v ldd >/dev/null 2>&1; then
  echo "whisper-server dynamic dependencies:"
  ldd_output="$(ldd "$payload_dir/whisper-server")"
  printf '%s\n' "$ldd_output"
  if printf '%s\n' "$ldd_output" | grep -q 'not found'; then
    echo "whisper-server has unresolved dynamic dependencies" >&2
    exit 1
  fi
fi

if ! "$out_dir/whisper-server" --version; then
  "$out_dir/whisper-server" --help >/dev/null
fi
