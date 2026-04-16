#!/usr/bin/env bash
# Stage a llama.cpp build output into a flat runtime directory for Docker images.
# Usage: stage-llama-runtime.sh <llama.cpp-build-dir> <destination-dir>

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <llama.cpp-build-dir> <destination-dir>" >&2
  exit 2
fi

build_dir="$1"
dest_dir="$2"

if [[ ! -d "$build_dir" ]]; then
  echo "llama.cpp build directory does not exist: $build_dir" >&2
  exit 1
fi

server_path=""
for candidate in \
  "$build_dir/bin/llama-server" \
  "$build_dir/bin/server" \
  "$build_dir/examples/server/llama-server" \
  "$build_dir/examples/server/server"; do
  if [[ -f "$candidate" ]]; then
    server_path="$candidate"
    break
  fi
done

if [[ -z "$server_path" ]]; then
  server_path="$(find "$build_dir" -type f \( -name llama-server -o -name server \) | head -n 1 || true)"
fi

if [[ -z "$server_path" || ! -f "$server_path" ]]; then
  echo "could not find llama-server in $build_dir" >&2
  exit 1
fi

mkdir -p "$dest_dir"
cp "$server_path" "$dest_dir/llama-server"
chmod 0755 "$dest_dir/llama-server"

# Copy llama.cpp's own shared libraries so the final image does not depend on build-tree paths.
while IFS= read -r lib_path; do
  cp -P "$lib_path" "$dest_dir/"
done < <(find "$build_dir" -type f \( -name '*.so' -o -name '*.so.*' \) | sort -u)

# Preserve any symlinked sonames that live next to the binary.
server_dir="$(dirname "$server_path")"
find "$server_dir" -maxdepth 1 -type l \( -name '*.so' -o -name '*.so.*' \) -exec cp -P {} "$dest_dir/" \;

file_count="$(find "$dest_dir" -maxdepth 1 \( -type f -o -type l \) | wc -l | tr -d ' ')"
echo "staged $file_count llama.cpp runtime file(s) into $dest_dir"
