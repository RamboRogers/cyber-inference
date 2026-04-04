#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p src/cyber_inference/web/static/css
npx --yes tailwindcss@3.4.17 \
  -c tailwind.config.js \
  -i src/cyber_inference/web/assets/app.css \
  -o src/cyber_inference/web/static/css/app.css \
  --minify
