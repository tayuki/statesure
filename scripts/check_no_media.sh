#!/usr/bin/env bash
# Fail if image or video files are tracked outside the synthetic test assets.
set -euo pipefail
found=$(git ls-files \
  | grep -Ei '\.(jpe?g|png|webp|gif|bmp|tiff?|heic|avif|mp4|mov|mkv|avi)$' \
  | grep -v '^tests/assets/synthetic/' || true)
if [ -n "$found" ]; then
  echo "Media files are not allowed in this repository:" >&2
  echo "$found" >&2
  exit 1
fi
