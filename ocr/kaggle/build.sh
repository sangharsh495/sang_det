#!/usr/bin/env bash
# Build the single-file Kaggle kernel (plate_format.py pasted into train_v6.py)
# and push it.  Usage: ocr/kaggle/build.sh [--push]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/build"
mkdir -p "$out"
{
  cat "$here/../plate_format.py" | sed 's/^from __future__ import annotations$//'
  echo
  echo "# ---------------------------------------------------------------- train_v6"
  cat "$here/train_v6.py"
} > "$out/train_v6.py"
cp "$here/kernel-metadata.json" "$out/"
python3 -m py_compile "$out/train_v6.py"
echo "built $out/train_v6.py"
if [[ "${1:-}" == "--push" ]]; then
  kaggle kernels push -p "$out"
fi
