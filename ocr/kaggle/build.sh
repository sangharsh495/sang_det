#!/usr/bin/env bash
# Build a single-file Kaggle kernel (plate_format.py pasted above the script)
# and optionally push it.
#   ocr/kaggle/build.sh v6 [--push]     -> kernel from train_v6.py + kernel-v6.json
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ver="${1:?usage: build.sh v6|v7 [--push]}"
out="$here/build/$ver"
mkdir -p "$out"
{
  sed 's/^from __future__ import annotations$//' "$here/../plate_format.py"
  echo
  echo "# ---------------------------------------------------------------- train_$ver"
  cat "$here/train_$ver.py"
} > "$out/train_$ver.py"
cp "$here/kernel-$ver.json" "$out/kernel-metadata.json"
python3 -m py_compile "$out/train_$ver.py"
echo "built $out/train_$ver.py"
if [[ "${2:-}" == "--push" ]]; then
  kaggle kernels push -p "$out"
fi
