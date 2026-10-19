#!/usr/bin/env bash
# Build (and optionally push) one CPU kernel per harvest shard: day1 day2 night.
#   ocr/kaggle/build_harvest.sh [--push]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
for shard in day1 day2 night; do
  out="$here/build/harvest-$shard"
  mkdir -p "$out"
  {
    sed 's/^from __future__ import annotations$//' "$here/../plate_format.py"
    echo
    sed "s/^SHARD = 'day1'$/SHARD = '$shard'/" "$here/train_ytharvest.py"
  } > "$out/train_ytharvest.py"
  python3 -m py_compile "$out/train_ytharvest.py"
  cat > "$out/kernel-metadata.json" <<JSON
{
  "id": "sangharshgedekar/plate-harvest-$shard",
  "title": "plate-harvest-$shard",
  "code_file": "train_ytharvest.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
JSON
  grep -q "SHARD = '$shard'" "$out/train_ytharvest.py"
  echo "built $out"
  if [[ "${1:-}" == "--push" ]]; then kaggle kernels push -p "$out"; fi
done
