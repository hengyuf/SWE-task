#!/bin/bash
# Copy the GLM-5.3 checkpoint to node-local /dev/shm (RAM) for fast reloads during experiments.
# Usage (on a compute node of your own job): stage_to_shm.sh [src] [dst]
set -euo pipefail
SRC="${1:-/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3}"
DST="${2:-/dev/shm/GLM-5.3}"
mkdir -p "$DST"
t0=$(date +%s)
find "$SRC" -maxdepth 1 -type f ! -name '*.safetensors' -exec cp -p {} "$DST/" \;
copy_one() {
  local f="$1" d="$2/$(basename "$1")"
  if [[ -f "$d" && "$(stat -c %s "$d")" == "$(stat -c %s "$f")" ]]; then return 0; fi
  cp "$f" "$d.tmp" && mv -f "$d.tmp" "$d"
}
export -f copy_one
ls "$SRC"/*.safetensors | xargs -P 16 -I{} bash -c 'copy_one "$1" "$2"' _ {} "$DST"
n=$(ls "$DST"/*.safetensors | wc -l)
echo "[$(date -Is)] $(hostname) staged $n shards to $DST in $(( $(date +%s) - t0 )) s; $(du -sh "$DST" | cut -f1)"
