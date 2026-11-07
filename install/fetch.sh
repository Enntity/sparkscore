#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Download the published NVFP4 checkpoint (the output of install/convert.sh)
# at its pinned revision and check every file against install/d3-ultra-nvfp4.sha256.
#
#   install/fetch.sh MODEL_ROOT
#
# Writes MODEL_ROOT/d3-ultra-NVFP4 (about 22 GB). Needs only curl; no
# Hugging Face login. To build the checkpoint yourself instead, run
# install/convert.sh (or FROM_SOURCE=1 ./start.sh).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=${1:?usage: install/fetch.sh MODEL_ROOT}
mkdir -p "$root"
out=$(cd "$root" && pwd)/d3-ultra-NVFP4
read -r repo rev < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["checkpoints"]["d3-ultra-nvfp4"]["published"]; print(p["repo"], p["revision"])' "$here/checkpoints.json")
[[ $rev =~ ^[0-9a-f]{40}$ ]] || { echo "install/checkpoints.json: no pinned published revision" >&2; exit 2; }
mkdir -p "$out.partial"
while read -r sum file; do
  file=${file#\*}
  dest=$out.partial/$file
  if [[ -f $dest ]] && echo "$sum  $dest" | sha256sum --quiet -c - 2>/dev/null; then continue; fi
  echo "downloading $file"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "https://huggingface.co/$repo/resolve/$rev/$file"
  echo "$sum  $dest" | sha256sum --quiet -c - || { echo "$file does not match its checksum" >&2; rm -f "$dest"; exit 1; }
done < "$here/d3-ultra-nvfp4.sha256"
rm -rf "$out" && mv "$out.partial" "$out" && touch "$out/.converted"
echo "$out: every file matches install/d3-ultra-nvfp4.sha256"
