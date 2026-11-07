#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Download the published NVFP4 checkpoint (the output of install/convert.sh)
# at its pinned revision and check every file against install/d3-ultra-nvfp4.sha256.
#
#   install/fetch.sh MODEL_ROOT
#
# Writes MODEL_ROOT/d3-ultra-NVFP4 (about 22 GB). Needs only Docker; no
# Hugging Face login. To build the checkpoint yourself instead, run
# install/convert.sh (or FROM_SOURCE=1 ./start.sh).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=${1:?usage: install/fetch.sh MODEL_ROOT}
mkdir -p "$root"
out=$(cd "$root" && pwd)/d3-ultra-NVFP4
read -r repo rev < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["checkpoints"]["d3-ultra-nvfp4"]["published"]; print(p["repo"], p["revision"])' "$here/checkpoints.json")
[[ $rev =~ ^[0-9a-f]{40}$ ]] || { echo "install/checkpoints.json: no pinned published revision" >&2; exit 2; }
# Hugging Face's own client downloads in parallel (several times faster than
# one curl stream); it runs in a pinned slim Python container, so nothing is
# installed on the host. The checksums are checked here, on the host.
image=python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54
mkdir -p "$out.partial"
docker run --rm -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 -v "$out.partial:/out" "$image" sh -euc "
  pip install -q --no-cache-dir --root-user-action=ignore 'huggingface_hub[hf_xet]>=1.0,<3'
  hf download $repo --revision $rev --local-dir /out --exclude README.md --exclude .gitattributes
  rm -rf /out/.cache
  chown -R $(id -u):$(id -g) /out"
(cd "$out.partial" && sha256sum --quiet -c "$here/d3-ultra-nvfp4.sha256") || { echo "the download does not match install/d3-ultra-nvfp4.sha256" >&2; exit 1; }
rm -rf "$out" && mv "$out.partial" "$out" && touch "$out/.converted"
echo "$out: every file matches install/d3-ultra-nvfp4.sha256"
