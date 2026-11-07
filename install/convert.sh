#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Download d3-ultra and the ModelOpt format reference at their pinned revisions,
# convert d3-ultra to NVFP4, validate the result and check it against the
# published checksums. Runs in NVIDIA's PyTorch container on this Spark's GPU.
#
#   install/convert.sh MODEL_ROOT
#
# Writes MODEL_ROOT/src/{d3-ultra,Qwen3.8-27B-NVFP4} (about 52 + 22 GB, kept so
# a rerun skips the download) and MODEL_ROOT/d3-ultra-NVFP4 (about 22 GB).
# Both checkpoints are Apache-2.0 and ungated; no Hugging Face login is needed.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=${1:?usage: install/convert.sh MODEL_ROOT}
mkdir -p "$root/src"
root=$(cd "$root" && pwd)
out=$root/d3-ultra-NVFP4
image=nvcr.io/nvidia/pytorch:25.10-py3@sha256:42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee

read -r src_id src_rev ref_id ref_rev < <(python3 - "$here/checkpoints.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))["checkpoints"]["d3-ultra-nvfp4"]
print(c["source"], c["source_revision"], c["format_reference"].split()[0], c["format_reference_revision"])
EOF
)
if [[ -f $out/.converted ]]; then
  echo "$out is already converted; delete it to convert again"
else
  docker run --rm --gpus all --ipc=host -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 \
    -v "$root:/work" -v "$here/../tools:/tools:ro" -v "$here/d3-ultra-nvfp4.sha256:/sums:ro" "$image" bash -euo pipefail -c "
      pip install -q --no-cache-dir 'huggingface_hub[cli]>=0.36,<2'
      hf download $src_id --revision $src_rev --local-dir /work/src/d3-ultra
      hf download $ref_id --revision $ref_rev --local-dir /work/src/Qwen3.8-27B-NVFP4
      rm -rf /work/d3-ultra-NVFP4.partial
      python3 /tools/nvfp4_convert.py convert --src /work/src/d3-ultra --ref /work/src/Qwen3.8-27B-NVFP4 \
        --out /work/d3-ultra-NVFP4.partial --readout /work/src/d3-ultra --drop-mtp
      python3 /tools/nvfp4_convert.py validate --src /work/src/d3-ultra --ref /work/src/Qwen3.8-27B-NVFP4 \
        --out /work/d3-ultra-NVFP4.partial --readout /work/src/d3-ultra
      rm -rf /work/d3-ultra-NVFP4 && mv /work/d3-ultra-NVFP4.partial /work/d3-ultra-NVFP4
      chown -R $(id -u):$(id -g) /work/src /work/d3-ultra-NVFP4"
  touch "$out/.converted"
fi
# The conversion is deterministic: a correct one matches the reference build
# byte for byte.
echo "checking $out against install/d3-ultra-nvfp4.sha256"
(cd "$out" && sha256sum --quiet -c "$here/d3-ultra-nvfp4.sha256") && echo "checksums match the reference build"
