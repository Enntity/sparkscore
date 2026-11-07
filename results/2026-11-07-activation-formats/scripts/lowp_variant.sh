#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# usage: lowp_variant.sh TAG=LOWP...   per variant: serve the RC with ATLAS_DENSE_PREFILL_LOWP=LOWP, oracle, sweep, JevBench, kev transfer-v4
set -uo pipefail
cd ~/sparkscore
ATLAS_TARGET_MODEL=qwen3.8-27b TARGET_DIR=/build/target/q38 JOBS=18 ./dev.sh build | tail -1
for tv in "$@"; do
  tag=${tv%%=*}; lowp=${tv#*=}; O=results/lowp-$tag; rm -rf $O; mkdir -p $O
  echo "== $tag LOWP=$lowp $(date -u +%T)"
  TARGET_DIR=/build/target/q38 MAXSEQ=16384 SLOTS=32 UTIL=0.80 \
    EXTRA_ENV="-e ATLAS_DENSE_PREFILL_BF16MMA=1 -e ATLAS_GDN_CHUNK_TC=1 -e ATLAS_ATTN_ROWS128=1 -e ATLAS_QWEN38_PDL=1 -e ATLAS_DENSE_PREFILL_LOWP=$lowp" \
    ./dev.sh serve d3-ultra-NVFP4 --max-prefill-tokens 16384 --lm-head-dtype bf16 || continue
  ./harness.sh start 8015 score http://127.0.0.1:8081 m $HOME/sparkscore/models/d3-ultra-NVFP4 --template d3-ultra \
    --temperature choice=1.636,noul=2.366,score=1.588 --model-id sparkscore-d3 || continue
  python3 d3_mask_check.py http://127.0.0.1:8081 d3-reference.jsonl models/d3-ultra-NVFP4/decision_config.json > $O/oracle.txt 2>&1; tail -2 $O/oracle.txt
  python3 score_sweep_d3.py http://127.0.0.1:8081 64,256,1024,4096 1 > $O/sweep.txt 2>&1; cat $O/sweep.txt
  (cd repo && .venv/bin/python -m tools.sparkscore.bench.run jevbench --harness ../jevbench --endpoint http://127.0.0.1:8015 \
     --model sparkscore-d3 --out ../$O/jevbench > ../$O/jevbench.log 2>&1)
  (cd repo && .venv/bin/python -m tools.sparkscore.bench.run kev --harness ../kev --python ../kev/.venv/bin/python --endpoint http://127.0.0.1:8015 \
     --model sparkscore-d3 --suite evals/v4/transfer-v4 --out ../$O/kev-transfer-v4 > ../$O/kev.log 2>&1)
  echo "$tag done $(date -u +%T)"
done
./harness.sh stop; ./dev.sh stop
echo VARIANTS_DONE
