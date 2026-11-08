#!/bin/sh
# serve_packs.sh TRACE   (re)start the packs Atlas build (W4A4 default levers, --score-packs 4) and the packs harness on 8016
docker exec sparkscore-packs bash -lc "pkill -f '^/build/target/q38/release/spark serve' || true"
for i in $(seq 60); do docker exec sparkscore-packs pgrep -f '^/build/target/q38/release/spark serve' >/dev/null || break; sleep 1; done
docker exec -d -e ATLAS_SSM_DECODE_RING=0 -e ATLAS_HYBRID_MOE_LAYOUT=0 -e ATLAS_GDN_BATCHED_FLA=1 -e ATLAS_SCORE_TRACE=${1:-0} \
  -e ATLAS_DENSE_PREFILL_BF16MMA=1 -e ATLAS_GDN_CHUNK_TC=1 -e ATLAS_ATTN_ROWS128=1 -e ATLAS_QWEN38_PDL=1 \
  -e ATLAS_DENSE_PREFILL_LOWP=gu,dn,qkv,o,qkvz,gout sparkscore-packs bash -lc "cd /build && RUST_LOG=info /build/target/q38/release/spark serve \
  --model-from-path /models/d3-ultra-NVFP4 --model-name m --port 8081 --kv-cache-dtype bf16 --max-seq-len 49152 --max-num-seqs 32 \
  --max-batch-size 32 --gpu-memory-utilization 0.80 --disable-thinking --max-prefill-tokens 49152 --lm-head-dtype bf16 --score-packs 4 > /build/serve.log 2>&1"
for i in $(seq 150); do curl -sf localhost:8081/health >/dev/null 2>&1 && break; sleep 2; done
for p in $(pgrep -f "tools.sparkscore.server --backend score --backend-url http://127.0.0.1:8081"); do kill $p; done
cd ~/sparkscore/repo-packs && (setsid nohup .venv/bin/python -m tools.sparkscore.server --backend score --backend-url http://127.0.0.1:8081 \
  --backend-model m --tokenizer $HOME/sparkscore/models/d3-ultra-NVFP4 --port 8016 --quiet --template d3-ultra \
  --model-id sparkscore-d3 > ../ss-8016.log 2>&1 < /dev/null &)
for i in $(seq 30); do curl -sf localhost:8016/health >/dev/null && { echo "atlas + harness up (trace ${1:-0})"; exit 0; }; sleep 1; done
echo "harness failed"; tail ../ss-8016.log; exit 1
