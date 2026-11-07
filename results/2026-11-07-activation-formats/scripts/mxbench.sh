#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# usage: mxbench.sh WEIGHTS "ACTS..." [kev]   one d3 server with fake-quantized weights; per activation scheme,
# JevBench public (and kev transfer-v4 with "kev"). Logs to ~/sparkscore/d3work/mxbench-WEIGHTS.log
cd "$HOME/sparkscore/d3work"
w=$1; acts=$2; kev=$3; actf=/w/d3work/mxbench-act-$w.txt
echo bf16 > mxbench-act-$w.txt
docker rm -f mxs-$w >/dev/null 2>&1
DECISION_API_KEY=$(cat server.key) ./d3run.sh mxs-$w --gpu --detach -- python -u /w/d3work/mxserve.py /w/models/d3-ultra /w/repo/tools $w file:$actf -- \
  --model /w/models/d3-ultra --device cuda:0 --host 127.0.0.1 --port 8095 --name d3-ultra
for i in $(seq 1 120); do curl -sf localhost:8095/health >/dev/null 2>&1 && break; docker ps -q -f name=mxs-$w | grep -q . || { echo "server died"; exit 1; }; sleep 10; done
echo "server up $(date -u +%T)"
export SPARKSCORE_API_KEY=$(cat server.key)
for a in $acts; do
  echo $a > mxbench-act-$w.txt
  echo "== $w $a jevbench $(date -u +%T)"
  (cd ~/sparkscore/repo && .venv/bin/python -m tools.sparkscore.bench.run jevbench --harness ../jevbench --endpoint http://127.0.0.1:8095 \
     --model d3-ultra --out ../results/jevbench-d3own-fq-$w-$a > ../d3work/mxbench-jev-$w-$a.log 2>&1); echo "jev exit $?"
  if [ "$kev" = kev ]; then
    echo "== $w $a kev $(date -u +%T)"
    (cd ~/sparkscore/repo && .venv/bin/python -m tools.sparkscore.bench.run kev --harness ../kev --python ../kev/.venv/bin/python \
       --endpoint http://127.0.0.1:8095 --model d3-ultra --suite evals/v4/transfer-v4 --out ../results/kev-transfer-v4-d3own-fq-$w-$a \
       > ../d3work/mxbench-kev-$w-$a.log 2>&1); echo "kev exit $?"
  fi
done
docker logs mxs-$w 2>&1 | grep "MXQ acts now"
docker rm -f mxs-$w >/dev/null
echo "MXBENCH_DONE $(date -u +%T)"
