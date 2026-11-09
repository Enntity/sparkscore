<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Reproducing the results

Everything here runs on one DGX Spark with SparkScore served by `./start.sh`
(see the [README](../README.md)). Nothing else should use the GPU while you
measure. The commands print or save the same tables as the receipts under
`results/`, so you can compare line by line.

## 1. Check the install

```bash
./start.sh status                # container up, limits printed
python3 install/smoke.py         # one typed request, checked
```

`install/convert.sh` has already checked that your converted weights match
the reference build byte for byte (`install/d3-ultra-nvfp4.sha256`).

## 2. Agreement with d3-ultra's own runtime (2 minutes)

d3's BF16 runtime recorded its answers to 49 questions with their exact
prompt tokens (`tools/sparkscore/bench/data/`). Feed the same tokens to Atlas
and compare:

```bash
docker exec sparkscore /opt/venv/bin/python tools/sparkscore/bench/oracle_check.py
```

| Build | `prefix_bidir`: same answer | mean KL |
|---|---:|---:|
| W4A4 (default) | 47/49 | 0.0148 |
| W4A8 (`./start.sh --precision w4a8`) | 48/49 | 0.0055 |

The second line it prints, plain causal attention, shows why the
prefix-bidirectional mask matters: d3-ultra was trained with it.

## 3. Engine time per pass (5 minutes)

```bash
docker exec sparkscore /opt/venv/bin/python tools/sparkscore/bench/engine_sweep.py http://127.0.0.1:8081 64,256,1024,4096 1
docker exec sparkscore /opt/venv/bin/python tools/sparkscore/bench/engine_sweep.py http://127.0.0.1:8081 2048 2
docker exec sparkscore /opt/venv/bin/python tools/sparkscore/bench/concurrency.py http://127.0.0.1:8081 1 2 4 8 16
```

Expected, W4A4, median engine time:

| Rows | 96 | 288 | 1,056 | 2,112 (N = 2) | 4,128 |
|---|---:|---:|---:|---:|---:|
| ms | 101 | 139 | 372 | 788 | 1,579 |

`concurrency.py`'s last column must be 0.0000: a request merged with others
scores exactly as it does alone.

## 4. JevBench public, three runs (about 15 minutes)

```bash
git clone https://github.com/fstandhartinger/jevbench ../jevbench
git -C ../jevbench checkout 609a664e81f46e03731600bd7bd9f1b7a61257c5   # JevBench v1.6.1, the pinned harness
python3 -m venv .venv && .venv/bin/pip install -r tools/sparkscore/requirements.txt -e ../jevbench
for i in 1 2 3; do
  .venv/bin/python -m tools.sparkscore.bench.run jevbench --harness ../jevbench \
    --endpoint http://127.0.0.1:8015 --model sparkscore-d3 --out results/mine/jevbench-$i
done
```

Expected: 210 of 231 right (90.9%), Brier 0.118, p50 146–148 ms and p95
1.05–1.06 s on every run (W4A4). The answers are deterministic, so all three
runs score the same. The latency spread is run-to-run noise.

## 5. The larger suites

kev-suites and the Decision Index sample take hours. Their exact commands
and harness versions are in
[tools/sparkscore/bench/README.md](../tools/sparkscore/bench/README.md), and
the expected numbers are in
[results/2026-11-07-d3-ultra-w4a4](../results/2026-11-07-d3-ultra-w4a4/RESULT.md).

## 6. Knowledge packs (about 20 minutes)

`results/2026-11-08-knowledge-packs/raw/pack-reference.jsonl` holds d3-ultra BF16's own answers for a handbook set
(40 tickets) and a 4-rule set (112 tickets). Each item has three answers: with the knowledge as an isolated
prefix (what a pack computes), with it in the state, and with no knowledge. Against a running SparkScore with
packs on, using a harness without `--temperature` so the probabilities compare with d3's:

```bash
S=results/2026-11-08-knowledge-packs/scripts
python3 $S/g3g4.py results/2026-11-08-knowledge-packs/raw/pack-reference.jsonl http://127.0.0.1:8016   # rules, agreement, merged == alone
python3 $S/g4_speed.py results/2026-11-08-knowledge-packs/raw/pack-reference.jsonl http://127.0.0.1:8016
python3 $S/g5_long.py                                                                                  # 3K and 12K handbooks
```

Expected:
- rules 35/35 and 128/128 on packs;
- merged vs alone max |Δp| 0.000000;
- p50 about 165 ms on the 1.2K pack, against about 533 ms in the state;
- p50 about 155 ms at 3.3K and 191 ms at 12K pack tokens (`g5_long.py`).

## What can differ

- **Latency** depends on GPU clocks and temperature. A Spark running another
  workload, or one that is thermally limited, is slower. Record
  `nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv` next to your
  numbers.
- **Accuracy** should match exactly. The conversion is byte-identical and
  every score pass is deterministic. If it doesn't match, check
  `./start.sh status` for the engine commit and the precision, and open an
  issue with your `results/mine/` directory.
