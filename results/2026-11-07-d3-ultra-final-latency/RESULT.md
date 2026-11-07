<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# d3-ultra NVFP4 on SparkScore: final latency runs

This is the latency half of the final benchmark for the SparkScore release
candidate. The accuracy half is `results/2026-11-07-d3-ultra-final-accuracy`.
Both halves give the same probabilities: the kernels added since the accuracy
build are bit-identical to it.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at `8434214d8`,
  built for `gb10 / qwen3.8-27b / nvfp4`.
- **Engine environment:** `ATLAS_SSM_DECODE_RING=0 ATLAS_GDN_BATCHED_FLA=1
  ATLAS_DENSE_PREFILL_BF16MMA=1 ATLAS_GDN_CHUNK_TC=1 ATLAS_ATTN_ROWS128=1
  ATLAS_QWEN38_PDL=1 ATLAS_DENSE_PREFILL_LOWP=1`.
- **Serve flags:** `--kv-cache-dtype bf16 --max-seq-len 16384
  --max-prefill-tokens 16384 --max-num-seqs 32 --max-batch-size 32
  --gpu-memory-utilization 0.80 --disable-thinking --lm-head-dtype bf16`.
- **Harness:** `tools/sparkscore`, template `d3-ultra`, `score` backend,
  `prefix_bidir`, with the fitted temperatures
  `choice=1.636,noul=2.366,score=1.588` (the defaults in
  `install/checkpoints.json`). The harness ran on the same Spark as the
  engine.
- **Hardware:** one DGX Spark, 2026-11-07. `raw/fingerprint.txt` records the
  run.
- **Earlier run:** one on `b61e0e37a`, a kernel round earlier, is in git
  history (p50 180 ms).

## JevBench v1.6.1, public items (231), three runs

| Run | Accuracy | Macro | Brier | ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 92.2% | 93.5% | 0.115 | 0.038 | 158 ms | 1,312 ms |
| 2 | 92.2% | 93.5% | 0.115 | 0.038 | 159 ms | 1,309 ms |
| 3 | 92.2% | 93.5% | 0.115 | 0.038 | 158 ms | 1,316 ms |

- **Accuracy:** the three runs give the same answers, because the score path
  is deterministic.
- **Latency:** p50 is 158–159 ms and p95 1.31–1.32 s. It is measured at the
  harness, per question, one request at a time.

Compared on the same items:

| Engine | Accuracy | p50 | p95 |
|---|---:|---:|---:|
| **d3-ultra, SparkScore** | **92.2%** | **158 ms** | **1,312 ms** |
| d3-ultra, d3's own BF16 runtime | 91.3% | 633 ms | 3,070 ms |
| KnowLine-35B-A3B, SparkScore | 87.4% | 176 ms | 1,211 ms |
| d3-ultra, SparkScore, first working build (2026-11-06) | 91.8% | 567 ms | 4,587 ms |

- **Against d3's own runtime:** 4.0× faster at p50 and 2.3× faster at p95.
- **Against KnowLine:** 4.8 points more accurate, and faster at p50 than the
  KnowLine model, which has only 3B active parameters. KnowLine's run is
  older, on an earlier engine build.

## Engine time per score pass

The harness is `score_sweep_d3.py`: one prefix plus N branches of 32 tokens,
`prefix_bidir`, median of 5 runs. Compute is 49 GFLOP per row.

| Rows | First working build | **Now** | Effective TFLOPS | Speedup |
|---:|---:|---:|---:|---:|
| 96 | 258 ms | **107 ms** | 44 | 2.4× |
| 288 | 570 ms | **152 ms** | 93 | 3.7× |
| 1,056 | 1,831 ms | **486 ms** | 106 | 3.8× |
| 2,112 (N = 2) | ~3,495 ms | **1,001 ms** | 103 | 3.5× |
| 4,128 | 6,820 ms | **1,952 ms** | 104 | 3.5× |

The p50 JevBench request is about 288 rows; the p90 is about 2,500.

### Distance from the hardware limit

- **Hardware peaks:** BF16 MMA 108 TFLOPS, FP8 215, stream read 234 GB/s,
  all measured on GB10.
- **At 288 rows:** the weight read is about 73 ms (about 17 GB) and FP8
  compute about 65 ms, so the limit is about 75 ms. The pass takes 152 ms,
  2.0× the limit. Its matrix multiplies take about 130 ms.
- **At 1,056 rows and up:** the pass runs at about 105 TFLOPS, about 48% of
  the FP8 peak.
- **What is in the way:** the 4-bit weights are re-encoded to 8-bit on every
  pass. The pace is set by how long one producer warp takes to re-encode one
  stage, not by memory, the tensor cores, or the total amount of re-encode
  work. Pre-encoding the weights at load is bit-identical. It does not help
  at 288 rows: 8-bit weights make the weight read the floor below about 465
  rows. For 1–5% on long passes it would cost 17 GB, so it is not used.

## Concurrent requests

The harness is `concurrency.py`. Each request has a distinct 300-token prefix
and 4 branches of 40 tokens, about 460 rows. K requests are sent at once.

| K | p50 latency | All done | Requests merged into one pass | Max difference vs solo |
|---:|---:|---:|---:|---:|
| 1 | 251 ms | 251 ms | 1 | 0 |
| 2 | 377 ms | 500 ms | 1 | 0 |
| 4 | 933 ms | 934 ms | 3 | 0 |
| 8 | 1,807 ms | 1,808 ms | 6 | 0 |
| 16 | 2,482 ms | 3,620 ms | 6 | 0 |

- **Correctness:** merged requests give exactly the probabilities they give
  alone.
- **Throughput:** a pass of this size is already compute-bound, so merging
  adds little. 16 requests finish in 3.6 s (4.4 requests/s), against 4.0
  requests/s one at a time.

## Against d3's BF16 oracle (d3's exact prompt tokens, 49 questions)

| Mask | Mean KL | Same answer |
|---|---:|---:|
| `prefix_bidir` | 0.00546 | 48/49 |
| Causal | 0.0386 | 45/49 |

These are unchanged from the accuracy build.

## Receipts

`raw/` holds the fingerprint, the oracle, sweep and concurrency outputs, and
the three JevBench runs with their ledgers and responses.
