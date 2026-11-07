<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# d3-ultra NVFP4 with the dense BF16-MMA prefill kernels

This run uses the same checkpoint, template and harness as
`results/2026-11-06-d3-ultra-nvfp4`. Only the engine changed. The 27B dense
prefill GEMMs now run on BF16 tensor cores, with weights dequantized exactly in
registers and activations kept in BF16. Before, the default kernels rounded
activations to E4M3.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at
  `ffa2abce7`, built for `gb10 / qwen3.8-27b / nvfp4`.
- **Serve flags:** `--kv-cache-dtype bf16 --max-seq-len 16384
  --max-prefill-tokens 16384 --max-num-seqs 32 --max-batch-size 32
  --gpu-memory-utilization 0.80 --disable-thinking --lm-head-dtype bf16`.
- **Environment:** `ATLAS_SSM_DECODE_RING=0 ATLAS_GDN_BATCHED_FLA=1
  ATLAS_DENSE_PREFILL_BF16MMA=1`.
- **Harness:** `tools/sparkscore`, template `d3-ultra`, `score` backend,
  `prefix_bidir`. The harness and the engine ran on the same Spark.
- **Date:** one DGX Spark, 2026-11-06.

## JevBench v1.6.1, public items (231), one run

| Engine | Accuracy | Macro | Brier | ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| **BF16-MMA kernels** | **91.3%** | **92.8%** | 0.122 | 0.032 | **288 ms** | **2,357 ms** |
| Earlier default kernels | 91.8% | 93.2% | 0.118 | 0.033 | 567 ms | 4,587 ms |
| d3's own BF16 runtime | 91.3% | 92.9% | 0.117 | 0.033 | 633 ms | 3,070 ms |

The accuracy difference from the earlier kernels is one item, which is within
run-to-run noise at n = 231. The two rows also ran on different Sparks.

## Engine time per score pass (median of 5)

The harness is `score_sweep_d3.py`: one branch of 32 tokens, `prefix_bidir`.

| Rows | Earlier kernels | BF16-MMA kernels |
|---:|---:|---:|
| 96 | 258 ms | 144 ms |
| 288 | 570 ms | 280 ms |
| 1,056 | 1,831 ms | 903 ms |
| 4,128 | 6,820 ms | 3,521 ms |

The p50 JevBench request is about 272 tokens, or about 288 rows; the p90 is
about 2,500 rows.

## Against d3's BF16 oracle (d3's exact prompt tokens, 49 questions)

| Mask | Mean KL, earlier kernels | Mean KL, BF16-MMA | Same answer |
|---|---:|---:|---:|
| `prefix_bidir` | 0.0163 | 0.0096 | 47/49 |
| Causal | 0.0494 | 0.0471 | 46/49 |

## Distance from the hardware limit (preliminary)

At 288 rows the weight-read floor is about 73 ms (about 17 GB at 234 GB/s).
Compute is about 131 ms at the BF16 peak of 108 TFLOPS, 65 ms at FP8 (215) and
33 ms at FP4 (431); all three peaks are mma.sync loops measured on GB10.

At 280 ms, this pass is at about 47% of its BF16 bound. The weight-read floor
can only be reached with lower-precision activations, which have to pass the
oracle first.
