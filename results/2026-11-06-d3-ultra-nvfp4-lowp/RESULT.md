<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# d3-ultra NVFP4: tensor-core GDN scan and W4A8 / W8A8 prefill GEMMs

This run uses the same checkpoint, template and harness as
`results/2026-11-06-d3-ultra-nvfp4-bf16mma`. Two engine levers are new:

- **`ATLAS_GDN_CHUNK_TC=1`:** a fused, chunked Gated DeltaNet scan on BF16
  tensor cores. It keeps the FP32 state in registers and splits FP32 operands
  into BF16 hi + lo pairs. The pass also does less arena and SSM-state zeroing,
  and the MRoPE `pow` is evaluated once per position.
- **`ATLAS_DENSE_PREFILL_LOWP=1`:**
  - The MLP and attention projections run W4A8 on `mma kind::mxf8f6f4`.
    NVFP4 weights are re-encoded per 64-K stage as E4M3 with a UE8M0 row scale.
    This is not exact: values with more than 4 significant bits round once.
  - The GDN projections run W8A8 (`m16n8k32`) on the checkpoint's per-tensor
    FP8 weights.
  - Tokens are quantized per token with a power-of-two scale.
  - Results do not depend on which requests share a pass: merged and solo
    passes matched on 36 of 36 requests.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at
  `2660f1b3c`, built for `gb10 / qwen3.8-27b / nvfp4`.
- **Serve flags:** `--kv-cache-dtype bf16 --max-seq-len 16384
  --max-prefill-tokens 16384 --max-num-seqs 32 --max-batch-size 32
  --gpu-memory-utilization 0.80 --disable-thinking --lm-head-dtype bf16`.
- **Environment:** `ATLAS_SSM_DECODE_RING=0 ATLAS_GDN_BATCHED_FLA=1
  ATLAS_DENSE_PREFILL_BF16MMA=1 ATLAS_GDN_CHUNK_TC=1
  ATLAS_DENSE_PREFILL_LOWP=1`.
- **Harness:** `tools/sparkscore`, template `d3-ultra`, `score` backend,
  `prefix_bidir`. The harness ran on the same Spark as the engine.
- **Date:** one DGX Spark, 2026-11-06.

## JevBench v1.6.1, public items (231), one run each

| Engine | Accuracy | Macro | Brier | ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| **This run** | **92.2%** | **93.5%** | 0.119 | 0.036 | **192 ms** | **1,493 ms** |
| BF16-MMA kernels | 91.3% | 92.8% | 0.122 | 0.032 | 288 ms | 2,357 ms |
| First default-kernel run | 91.8% | 93.2% | 0.118 | 0.033 | 567 ms | 4,587 ms |
| d3's own BF16 runtime | 91.3% | 92.9% | 0.117 | 0.033 | 633 ms | 3,070 ms |

The accuracy differences between these rows are 1–2 items, which is within
noise at n = 231.

## Engine time per score pass (median of 5, `score_sweep_d3.py`, one 32-token branch)

| Rows | Default kernels | BF16-MMA | This run |
|---:|---:|---:|---:|
| 96 | 258 ms | 144 ms | 113 ms |
| 288 | 570 ms | 280 ms | 186 ms |
| 1,056 | 1,831 ms | 903 ms | 545 ms |
| 4,128 | 6,820 ms | 3,521 ms | 2,308 ms |

## Against d3's BF16 oracle (d3's exact prompt tokens, 49 questions)

| Mask | Mean KL | Same answer |
|---|---:|---:|
| `prefix_bidir` | 0.0055 | 48/49 |
| Causal | 0.0386 | 45/49 |

These 49-answer readings move by about ±0.005 under small perturbations.
Read the result as "at the BF16-activation level", not as better than it.

W4A4 on the MLP (`mxf4nvf4`) was 15% faster at 288 rows but read mean KL
0.0205, above the 0.0163 bar, so it stays off.

## Distance from the hardware limit (preliminary)

At 288 rows the floor is about 75 ms. Reading about 17 GB of weights at
234 GB/s takes about 73 ms. FP8 compute at the measured 215 TFLOPS takes
about 65 ms.

This pass takes 186 ms, of which the GEMMs are about 158 ms. W4A8 currently
reaches about 58% of the FP8 peak; it is limited by the weight re-encode in
the ALUs.
