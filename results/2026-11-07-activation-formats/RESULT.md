<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Which 4-bit formats d3-ultra tolerates (2026-11-07)

These are sanity checks for the next kernel lever. Each format below names
an MMA instruction that could consume it without re-encoding weights on
every pass, which is what holds the release candidate's 288-row pass at
152 ms against a ~75 ms limit. Every number is from a single run
(preliminary); the score paths are deterministic, so repeat runs give the
same answers.

## Part 1: fake quantization on d3's own BF16 runtime

`scripts/mxquant.py` fake-quantizes the weights, and optionally the inputs,
of the matrices the engine runs in 4-bit: MLP gate/up/down and attention
q/k/v/o. Everything else stays BF16. The suites run against d3's own server
(`scripts/mxserve.py`, `scripts/mxbench.sh`), with d3's own temperatures.

- **Weights:**
  - NVFP4: the SparkScore converter. Attention goes through FP8 first, as the
    engine loads it.
  - MXFP4 + row: E2M1 with a UE8M0 scale per 32. The exponent is chosen per
    block by squared error, after a per-row FP32 scale picked from 8
    candidates; the row scale is applied in the epilogue.
- **Activations:**
  - E4M3 tok: E4M3 with one FP32 scale per token (today's engine).
  - MX8: E4M3 with a UE8M0 scale per 32.
  - NVFP4: E2M1 with E4M3 scales per 16 and a per-token FP32 scale.
  - NVFP4 ×2: that, plus a second NVFP4 term for the remainder under the same
    per-token scale. It maps to two FP4 MMAs into one accumulator.

| Weights / activations | JevBench (231) | Brier | kev v4 acc | NLL | ECE | cov@5% |
|---|---:|---:|---:|---:|---:|---:|
| BF16 / BF16 (d3, reference) | 211 | 0.117 | | | | |
| NVFP4 / BF16 | 210 | 0.122 | | | | |
| NVFP4 / E4M3 tok | 210 | 0.123 | 0.877 | 0.448 | 0.087 | 0.793 |
| NVFP4 / NVFP4 ×2 | 211 | 0.122 | 0.872 | 0.451 | 0.090 | 0.794 |
| NVFP4 / NVFP4 | 211 | 0.123 | 0.870 | 0.448 | 0.090 | 0.794 |
| MXFP4 + row / BF16 | 209 | 0.132 | | | | |
| MXFP4 + row / E4M3 tok | 207 | 0.134 | 0.872 | 0.463 | 0.093 | 0.771 |
| MXFP4 + row / MX8 | 208 | 0.130 | 0.869 | 0.458 | 0.094 | 0.796 |

**Weight error:** relative RMS 0.095 for NVFP4. For MXFP4 it is 0.117 with
OCP scales, 0.113 with the best exponent per block, and 0.110 with the row
scale as well.

**Oracle, against d3's BF16 probabilities (72 questions):** this cannot
separate the formats. Mean KL is driven by two or three hard questions, so
every format lands at 0.005–0.020.

- MXFP4 + row / BF16 scores best: 0.0062, 71/72 agree.
- Single-term NVFP4 activations have the best trimmed mean, 0.0009 without
  the worst three, and the worst single question, 0.64.

**Findings:**
- MXFP4 weights cost a little with every activation format: 2–4 JevBench
  items, about +0.01 Brier, and +0.01–0.015 kev NLL against NVFP4 weights.
- Activations in NVFP4, one term or two, show no loss against today's E4M3
  per-token scheme.

## Part 2: the engine's existing W4A4 kernels

These ran on the release candidate (Atlas `8434214d8`) with every serve
setting unchanged except `ATLAS_DENSE_PREFILL_LOWP`, using the fitted
temperatures (`scripts/lowp_variant.sh`). The W4A4 kernels are the older
8-warp design (`kind::mxf4nvf4` m16n8k64), not the warp-specialized one.

| Engine (LOWP) | JevBench | Brier | ECE | p50 | p95 | kev v4 acc | NLL | ECE | Oracle KL (49) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RC: W4A8 (`1`) | 213 | 0.115 | 0.038 | 158 ms | 1,312 ms | 0.870 | 0.334 | 0.033 | 0.0055, 48/49 |
| W4A4 everywhere (`gu,dn,qkv,o,qkvz,gout`) | 210 | 0.118 | 0.029 | 152 ms | 1,081 ms | 0.877 | 0.334 | 0.030 | 0.0148, 47/49 |
| W4A4 MLP, W4A8 attention (`gu,dn,qkv8,o8,qkvz,gout`) | 212 | 0.118 | 0.026 | 153 ms | 1,103 ms | 0.870 | 0.328 | 0.033 | 0.0169, 48/49 |

The RC kev row is the fitted-temperature readout from
`results/2026-11-07-d3-ultra-final-accuracy`.

Engine time per pass (median of 5):

| Rows | RC | W4A4 everywhere | W4A4 MLP |
|---:|---:|---:|---:|
| 96 | 107 ms | 103 ms | 104 ms |
| 288 | 152 ms | 144 ms | 146 ms |
| 1,056 | 486 ms | 398 ms | 403 ms |
| 4,128 | 1,952 ms | 1,673 ms | 1,692 ms |

**Accuracy:** the engine's W4A4 matches the RC on the suites. kev is equal,
and JevBench is within 1–3 items. Its oracle KL is 2.7–3.1× higher, but most
of that comes from one question (`smoke/date/overdue_mar_20`, max |dp| 0.47).

**Speed:** the old W4A4 kernels are already 5% faster at 288 rows and 14–18%
faster on long passes. They skip the re-encode but were never tuned. Rounds
5–6 replaced the 8-warp design they share with the faster warp-specialized,
144-row tiling, but only for W4A8.

## What this means

- **Native NVFP4 × NVFP4 (single term)** is the strongest option. It
  consumes the checkpoint's bytes as they are, at 431 TFLOPS, with no
  re-encode. The compute floor at 288 rows halves to about 33 ms, so the pass
  becomes bound by its weight read.
- **NVFP4 ×2 activations** are the safe fallback. Accuracy matches W4A8
  exactly as far as these suites can tell, with the FP8 rate and no re-encode.
  It is the same kernel with a second MMA.
- **MXFP4 weights** are the weakest option: a small but consistent accuracy
  cost, plus a new checkpoint.
