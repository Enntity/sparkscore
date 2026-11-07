<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# BF16-MMA MoE prefill kernel (`ATLAS_MOE_PREFILL_BF16MMA=1`) on KnowLine

This is a one-variable comparison against
`results/2026-11-06-knowline-35b-a3b-gen1.5-nvfp4` (read twice, port 8010).
Only the MoE prefill kernel differs.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at `4f4c0baab`.
  It includes the kernel commits `fd96109ad`, `b1242a88e` and `201f62fd9` and
  query merging (`8600f8723`).
- **Environment:** the same as the baseline, plus `ATLAS_MOE_PREFILL_BF16MMA=1`.
- **Checkpoint and harness:** unchanged (KnowLine NVFP4, `--lm-head-dtype bf16`,
  `knowline-gen1.5`, read twice).
- **What the kernel changes:** NVFP4 weights are dequantized exactly to BF16
  and the MMA runs in BF16 (`mma.sync m16n8k16`). Activations stay BF16. The
  default kernel dequantizes to FP8 and also rounds activations to FP8.

## Latency

`raw/sweep-knowline-bf16mma.txt`: engine time, median of 5 warm repeats.
32-token branches. The "before" column is the zero-shot sweep, so it was
measured on a different checkpoint with the same architecture.

| State × questions | Rows | Default kernel | BF16-MMA |
|---|---:|---:|---:|
| 64 × 1 | 96 | 99.5 ms | 79.7 ms |
| 256 × 4 | 384 | 161.5 | 135.9 |
| 1,024 × 1 | 1,056 | 304.4 | 243.8 |
| 1,024 × 16 | 1,536 | 439.7 | 363.6 |
| 4,096 × 64 | 6,144 | 1,681.7 | 1,408.0 |

## Accuracy (one run each)

| | Default kernel | BF16-MMA |
|---|---:|---:|
| JevBench public: accuracy / Brier / ECE | 87.4% / 0.165 / 0.025 | 86.1% / 0.167 / 0.034 |
| JevBench p50 / p95 | 176 / 1,211 ms | 177 / 1,047 ms |
| kev decision-v7: accuracy / NLL | 87.6% / 0.340 | 87.5% / 0.339 |
| kev transfer-v4: accuracy / NLL | 86.4% / 0.357 | 85.7% / 0.359 |
| kev transfer-v9: accuracy / NLL | 83.3% / 0.472 | 82.4% / 0.476 |

## Reading

- **Accuracy: not yet determined.** BF16-MMA is 0.1–1.3 points lower on all
  four suites (about 18 of 3,197 items), and NLL and Brier are unchanged. A
  single run each cannot separate that from noise.
- **One candidate mechanism, untested:** KnowLine was evaluated with FP8 at
  load, and the default kernel's FP8 activation rounding is closer to that.
- **JevBench p50 did not improve, although the sweep is 15–20% faster at
  these row counts.** Not yet explained.
- **Decision:** the kernel stays opt-in. Next is a paired A/B on more items
  (per-item McNemar test), plus end-to-end timing of the harness.
