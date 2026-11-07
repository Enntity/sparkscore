<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# KnowLine-35B-A3B-Gen1.5, NVFP4, on the one-pass score path

`PelaAI/KnowLine-35B-A3B-Gen1.5` (Apache-2.0, revision
`9adcf717361b2f7d2946d093f1e85efdf1e59010`) converted to NVFP4 with
`tools/nvfp4_convert.py` (NVIDIA's MIXED_PRECISION recipe, LM head kept BF16),
served from one DGX Spark on 2026-11-06.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at `da0d88b85`,
  built for `gb10 / qwen3.6-35b-a3b / nvfp4`. The checkpoint loads as
  `qwen3_6_moe`.
- **Serve flags:** as in the zero-shot baseline, plus `--lm-head-dtype bf16`,
  so the score head computes the option logits in FP32 from the BF16 head.
- **Harness:** `tools/sparkscore` at `8725fe9`, `--template knowline-gen1.5`,
  rotations off. Port `8010` reads twice (KnowLine's default); `8011` reads
  once.
- **Comparison:** the zero-shot rows are
  `results/2026-11-06-zero-shot-qwen3.6-35b-a3b-nvfp4`.
- **Converter check:** converting the official BF16 Qwen3.6-35B-A3B
  reproduces all three shards of `nvidia/Qwen3.6-35B-A3B-NVFP4` byte for
  byte, by SHA-256.

## JevBench v1.6.1, public items (231)

| | Accuracy | Macro | Brier | ECE | Ordinal MAE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B zero-shot | 84.8% | 84.1% | 0.232 | 0.047 | 0.241 | 138 ms | 661 ms |
| **KnowLine, read twice** | **87.4%** | **89.4%** | **0.165** | **0.025** | **0.182** | 176 ms | 1,211 ms |
| KnowLine, read once | 87.4% | 88.9% | 0.165 | 0.030 | 0.244 | 147 ms | 644 ms |

## kev-suites, development partitions (clean)

| Suite | Variant | Accuracy | NLL | Brier | ECE | Coverage at 5% error |
|---|---|---:|---:|---:|---:|---:|
| decision-v7 (1,264) | zero-shot | 81.5% | 0.654 | 0.286 | 0.086 | 0.459 |
| | **KnowLine, read twice** | **87.6%** | **0.340** | **0.180** | **0.019** | **0.793** |
| | KnowLine, read once | 87.7% | 0.341 | 0.181 | 0.024 | 0.786 |
| transfer-v4 (656) | zero-shot | 80.8% | 0.622 | 0.297 | 0.096 | 0.247 |
| | **KnowLine, read twice** | **86.4%** | **0.357** | **0.193** | **0.034** | **0.739** |
| | KnowLine, read once | 84.0% | 0.412 | 0.225 | 0.040 | 0.642 |
| transfer-v9 (1,046) | zero-shot | 76.3% | 0.762 | 0.350 | 0.093 | 0.367 |
| | **KnowLine, read twice** | **83.3%** | **0.472** | **0.238** | 0.036 | **0.637** |
| | KnowLine, read once | 81.4% | 0.512 | 0.261 | 0.023 | 0.575 |

**Per-type temperatures** (`tools.sparkscore.calibrate`, fitted on
decision-v7, evaluated on transfer-v4 and transfer-v9) add little here: with
read twice, ECE goes from 0.046 to 0.030 and NLL from 0.482 to 0.480.
KnowLine's folded temperature (T = 1.2) already calibrates it.

## Reading

- KnowLine gains 5–7 points of accuracy over zero-shot on every suite.
- It roughly halves NLL and doubles to triples the share of decisions that
  can be automated at a 5% error budget.
- Read twice adds about 2 points on the transfer suites. Its cost is latency
  (p50 +29 ms, p95 about 2× on JevBench), so it stays the default.
- Faster MoE prefill is what will shrink that cost.

## Measurement notes

- **One run per configuration.** Each accuracy row is one full pass over its
  suite. The latencies are observed values from that same run, with requests
  sent one at a time on one Spark. Treat them as preliminary, single-harness
  numbers.
- **Zero-shot vs KnowLine is a product comparison, not an A/B.** The two
  rows differ in checkpoint, template (`chat-v1` vs `knowline-gen1.5`) and
  LM-head precision (NVFP4 vs BF16).
- **Read twice vs read once is a one-variable comparison.** Only
  `--read-twice` differs.
- The score path itself was checked separately: on KnowLine it agrees with
  Atlas's single-stream echo path with mean KL 0.0045 (40 decisions,
  `battery.py`).
