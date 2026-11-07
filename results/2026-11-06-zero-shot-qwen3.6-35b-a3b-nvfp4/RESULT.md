<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Zero-shot baseline: Qwen3.6-35B-A3B NVFP4 on the one-pass score path

This is the starting point for SparkScore: an untrained model, NVIDIA's
`nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint, the `chat-v1` template. It was
measured on one DGX Spark on 2026-11-06.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at `f5287b8d1`
  (latency sweep at `da0d88b85`), built for `gb10 / qwen3.6-35b-a3b / nvfp4`.
- **Serve flags:** `--kv-cache-dtype bf16 --max-seq-len 16384 --max-num-seqs 96
  --max-batch-size 96 --gpu-memory-utilization 0.80 --disable-thinking`.
- **Environment:** `ATLAS_GDN_BATCHED_FLA=1 ATLAS_SSM_DECODE_RING=0`.
- **Harness:** `tools/sparkscore` on the `harness` branch (`5856ba9`), `score`
  backend; the BF16 reference rows use the `vllm` backend against vLLM 0.24.0
  serving `Qwen/Qwen3.6-35B-A3B` in BF16.
- No fine-tuning, no temperature and no option rotations.

## Accuracy and calibration

**JevBench v1.6.1, public items (231).**

| | Accuracy | Macro accuracy | Brier | ECE | p50 latency |
|---|---:|---:|---:|---:|---:|
| Atlas score path, NVFP4 | 84.8% | 84.1% | 0.232 | 0.047 | 138 ms |
| vLLM, BF16 | 84.8% | 83.4% | 0.211 | 0.052 | 272 ms |

**kev-suites, development partitions, score path.**

| Suite | n | Accuracy | ECE | Brier | Confidence bias |
|---|---:|---:|---:|---:|---:|
| decision-v7 | 1,264 | 81.5% | 0.086 | 0.286 | +0.086 |
| transfer-v4 | 656 | 80.8% | 0.096 | 0.297 | +0.080 |
| transfer-v9 | 1,046 | 76.3% | 0.093 | 0.350 | +0.081 |

**Score path against Atlas's single-stream path and BF16** (40 decisions, 10
states × 4 questions, `raw/battery-a3b.json`):
- The score path and the single-stream path pick the same answer on 40/40,
  with mean KL 0.021.
- Each NVFP4 path picks the same answer as BF16 on 34/40, with mean KL 0.063
  for the score path and 0.075 for the single-stream path.
- The differences come from the NVFP4 weights, not from the score path.

## Latency

`raw/latency-sweep.txt` gives engine time on a warm server, the median of 5
runs. Branches are 32 tokens each.

| State tokens | 1 question | 4 | 16 | 64 |
|---:|---:|---:|---:|---:|
| 64 | 99.5 ms | 123.7 | 234.8 | 682.1 |
| 256 | 141.1 | 161.5 | 273.0 | 722.3 |
| 1,024 | 304.4 | 325.4 | 439.7 | 912.7 |
| 4,096 | 1,011.3 | 1,030.2 | 1,156.3 | 1,681.7 |

## Reading

- **Quantization:** NVFP4 matches BF16 accuracy at half the latency, so it is
  the serving format.
- **Calibration:** the untrained model is overconfident by about 0.08 on
  every kev suite.
- **Latency:** at small row counts a pass sits near the weight-read floor
  (about 100 ms). Larger requests are limited by MoE prefill compute.

## Measurement notes

- **One run per configuration.** Each benchmark row is one run.
- **Latency sweep.** The table is the median of 5 warm repeats per shape,
  each shape warmed once first. Treat it as observed under the fingerprint
  above.
- **NVFP4 vs BF16 is not a one-variable A/B.** The two rows differ in
  precision, engine and backend (Atlas `/v1/score` vs vLLM 0.24). Accuracy is
  comparable across them; latency is not.
