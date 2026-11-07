<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# KnowLine-35B-A3B-Gen1.5 in BF16 (vLLM): what our NVFP4 conversion costs

This run serves the source BF16 weights (`PelaAI/KnowLine-35B-A3B-Gen1.5`, revision
`9adcf717361b2f7d2946d093f1e85efdf1e59010`) with vLLM 0.24.0 on one DGX Spark
(`--max-model-len 8192 --gpu-memory-utilization 0.72 --enforce-eager`). Requests
go through the same harness and template as the NVFP4 rows: `knowline-gen1.5`,
read twice, `--backend vllm`. The `vllm` backend reads the top-20 logprobs at the
answer slot and falls back to one prompt per option. The decision-v7 suite was
stopped unfinished: its 77-option questions take that fallback, which would have
taken hours. The NVFP4 rows come from
`results/2026-11-06-knowline-35b-a3b-gen1.5-nvfp4` (Atlas `/v1/score`, port 8010).

| | Engine and weights | Accuracy | Macro / NLL | Brier | ECE | Coverage at 5% |
|---|---|---:|---:|---:|---:|---:|
| JevBench public | vLLM, BF16 | 87.0% | macro 88.8% | 0.157 | 0.037 | — |
| | Atlas, NVFP4 | 87.4% | macro 89.4% | 0.165 | 0.025 | — |
| kev transfer-v4 | vLLM, BF16 | 87.2% | NLL 0.357 | 0.194 | 0.038 | 0.741 |
| | Atlas, NVFP4 | 86.4% | NLL 0.357 | 0.193 | 0.034 | 0.739 |
| kev transfer-v9 | vLLM, BF16 | 83.9% | NLL 0.458 | 0.233 | 0.030 | 0.662 |
| | Atlas, NVFP4 | 83.3% | NLL 0.472 | 0.238 | 0.036 | 0.637 |

Two variables change together here, the weights' precision and the engine, so
this is an observation, not a one-variable A/B. Each row is a single run.

The conversion costs little:

- On the kev transfer suites, accuracy is within 0.6–0.8 points and NLL within
  0.014.
- On JevBench, the NVFP4 path is slightly ahead on accuracy and calibration
  and slightly behind on Brier.

Latency is not comparable across these engines. vLLM's JevBench p50 was 378 ms
against Atlas's 176 ms, but vLLM ran eager with no shared-prefix pass.
