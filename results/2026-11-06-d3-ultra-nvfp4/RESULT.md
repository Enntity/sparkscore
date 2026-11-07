<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# d3-ultra (Qwen3.8-27B decision model) on the one-pass prefix-bidirectional score path

`vllm-sr/d3-ultra` (Apache-2.0, revision `ccdcfa611649a59e095e0b5e1732a15cedafadd6`) was
converted to NVFP4 with `tools/nvfp4_convert.py`:

- Format reference: a2genesis Qwen3.8-27B NVFP4 (MLP NVFP4 W4A16; attention and
  GDN projections FP8).
- LM head: d3's FP32 readout rows, cast to BF16, at the code token ids; zeros
  elsewhere.
- No MTP.

It was served from one DGX Spark on 2026-11-06.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at `35779dcdc`,
  built for `gb10 / qwen3.8-27b / nvfp4`. It includes the `prefix_bidir` mask
  (`87bc7daad`).
- **Serve flags:** `--kv-cache-dtype bf16 --max-seq-len 16384
  --max-prefill-tokens 16384 --max-num-seqs 32 --max-batch-size 32
  --gpu-memory-utilization 0.84 --disable-thinking --lm-head-dtype bf16`.
- **Environment:** `ATLAS_SSM_DECODE_RING=0 ATLAS_GDN_BATCHED_FLA=1`.
- **Harness:** `tools/sparkscore`, template `d3-ultra` (`76f4ea6`), `score` backend.
  It sends `"attention": "prefix_bidir"`.
- **Reference row:** d3's own runtime and server (`d3_server.py`, BF16, its
  defaults) on the other Spark, driven by the same JevBench runner. The two rows
  differ in engine and weight precision, so they are two observations, not a
  one-variable A/B.

## JevBench v1.6.1, public items (231), one run each

| | Accuracy | Macro | Brier | ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| **d3-ultra NVFP4, Atlas `/v1/score`** | **91.8%** | **93.2%** | 0.118 | 0.033 | 567 ms | 4,587 ms |
| d3-ultra BF16, d3's own runtime | 91.3% | 92.9% | 0.117 | 0.033 | 633 ms | 3,070 ms |
| KnowLine-35B-A3B NVFP4, Atlas (earlier) | 87.4% | 89.4% | 0.165 | 0.025 | 176 ms | 1,211 ms |

## Against the BF16 oracle

The oracle is d3's own runtime: 41 requests, 72 questions. The checks are in
the session's `d3_oracle.py` and `d3_mask_check.py`.

- **Through the harness:** 69 questions compared. Same answer on 66/69, mean
  |Δp| 0.024, mean KL 0.015, worst |Δp| 0.47 (`smoke/date/overdue_mar_20`).
  The 21.8k-token request did not run, because this server's arena was 16k.
- **With d3's exact prompt tokens** (49 questions):
  - `prefix_bidir`: mean KL 0.016, same answer on 47/49.
  - Plain causal: mean KL 0.049, same answer on 46/49.

  So the bidirectional mask brings Atlas about 3× closer to the oracle. The
  rest is consistent with NVFP4 weights: relative dequantization error is
  9.5% on NVFP4 modules and 2.7% on FP8 modules. d3's own two BF16 paths
  differ by up to 0.024.

## Reading

- On JevBench public, d3-ultra is the accuracy leader on a Spark: +4.4 points
  of accuracy and a 29% lower Brier score than KnowLine.
- The NVFP4 port matches d3's own BF16 runtime.
- **Latency is the cost.** The p50 is about 3× KnowLine's, and the p95 is set
  by long states. The dense 27B prefill kernels run at 22–29% of a BF16-MMA
  roofline, so faster dense kernels are the next lever.
