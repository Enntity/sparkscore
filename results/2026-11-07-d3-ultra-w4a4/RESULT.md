<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# d3-ultra NVFP4 on SparkScore with W4A4: the new default

The release candidate now runs its MLP and attention matrix multiplies as
W4A4: NVFP4 weights times NVFP4 tokens, on the native FP4 MMA (`mma.sync
kind::mxf4nvf4`), so the weights are no longer re-encoded on every pass. The
GDN projections stay W8A8, matching their FP8 weights. The W4A8 build that this
replaces is in `results/2026-11-07-d3-ultra-final-accuracy` and
`results/2026-11-07-d3-ultra-final-latency`. Why W4A4 and not the other 4-bit
formats is in `results/2026-11-07-activation-formats`.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at
  `6e8c06a44`, built for `gb10 / qwen3.8-27b / nvfp4`.
- **Engine environment:** as for the W4A8 build, except
  `ATLAS_DENSE_PREFILL_LOWP=gu,dn,qkv,o,qkvz,gout`.
- **Serve flags:** as for the W4A8 build, except `--max-seq-len 49152
  --max-prefill-tokens 49152`. The larger prefill arena lets passes of up to
  ~38K rows run in one piece. It measured no speed cost: every number was
  within 1% of a 16K-arena run on the same commit.
- **Harness:** `tools/sparkscore`, template `d3-ultra`, `score` backend,
  `prefix_bidir`. Temperatures are noted per suite below.
- **Hardware:** one DGX Spark, 2026-11-07. `install/checkpoints.json` has the
  full serve block.

### Which build produced each run

The W4A4 kernels changed during the day. Each kernel round was checked
bit-identical to the one before it in W4A4: 115 score distributions, merged vs
solo, and the oracle. So every accuracy number below holds for `6e8c06a44`.

| Run | Build | Arena | Temperatures |
|---|---|---|---|
| Latency, JevBench ×3, oracle, concurrency | `6e8c06a44` | 48K | fitted |
| kev transfer-v4 | `8434214d8` (first W4A4 kernels) | 16K | fitted |
| kev transfer-v9, decision-v7 | `8434214d8` | 32K | d3's own |
| Decision Index sample | `8434214d8` at 32K; the 4 failed requests retried on `2d87e1c24` at 48K | 32K / 48K | d3's own |

The temperature column matches how each suite was run for the W4A8 build, so
each row compares like with like.

## JevBench v1.6.1 public items (231), three runs

| Engine | Accuracy | Macro | Brier | ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| **d3-ultra, SparkScore W4A4** | 90.9% (210) | 92.5% | 0.118 | 0.029 | **146–148 ms** | **1.05–1.06 s** |
| d3-ultra, SparkScore W4A8 | 92.2% (213) | 93.5% | 0.115 | 0.038 | 158 ms | 1.31 s |
| d3-ultra, d3's own BF16 runtime | 91.3% (211) | 92.9% | 0.117 | 0.033 | 633 ms | 3.07 s |
| KnowLine-35B-A3B, SparkScore | 87.4% (202) | 89.4% | 0.165 | 0.025 | 176 ms | 1.21 s |

**Against W4A8, item by item:** 4 items are right only under W4A8, and 1 only
under W4A4 (exact McNemar p = 0.38).

- All five are hard items where W4A8's top probability was 0.39–0.80.
- Two of them (`hard-sol-b-judge_hard-02` and `hard-sol-c-judge_hard-07`) also
  flip in a W4A4-MLP-only run. W4A4 does shift d3's probabilities, and the
  oracle KL below shows the same thing.
- The larger suites below show no average loss.

## kev (clean rows)

| Suite | n | W4A8 acc | W4A4 acc | W4A8 NLL / ECE | W4A4 NLL / ECE | W4A8 cov@5% | W4A4 cov@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| transfer-v4 (fitted temperatures) | 656 | 87.0% | 87.7% | 0.334 / 0.033 | 0.334 / 0.030 | | 0.793 |
| transfer-v9 (d3's own temperatures) | 1046 | 83.6% | 83.7% | 0.558 / 0.093 | 0.549 / 0.089 | 0.672 | 0.682 |
| decision-v7 (d3's own temperatures) | 1264 | 87.1% | 86.9% | 0.420 / 0.061 | 0.425 / 0.064 | 0.794 | 0.795 |

The W4A8 transfer-v4 numbers are the fitted readout from the accuracy
results; that run reported no coverage with fitted temperatures. Across the
2,966 rows, W4A4 gets 3 more rows right than W4A8 (+5, +1, −3).

## Decision Index 0.3, 1,000-row sample

- **Macro:** W4A4 0.756, against W4A8's 0.749 and KnowLine's 0.747 on the same
  rows. W4A4 is better than W4A8 on 9 benchmarks and worse on 7.
- **Failures:** 1 of 1,000. The 48K arena runs the three 22–30K-token BRIGHT
  requests that failed at 32K. The one left exceeds the harness's own
  32,768-token limit per question, which KnowLine's run fails too.

## Engine time per score pass (median of 5)

| Rows | W4A8 (`8434214d8`) | **W4A4** | Effective TFLOPS |
|---:|---:|---:|---:|
| 96 | 107 ms | **101 ms** | 46 |
| 288 | 152 ms | **139 ms** | 102 |
| 1,056 | 486 ms | **372 ms** | 139 |
| 2,112 (N = 2) | 1,001 ms | **788 ms** | 131 |
| 4,128 | 1,952 ms | **1,579 ms** | 128 |

- **At 288 rows:** the limit is about 75 ms. W4A4's matrix multiplies take
  about 117 ms; W4A8's took about 130.
- **What bounds them now:** the 4-bit weights stream from memory at about
  140 GB/s inside the kernels. A copy-only test of the same access pattern
  reaches about 190 GB/s, against 234 GB/s for a plain stream.
- **What W4A4 changed:** at this size the multiplies are now bound by memory,
  not compute.

## Concurrent requests

The harness is `concurrency.py`: distinct 300-token prefixes, 4 branches of
40 tokens each.

| K | p50 latency | All done | Merged per pass | Max difference vs solo |
|---:|---:|---:|---:|---:|
| 1 | 201 ms | 202 ms | 1 | 0 |
| 2 | 298 ms | 397 ms | 1 | 0 |
| 4 | 729 ms | 731 ms | 3 | 0 |
| 8 | 1,325 ms | 1,523 ms | 6 | 0 |
| 16 | 2,448 ms | 2,984 ms | 6 | 0 |

W4A8 took 251 ms for one request and 3.62 s for all 16.

## Against d3's BF16 oracle (49 questions)

| Mask | W4A8 mean KL | W4A4 mean KL | W4A4 same answer |
|---|---:|---:|---:|
| `prefix_bidir` | 0.00546 | 0.01475 | 47/49 |

One question accounts for much of the increase: `smoke/date/overdue_mar_20`,
with max |dp| 0.47.

## Receipts

`raw/` holds the latency run (fingerprint, oracle, sweep, concurrency, three
JevBench runs), the three kev runs and the Decision Index run.

- **Decision Index:** the run's results file has the retried requests appended
  to it; the last record per request is the one scored. Its own checksum file
  went stale when the retry appended to it and was removed.
- **Fingerprint:** the latency fingerprint's flags line was corrected after
  the run, and the file says so.
