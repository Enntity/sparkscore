<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# d3-ultra NVFP4 on SparkScore: final accuracy runs

This is the accuracy half of the final benchmark for the SparkScore release
candidate. The latency half is measured separately on the final kernels.

The candidate kernels later merged on top of this build are bit-identical to
it: 0 difference across 115 score distributions, and merged and solo passes
give the same probabilities. Every number below therefore holds for the
release candidate.

## Setup

- **Engine:** Enntity/atlas `upstream/systemone-score-20261105` at `30c434a2c`,
  built for `gb10 / qwen3.8-27b / nvfp4`.
- **Engine environment:** `ATLAS_SSM_DECODE_RING=0 ATLAS_GDN_BATCHED_FLA=1
  ATLAS_DENSE_PREFILL_BF16MMA=1 ATLAS_GDN_CHUNK_TC=1 ATLAS_ATTN_ROWS128=1
  ATLAS_QWEN38_PDL=1 ATLAS_DENSE_PREFILL_LOWP=1`.
- **Serve flags:** `--kv-cache-dtype bf16 --max-seq-len 16384
  --max-prefill-tokens 16384 --max-num-seqs 32 --max-batch-size 32
  --gpu-memory-utilization 0.80 --disable-thinking --lm-head-dtype bf16`. The
  Decision Index rows that failed were retried with 32768 (see below).
- **Harness:** `tools/sparkscore`, template `d3-ultra`, `score` backend,
  `prefix_bidir`. The harness ran on the same Spark as the engine.
- **Hardware:** one DGX Spark per run (two were used in parallel),
  2026-11-06/07.
- **Reference rows:**
  - KnowLine-35B-A3B NVFP4 on Atlas, from `results/2026-11-06-knowline-*`.
  - d3's own BF16 runtime (`d3_server.py`) on the same hardware.

## JevBench v1.6.1, public items (231), one run each

| Engine | Accuracy | Macro | Brier | ECE | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| **d3-ultra, SparkScore** | **92.2%** | **93.5%** | 0.119 | 0.036 | **185 ms** | **1,376 ms** |
| d3-ultra, SparkScore, fitted temperatures | 92.2% | 93.5% | **0.115** | 0.038 | 186 ms | — |
| d3-ultra, d3's own BF16 runtime | 91.3% | 92.9% | 0.117 | 0.033 | 633 ms | 3,070 ms |
| KnowLine-35B-A3B, SparkScore | 87.4% | 89.4% | 0.165 | 0.025 | 176 ms | 1,211 ms |

The accuracy gap to d3's own runtime is two items, which is noise at n = 231.
The latency gap is 3.4×.

## kev (clean rows), one run each

| Suite | Engine | n | Accuracy | NLL | Brier | ECE | Coverage at 5% error |
|---|---|---:|---:|---:|---:|---:|---:|
| transfer-v4 | d3-ultra | 656 | 87.0% | 0.447 | 0.208 | 0.088 | **0.802** |
| | KnowLine | 656 | 86.4% | 0.357 | 0.193 | 0.034 | 0.739 |
| transfer-v9 | d3-ultra | 1046 | 83.6% | 0.558 | 0.254 | 0.093 | **0.672** |
| | KnowLine | 1046 | 83.3% | 0.472 | 0.238 | 0.036 | 0.637 |
| decision-v7 | d3-ultra | 1264 | 87.1% | 0.420 | 0.192 | 0.061 | 0.794 |
| | KnowLine | 1264 | 87.6% | 0.340 | 0.180 | 0.019 | 0.793 |

- **Accuracy:** d3 and KnowLine tie.
- **Ranking:** d3 orders its answers by confidence better, so it can answer
  more of each suite before its error rate reaches 5%.
- **Calibration:** with its own temperature, d3 is overconfident on kev. Its
  mean confidence is about 0.96 at 87% accuracy.

### Fitted readout temperatures

`tools.sparkscore.calibrate` fits one temperature per question type on kev
decision-v4 (clean rows) only. decision-v4 is not reported anywhere above, so
every row in this table is held out.

- **Fitted:** choice 1.636, noul 2.366, score 1.588. These multiply d3's own
  temperature.
- **What changes:** NLL, Brier and ECE. Accuracy cannot change.
- **The `n` column:** calibrate counts every clean row that has an integer
  label, so its `n` and accuracy can differ from the kev report above.

| Suite | n | d3 NLL / ECE | d3 fitted NLL / ECE | KnowLine NLL / ECE |
|---|---:|---:|---:|---:|
| transfer-v4 | 656 | 0.447 / 0.088 | **0.334 / 0.033** | 0.357 / 0.034 |
| transfer-v9 | 1156 | 0.662 / 0.109 | **0.525 / 0.046** | 0.553 / 0.052 |
| decision-v7 | 1264 | 0.420 / 0.061 | 0.341 / **0.009** | 0.340 / 0.019 |

With the fitted temperatures, d3 matches or beats KnowLine on NLL and ECE on
all three suites. Its accuracy and coverage are unchanged. On JevBench the
fitted temperatures lower Brier from 0.119 to 0.115, and ECE moves by noise
only. They are the default in `install/checkpoints.json`.

## Decision Index 0.3, 1,000-row sample, against KnowLine on the same rows

The table is from `di_compare.py`.

- **Metrics:** each benchmark is scored on its native metric. These are not the
  board's chance-corrected index, and with about 20 rows per benchmark single
  differences are noisy.
- **Macro:** d3 0.749, KnowLine 0.747, over 42 benchmarks. d3 is better on 21
  and worse on 14.
- **d3 clearly ahead:** BBH (+0.21), CRUXEval (+0.16), HLE (+0.11), GPQA
  Diamond (+0.10), ANLI (+0.10), BANKING77 (+0.10), VAST (+0.10), ToolRet
  (+0.09), ChessBench (+0.10).
- **KnowLine clearly ahead:** SGD/SGD-X (−0.36), Amazon ESCI (−0.27),
  SATA-Bench (−0.21), PhishNChips (−0.16), POP909-CL (−0.15), cfcolor (−0.10).

**Fidelity check.** We ran d3's own BF16 runtime on the 96 sample rows of the
four weakest benchmarks plus BBH (`raw/di-fidelity5-d3-own`).

- **Same answers:** 95% on ESCI, BBH and SGD; 99% on PhishNChips and SATA.
- **Same scores:** benchmark scores match within one row. ESCI 0.463 against
  ours 0.459, SGD 0.410 against 0.365, BBH 0.895 against 0.842, PhishNChips and
  SATA equal.

So those losses are d3-ultra's own, not the port's.

**Failures.** Four of the 1,000 requests did not run.

- **Too long for one pass (3, all BRIGHT, 22–30K tokens).** The first run's
  16K prefill arena failed 13 requests. A retry with a 32K arena (`--max-seq-len
  32768 --max-prefill-tokens 32768`) recovered 10 of them. The three left need
  more than the 25.4K rows one score pass allows, so their state must be split
  inside the engine (layer by layer); that is not built.
- **Over the per-question limit (1).** It exceeds the harness's 32,768-token
  limit, and KnowLine's run fails it too.

Every failure is scored as wrong.

For reference, d3-ultra's reported public Decision Index score is 71.57 (a
vendor submission) and KnowLine's is 65.8. This sample cannot be compared with
those figures.

## Full Decision Index sample table

```
A = KnowLine-35B-A3B NVFP4, B = d3-ultra NVFP4 (SparkScore)

benchmark                       n       A       B     B-A  metric
ACOS                           89   0.466   0.487  +0.021  per-review F1
ANLI                           20   0.630   0.732  +0.102  macro-F1
API-Bank                       20   0.900   0.900  +0.000  accuracy
ARC-Challenge                  20   0.900   0.900  +0.000  accuracy
ARC-Easy                       20   0.950   1.000  +0.050  accuracy
Amazon ESCI                    19   0.724   0.459  -0.265  macro-F1
BANKING77                      20   0.896   1.000  +0.104  macro-F1
BBH fixed-option tasks         19   0.632   0.842  +0.211  accuracy
BFCL                           20   1.000   1.000  +0.000  case exact accuracy
BPoMP                          79   0.975   0.987  +0.013  accuracy
BRIGHT                         18   0.416   0.411  -0.005  nDCG@10
CLINC150+OOS                   20   0.733   0.794  +0.061  macro-F1
CLadder                        19   1.000   1.000  +0.000  accuracy
CRUXEval                       19   0.684   0.842  +0.158  accuracy
ChessBench                     20   0.250   0.350  +0.100  accuracy
ContractNLI                    20   0.862   0.835  -0.027  macro-F1
FinEntity                      19   0.960   0.944  -0.016  macro-F1
GPQA Diamond                   20   0.450   0.550  +0.100  accuracy
GSM8K                          40   0.850   0.875  +0.025  accuracy
HLE                            19   0.105   0.211  +0.105  accuracy
Habermas Machine               19   0.737   0.684  -0.053  accuracy
HellaSwag                      20   0.950   1.000  +0.050  accuracy
HoVer claim verification       19   0.789   0.789  +0.000  accuracy
Home appliance simulator       20   1.000   1.000  +0.000  case exact accuracy
Humicroedit                    20   0.700   0.750  +0.050  accuracy
MMLU                           20   0.850   0.900  +0.050  accuracy
MMLU-Pro                       19   0.684   0.737  +0.053  accuracy
MuSR                           20   0.700   0.750  +0.050  accuracy
NLI4CT                         19   0.842   0.782  -0.060  macro-F1
New Yorker caption matching    19   0.684   0.632  -0.053  accuracy
POP909-CL                      20   0.850   0.700  -0.150  accuracy
PhishNChips phishing decisions   19   1.000   0.842  -0.158  accuracy
RAGTruth response-level hallucination   19   0.889   0.842  -0.047  F1 on hallucinated class
RouterBench                    40   0.762   0.725  -0.037  selected quality (quality objective)
SATA-Bench                     19   0.421   0.211  -0.211  case exact accuracy
SGD/SGD-X                      20   0.722   0.365  -0.357  macro-F1
SimpleBench                    10   0.300   0.400  +0.100  accuracy
ToolRet                        20   0.747   0.840  +0.092  nDCG@10
VAST                           19   0.841   0.944  +0.103  macro-F1
When2Call MCQ                  19   1.000   1.000  +0.000  accuracy
WinoGrande                     20   0.950   1.000  +0.050  accuracy
cfcolor                        20   0.550   0.450  -0.100  accuracy

benchmarks 42: macro A 0.747  B 0.749  B better on 21, worse on 14
```
