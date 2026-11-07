<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Competitors, 2026-11-05

**[I]** marks an independent measurement: Decision Index 0.4.1, JevBench
v1.6.1, or an arXiv paper. **[V]** marks a vendor or author claim. None of
these numbers were measured on a DGX Spark. The boards use different hardware
(Decision Index: one RTX PRO 6000; JevBench: mixed GPUs, loopback), so
compare latencies only within one board.

## The leaderboards (reuse these)

| Harness | Source | What it measures |
|---|---|---|
| Decision Index 0.4.1 | github.com/apolinario/decision-index (MIT) | 37 public benchmarks, 110k requests; the "Full" score is private (60% same-skill, 40% new domains); chance-corrected; includes Jev reference answers |
| JevBench v1.6.1 | github.com/fstandhartinger/jevbench (MIT), benchmarkheaven.com | 534 decisions plus a sealed set; Intelligence, Calibration (ECE plus fidelity to gold distributions), Speed, Cost |
| kev-suites | hf.co/datasets/jaredpalmer/kev-suites | `kev.benchmark --remote` scores any `/v1/systemone` endpoint: accuracy, Brier, ECE, option-order and isolation checks |
| RLCDAlignBench | github.com/sumleo/RLCDAlignBench (code MIT, data CC-BY-4.0) | 44 benchmarks, 7,193 instances, cached Jev answers |
| typed-decisions | hf.co/datasets/LocalLLaMA/typed-decisions (Apache-2.0) | gold answers are soft distributions from a ~4B teacher, so it measures agreement with that teacher, not correctness |

## Jev itself

| | Number | Source |
|---|---|---|
| Decision Index Full (1.14.0) | 60.24; ECE 0.080; median 196 ms, p95 297 ms, network included | [I] |
| JevBench (1.13.0) | Capability 77.1; Intelligence 63.6; Calibration 90.6; p50 239 ms | [I] |
| Per-dataset calibration | median ECE 0.168 vs 0.047 pooled; 24 of 31 benchmarks fail a perfect-calibration null; ranking is good (median AUROC 0.905) | [I] arXiv 2609.29429 |
| Latency vs length | p50 274 ms at base length, 343 ms at 8k tokens, 418 ms at 16k | [I] arXiv 2609.22753 |
| Marketed latency | 70–500 ms | [V] |

## Hosted

| Service | Weights | Numbers |
|---|---|---|
| Perplexity Decisions (pplx-decider v1.1/v1.2, Qwen3.8-27B with noncausal full-attention layers and a readout head) | v1.1 open, Apache-2.0 | JevBench capability 82.8 (#1 overall) [I]; Decision Index v1.2 Full 67.98, ECE 0.055 [I] |
| Fastino GLiDE 1.2 (Qwen3.8-27B) | listed, possibly gated | Decision Index Full 71.52, ECE 0.020, median 93 ms [I] |
| Mercury Decide (Inception) | closed | JevBench capability 73.6, p50 332 ms [I] |
| Levanto Sage 1.3 | closed | JevBench composite 74.0, p50 152 ms [I] |
| Cloudflare Clef (Qwen3.8-27B / Qwen3.5-9B / Qwen3-Omni-30B-A3B) | Apache-2.0 | Decision Index Full 54.9 / 44.4 / 45.0 [I] |
| Liquid d1-3B | LFM Open License | Decision Index Full 31.1, median 16 ms [I] |
| OpenAI Decisions (`gpt-6-luna`) | closed, not Jev-shaped | no independent numbers found |

## Open

| Project | Backbone | Method | Numbers |
|---|---|---|---|
| KnowLine-35B-A3B Gen1.5 | Qwen3.5-35B-A3B, LoRA r32 | label logprobs, "read twice" prompt, T=1.20 | Decision Index Full 65.77, ECE 0.044, median 47 ms [I] — **the bar in our class** |
| Stock Qwen3.6-35B-A3B (via decisio) | — | zero-shot label logprobs | Decision Index Full 50.8, median 70.5 ms [I] — **our zero-shot baseline** |
| Bespoke Nimble v4.1 Max | Qwen3.8-27B | contrastive data, hard-label CE, T=2.179 | Decision Index Full 70.03 [I]; CC-BY-NC |
| Kev (Apache-2.0) | Qwen3.5 0.8–9B, Qwen3.8-27B | LoRA plus pointer head, per-checkpoint temperature | JevBench Calibration 93.7 (best) [I]; Decision Index Full 59.6 (27B) [I]; 4B 18 ms on H100 [V] |
| decider (Apache-2.0) | Qwen3.5 2B–35B-A3B, Gemma-4 | one slot per question, all slots in one pass, single-token labels up to 255 | 35B-A3B Decision Index Full 46.6, JevBench p50 49 ms [I] |
| openjev-sglang | Qwen3.6-35B-A3B NVFP4, SGLang | state prefilled once, then N concurrent one-token calls | no published numbers; the author says the probabilities are not calibrated |
| SGLang `/v1/systemone` (nightly) | Qwen3.8-27B, Qwen3.5-35B-A3B | one prefill per question | its docs warn that probabilities drift by up to 0.07 between cold and cached on hybrid models |
| Laya, CLM, Verdict, GLiNER2.5-Decide | encoders, 150M–8B | encoder heads or contrastive scoring | near chance on both boards [I] |

## What nobody does

1. **One weight read per request on a hybrid MoE.** Everyone runs the state
   and the questions as separate passes. On a Spark each pass costs about
   100 ms in weight reads. SparkScore puts the state and every branch in one
   pass. The Gated DeltaNet state is forked inside the layer, so branches are
   exactly isolated and the probabilities are deterministic. That also
   removes the cold-versus-cached drift SGLang warns about.
2. **Soft-label training with a proper scoring rule.** Nimble, Kev and
   KnowLine train on hard labels and then fit a temperature. Jev's
   distributions sit far from gold (KL 1.44 on typed-decisions [I]).
3. **Per-tenant base-rate and threshold correction.** This addresses the main
   finding of arXiv 2609.29429; a threshold fitted on ten labelled items
   recovers most of the F1 gap.
4. **Option-order debiasing at almost no cost.** Permuted copies of a
   question become extra branches in the same pass. Small models flip their
   answer 22–44% of the time when the options are reordered [I].
5. **Published DGX Spark latency.** No competitor reports GB10 numbers.

Sources: the TypeSafe API docs (docs.typesafe.ai/api); each project's
repository or model card; Decision Index `data/index.json` (generated
2026-11-04); the JevBench results CSV; arXiv 2609.29429, 2609.22753 and
2610.07177.
