<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# The engine

SparkScore asks one thing of Atlas: a **score pass**. One forward pass reads
a request's state and every one of its questions, and returns each
question's probabilities over its option tokens. Nothing is sampled or
decoded.

This page describes the score path and the dense 27B prefill kernels that
SparkScore serves. All of it is Atlas code.

## Where the code is

- **Pin:** `install/atlas-source.json` pins the engine: `Enntity/atlas`, branch
  `sparkscore/score-packs-20261108`, commit `2d26979c0`. It is the score path at
  `b594151da` (built on Atlas-Inf `main` at `999bfc48d`) plus knowledge packs and
  their attention kernel. `install/build.sh` fetches exactly that commit and
  checks its git tree.
- **Target:** `gb10 / qwen3.8-27b / nvfp4`. d3-ultra is a Qwen3.8-27B
  fine-tune: 64 layers, 16 with full attention and 48 with Gated DeltaNet
  (GDN), hidden size 5120.
- **Upstream:** the score path is proposed upstream in Atlas-Inf/atlas
  ([Atlas-Inf/atlas#331](https://github.com/Atlas-Inf/atlas/pull/331)); knowledge packs are not yet. When it merges, `install/repin.sh` moves the pin.
- **Measured builds:** most measurements on this page are from `6e8c06a44`.
  `b594151da` adds an L2 evict-first hint on the W8A8 GDN weight copies
  and the review fixes. `results/2026-11-07-release-replication` reruns the
  headline checks at the pin. Its answers are identical, and its engine times
  are within 1% in an interleaved A/B. The pack attention kernel (below) gives
  bit-identical output for every request without a long pack.

## Why one pass

Every row in a pass multiplies against every weight of the dense 27B, so the
cost of a pass has two parts:

- **The weight read.** About 17 GB at the GB10's 234 GB/s stream rate, which
  is about 73 ms whatever the number of rows.
- **Compute.** About 49 GFLOP per row.

The numbers are from `results/2026-11-07-d3-ultra-final-latency`.

A decision request is a long state followed by short questions. The p50
JevBench request is about 288 rows, and the p90 about 2,500 (same receipt).
At those sizes the weight read is a large part of the pass, so the number of
passes sets the cost. So does reading the state once instead of once per
question.

For a state of L tokens and N questions of q_i tokens:

| Design | Weight reads | Rows computed |
|---|---:|---|
| One prompt per question | N | N·L + Σ q_i |
| Cached state, then one batched pass of the questions | 2 | L + Σ q_i |
| **SparkScore: state and questions in one pass** | **1** | L + Σ (q_i + r), r < 16 |

The r rows are the state's last partial KV block, which each question
replays (see below).

## The score pass

### `POST /v1/score`

The harness calls this Atlas route (`crates/spark-server/src/api/score.rs`).

```json
{"prefix": [ids], "branches": [[ids], ...], "options": [ids] | [[ids], ...], "attention": "prefix_bidir"}
```

- **Request:** `prefix` is the state block. Each branch is one question
  block, and each option list holds that branch's single-token options. A
  flat `options` list applies to every branch. `attention` is `"causal"`
  (the default) or `"prefix_bidir"`.
- **Response:** `{"probs": [[p, ...], ...], "engine_us", "queue_us",
  "batched"}`, one softmax per branch over exactly its option logits.
  `batched` is the number of requests that shared the pass.
- **Limits:** 1 to 256 branches, and 1 to 256 options per branch. Every
  token id must be inside the vocabulary.
- **Errors:** a malformed request is 400. A request that cannot run is 422,
  for example when it is too large for one pass. Scoring disabled is 501.

### Row layout

`crates/spark-model/src/model/trait_impl/prefill_b/score_plan.rs` lays out
the rows. For a prefix of L tokens and KV blocks of 16 rows, F = 16·⌊L/16⌋.
The rows are:

```
[prefix[0..F)]  ++  [prefix[F..L) ++ branch_1]  ++ … ++  [prefix[F..L) ++ branch_N]
     head                 branch 1                          branch N
```

- **Head:** the state's whole blocks. It is omitted when F = 0.
- **Branches:** each starts at position F. Its block table shares the head's
  F/16 physical blocks, reference-counted as a prefix-cache hit would be. It
  owns the rest of its blocks.
- **Tail replay:** each branch replays the state's last partial block itself,
  fewer than 16 rows. So no partly filled block is ever shared.

### What runs per layer

`prefill_b/score.rs` and `score_ssm.rs` run the pass.

- **Once over all rows:** the norms, the projections and the MLP, so each
  weight is read once per pass.
- **Full-attention layers.** Every row's K and V are written before the
  batched varlen attention reads them. Each stream's offset comes from its
  own KV length. A branch reads the head's blocks in place, and branches
  never see each other.
- **GDN layers.** The recurrence is the one ordered step. The heads scan
  first, together, from the zero state. Then every branch scans from its
  head's final state, all branches together. With `ATLAS_GDN_CHUNK_TC=1`
  each scan is one launch, and the branches read the head's recurrent state
  and conv window in place, with no copy. Without it, the head's state is
  copied into each branch first. Either way, each branch sees exactly the
  state it would have had if run alone.

### `prefix_bidir`

d3-ultra was trained with this mask (its `decision_config.json` sets
`attention_mode: "prefix_bidir"`). The template sends it with every request.

- **State rows:** in the 16 full-attention layers, a row of the state block
  reads every state key, in both directions, and never a question's key.
- **Question rows:** a row of a question block reads the whole state block
  and every row of its own block, in both directions.
- **GDN layers:** they stay causal.

The state's hidden states therefore do not depend on any question, which is
why one state block can serve every branch.

In the kernel, each stream carries a state end S, the prefix length. A
query at a slot below S reads keys below S. Any other query reads every key
of its stream. A stream without S is causal, byte-identical to the build
without the mask (`kernels/gb10/common/prefill_paged_compute.cuh`,
`prefill_paged_rows128.cu`). The replayed tail rows are state rows, so they
keep the state mask. A head whose state ends in a partial block reads that
block from its first branch, because every branch writes the same tail K
and V.

Against d3's own BF16 runtime, on d3's exact prompt tokens for 49
questions, the W4A8 build measures mean KL 0.00546 with `prefix_bidir` and
0.0386 with a causal mask (`results/2026-11-07-d3-ultra-final-latency`).

### The option head

`prefill_b/score_head.rs` computes the option logits:

1. The final norm runs on each branch's last row only.
2. Those rows go to the host.
3. Each option logit is an FP32 dot product with the LM-head row of its
   token id. The rows are read from the BF16 LM head once and cached.

The full-vocabulary LM head never runs. The softmax is taken over each
branch's own options. This needs `--lm-head-dtype bf16`. The converted
d3-ultra checkpoint's LM head is d3's FP32 code-readout rows, cast to BF16,
at the code token ids, and zeros elsewhere (`install/checkpoints.json`).

### What the pass skips

Nothing is sampled, decoded, inserted into the prefix cache or snapshotted.
The pass allocates its own sequences and frees them before it returns.

## Merging and splitting requests

`crates/spark-server/src/scheduler/score.rs` schedules score jobs.

- **When jobs run:** on their own channel, at a quiescent point between
  scheduler steps.
- **Merging:** jobs queued together run as one pass, and so share one read
  of the weights. They are packed in arrival order up to two budgets: the
  row arena, and the SSM state pool capped at `--max-batch-size`. Each head
  and each branch is one sequence.
- **Failed merges:** if a merged pass fails, each of its jobs is retried
  alone, so one bad request cannot fail its neighbours.
- **Splitting:** a job with several branches that does not fit one pass runs
  as several passes. Each pass covers the prefix and a group of branches,
  and the results are reassembled in order. A single branch that does not
  fit is an error; the state is not split inside a pass.

The row budget is the smallest of three limits: the prefill arena, the GDN
staging buffers and the metadata scratch. Under the shipped
`--max-seq-len 49152 --max-prefill-tokens 49152`, passes of up to about 38K
rows ran in one piece (`results/2026-11-07-d3-ultra-w4a4`).

Merged and solo passes give identical probabilities. In the concurrency runs
of `results/2026-11-07-d3-ultra-w4a4`, distinct 300-token prefixes with 4
branches of 40 tokens each:

| Requests at once | p50 latency | All done | Merged per pass | Max difference vs solo |
|---:|---:|---:|---:|---:|
| 1 | 201 ms | 202 ms | 1 | 0 |
| 4 | 729 ms | 731 ms | 3 | 0 |
| 16 | 2,448 ms | 2,984 ms | 6 | 0 |

## Knowledge packs

A knowledge pack is a token block that requests continue without recomputing it: a policy or a handbook
(`crates/spark-model/src/model/trait_impl/prefill_b/score_pack.rs`).

- **Build.** One score pass over the pack's P tokens, as one bidirectional block under `prefix_bidir`. Atlas keeps
  that pass's sequence instead of freeing it: its KV blocks, and its SSM slot holding the GDN state and conv
  window after the last token.
- **Request.** A request on a pack is planned at positions shifted by P (`score_plan.rs`). A stream that reads
  the pack directly (the request's head, or a branch without one):
  - shares the pack's whole KV blocks by reference count;
  - copies the pack's partial last block into a block it owns;
  - enters the GDN layers from the pack's state through the pass's pointer tables. The single-launch scan and
    conv1d read it in place, so no state is copied.
  - Branches share their head's blocks and fork from it as before.
- **Mask.** The state block's `state_end` now spans the pack, so the state and question blocks read the pack's
  keys. The pack never reads anything after it. No kernel changes.
- **Holding.** Packs are content-addressed by their tokens and attention mode. `--score-packs N` bounds how many
  are held, evicting the least recently used after a new one is built. Each pack keeps one SSM slot, so the
  scheduler admits that many fewer sequences.
- **API.** `/v1/score` takes `"pack": [ids]` or `"pack_id"`, answers with `"pack_id"` and
  `"pack": "built" | "held"`, and answers 404 for an id it no longer holds.

**Attention over long keys.** Score streams are short (a ~240-row state, ~20-row branches), so per-head 32-row
attention tiles ran at about 10 TFLOPS however long the keys. Score passes now use 128-row tiles holding one KV
head's six query heads (`prefill_paged_rows128.cu`, `_gqa`), so every K/V tile is shared and the output is
bit-identical. A stream with more than 4,096 keys and at least 4 keys a row (a pack under a short state) splits
each tile's key range across a 2-CTA cluster, combined through distributed shared memory (`_gqa_split2`); split rows
are within one bf16 ulp. The rule reads each stream alone, so a merged pass still scores as each request alone.

| Pack tokens | p50, pack | p50, knowledge in the state | p50, no knowledge |
|---:|---:|---:|---:|
| 1,179 | 165 ms | 533 ms | 167 ms |
| 3,272 | 155 ms | 1,317 ms | — |
| 12,051 | 191 ms | 5,432 ms | — |

Attention over the 16 full-attention layers, microbenchmarked at d3's geometry: 16.6 → 5.1 ms on a 1.2K pack,
143 → 32 ms on a 12K pack, 1,040 → 212 ms for eight 12K-pack requests merged (results/2026-11-08-knowledge-packs).

## Dense 27B prefill kernels

The prefill matrix multiplies of the Qwen3.6/3.8-27B hybrids are in
`kernels/gb10/qwen3.6-27b/nvfp4/dense_bf16mma_prefill.cu`. Their dispatch is
`crates/spark-model/src/layers/ops/dense_bf16mma*.rs`. They serve:

- **The MLP:** gate+up, with SiLU·up in the epilogue, then down. NVFP4
  weights.
- **The attention projections:** q/k/v and o. FP8 in the checkpoint;
  Atlas re-encodes them to NVFP4 at load.
- **The GDN projections:** `in_proj_qkvz` and `out_proj`. FP8 weights with a
  per-tensor scale.

`ATLAS_DENSE_PREFILL_LOWP` picks the activation format per projection group:

| Group names | Projections | Format | MMA |
|---|---|---|---|
| `gu`, `dn`, `qkv`, `o` | MLP and attention | **W4A4**: NVFP4 weights × NVFP4 tokens | `mma.sync kind::mxf4nvf4` m16n8k64; the MMA applies both E4M3 block scales (per 16 K) |
| `gu8`, `dn8`, `qkv8`, `o8` | MLP and attention | **W4A8**: NVFP4 weights re-encoded per stage × E4M3 tokens | `kind::mxf8f6f4` m16n8k32 with a UE8M0 row scale |
| `qkvz`, `gout` | GDN projections | **W8A8**: FP8 weights × E4M3 tokens | m16n8k32 E4M3, on a transposed `[K, N]` copy of the FP8 weight |
| `dnmx` | down, with `gu8` | gate+up writes the down projection's tokens as E4M3 with a scale per 32 K | opt-in; new numerics |

- **Shorthands:** `1` means `gu8,dn8,qkv8,o8,qkvz,gout`, and `all` means
  `gu,dn,qkv,o,qkvz,gout`.
- **The default:** `install/checkpoints.json` sets the `all` mix: W4A4 for
  the MLP and attention, W8A8 for GDN. `./start.sh --precision w4a8` sets
  `1`.

**Token quantizers.** `dl_quant_nvfp4` and `dl_quant_fp8` give each token row
one CTA and a power-of-two scale, applied once in the FP32 epilogue.

- **E4M3:** the codes are E4M3 of x / 2^e, with 2^e the smallest power of
  two that fits the row's maximum.
- **NVFP4:** ModelOpt's block quantizer, with an E4M3 scale per 16 values.

**W4A8 re-encode.** Each 64-wide K stage of NVFP4 weights is re-encoded per
feature as E4M3 of w·s/2^E. Values with at most 4 significant bits are
exact. The rest round once to E4M3.

The W4A4 kernels need no re-encode: the weights are the bytes the MMA
consumes.

**Kernel variants by row count.** Within one format the variants give
bit-identical results. So the row count, and therefore merging, never
changes a result.

| Format | Kernel | Rows |
|---|---|---|
| W4A4 | 8-warp `dense_w4a4_*` | below 96 |
| W4A4 | persistent, a loader warpgroup and 8 MMA warps, 4 stages (`dense_w4a4wsl_*`) | 96 to 383 |
| W4A4 | the same with another token layout, 6 stages (`dense_w4a4ws_*`) | from 384 |
| W4A8 | 8-warp `dense_w4a8_*`; re-encodes the next stage while the current stage's MMAs run | below 129, and `dnmx` |
| W4A8 | warp-specialized, persistent (`dense_w4a8ws_*`): 8 producer warps re-encode, 8 MMA warps | from 129 |

- **Tiles:** the persistent kernels use token tiles of up to 144 rows and
  mbarrier rings. They launch min(tiles, SMs) CTAs.
- **Prefetch:** the W4A4 gate+up loader prefetches weight lines into L2 four
  stages ahead.
- **Determinism:** one CTA computes each output in a fixed K order. Results
  never depend on the grid or the chunking.

**Tests.** GPU tests compare every format and kernel variant with an exact
reference, at 33, 135, 1,100 and 2,100 rows. They need a GB10 and are
`#[ignore]`d by default:

- `dense_bf16mma_matches_exact_dequant_reference`
- `dense_lowp_matches_exact_reference_on_quantized_activations`
- `dense_mx_handoff_matches_reference`

Every kernel round was also checked bit-identical to the one before it:
115 score distributions, merged vs solo, and the oracle
(`results/2026-11-07-d3-ultra-w4a4`).

## Serve settings

The `serve` block of `install/checkpoints.json`:

| Setting | What it does |
|---|---|
| `ATLAS_SSM_DECODE_RING=0` | Does not allocate the plain-decode rollback ring. The score path never decodes. The ring is 8 slots × max batch × the 27B's 158.9 MB SSM state, about 38 GB at batch 32 (`crates/spark-model/src/ssm_reserve.rs`). |
| `ATLAS_GDN_BATCHED_FLA=1` | Runs batched GDN prefill on the chunk-parallel FLA kernels, with a per-request state pointer table, instead of the slower persistent wy64 kernel. With `ATLAS_GDN_CHUNK_TC=1` the score pass uses the fused scan instead; this is the batched path where the fused scan does not apply. |
| `ATLAS_DENSE_PREFILL_BF16MMA=1` | Turns on the dense prefill GEMMs above for 32 rows and more; fewer rows keep the decode kernels. Alone it keeps activations in BF16, with weights dequantized exactly on BF16 tensor cores. `ATLAS_DENSE_PREFILL_LOWP` builds on it. The kernels ship only in the 27B target. |
| `ATLAS_GDN_CHUNK_TC=1` | One fused launch per GDN scan, on BF16 tensor cores, in place of the FLA path's three. Includes the score pass's multi-stream conv1d. Used when the heads are 128-dimensional and the pass is not an exact replay. |
| `ATLAS_ATTN_ROWS128=1` | Batched paged prefill attention on 128-row query tiles, FlashAttention-2 layout, K and V loaded by TMA. It replaces the 64-row kernel for BF16 KV with head dimension 256, with bit-identical output. Hence `--kv-cache-dtype bf16`. |
| `ATLAS_QWEN38_PDL=1` | Programmatic dependent launch for the qwen3.8-27b target's listed kernels: the dense GEMMs, the token quantizers, the GDN gates and some prefill glue (`crates/spark-runtime/src/cuda_backend/gpu_impl/pdl.rs`). A kernel can launch while its predecessor finishes. Results are unchanged. |
| `ATLAS_DENSE_PREFILL_LOWP=gu,dn,qkv,o,qkvz,gout` | The activation formats above: W4A4 MLP and attention, W8A8 GDN. |
| `--lm-head-dtype bf16` | Keeps the BF16 LM head that the gathered FP32 option head reads. |
| `--max-batch-size 32` | Caps the sequences per pass: each head and each branch takes one SSM slot. |
| `--max-seq-len 49152 --max-prefill-tokens 49152` | The prefill arena. It sets the rows per pass. Every number in `results/2026-11-07-d3-ultra-w4a4` was within 1% of a 16K-arena run. |

`ATLAS_SCORE_TRACE=1`, set in the receipts' fingerprints, only logs each
pass's setup, forward and head times.

## Where the time goes

The engine time per score pass is from `results/2026-11-07-d3-ultra-w4a4`:
one prefix plus N branches of 32 tokens, `prefix_bidir`, median of 5.
`tools/sparkscore/bench/engine_sweep.py` reruns it, and `concurrency.py`
next to it reruns the merging table above ([REPRODUCE.md](REPRODUCE.md)).

| Rows | W4A8 (`8434214d8`) | **W4A4 (default)** | Effective TFLOPS (W4A4) |
|---:|---:|---:|---:|
| 96 | 107 ms | **101 ms** | 46 |
| 288 | 152 ms | **139 ms** | 102 |
| 1,056 | 486 ms | **372 ms** | 139 |
| 2,112 (N = 2) | 1,001 ms | **788 ms** | 131 |
| 4,128 | 1,952 ms | **1,579 ms** | 128 |

The first working d3-ultra build took 570 ms at 288 rows
(`results/2026-11-07-d3-ultra-final-latency`).

### The roofline at 288 rows

The GB10 peaks, measured: BF16 MMA 108 TFLOPS, FP8 215, stream read
234 GB/s (`results/2026-11-07-d3-ultra-final-latency`). The FP4 rate is
431 TFLOPS (`results/2026-11-07-activation-formats`).

- **Weight read:** about 17 GB, about 73 ms.
- **Compute:** 288 × 49 GFLOP is about 14 TFLOP. That is about 131 ms at the
  BF16 rate, 66 ms at FP8 and 33 ms at FP4. This is arithmetic on the
  numbers above.
- **The limit:** about 75 ms. On FP4 the pass is bound by its weight read.
- **Measured:** 139 ms, about 1.9× the limit. Its matrix multiplies take
  about 117 ms.

### What bounds it now

- **The weight stream.** Inside the kernels, the 4-bit weights stream from
  memory at about 140 GB/s. A copy-only test with the same access pattern
  reaches about 190 GB/s, against 234 GB/s for a plain stream.
- **What W4A4 changed.** At 288 rows the W4A4 multiplies are bound by
  memory, not compute.
- **Before it.** W4A8 was bound by how long one producer warp took to
  re-encode one stage (`results/2026-11-07-d3-ultra-final-latency`).

### Not used

- **Pre-encoded 8-bit weights.** They were bit-identical but cost 17 GB.
  They gain 1–5% on long passes and nothing at 288 rows
  (`results/2026-11-07-d3-ultra-final-latency`).
- **A two-term NVFP4 activation mode** is not on the pinned branch.
- **Splitting one over-long state across passes, layer by layer,** is not
  built.

## Accuracy of the formats

All measurements are against d3's BF16 runtime, on d3's exact prompt
tokens: 49 questions, `prefix_bidir`.

| Build | Mean KL | Same answer |
|---|---:|---:|
| W4A8 (`LOWP=1`) | 0.00546 | 48/49 |
| W4A4 (`all`, the default) | 0.01475 | 47/49 |

One question accounts for much of W4A4's increase:
`smoke/date/overdue_mar_20`, max |dp| 0.47. On the benchmark suites W4A4
shows no average loss (`results/2026-11-07-d3-ultra-w4a4`):

- **kev:** 3 more of 2,966 rows right.
- **Decision Index:** macro 0.756 against 0.749 on the 1,000-row sample.
- **JevBench:** 3 fewer of 231 items right; exact McNemar p = 0.38.
