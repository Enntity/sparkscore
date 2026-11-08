<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Knowledge packs: a handbook read once, at almost no cost per request

`/v1/systemone` takes `knowledge`: text the model reads after d3-ultra's system prompt and before the state, such as
a policy, a handbook or a catalog. On an Atlas started with `--score-packs N`, the text is a **pack**. Atlas encodes
it once and keeps two things:

- the K/V of its tokens in the 16 full-attention layers;
- the GDN state and conv window after its last token.

Every later request on the same text starts from that state. Under `prefix_bidir` the pack is one bidirectional
block that never reads the state, so what it holds is the same for every request.

## Setup

- **Engine:** Enntity/atlas branch `sparkscore/score-packs-20261108`, on `upstream/systemone-score-20261105` at
  `b594151da`. Built for `gb10 / qwen3.8-27b / nvfp4`.
- **Engine environment and flags:** the shipped W4A4 default, plus `--score-packs 4`. Score tracing was off.
- **Harness:** `tools/sparkscore`, template `d3-ultra`, `score` backend. Temperatures were left at d3's own (no
  `--temperature`), so the probabilities compare directly with d3's runtime.
- **Reference:** `raw/pack-reference.jsonl`, written by `scripts/g1_fixture.py` from d3-ultra BF16 on d3's own
  runtime. Each item has three answers:
  - with the knowledge as an isolated system prefix (what a pack computes);
  - with the knowledge in the state;
  - with no knowledge.
- **Sets:**
  - `handbook`: a 1,142-token support handbook (1,179 pack tokens with the system prompt) with 7 override rules
    spread through it, and 40 tickets × 3 questions.
  - `rules4`: a 103-token, 4-rule policy (140 pack tokens), and 112 tickets × 3 questions.
  - Tickets use templates held out from any tuning.
- **Hardware:** one DGX Spark, 2026-11-08.

## Accuracy (`scripts/g3g4.py`)

| Set | Condition | Rule cases right | Argmax vs BF16 reference | Mean KL vs reference |
|---|---|---:|---:|---:|
| handbook | no knowledge | 6/35 | 118/120 (none) | 0.0100 |
| handbook | knowledge in state | 35/35 | 120/120 (in-state) | 0.0024 |
| handbook | **pack** | **35/35** | 117/120 (isolated prefix) | 0.0151 |
| rules4 | no knowledge | 28/128 | 328/336 (none) | 0.0116 |
| rules4 | knowledge in state | 128/128 | 328/336 (in-state) | 0.0343 |
| rules4 | **pack** | **128/128** | 324/336 (isolated prefix) | 0.0379 |

- **Pack vs in-state on Atlas:** answers agree on 117/120 (handbook) and 321/336 (rules4).
- **Merging:** pack requests merged eight at a time score exactly as they do alone (max |Δp| 0.000000).
- **For scale:** W4A4 against BF16 on d3's 49-question oracle is mean KL 0.0148.

## Speed (`scripts/g4_speed.py`, p50 at the client, 3 rounds)

| Set | No knowledge | Knowledge in state | Pack |
|---|---:|---:|---:|
| handbook (1,179 pack tokens) | 168.9 ms | 537.8 ms | **173.8 ms** |
| rules4 (140 pack tokens) | 169.6 ms | 184.8 ms | **164.4 ms** |

A pack request computes fewer rows than a request without knowledge, because d3's system prompt is in the pack. So a
short pack is faster than no knowledge at all.

The 1.2K-token pack costs about 5 ms over no knowledge. That is attention over its keys in the 16 full-attention
layers: the phase profile shows the attention layers at 41–50 ms with the pack against 35–38 ms without, and GDN
unchanged. That attention runs at roughly 20–35 TFLOPS, well under the 108 TFLOPS BF16 MMA peak, so a long-key
attention kernel is the next lever. `ATLAS_ATTN_ROWS128=0` made no difference (172.4 ms).

## Long handbooks (`scripts/g5_long.py`)

The same 7 override rules were placed through a handbook padded with ordinary sections: regional notes,
integrations, macros and known issues.

| Pack tokens | Rule cases, in state | Rule cases, pack | Pack vs in-state argmax | p50 in state | p50 pack |
|---:|---:|---:|---:|---:|---:|
| 3,272 | 35/35 | 35/35 | 117/120 | 1,333 ms | **175 ms** |
| 12,051 | 35/35 | 35/35 | 114/120 | 5,452 ms | **280 ms** |

## Engine checks

- **Unit tests:** `cargo test -p spark-model --lib score_plan` (7, including 4 new pack-geometry tests) and
  `cargo test -p spark-server --bin spark score` (28, including the new pack API, error and cost tests).
- **Chat:** with 4 packs held, the scheduler admits 28 of 32 sequences, since each pack keeps one SSM slot.
  A chat completion still answers.
