<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Two knowledge packs in one request: built together, stacked, or apart

Issue [#1](https://github.com/Enntity/sparkscore/issues/1) asks whether a request can read two packs, such as a
handbook every customer shares and one customer's contract. This is the accuracy check it calls for, run before
any engine work. It asks whether packs built separately can be read together, and whether stacking one pack on
another works better.

**Result:**
- Every layout keeps every override rule: 35/35.
- A pack built on top of another (a **pack tree**) is close to one combined pack.
- Packs built **apart** spread one document's rule onto tickets it does not cover. B's "Osprey tickets can wait"
  rule changes the urgency of 10–12 of 30 other tickets, against 0–3 for one pack or a tree.

Combining the recurrent (GDN) state exactly makes no difference to that leak. It comes from the attention side: B's
text was encoded without ever reading A's.

## Setup

- **Model and runtime:** d3-ultra BF16 on d3's own runtime (no Atlas). The job ran in the persistent session of
  `../2026-11-08-knowledge-packs/scripts/runner.py`, with that result's `wsp.py`, `handbook.py`, `tickets.py` and
  `rules4.py`.
- **Block runner:** `scripts/p1_combo.py` runs d3's layers the way `StateOnce` does (same projections, RoPE,
  convolution and `torch_chunk_gated_delta_rule`). What changes per layout is which earlier blocks each block's
  attention reads, and which GDN state it starts from.
  - **Sanity check:** with d3's own mask, the runner matches d3's logits path: argmax 120/120, mean KL 9.5e-6,
    max KL 4.3e-4.
- **Documents:** the 7-rule handbook of the 2026-11-08 result, split in two:
  - **A:** title and sections 1–7, with the wisteria, helios, kestrel and lattice rules.
  - **B:** sections 8–13 under their own title, with the osprey, tamarind and orchard rules.
  - Both sit after d3's system prompt: 37 tokens of system prompt, then A (738 tokens), then B (425 tokens),
    1,200 pack tokens in all.
- **Tickets:** the handbook's 40 test tickets × 3 questions (team, urgency, legal).
  - 5 tickets trigger no rule and 35 trigger one: 20 rule cases are in A and 15 in B.
- **Reference:** A and B in the request's state, which is the regime d3 was trained on.
- **Hardware:** one DGX Spark, 2026-11-10.

## Layouts

All layouts read the same tokens, `[system prompt | A | B | state | question]`. They differ only in what each
block reads.

| Layout | What is encoded with what | GDN state the request starts from |
|---|---|---|
| in-state | Everything in the state block (d3's own mask) | d3's causal scan |
| one-pack | System prompt + A + B as one pack (what SparkScore supports today) | the pack's |
| chain | System prompt + A as a pack; B reads it and itself | the exact scan through A then B |
| chain-base | System prompt alone; A reads it; B reads both | the exact scan |
| iso-composed | System prompt alone; A and B each read only it (B never sees A) | `M_B·S_A + (S_B − M_B·S_sys)` |
| iso-naive | as iso-composed | B's own, with A's dropped |
| `-rev` | B before A | |

**Composition.** The gated delta rule is affine in the state it starts from. B's per-token keys, values and gates
do not depend on A when B is built apart. So scanning B from A's state equals `M_B·S_A + C_B`, where:
- `M_B` is B's scan from the identity state with zero values;
- `C_B` is B's scan from the base state, minus `M_B·S_sys`.

This matched a direct scan of B's features from A's state to 0.32% (0.53% reversed), relative to the state's
largest entry.

## Results (`raw/p1_combo.out`, `scripts/p1_leak.py`)

| Layout | Rule cases A | Rule cases B | Argmax vs in-state | Mean KL | Max KL | Other tickets answered "Can wait" |
|---|---:|---:|---:|---:|---:|---:|
| in-state (reference) | 20/20 | 15/15 | — | — | — | 1/30 |
| no knowledge | 5/20 | 1/15 | 88/120 | 1.920 | 15.99 | 0/30 |
| **one-pack** | 20/20 | 15/15 | **119/120** | 0.026 | 1.32 | **0/30** |
| **chain** | 20/20 | 15/15 | **116/120** | 0.041 | 1.64 | **3/30** |
| chain-base | 20/20 | 15/15 | 117/120 | 0.038 | 1.66 | 2/30 |
| **iso-composed** | 20/20 | 15/15 | **109/120** | 0.147 | 3.71 | **12/30** |
| iso-naive | 20/20 | 15/15 | 110/120 | 0.158 | 4.45 | 11/30 |
| in-state-rev (reference) | 20/20 | 15/15 | — | — | — | 2/30 |
| one-pack-rev | 20/20 | 15/15 | 118/120 | 0.021 | 1.04 | 2/30 |
| iso-composed-rev | 20/20 | 15/15 | 112/120 | 0.115 | 3.26 | 10/30 |

- **"Other tickets"** are the 30 that trigger neither urgency rule (neither osprey nor kestrel).
- **Every changed answer, in every layout, is an urgency answer.** In the separate-pack layouts each one moves to
  "Can wait" (0), which is the Osprey override in B. Most are confident. Of the 11 changed answers in iso-composed, 10
  had the reference's answer at 0.95 or more, and 8 put 0.6 or more on "Can wait".
- **The GDN state is not the cause.** iso-composed and iso-naive score alike (109 and 110/120; 12 and 11 leaks).
- **Order barely matters.** iso-composed-rev agrees with iso-composed on 118/120.
- **A rule-case count alone hides this.** Every layout is 35/35. A check of rules applied where they should not be
  is needed in any future pack test.

A likely reason: B defines the Osprey tier but not the other tiers, which are in A (section 6). Read without A,
the override has nothing to scope it. Two real documents, such as a handbook and a customer contract, may be more
or less independent than two halves of one handbook. This is one fixture and one split.

## What it means for the engine

- **Pack trees are safe to build.** A child pack is encoded reading its parent and holds only its own K/V and
  final state. It keeps one pack's accuracy within a few answers, and it covers the shared-handbook-then-customer
  case.
- **Arbitrary sets of packs are not usable as they stand.** They would need cross-document context, such as
  re-encoding part of each later pack against the earlier ones, and then this check again.
- **The composition math is not needed.** It is exact enough but changes nothing.

## Files

- `scripts/p1_combo.py`: the job, as it ran.
- `scripts/p1_leak.py`: the per-question and "Can wait" breakdown from `raw/combo.jsonl`.
- `raw/p1_combo.out`: the job's output.
- `raw/combo.jsonl`: per ticket, the state, triggers, gold answers and every layout's probabilities, in d3's code
  order: team `[billing, infrastructure, sales]`, urgency `0..3`, legal `[false, true]`.
