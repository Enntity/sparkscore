<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# From weight-space programming to knowledge packs

*Teaching a decision model a private rulebook, and making it cheap*

Enntity · 2026-11-08 · SparkScore (d3-ultra 27B, NVFP4) on one NVIDIA DGX Spark

## Abstract

We asked whether a decision model can be given domain knowledge it lacks by writing it into its weights:
**weight-space programming**, as recently demonstrated by MIT CSAIL [1]. The test bed was d3-ultra, a 27B
hybrid-attention model that turns a state plus typed questions into calibrated probabilities. The knowledge was a
private routing rule it cannot know.

- **The naive edit fails.** A closed-form edit that copies the prompt's effect into one MLP layer reaches only 8/12
  on held-out tickets and leaks onto tickets the rule does not cover.
- **A designed edit works.** Reading the model first (activation patching and linear probes) shows why, and points
  to a two-site circuit: an early detector writing a marker, and a late actuator reading it. This circuit gets 12/12
  on the rule and 31/32 overall, beating the rule written in the prompt (30/32). Its effect on unrelated requests is
  about the size of 4-bit quantization noise, and it fits in two rank-8 adapters.
- **It does not scale.** Four rules in one circuit interfere.

The same experiments pointed to a better object to move: the prompt's own internal state.

- **The isolated state is exact.** A policy placed at the start of the prompt and barred from reading what follows
  gives the same answers as the policy in place: 51/51 rules on a 4-rule policy, 29/29 on a 7-rule, 1.1K-token
  handbook.
- **Knowledge packs.** We built this into the Atlas inference engine. A pack is encoded once, and every later
  request reads its attention keys and recurrent state without recomputing them.
- **Speed.** On one DGX Spark, a 1.2K-token handbook costs 165 ms per request against 533 ms with the handbook in
  the request, and 167 ms with no handbook. At 12K tokens: 191 ms against 5.43 s.

## 1. Background

**SparkScore.** SparkScore serves d3-ultra [2], a Qwen3.8-27B decision model, on the Atlas engine [3]:
- **Weights:** NVFP4, with W4A4 matrix multiplies.
- **Shape of a request:** the input is a state and typed questions (`noul` yes/no, `choice`, `score`). The output
  is one probability distribution per question, read in one prefill pass. Nothing is generated.
- **Attention:** d3 was trained with `prefix_bidir`. In the 16 full-attention layers, the state block attends to
  itself in both directions, and each question block reads the state and itself. Its 48 Gated DeltaNet (GDN)
  layers stay causal.

**Weight-space programming.** Takehi, Teoh, Zhang and Khattab [1] call it weight-space programming: a
general-purpose LLM reads another network's weights and writes new ones directly, with no optimizer. Their
headline result traced the repetition loops of Qwen3.5-0.8B to one direction written by two MLP layers, and
removed it by editing 0.016% of the weights. MATH-500 went from 7.9% to 48.1%.

They also report a limit: on a model without the loop problem, "editing weights could remove an obstacle that hid
an ability, but it could not add an ability the model lacked." A skeptical follow-up [4] finds that reading
weights zero-shot is mostly arithmetic at the network's edges plus dataset priors, and that semantic tracing
through the middle fails.

**Our question.** Our question sits between the two. The knowledge was missing: a company's routing rule.
The ability was not: d3 applies such a rule perfectly when the rule is in the prompt. Can the prompt be moved into
the weights?

## 2. Task

**Synthetic support tickets.** Each ticket carries a team question (`choice`: billing, infrastructure or sales).
The private rule:

> Every ticket that mentions **Helios** goes to Infrastructure, whatever it is about.

**Cells:**
- **Rule cases:** Helios with a billing or sales topic; the rule flips the default answer.
- **Controls:** the same tickets with another product.
- **Infrastructure tickets:** either product, where the answer is Infrastructure anyway.

Ticket templates were split into train and test halves, so no test ticket shares wording with a training ticket.

**Collateral.** We measured it on d3's 41 recorded reference requests (72 questions) as KL divergence against the
unedited model, and as argmax flips.

**Baseline and ceiling.** Without the rule, d3 gets the rule cases 0/12 (p = 0.00). With the rule in the state, it
gets 12/12, at the cost of two control tickets it now over-routes. So the knowledge is missing from the weights, and
d3 can use it from context.

## 3. Writing the rule into the weights

All edits are closed-form: forward passes plus one linear solve (MEMIT-style normal equations [5]), with no gradient
descent. The targets come from the model's own activations with the rule in the prompt. The keys are MLP inputs
without it. Generic keys from the reference requests anchor the solve, so ordinary inputs map to no change.

**3.1 Naive context compilation.** We copied the rule's effect into one `down_proj` at every position: the residual
with the rule in the prompt, minus the residual without it. The edit fits its training targets (R² up to 0.99) and
leaves the reference requests nearly untouched (≤0.002 mean KL), but early layers transfer nothing (0/12). Later
layers transfer part of the rule and leak onto the controls: layer 48 gets 8/12 on the rule, but only 3/6 on sales
controls. Restricting the targets to the question block did not help.

**3.2 Reading before writing.** Three diagnostics explained the failure:

- **Activation patching.** Pasting the with-rule residual into the no-rule run changes nothing at any position
  before about layer 34. From layer 40 to 46, pasting only the question block carries the decision fully
  (p(Infrastructure) 0.71 → 1.00, controls 0.00). Pasting at the "Helios" tokens never does. In context, d3 does
  not tag the product; its question block looks the rule up late, by attention.
- **Linear probe.** "This ticket mentions Helios" is linearly readable at the question block early, at 0.97
  held-out accuracy at layer 8. It is near chance (0.62–0.72) at layers 38–50, exactly where the rule acts. So one
  edit can't be both effective and specific: where the rule acts, the model no longer says which product it's
  looking at.
- **Steering.** A single averaged direction does not move the answer. The effect is position-specific.

**3.3 A programmed circuit.** We separated *when* from *what*:

- **Site A (layer 8 or 14):** a rank-1 edit that writes a marker `u` into the question block when Helios is
  present. `u` is the lowest-variance direction of ordinary residuals.
- **Site B (layer 46):** a closed-form edit from marker-bearing keys to the per-position effect of the rule in
  context. Every other key maps to zero.

| Held-out test (32 tickets) | All | Rule cases | Collateral KL (mean) | Flips (of 72) |
|---|---:|---:|---:|---:|
| No edit | 20 | 0/12 | — | — |
| Rule in the prompt | 30 | 12/12 | — | — |
| One-layer compile (L48) | 23 | 8/12 | 0.0012 | 0 |
| Circuit A14, \|u\| 20 → B46 | 29 | 10/12 | 0.0040 | 0 |
| **Circuit A8, \|u\| 10 → B46** | **31** | **12/12** | 0.0144 | 1 |

The circuit beats the prompt. It also survives 4-bit storage:
- **Folded into the weights and quantized to NVFP4:** it keeps 10/12.
- **As a separate BF16 delta beside the NVFP4 weights:** 0 flips.
- **Truncated to rank 8 at both sites:** 29/32 with 0 flips, about 0.7 MB for the rule.

For scale: SparkScore's shipped W4A4 build differs from BF16 by a mean KL of 0.015.

**3.4 Where it stops.** The edit is weaker than the prompt on surface forms it never saw: "the Helios product" gets
9/12 against 12/12. Four rules compiled into one circuit interfere:
- Two rules on the same question with opposite answers cancel: Helios 2/12, Orchard 0–2/10.
- The detectors are weak: about 5 on their own marker against 1–2 on the others, where the target is 10.
- Answers the rules shouldn't touch change: urgency on non-trigger tickets stays put only 34–46/66 times.

This matches the boundary in [1]. The circuit moved knowledge the model could already *use*, and it needed a
specific design per rule.

## 4. Moving the prompt's state instead

Patching had already shown that the rule's whole effect flows from the policy's representation, read late by
attention. That suggests moving the representation itself: compute the policy's internal state once and reuse it.
In d3, that state is:
- the attention keys and values of the policy tokens in the 16 full-attention layers;
- the GDN recurrent state and conv window after the policy in the 48 linear layers.

**The isolation question.** Under `prefix_bidir` a policy normally reads the ticket too. Its state is reusable only
if the answers survive barring that. We tested it by changing the mask in d3's own runtime: the policy's queries
read only the system prompt and the policy.

| 4-rule set (112 tickets × 3 questions) | Rule cases | Team kept | Urgency kept |
|---|---:|---:|---:|
| No policy | 0/51 | 58/58 | 66/66 |
| Policy in the state | 51/51 | 30/58 | 2/66 |
| Policy isolated from the ticket | 51/51 | 29/58 | 2/66 |
| **Policy as an isolated system prefix** | **51/51** | 44/58 | 7/66 |

With the policy at the very start of the prompt (after d3's system prompt) and isolated, its K/V and GDN state
depend on nothing that follows. This masked computation equals computing the prefix once and reusing it.

**A realistic handbook.** On a 1,142-token support handbook with seven override rules spread through ordinary
content, the isolated prefix gets 29/29 rule cases, the same as the handbook in the state. It agrees with the
in-state answers on 118/120 questions (mean KL 0.030).

**A side finding.** The terse 4-rule policy over-applies: urgency changes on 64 of 66 tickets it shouldn't touch.
The fuller handbook does not over-apply. How knowledge is written matters as much as where it lives.

## 5. Knowledge packs in Atlas

**What existed.** Atlas's score path already did one pass over many requests:
- requests share KV blocks by reference count;
- GDN state is forked between streams by pointer tables;
- the bidirectional mask is a per-stream `state_end`.

**A pack, built on that:**
- **Building:** one score pass over the pack's tokens as a bidirectional block. Atlas keeps its KV blocks and its
  SSM slot.
- **A request on a pack** is planned at positions shifted by the pack's length P:
  - its streams share the pack's whole KV blocks and copy the partial last one;
  - their entry GDN state and conv window are the pack's, read in place through the pointer tables, so no state is
    copied.
- **No mask kernel change.** The state block's mask now spans the pack.
- **Holding packs:** content-addressed and evicted least recently used first. Each pack holds one SSM slot, so the
  scheduler admits that many fewer sequences.
- **Using them:** SparkScore exposes them as `knowledge` (text) or `knowledge_name` (a registered document).

**Accuracy on Atlas W4A4.** Against the d3 BF16 reference with the knowledge as an isolated prefix:

| Set | Rule cases | Argmax agreement | Mean KL |
|---|---:|---:|---:|
| handbook (1,179 pack tokens) | 35/35 | 117/120 | 0.015 |
| 4-rule policy (140 pack tokens) | 128/128 | 324/336 | 0.038 |

The in-state baseline's own W4A4 error on the 4-rule set is about the same (0.034). Requests on a pack merged into
one pass score exactly as they do alone.

**Speed.** p50 per request (40 tickets × 3 questions), with the attention kernel below:

| Pack tokens | Knowledge in the request | Knowledge pack | No knowledge |
|---:|---:|---:|---:|
| 140 | 184 ms | 163 ms | 168 ms |
| 1,179 | 533 ms | 165 ms | 167 ms |
| 3,272 | 1,317 ms | 155 ms | — |
| 12,051 | 5,432 ms | 191 ms | — |

A pack request computes fewer rows than a request with no knowledge, because d3's system prompt lives in the pack.
So a short pack is faster than none. What a long pack does cost is attention over its keys in the 16 full-attention
layers, which ran at only 13–35 TFLOPS against the GPU's 108 TFLOPS BF16 peak: before the kernel below, a 1.2K pack cost
174 ms and a 12K pack 280 ms.

**A long-key attention kernel.** Score streams are short, so their attention ran per-head 32-row tiles at about
10 TFLOPS whatever the key length. The kernel packs one KV head's six query heads into each 128-row tile, which
shares every K/V tile and is bit-identical. A stream with more than 4,096 keys and at least 4 keys a row splits its
key range across a 2-CTA cluster, within one bf16 ulp. Because the split rule reads only the stream's own lengths,
merged passes still score exactly as alone. Attention over a 12K pack fell from 143 to 32 ms, and the request from
280 to 191 ms. Accuracy is unchanged: 35/35 and 128/128 rules, and at 12K the pack now agrees with the in-request
answer on 117/120 questions (114 before). Of the remaining 24 ms at 12K, the kernel runs at about 45 TFLOPS, half
of peak.

## 6. What people do today

A decision service with a domain handbook has five options without packs. None of them reuses the handbook
across requests on a model like d3.

**The handbook in every request.** This is the documented practice:
- **Jev:** TypeSafe's docs say Jev is not fine-tuned or adapted with customer data. Reference material goes in
  the state, and rules go in each question's instructions and criteria. Every input token is billed on every
  call, with no cached rate [6].
- **d3:** its own API takes only the state and the questions [7].
- **Cost:** the handbook is recomputed for every ticket: 5.43 s per ticket at 12K tokens on one Spark (§5).

**A prompt cache.** Prefix caching works because in a causal model a token's state depends only on the tokens
before it. Under `prefix_bidir` the handbook sits in the state block and reads the ticket:
- **Mixing:** the first full-attention layer mixes the ticket into every handbook token, so every deeper layer
  differs per ticket.
- **What stays the same:** only the three Gated DeltaNet layers before that one, which are cheap.
- **Serving projects agree:**
  - vLLM's proposal for caching d3-family requests stops every cache hit at the end of the bidirectional
    block. It reuses the state for more questions about the same ticket, not across tickets [8].
  - oMLX's decision cache leaves d3 out because of its non-causal attention [9].

**A causal decision model with ordinary caching.** This works, and oMLX does it for causal decision models [9].
It gives up the bidirectional read that d3 was trained for: d3 scored with a causal mask drifts about seven
times further from its own answers (mean KL 0.039 against 0.0055).

**Per-customer fine-tuning.**
- **Jev:** not offered [6].
- **d3:** its weights are open, so it is possible. It needs labeled examples for every rule and a training run
  for every edit, and the result approximates the rules rather than following the text.
- **Closed-form edits:** these interfere past a few rules (§3.4).

**Isolated blocks in causal models.** Block-Attention [10] encodes retrieved passages in isolation so their
states can be reused, and it needed fine-tuning to recover accuracy.

**What a pack changes.** A pack changes one thing in the mask: the knowledge block no longer reads the ticket,
while the ticket and the questions still read it. That makes the handbook's state the same for every request,
so it is built once (5.7 s for 12K tokens) and read by every later request. d3 needs no retraining for it:
rules 35/35, and 117/120 answers match the in-request version.

## 7. Discussion

**Read before you write.** The failed edit (§3.1) and the working one (§3.3) differ only in what we learned by
reading the model: where the rule acts, and where the trigger is visible. That is the lesson of weight-space
programming [1], and of its critique [4]. Activation-level reading (patching, probes) located the circuit; static
weight reading would not have.

**The prompt is already a program.** In-context, d3 executes a rule by looking it up late through attention. A
weight edit has to re-implement that lookup as a detector plus an actuator, one rule at a time, and the rules then
compete for the same few directions. The prompt's own state is that program, compiled by the model itself. Moving
the state needs no solve, no per-rule design and no tuning, and it is exact. It is also exactly what the patching
experiment pointed at.

**Weights or state.** Packs are parameters in all but name: a few hundred MB of keys and recurrent state per
document. They are not edits to existing matrices. A weight circuit remains useful where a pack can't go:
- **Specificity:** an edit can be told "change nothing else", where a prompt over-applies (§4);
- **Deployment:** a model shipped without a state cache.

For a scoring service, packs win on every axis we measured.

## 8. Limitations

- **Synthetic tasks:** the tasks are synthetic routing, urgency and legal rules on support tickets, tested on
  held-out templates. We did not measure JevBench, kev or Decision Index with packs, because a pack is attached
  only to its own domain's requests.
- **Small weight-edit study:** one model (d3-ultra), at most four rules in one circuit, and 12 rule cases per
  test.
- **Long packs:** past 12K tokens, the isolated-prefix equivalence is untested; at 12K the pack agrees with
  in-state on 117/120 answers.
- **Isolation can change answers.** Barring the knowledge from reading the ticket is a change to d3's mask, and on
  terse policies it shifts some answers (urgency kept 7/66 against 2/66).

## 9. Reproducing

- **Weight-edit and isolation experiments:** `experiments/wsp-d3-20261108` (scripts and every output). They run on
  d3's own BF16 runtime.
- **Knowledge packs:** `results/2026-11-08-knowledge-packs` (reference fixture, drivers, numbers). The engine is
  Enntity/atlas `sparkscore/score-packs-20261108`; the API is in `docs/API.md`.

## References

1. R. Takehi, J. Teoh, A. Zhang, O. Khattab. *Sparks of Weight-Space Programming: LLMs can natively read & write
   neural network weights.* MIT CSAIL, 2026-11-02. https://rikiyat.github.io/blog/weight-space-programming/
2. vLLM Semantic Router team. *d3-ultra.* https://huggingface.co/vllm-sr/d3-ultra
3. Atlas inference engine. https://github.com/Atlas-Inf/atlas
4. *Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids?
   Exploring "Weight-Space Programming".* arXiv:2611.08070, 2026.
5. K. Meng, A. S. Sharma, A. Andonian, Y. Belinkov, D. Bau. *Mass-Editing Memory in a Transformer (MEMIT).* ICLR
   2023.
6. TypeSafe. *Jev models: customizing Jev*; *State*. https://docs.typesafe.ai/models,
   https://docs.typesafe.ai/concepts/state.md
7. vLLM Semantic Router. *API docs.* https://app.vllm-sr.ai/docs
8. G. Zamoscinski. *[RFC]: Prefix caching of Decision 3.0 `prefix_bidir` state blocks.* vllm-project/vllm#64267,
   2026-10-28.
9. *feat(decision): reuse prompt prefixes across /v1/systemone requests.* jundot/omlx#4586, 2026-10-15.
10. D. Ma, Y. Wang, T. Lan. *Block-Attention for Efficient Prefilling.* ICLR 2025. arXiv:2409.15355.
