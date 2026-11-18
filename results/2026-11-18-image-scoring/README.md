# Image scoring on one Spark: gate, speed and merged passes (2026-11-18)

Issue #6, engine step 1. Atlas `/v1/score` takes images (Enntity/atlas branch
`sparkscore/score-images` on `afd71f444`, uncommitted at the time of these runs; binary
`spark-rc4` on Spark 04), d3-ultra-NVFP4, serve settings below.

## Gate: Atlas against d3's own runtime

`img_ref.py` runs d3-ultra's BF16 runtime (`vllm-sr/d3-ultra` @ `ccdcfa61`, its own processor and
state-once path) and records each question's expanded prompt ids, state block size and
probabilities. `img_atlas.py` sends Atlas exactly those ids (one request per image: shared state
block, one branch per question, `prefix_bidir`) with the image bytes; Atlas decodes and resizes
the image itself.

| Set | Questions | Argmax flips | Mean KL | Max abs dp |
|---|---|---|---|---|
| synthetic patterns, 8 images x 5 q (`img_cases.py`, `gate-rc4.json`) | 40 | 0 | 0.0010 | 0.063 |
| COCO val2017 photos, 24 x 6 q (`coco_cases.py`, `gate-coco-rc4.json`) | 144 | 5 | 0.0089 | 0.262 |
| same photos pre-resized to the target size as PNG (`gate-coco-aligned-rc2.json`) | 144 | 3 | 0.0094 | 0.269 |

- Every COCO flip is a near-tie in d3 itself (e.g. 0.40 / 0.56 between "Good" and "Excellent").
- The `count` question carries most of the KL (0.032 mean); `person` and `indoors` are at 0.0009.
- Pre-resizing the photos so neither engine resizes leaves the KL where it was. So the gap is the
  NVFP4 language model, not Atlas's JPEG decode or resize. The text path's own W4A4 oracle KL
  was 0.0148.

The old per-head ViT attention gave 0.0013 on the synthetic set (`gate-gemm.json`). The fused
flash kernel matches an FP64 reference to BF16 rounding (`test-flash72.log`).

## Speed (engine ms, median)

1 question: `one_q.py` on the fused-epilogue build (`spark-rc1`, same kernels as rc4). 5 questions:
`gate-rc4.json`.

| Image | Rows (1 q) | 1 question | 5 questions | d3 runtime, 5 questions |
|---|---|---|---|---|
| 224 square | 161 | 132 | 192 | 855 |
| 512x384 | 301 | 185 | 261 | 890 |
| 1024x576 | 685 | 329 | 434 | 1,280 |
| 1280x720 | 977 | 478 | 576 | 1,683 |
| 1600x900 (clamped) | 1,497 | 773 | 899 | 2,362 |

Before this work, the per-head attention loop took 757 ms at 1024x576 and 2,876 ms at
1600x900 (5 questions).

## Merged image passes (`merge_check.py`)

Image requests now share a score pass up to the tower's output window, all images encoded in one
tower batch. Merged answers are bit-identical to each request run alone (max dp 0.0). The single
and batched ViT blocks share one fused tail, so a request's rows never depend on its batch.

| 24 COCO photos | Sequential | 24 concurrent |
|---|---|---|
| 1 question each | 4.41 img/s | 5.57 img/s (16 per pass) |
| 6 questions each | 3.12 img/s | 3.38 img/s (4 per pass) |

## Where the time goes (nsys, 1024x576, 5 q, before the epilogue fusion; `prof/`)

434 ms of GPU kernels:
- The language model takes 346 ms, exactly its text twin's (`wide-text-kern.csv`). Image rows
  cost what text rows cost.
- The vision tower takes ~86 ms:
  - BF16 GEMMs 44 ms at ~44 TFLOPS
  - flash attention 17 ms
  - elementwise passes ~25 ms; the fused GEMM epilogue has since folded in bias, GELU and
    residual

Tried and rejected: `ATLAS_NO_GDN_FP8=1` (GDN projections requantized to NVFP4 so W4A4 applies)
is slower, 602 vs 441 ms at 1024x576.

## Serve

`spark serve --model-from-path d3-ultra-NVFP4 --kv-cache-dtype bf16 --max-seq-len 49152
--max-num-seqs 32 --max-batch-size 32 --gpu-memory-utilization 0.80 --disable-thinking
--max-prefill-tokens 49152 --lm-head-dtype bf16 --score-packs 0 --vision-max-pixels 1638400`,
env `ATLAS_SSM_DECODE_RING=0 ATLAS_HYBRID_MOE_LAYOUT=0 ATLAS_GDN_BATCHED_FLA=1
ATLAS_DENSE_PREFILL_BF16MMA=1 ATLAS_GDN_CHUNK_TC=1 ATLAS_ATTN_ROWS128=1 ATLAS_QWEN38_PDL=1
ATLAS_DENSE_PREFILL_LOWP=gu,dn,qkv,o,qkvz,gout`.
