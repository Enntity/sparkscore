# GB10 limits for SparkScore prefill GEMMs (2026-11-18, one DGX Spark)

Question: what would a no-holds-barred SparkScore / Qwen3.8-27B prefill do,
and does the "blocks share one copy of the weights" GEMM design work on GB10?

Hardware: GB10 (sm_121a), 48 SMs, SM clock 2.15–2.18 GHz during runs,
CUDA 13.0, driver 580.178. Benchmarks: `cb.cu` (copy/sharing) and `mma.cu`
(tensor-core peak), built with `nvcc -O3 -gencode arch=compute_121a,code=sm_121a`.
An idle vLLM embedding model was resident on that Spark during the runs.

## Tensor-core peak (mma.sync, register operands, 32 warps/SM)

| Format | TFLOPS | per SM |
|---|---|---|
| FP4 (mxf4nvf4 m16n8k64, block-scaled) | 428 | 8.92 |
| FP8 (e4m3 m16n8k32) | 214 | 4.47 |
| BF16 (m16n8k16) | 107 | 2.23 |

The MLP gate+up kernel (`dense_w4a4ws_gate_up_silu`, 685 tokens, 244 GFLOP)
takes 1.17 ms = 209 TFLOPS = **49% of FP4 peak**.

## Block sharing inside a cluster: dead on GB10

| Mechanism | Result |
|---|---|
| Cluster of 8 | only 4 co-resident (32 of 48 SMs); clusters of 4 fill all 48 |
| `cp.async.bulk ... .multicast::cluster` | works, ~120–240 µs per copy (ptxas warns "substantially reduced performance") — ~1000x too slow |
| `ld.shared::cluster` (read a peer's smem) | 5–8 GB/s per SM vs 51–56 local |
| bulk smem→peer smem push | 2–5 GB/s per SM, ~3.4 µs per copy, no gain from pipelining |

None of these can feed tensor cores. A GEMM core built on shared tiles is not
worth building on GB10.

## L2 → SM bandwidth: the real ceiling is request size, not bytes

8 MB working set (L2-resident, 99.5% hit rate), consumers nearly idle:

| Load pattern | GB/s total | per SM |
|---|---|---|
| bulk copy 16 KB tiles, 4 stages | 1,715 | 35.7 |
| bulk copy, 2 CTAs/SM | 1,749 | 36.4 |
| per-thread 16 B cp.async, contiguous, 2 CTAs/SM | 1,651 | 34.4 |
| same, 128 B runs from rows 2,560 B apart | 1,679 | 35.0 |
| same, 32 B runs | 955 | 19.9 |
| 1 CTA/SM, 16 B runs | 481 | 10.0 |

ncu: the bulk-copy run at 1.61 TB/s reads `lts__t_sectors` 81.5% of peak.

The real MLP kernel (a second Spark, `bench685.ncu-rep`): 696 GB/s delivered, but 22.8 M
L2 read requests for 27.5 M sectors = **1.21 sectors per request**, and
`lts__throughput` 72%. The weights arrive in full 128 B lines, but the NVFP4
tokens are row-major `[M, K/2]`, so every 64-wide K stage reads 32 B from
each token row plus a 4 B scale (`cp.async.ca` 4 B). The kernel sits at the
small-request ceiling, not the byte ceiling.

## What the numbers say a fast prefill does

At FP4 peak, the gate+up tile (256 features x 144 tokens, K stage 64) needs
14.4 KB per 4.72 MFLOP = 27 GB/s per SM: under the 35 GB/s full-line ceiling,
over the ~20 GB/s small-run ceiling. So:

1. **Token operand in a K-tiled layout** (`[K/64][M][32 B]` data, scales the
   same way), written by the quantizer, loaded as full lines or one bulk copy
   per stage. Weights pre-tiled at load into the exact staged image. The MMA
   code is unchanged, so outputs stay bit-identical. Expected: gate+up 1.17 →
   ~0.70 ms (82% of peak); the same fix for down, the FP8 GDN projections and
   the attention projections takes the LM GEMMs from ~175 to ~120 ms.
2. **Fusion**: norm + quantize in one pass writing the tiled layout; residual
   add in the GEMM epilogue. This is the same quantizer the layout change
   rewrites, so the two land together.
3. Then the per-layer persistent kernel, GDN scan and ViT work, and the
   optional lower-precision steps (each needs a quality gate).
