// GB10 (sm_121) cluster data-sharing microbenchmark.
// Question: can a cluster of C CTAs share one L2 read of a tile, so a GEMM
// stops being L2-bound?  Modes:
//   0 own      : every CTA bulk-copies every tile into its own smem (today's pattern)
//   1 mcast    : CTA r issues tiles t%C==r with .multicast::cluster to all C CTAs
//   2 dsmem    : static smem, consumers read peer CTAs' smem via ld.shared::cluster
//   3 local    : static smem, consumers read own smem (reference)
// Consumers (all warps but the producer) read every byte of each tile with
// 16-byte loads and fold it into a checksum, like a GEMM main loop would.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); exit(1);} } while (0)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t cta_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(n));
}
__device__ __forceinline__ void mbar_expect_tx_arrive(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
  asm volatile(
    "{\n.reg .pred p;\nW%=:\n"
    "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
    "@!p bra W%=;\n}\n" :: "r"(smem_u32(b)), "r"(phase) : "memory");
}
// arrive on the same-offset mbarrier in CTA `rank` of this cluster
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* b, uint32_t rank) {
  uint32_t a = smem_u32(b), ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(ra) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
    :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s_mcast(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint16_t mask) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
    :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "h"(mask) : "memory");
}
__device__ __forceinline__ void bulk_s2peer(void* local_dst_same_off, uint32_t peer, const void* local_src, uint32_t bytes, uint64_t* local_bar_same_off) {
  uint32_t d, b;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(smem_u32(local_dst_same_off)), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(smem_u32(local_bar_same_off)), "r"(peer));
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
    :: "r"(d), "r"(smem_u32(local_src)), "r"(bytes), "r"(b) : "memory");
}
__device__ __forceinline__ uint4 ld_dsmem(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}

constexpr int MAXST = 16;

// smem layout: [stages * tile] data, then full[MAXST], empty[MAXST]
extern "C" __global__ void __launch_bounds__(1024, 1)
bench(const uint8_t* __restrict__ src, size_t src_bytes, int tiles, int tile_bytes, int stages,
      int mode, int csize, int read_bytes, int piece, unsigned long long* out) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint64_t* full = (uint64_t*)(sm + (size_t)stages * tile_bytes);
  uint64_t* empty = full + MAXST;
  const int tid = threadIdx.x, nthr = blockDim.x;
  const int cons_threads = nthr - 32;            // warp 0 = producer
  const uint32_t rank = csize > 1 ? cta_rank() : 0;
  const int cluster_id = blockIdx.x / csize;
  // each cluster walks a different offset so clusters don't all hit the same lines
  const size_t ntiles_src = src_bytes / tile_bytes;
  uint32_t acc = 0;

  if (mode == 2 || mode == 3) {   // static smem read test
    for (int i = tid * 16; i < stages * tile_bytes; i += nthr * 16) *(uint4*)(sm + i) = make_uint4(i, i ^ 7, i + 3, rank);
    if (csize > 1) cluster_sync(); else __syncthreads();
    const uint32_t base = smem_u32(sm);
    for (int t = 0; t < tiles; ++t) {
      uint32_t peer = (mode == 2) ? (uint32_t)((rank + 1 + t) % csize) : rank;
      uint32_t pb;
      if (mode == 2) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pb) : "r"(base), "r"(peer));
      const int off0 = (t % stages) * tile_bytes;
      #pragma unroll 4
      for (int i = tid * 16; i < read_bytes; i += nthr * 16) {
        uint4 v;
        if (mode == 2) v = ld_dsmem(pb + (uint32_t)(off0 + i));
        else v = *(const uint4*)(sm + off0 + i);
        acc ^= v.x + v.y * 3 + v.z * 5 + v.w * 7;
      }
    }
    if (csize > 1) cluster_sync();
    if (acc == 0x12345678u) out[blockIdx.x] = acc;
    return;
  }

  if (tid == 0) {
    for (int s = 0; s < stages; ++s) { mbar_init(&full[s], mode == 6 ? 32 : 1); mbar_init(&empty[s], (mode != 0 ? csize : 1) * (cons_threads / 32)); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (mode != 6) for (int s = 0; s < stages && s < tiles; ++s) mbar_expect_tx_arrive(&full[s], tile_bytes);  // arm round 0
  }
  if (csize > 1) cluster_sync(); else __syncthreads();

  if (tid < 32 && mode == 6) {   // per-thread 16-byte cp.async loader warp (today's GEMM pattern)
    for (int t = 0; t < tiles; ++t) {
      const int s = t % stages;
      const size_t st = (((size_t)cluster_id * 7919 + t) % ntiles_src) * tile_bytes;
      if (t >= stages) mbar_wait(&empty[s], ((t / stages) & 1) ^ 1);
      const uint32_t d = smem_u32(sm + (size_t)s * tile_bytes);
      // piece > 0: tile gathered as 2^piece-byte runs from rows 2560 B apart (row-major tokens, one K stage);
      // bases stay below 4 MB and gathers below 2.6 MB more, inside the 8 MB source
      const uint32_t base = ((uint32_t)(cluster_id * 7919 + t) & 1023u) << 12;
      const uint8_t* sb = piece ? src + base : src + st;
      const uint32_t pmask = (1u << piece) - 1;
      for (int i = tid * 16; i < tile_bytes; i += 512) {
        const uint32_t g = piece ? ((uint32_t)i >> piece) * 2560u + ((uint32_t)i & pmask) : (uint32_t)i;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d + i), "l"(sb + g) : "memory");
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(smem_u32(&full[s])) : "memory");
    }
  } else if (tid < 32) {          // producer warp
    if (tid == 0) {
      for (int t = 0; t < tiles; ++t) {
        const int s = t % stages;
        const uint32_t ph = (t / stages) & 1;
        const size_t st = (((size_t)cluster_id * 7919 + t) % ntiles_src) * tile_bytes;
        if (mode == 0) {
          if (t >= stages) mbar_wait(&empty[s], ph ^ 1);
          bulk_g2s(sm + (size_t)s * tile_bytes, src + st, tile_bytes, &full[s]);
        } else if ((uint32_t)(s % csize) == rank) {     // I issue this stage for the whole cluster
          if (t >= stages) mbar_wait(&empty[s], ph ^ 1);
          if (mode == 1) bulk_g2s_mcast(sm + (size_t)s * tile_bytes, src + st, tile_bytes, &full[s], (uint16_t)((1u << csize) - 1));
          else if (mode == 5) bulk_g2s_mcast(sm + (size_t)s * tile_bytes, src + st, tile_bytes, &full[s], (uint16_t)(1u << rank));
          else bulk_g2s(sm + (size_t)s * tile_bytes, src + st, tile_bytes, &full[s]);  // mode 4: pusher lane forwards
        }
      }
    } else if (tid == 1 && mode == 4) {   // pusher: forward my landed tiles to peers
      for (int t = 0; t < tiles; ++t) {
        const int s = t % stages;
        if ((uint32_t)(s % csize) != rank) continue;
        mbar_wait(&full[s], (t / stages) & 1);
        for (uint32_t r = 1; r < (uint32_t)csize; ++r)
          bulk_s2peer(sm + (size_t)s * tile_bytes, (rank + r) % csize, sm + (size_t)s * tile_bytes, tile_bytes, &full[s]);
      }
    }
  } else {                 // consumers
    const int ct = tid - 32;
    for (int t = 0; t < tiles; ++t) {
      const int s = t % stages;
      mbar_wait(&full[s], (t / stages) & 1);
      const uint8_t* tile = sm + (size_t)s * tile_bytes;
      #pragma unroll 4
      for (int i = ct * 16; i < read_bytes; i += cons_threads * 16) {
        uint4 v = *(const uint4*)(tile + i);
        acc ^= v.x + v.y * 3 + v.z * 5 + v.w * 7;
      }
      __syncwarp();
      if (mode != 6 && ct == 0 && t + stages < tiles) mbar_expect_tx_arrive(&full[s], tile_bytes);  // arm next round
      if ((ct & 31) == 0) {
        if (mode == 0) asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(smem_u32(&empty[s])) : "memory");
        else mbar_arrive_remote(&empty[s], (uint32_t)(s % csize));
      }
    }
  }
  if (csize > 1) cluster_sync(); else __syncthreads();   // keep smem alive for in-flight remote ops
  if (acc == 0x12345678u) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
  const int mode = argc > 1 ? atoi(argv[1]) : 0;
  const int csize = argc > 2 ? atoi(argv[2]) : 1;
  const int tile_bytes = argc > 3 ? atoi(argv[3]) : 8192;
  const int stages = argc > 4 ? atoi(argv[4]) : 8;
  const int ctas_per_sm = argc > 5 ? atoi(argv[5]) : 1;
  const size_t src_mb = argc > 6 ? atoi(argv[6]) : 8;
  const int threads = argc > 7 ? atoi(argv[7]) : 288;
  int tiles = argc > 8 ? atoi(argv[8]) : 4096;
  int read_bytes = argc > 9 ? atoi(argv[9]) : tile_bytes;
  int piece = argc > 10 ? atoi(argv[10]) : 0;

  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  const size_t src_bytes = src_mb << 20;
  uint8_t* src; CK(cudaMalloc(&src, src_bytes)); CK(cudaMemset(src, 1, src_bytes));
  unsigned long long* out; CK(cudaMalloc(&out, 8 * 4096));
  const size_t smem = (size_t)stages * tile_bytes + 2 * MAXST * 8;
  CK(cudaFuncSetAttribute(bench, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CK(cudaFuncSetAttribute(bench, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  int grid = p.multiProcessorCount * ctas_per_sm;
  grid -= grid % csize;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = csize; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  int maxc = 0; CK(cudaOccupancyMaxActiveClusters(&maxc, (void*)bench, &cfg));

  auto run = [&]() { CK(cudaLaunchKernelEx(&cfg, bench, (const uint8_t*)src, src_bytes, tiles, tile_bytes, stages, mode, csize, read_bytes, piece, out)); };
  run(); CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  const int reps = 10;
  CK(cudaEventRecord(a)); for (int i = 0; i < reps; ++i) run(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= reps;
  const double delivered = (double)grid * tiles * tile_bytes;          // bytes landing in (or read from) smem
  const double l2 = (mode == 1 || mode == 4) ? delivered / csize : ((mode == 0 || mode == 6) ? delivered : 0);
  static const char* names[] = {"own", "mcast", "dsmem", "local", "push", "mc1", "cpasync"};
  printf("piece=%3d rd=%5d mode=%-5s C=%d tile=%5d st=%2d grid=%3d thr=%d maxActiveClusters=%3d  %.3f ms  delivered %7.1f GB/s  (%6.1f GB/s/SM)  L2 reads %7.1f GB/s\n",
         piece, read_bytes, names[mode], csize, tile_bytes, stages, grid, threads, maxc, ms, delivered / ms / 1e6,
         delivered / ms / 1e6 / p.multiProcessorCount, l2 / ms / 1e6);
  return 0;
}
