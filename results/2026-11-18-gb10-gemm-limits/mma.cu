// Peak mma.sync throughput on GB10: FP4 (mxf4nvf4 m16n8k64), FP8 (e4m3 m16n8k32), BF16 (m16n8k16).
#include <cstdio>
#include <cuda_runtime.h>
template <int KIND> __global__ void __launch_bounds__(256) k(float* out, int iters) {
  unsigned a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 0x22222222u, a2 = a0 + 7, a3 = a0 * 3, b0 = a0 ^ 0x5a5a5a5au, b1 = a0 + 0x11111111u;
  unsigned sa = 0x38383838u, sb = 0x38383838u;
  float c[8][4] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (KIND == 0)
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
      else if (KIND == 1)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  float s = 0; for (int j = 0; j < 8; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 1234.5f) out[0] = s;
}
template <int KIND> void run(const char* name, int kk) {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  float* o; cudaMalloc(&o, 4); const int iters = 4096; dim3 g(p.multiProcessorCount * 4);
  k<KIND><<<g, 256>>>(o, 16); cudaDeviceSynchronize();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); k<KIND><<<g, 256>>>(o, iters); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  double flop = 2.0 * 16 * 8 * kk * 8.0 * iters * (g.x * 256 / 32);
  printf("%-5s %8.1f TFLOPS dense (%.2f TFLOPS/SM) err=%s\n", name, flop / ms / 1e9, flop / ms / 1e9 / p.multiProcessorCount, cudaGetErrorString(cudaGetLastError()));
}
int main() { run<0>("fp4", 64); run<1>("fp8", 32); run<2>("bf16", 16); }
