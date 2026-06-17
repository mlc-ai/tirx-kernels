#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define M 256
#define MH 128
#define N 32
#define NH 16
#define K 16
__device__ __forceinline__ int ci(int r, int k) {
  return ((r % 8) + 8 * (k / 8) + 16 * (r / 8)) * 8 + (k % 8);
}
__device__ __forceinline__ uint64_t sd(uint32_t a) {
  return ((uint64_t)((a >> 4) & 0x3FFF)) | ((uint64_t)8 << 16) |
         ((uint64_t)16 << 32) | ((uint64_t)1 << 46);
}
__device__ __forceinline__ float val(int r, int c, int s) {
  return (float)(((r * 3 + c * 5 + s) % 5) - 2);
}
__global__ void __cluster_dims__(2, 1, 1) kk(float *dump) {
  unsigned tid = threadIdx.x, warp = tid >> 5, lb = warp << 5;
  uint32_t rank;
  asm volatile("mov.u32 %0,%%cluster_ctarank;" : "=r"(rank));
  __shared__ __align__(16) __half As[MH * K];
  __shared__ __align__(16) __half Bs[NH * K];
  __shared__ uint32_t tb;
  __shared__ uint64_t mbar;
  if (tid < MH)
    for (int k = 0; k < K; ++k)
      As[ci(tid, k)] = __float2half(val(rank * MH + tid, k, 0)); // A split by M
  if (tid < NH)
    for (int k = 0; k < K; ++k)
      Bs[ci(tid, k)] = __float2half(val(rank * NH + tid, k, 1)); // B split by N
  __syncthreads();
  if (warp == 0) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(&tb);
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.b32 [%0], %1;\n" ::"r"(s),
        "n"(N));
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
  }
  asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n");
  uint32_t tc = tb;
  if (rank == 0 && tid == 0) {
    uint32_t mb = (uint32_t)__cvta_generic_to_shared(&mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;\n" ::"r"(mb));
    asm volatile("fence.proxy.async.shared::cta;\n");
    uint64_t da = sd((uint32_t)__cvta_generic_to_shared(&As[0])),
             db = sd((uint32_t)__cvta_generic_to_shared(&Bs[0]));
    uint32_t id = (1u << 4) | ((unsigned)(N >> 3) << 17) |
                  ((unsigned)(M >> 4) << 24),
             z = 0;
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 "
                 "p,%4,0;\ntcgen05.mma.cta_group::2.kind::f16 "
                 "[%0],%1,%2,%3,{%5,%6,%7,%8,%9,%10,%11,%12},p;\n}\n" ::"r"(tc),
                 "l"(da), "l"(db), "r"(id), "r"(z), "r"(z), "r"(z), "r"(z),
                 "r"(z), "r"(z), "r"(z), "r"(z), "r"(z));
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];\n" ::"r"(
            mb));
    asm volatile(
        "{\n.reg .pred p;\nLw:\nmbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],0;\n@!p bra Lw;\n}\n" ::"r"(mb));
  }
  asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n");
  uint32_t ta = tc + (lb << 16);
  for (int c = 0; c < N; c += 8) {
    uint32_t r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7},[%8];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]),
                   "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(ta + c));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
    for (int j = 0; j < 8; ++j)
      dump[(rank * 128 + tid) * N + (c + j)] = __uint_as_float(r[j]);
  }
  asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n");
  if (warp == 0)
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;\n" ::"r"(tc),
        "n"(N));
}
int main() {
  float *d;
  cudaMallocManaged(&d, 2 * 128 * N * 4);
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(2, 1, 1);
  c.blockDim = dim3(128, 1, 1);
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim = {2, 1, 1};
  c.attrs = a;
  c.numAttrs = 1;
  cudaLaunchKernelEx(&c, kk, d);
  cudaError_t e = cudaDeviceSynchronize();
  if (e) {
    printf("ERR %s\n", cudaGetErrorString(e));
    return 1;
  }
  for (int cta = 0; cta < 2; ++cta)
    for (int lane = 0; lane < 128; ++lane)
      for (int col = 0; col < N; ++col)
        printf("%d %d %d %g\n", cta, lane, col,
               d[(cta * 128 + lane) * N + col]);
  return 0;
}
