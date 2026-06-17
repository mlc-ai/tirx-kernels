// Hand-written tcgen05.mma harness: M=128, N=128, K=16, cta_group=1, kind::f16.
// Lays A/B in SMEM in the canonical no-swizzle K-major form, builds the SMEM
// and instruction descriptors by hand (per the PTX table / CUTLASS
// make_umma_desc), issues the MMA, then reads the accumulator back with
// hand-written tcgen05.ld and compares against a numpy-style A@B^T computed on
// the host.
//
// Canonical no-swizzle Major-K layout (units of uint128_t = 8 f16):
//   ((8,n_mn),2):((1,SBO),LBO), with LBO=8, SBO=16 for M=128,K=16.
//   u128 index of (m,k) = (m%8) + 8*(k/8) + 16*(m/8); the f16 sits at lane k%8.

#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define M 64
#define N 128
#define K 16

__device__ __forceinline__ int canon_idx(int row, int k) {
  // f16 element offset within an (row x K) canonical K-major SMEM tile
  return ((row % 8) + 8 * (k / 8) + 16 * (row / 8)) * 8 + (k % 8);
}

__device__ __forceinline__ uint64_t smem_desc(uint32_t smem_addr) {
  // no swizzle, K-major, LBO=8, SBO=16 (uint128 units), version=1,
  // base_offset=0
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr >> 4) & 0x3FFF); // bits[0,14) start address
  d |= (uint64_t)(8u & 0x3FFF) << 16;         // bits[16,30) leading byte offset
  d |= (uint64_t)(16u & 0x3FFF) << 32;        // bits[32,46) stride byte offset
  d |= (uint64_t)1 << 46;                     // bits[46,48) version = 1
  // base_offset=0 [49,52), lbo_mode=0 [52], swizzle=0 [61,64)
  return d;
}

__global__ void mma_kernel(const __half *gA, const __half *gB, float *out) {
  unsigned tid = threadIdx.x, warp = tid >> 5, lanebase = warp << 5;
  __shared__ __align__(16) __half As[M * K];
  __shared__ __align__(16) __half Bs[N * K];
  __shared__ uint32_t tmem_base_smem;
  __shared__ uint64_t mbar;

  // cooperative canonical fill: thread t owns row t of A and B
  for (int k = 0; k < K; ++k) {
    As[canon_idx(tid, k)] = gA[tid * K + k];
    Bs[canon_idx(tid, k)] = gB[tid * K + k];
  }
  __syncthreads();

  if (warp == 0) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(&tmem_base_smem);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;\n" ::"r"(s),
        "n"(N));
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  __syncthreads();
  uint32_t tmem_c = tmem_base_smem;

  if (tid == 0) {
    // init mbarrier
    uint32_t mb = (uint32_t)__cvta_generic_to_shared(&mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(mb));
    asm volatile("fence.proxy.async.shared::cta;\n");

    uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(&As[0]);
    uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(&Bs[0]);
    uint64_t da = smem_desc(a_addr), db = smem_desc(b_addr);
    uint32_t idesc =
        (1u << 4) | ((unsigned)(N >> 3) << 17) | ((unsigned)(M >> 4) << 24);
    uint32_t scaleC = 0; // p=false => D = A*B (ignore input D)
    uint32_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, "
                 "%6, %7, %8}, p;\n\t}\n" ::"r"(tmem_c),
                 "l"(da), "l"(db), "r"(idesc), "r"(scaleC), "r"(m0), "r"(m1),
                 "r"(m2), "r"(m3));
    // commit MMA completion to the mbarrier
    uint32_t mb2 = (uint32_t)__cvta_generic_to_shared(&mbar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n" ::"r"(
            mb2));
  }
  __syncthreads();

  // wait for MMA completion (phase 0)
  if (tid == 0) {
    uint32_t mb = (uint32_t)__cvta_generic_to_shared(&mbar);
    asm volatile("{\n\t.reg .pred p;\n\tLwait:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
                 "@!p bra Lwait;\n\t}\n" ::"r"(mb));
  }
  __syncthreads();

  // read accumulator: warpgroup, thread tid reads lane tid, cols 0..N-1 via
  // 32x32b.x8 loops
  uint32_t ta = tmem_c + (lanebase << 16);
  for (int c = 0; c < N; c += 8) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
    uint32_t a = ta + c;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5),
                   "=r"(r6), "=r"(r7)
                 : "r"(a));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
    uint32_t rr[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
    for (int j = 0; j < 8; ++j)
      out[tid * N + (c + j)] = __uint_as_float(rr[j]);
  }
  __syncthreads();
  if (warp == 0)
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem_c),
        "n"(N));
}

int main() {
  __half *gA, *gB;
  float *out;
  cudaMallocManaged(&gA, M * K * sizeof(__half));
  cudaMallocManaged(&gB, N * K * sizeof(__half));
  cudaMallocManaged(&out, M * N * sizeof(float));
  auto val = [](int r, int c, int s) {
    return (float)(((r * 3 + c * 5 + s) % 5) - 2);
  };
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      gA[m * K + k] = __float2half(val(m, k, 0));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      gB[n * K + k] = __float2half(val(n, k, 1));

  mma_kernel<<<1, 128>>>(gA, gB, out);
  cudaError_t e = cudaDeviceSynchronize();
  if (e) {
    printf("CUDA ERROR: %s\n", cudaGetErrorString(e));
    return 1;
  }

  for (int lane = 0; lane < 128; ++lane)
    for (int col = 0; col < 128; ++col)
      printf("0 %d %d %g\n", lane, col, out[lane * 128 + col]);
  return 0;
}
