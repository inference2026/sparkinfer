// int8 tensor-core GEMM for Qwythos batched prefill (see prefill_i8.h).
//
// C[M,N] = A[M,K] @ W^T, W native GGUF [N,K] row-major. int8 x int8 -> int32 wmma with the dequant
// (per-token sx[m] * per-channel sw[n]) folded into the store, emitting bf16 C. The tiling mirrors
// the bf16 batched-prefill GEMM exactly (128x128 tile, 8 warps, 2x4 frags, BK=32, cp.async
// double-buffer) so this is a drop-in replacement that keeps the rest of the prefill pipeline bf16.
// On sm_120 int8 tensor cores run ~2x the bf16 kernel at identical output fidelity (GGUF weights are
// already 4-6 bit, so int8 weight-quant is lossless vs what is stored).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "sparkinfer/kernels/prefill_i8.h"

namespace sparkinfer { namespace kernels {

namespace {
constexpr int PF_BM = 128;
constexpr int PF_BN = 128;
constexpr int PF_BK = 32;

__device__ __forceinline__ void pf_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

// Per-row symmetric int8 quantize, one warp per row.
__global__ void pf_quantize_rows_i8(const __nv_bfloat16* __restrict__ x, signed char* __restrict__ q,
                                    float* __restrict__ scale, int rows, int cols) {
    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= rows) return;
    float amax = 0.f;
    for (int c = lane; c < cols; c += 32) amax = fmaxf(amax, fabsf(__bfloat162float(x[(size_t)r * cols + c])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float d = amax / 127.0f;
    if (lane == 0) scale[r] = d;
    for (int c = lane; c < cols; c += 32)
        q[(size_t)r * cols + c] = (signed char)((amax == 0.f) ? 0 : (int)roundf(__bfloat162float(x[(size_t)r * cols + c]) / d));
}

__global__ void pf_gemm_i8_kernel(const signed char* __restrict__ A, const signed char* __restrict__ W,
                                  const float* __restrict__ sx, const float* __restrict__ sw,
                                  __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    using namespace nvcuda;
    __shared__ signed char As[2][PF_BM][PF_BK];
    __shared__ signed char Bs[2][PF_BN][PF_BK];
    __shared__ int Cs[8][16][16];                    // per-warp int32 fragment staging

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp & 3;                        // rows [wm*32, +32)
    const int wn   = warp >> 2;                       // cols [wn*64, +64)
    const int m0   = blockIdx.y * PF_BM;
    const int n0   = blockIdx.x * PF_BN;
    const int nk   = (K + PF_BK - 1) / PF_BK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) wmma::fill_fragment(cf[i][j], 0);

    // int8 tile is 128x32 = 4096 B = 256 x 16B slots; 256 threads stage 1 A-slot + 1 B-slot each.
    auto stage = [&](int buf, int k0) {
        const int r = tid >> 1, c16 = (tid & 1) * 16;
        const int gm = m0 + r, gk = k0 + c16;
        pf_cp16(&As[buf][r][c16], &A[(size_t)gm * K + gk], gm < M && gk < K);
        const int gn = n0 + r;
        pf_cp16(&Bs[buf][r][c16], &W[(size_t)gn * K + gk], gn < N && gk < K);
        __pipeline_commit();
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PF_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < PF_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf[4];
            #pragma unroll
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(af[i], &As[buf][wm * 32 + i * 16][kk], PF_BK);
            #pragma unroll
            for (int j = 0; j < 4; j++) wmma::load_matrix_sync(bf[j], &Bs[buf][wn * 64 + j * 16][kk], PF_BK);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        __syncthreads();
        buf ^= 1;
    }
    // Store 8 fragments via per-warp int32 staging; fold the dequant sx[m]*sw[n] into the bf16 write.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int gm = m0 + wm * 32 + i * 16, gn = n0 + wn * 64 + j * 16;
            wmma::store_matrix_sync(&Cs[warp][0][0], cf[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, cc = e & 15;
                const int rm = gm + r, rn = gn + cc;
                if (rm < M && rn < N)
                    C[(size_t)rm * N + rn] = __float2bfloat16((float)Cs[warp][r][cc] * sx[rm] * sw[rn]);
            }
            __syncwarp();
        }
    }
}
} // namespace

void launch_prefill_quantize_rows_i8(const void* x_bf16, signed char* q, float* scale,
                                     int rows, int cols, cudaStream_t stream) {
    pf_quantize_rows_i8<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16), q, scale, rows, cols);
}

void launch_prefill_gemm_i8(const signed char* A, const signed char* W,
                            const float* sx, const float* sw, void* C,
                            int M, int N, int K, cudaStream_t stream) {
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM);
    pf_gemm_i8_kernel<<<grid, 256, 0, stream>>>(
        A, W, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N, K);
}

}} // namespace sparkinfer::kernels
