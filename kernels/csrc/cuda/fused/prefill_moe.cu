// Batched (token-parallel) MoE FFN kernels for Qwen3.6-35B-A3B prompt prefill.
// See prefill_moe.h for the pipeline; qwen35_prefill.cpp orchestrates per layer.
//
// The grouped GEMM mirrors the merged int8 prefill GEMM tiling (prefill_gemm_i8.cu:
// 128x128 tile, 8 warps, BK=32, cp.async double buffer) with two changes: the M
// dimension is a per-expert slice of the expert-bucketed pair list (a tile never
// spans experts), and the A rows / C rows can be indirected through pair_tok so
// gate/up read per-token activations and down scatter-adds per-token outputs.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include "sparkinfer/kernels/prefill_moe.h"

namespace sparkinfer {
namespace kernels {
namespace {

__device__ __forceinline__ float pm_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float pm_silu(float x) { return x / (1.f + __expf(-x)); }

// ---- router logits: one warp per (token, expert), gemv_f32-order dot ----
__global__ void pfm_router_logits_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ W,
                                         float* __restrict__ logits,
                                         int n_tokens, int n_experts, int H) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // tokens on grid.x (up to 2^31), expert-groups on grid.y (=32): grid.y = n_tokens would
    // overflow CUDA's 65535 grid-dim limit at N >= 64k and silently fail to launch.
    const int e = blockIdx.y * (blockDim.x >> 5) + warp;
    const int t = blockIdx.x;
    if (e >= n_experts || t >= n_tokens) return;
    const __nv_bfloat16* xr = x + (size_t)t * H;
    const __nv_bfloat16* wr = W + (size_t)e * H;
    float acc = 0.f;
    for (int i = lane; i < H; i += 32) acc += pm_to_f(xr[i]) * pm_to_f(wr[i]);
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) logits[(size_t)t * n_experts + e] = acc;
}

// ---- bucketing: exclusive scan of counts + tile map (one block), then pair scatter ----
constexpr int PM_BM = 128;   // GEMM rows per tile (must match pfm_moe_gemm_i8)

__global__ void pfm_scan_tiles_kernel(const int* __restrict__ counts,
                                      int* __restrict__ offsets, int* __restrict__ cursors,
                                      int* __restrict__ tilemap, int* __restrict__ d_ntiles,
                                      int n_experts) {
    // single block, n_experts <= 1024 threads; simple shared-memory scan
    __shared__ int s_off[1025];
    const int t = threadIdx.x;
    if (t == 0) {
        int run = 0;
        for (int e = 0; e < n_experts; e++) { s_off[e] = run; run += counts[e]; }
        s_off[n_experts] = run;
        int nt = 0;
        for (int e = 0; e < n_experts; e++) {
            const int tiles = (counts[e] + PM_BM - 1) / PM_BM;
            for (int i = 0; i < tiles; i++) { tilemap[2 * nt] = e; tilemap[2 * nt + 1] = i; nt++; }
        }
        d_ntiles[0] = nt;
    }
    __syncthreads();
    if (t <= n_experts) offsets[t] = s_off[t];
    if (t < n_experts) cursors[t] = 0;
}

__global__ void pfm_scatter_kernel(const int* __restrict__ expert_ids,
                                   const float* __restrict__ expert_weights,
                                   const int* __restrict__ offsets, int* __restrict__ cursors,
                                   int* __restrict__ pair_tok, float* __restrict__ pair_w,
                                   int n_pairs, int top_k) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const int e = expert_ids[p];
    const int slot = offsets[e] + atomicAdd(&cursors[e], 1);
    pair_tok[slot] = p / top_k;
    pair_w[slot]   = expert_weights[p];
}

// ---- grouped int8 GEMM over expert-partitioned pair tiles ----
#define PM_BN 128
#define PM_BK 32
__device__ __forceinline__ void pm_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

template <bool A_INDIRECT, bool C_SCATTER>
__global__ void pfm_moe_gemm_i8_kernel(const signed char* __restrict__ A_i8,
                                       const float* __restrict__ sx,
                                       const signed char* __restrict__ W_i8,
                                       const float* __restrict__ sw,
                                       const int* __restrict__ pair_tok,
                                       const float* __restrict__ pair_w,
                                       const int* __restrict__ offsets,
                                       const int* __restrict__ tilemap,
                                       const int* __restrict__ d_ntiles,
                                       __nv_bfloat16* __restrict__ C,
                                       float* __restrict__ out_f32,
                                       int N, int K) {
    using namespace nvcuda;
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * PM_BM;        // first pair row of this tile
    const int cnt = offsets[e + 1] - offsets[e];    // pairs for this expert
    const int M   = min(PM_BM, cnt - mt * PM_BM);   // valid rows in tile

    __shared__ signed char As[2][PM_BM][PM_BK];
    __shared__ signed char Bs[2][PM_BN][PM_BK];
    __shared__ int Cs[8][16][16];
    __shared__ int s_tok[PM_BM];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;
    const int n0 = blockIdx.x * PM_BN;
    const int nk = (K + PM_BK - 1) / PM_BK;
    const signed char* We = W_i8 + (size_t)e * N * K;
    const float*       swe = sw + (size_t)e * N;

    for (int r = tid; r < PM_BM; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) wmma::fill_fragment(cf[i][j], 0);

    auto stage = [&](int buf, int k0) {
        // 256 threads, 16B each: A rows via s_tok, B rows from the expert's weight slice
        const int r = tid >> 1, c16 = (tid & 1) * 16;
        const int gk = k0 + c16;
        const int arow = s_tok[r];
        pm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + gk], arow >= 0 && gk < K);
        const int gn = n0 + r;
        pm_cp16(&Bs[buf][r][c16], &We[(size_t)gn * K + gk], gn < N && gk < K);
        __pipeline_commit();
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PM_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < PM_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf[4];
#pragma unroll
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(af[i], &As[buf][wm * 32 + i * 16][kk], PM_BK);
#pragma unroll
            for (int j = 0; j < 4; j++) wmma::load_matrix_sync(bf[j], &Bs[buf][wn * 64 + j * 16][kk], PM_BK);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        __syncthreads();
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int rm0 = wm * 32 + i * 16, gn0 = n0 + wn * 64 + j * 16;
            wmma::store_matrix_sync(&Cs[warp][0][0], cf[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int el = lane; el < 256; el += 32) {
                const int r = el >> 4, cc = el & 15;
                const int rm = rm0 + r, rn = gn0 + cc;
                if (rm < M && rn < N) {
                    const int p = p0 + rm;
                    const float v = (float)Cs[warp][r][cc]
                                    * sx[A_INDIRECT ? s_tok[rm] : p] * swe[rn];
                    if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                    else           C[(size_t)p * N + rn] = __float2bfloat16(v);
                }
            }
            __syncwarp();
        }
    }
}

// ---- shared-expert helpers ----
__global__ void pfm_shared_gate_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ w,
                                       float* __restrict__ dw, int n_tokens, int H) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    if (t >= n_tokens) return;
    const __nv_bfloat16* xr = x + (size_t)t * H;
    float acc = 0.f;
    for (int i = lane; i < H; i += 32) acc += pm_to_f(xr[i]) * pm_to_f(w[i]);
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) dw[t] = 1.f / (1.f + __expf(-acc));
}

__global__ void pfm_shared_swiglu_kernel(const __nv_bfloat16* __restrict__ gate,
                                         const __nv_bfloat16* __restrict__ up,
                                         const float* __restrict__ dw,
                                         __nv_bfloat16* __restrict__ h, long n, int ffn) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d = dw ? dw[(int)(i / ffn)] : 1.f;
    h[i] = __float2bfloat16(pm_silu(pm_to_f(gate[i])) * pm_to_f(up[i]) * d);
}

__global__ void pfm_resid3_kernel(const __nv_bfloat16* __restrict__ h,
                                  const float* __restrict__ routed,
                                  const __nv_bfloat16* __restrict__ shared,
                                  __nv_bfloat16* __restrict__ x, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = pm_to_f(h[i]) + routed[i];
    if (shared) v += pm_to_f(shared[i]);
    x[i] = __float2bfloat16(v);
}

} // namespace

void launch_pfm_router_logits(const void* x, const void* W, float* logits,
                              int n_tokens, int n_experts, int H, cudaStream_t stream) {
    dim3 grid(n_tokens, (n_experts + 7) / 8);
    pfm_router_logits_kernel<<<grid, 8 * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(W),
        logits, n_tokens, n_experts, H);
}

void launch_pfm_bucket_pairs(const int* expert_ids, const float* expert_weights,
                             const int* counts, int* offsets, int* cursors,
                             int* pair_tok, float* pair_w,
                             int* tilemap, int* d_ntiles,
                             int n_tokens, int n_experts, int top_k, cudaStream_t stream) {
    pfm_scan_tiles_kernel<<<1, n_experts + 1, 0, stream>>>(counts, offsets, cursors,
                                                           tilemap, d_ntiles, n_experts);
    const int P = n_tokens * top_k;
    pfm_scatter_kernel<<<(P + 255) / 256, 256, 0, stream>>>(
        expert_ids, expert_weights, offsets, cursors, pair_tok, pair_w, P, top_k);
}

void launch_pfm_moe_gemm_i8(const signed char* A_i8, const float* sx,
                            const signed char* W_i8, const float* sw,
                            const int* pair_tok, const float* pair_w,
                            const int* offsets, const int* tilemap, const int* d_ntiles,
                            void* C_bf16, float* out_f32,
                            int N_out, int K, int max_tiles,
                            bool a_indirect, bool c_scatter, cudaStream_t stream) {
    dim3 grid((N_out + PM_BN - 1) / PM_BN, max_tiles);
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_bf16);
    if (a_indirect && !c_scatter)
        pfm_moe_gemm_i8_kernel<true, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K);
    else if (!a_indirect && c_scatter)
        pfm_moe_gemm_i8_kernel<false, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K);
    else if (a_indirect && c_scatter)
        pfm_moe_gemm_i8_kernel<true, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K);
    else
        pfm_moe_gemm_i8_kernel<false, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K);
}

void launch_pfm_shared_gate(const void* x, const void* w, float* dw,
                            int n_tokens, int H, cudaStream_t stream) {
    pfm_shared_gate_kernel<<<n_tokens, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(w),
        dw, n_tokens, H);
}

void launch_pfm_shared_swiglu(const void* gate, const void* up, const float* dw,
                              void* h, int n_tokens, int ffn, cudaStream_t stream) {
    const long n = (long)n_tokens * ffn;
    pfm_shared_swiglu_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
        dw, reinterpret_cast<__nv_bfloat16*>(h), n, ffn);
}

void launch_pfm_resid3(const void* h, const float* routed_f32, const void* shared,
                       void* x, long n, cudaStream_t stream) {
    pfm_resid3_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h), routed_f32,
        reinterpret_cast<const __nv_bfloat16*>(shared),
        reinterpret_cast<__nv_bfloat16*>(x), n);
}

} // namespace kernels
} // namespace sparkinfer
