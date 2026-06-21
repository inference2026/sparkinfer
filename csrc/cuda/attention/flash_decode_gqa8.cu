#include "blackwell/kernels/attention.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Flash decode for 8:1 GQA — specialized for Qwen3.5-35B-A3B.
//
// Key difference from generic flash_decode.cu:
//   num_q_heads = 16, num_kv_heads = 2 → GQA ratio = 8
//   With only 2 KV heads, the KV cache is 8× smaller than a standard model.
//   This shifts the bottleneck: KV bandwidth is so low that Q projection
//   and routing overhead become comparatively significant.
//
// Strategy: assign 8 Q-heads per KV-head in a single CTA.
//   Each warp handles one Q-head; 8 warps share the same KV tile load.
//   This amortizes shared memory bandwidth for KV across 8 consumers.

namespace blackwell {
namespace kernels {

// GQA ratio hardcoded to 8 for register pressure control.
// For other ratios use the generic kernel in flash_decode.cu.
static constexpr int GQA_RATIO = 8;

template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_decode_gqa8_kernel(
    const scalar_t* __restrict__ q,      // [num_seqs, 16, head_dim]
    const scalar_t* __restrict__ k_pool, // [num_blocks, block_size, 2, head_dim]
    const scalar_t* __restrict__ v_pool,
    const int*      __restrict__ block_table, // [num_seqs, max_blocks]
    const int*      __restrict__ seq_lens,
    scalar_t*       __restrict__ out,    // [num_seqs, 16, head_dim]
    const float scale,
    const int max_blocks_per_seq
) {
    // blockIdx.x = seq_id
    // blockIdx.y = kv_head_id  (0 or 1 for this model)
    // 8 warps per block, one per Q-head
    const int seq_id    = blockIdx.x;
    const int kv_head   = blockIdx.y;
    const int warp_id   = threadIdx.x / 32;   // which Q-head within this KV group
    const int lane      = threadIdx.x % 32;
    const int q_head    = kv_head * GQA_RATIO + warp_id;

    const int seq_len    = seq_lens[seq_id];
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared KV tiles — loaded once per block, consumed by all 8 warps
    extern __shared__ float smem[];
    float* s_k = smem;                         // [BLOCK_SIZE, HEAD_DIM]
    float* s_v = s_k + BLOCK_SIZE * HEAD_DIM;  // [BLOCK_SIZE, HEAD_DIM]

    // Per-warp Q registers
    float q_reg[HEAD_DIM / 32] = {};
    const scalar_t* q_ptr = q + (seq_id * 16 + q_head) * HEAD_DIM;
    for (int d = lane; d < HEAD_DIM; d += 32)
        q_reg[d / 32] = __bfloat162float(((const __nv_bfloat16*)q_ptr)[d]);

    float m = -1e9f, l = 0.f;
    float acc[HEAD_DIM / 32] = {};

    for (int blk = 0; blk < num_blocks; blk++) {
        const int phys = block_table[seq_id * max_blocks_per_seq + blk];
        const scalar_t* k_blk = k_pool + (phys * BLOCK_SIZE * 2 + kv_head) * HEAD_DIM;
        const scalar_t* v_blk = v_pool + (phys * BLOCK_SIZE * 2 + kv_head) * HEAD_DIM;

        // All 8 warps cooperate to load KV tile into shared memory
        // Each warp loads (BLOCK_SIZE * HEAD_DIM / 8) elements
        const int elems = BLOCK_SIZE * HEAD_DIM;
        for (int i = threadIdx.x; i < elems; i += blockDim.x) {
            s_k[i] = __bfloat162float(((const __nv_bfloat16*)k_blk)[i]);
            s_v[i] = __bfloat162float(((const __nv_bfloat16*)v_blk)[i]);
        }
        __syncthreads();

        const int valid = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);
        for (int t = 0; t < valid; t++) {
            float dot = 0.f;
            for (int d = lane; d < HEAD_DIM; d += 32)
                dot += q_reg[d / 32] * s_k[t * HEAD_DIM + d];
            // Warp reduce
            for (int mask = 16; mask > 0; mask >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, mask);
            dot *= scale;

            const float m_new   = fmaxf(m, dot);
            const float exp_m   = __expf(m - m_new);
            const float exp_dot = __expf(dot - m_new);
            l = l * exp_m + exp_dot;
            m = m_new;
            for (int d = lane; d < HEAD_DIM; d += 32)
                acc[d / 32] = acc[d / 32] * exp_m + exp_dot * s_v[t * HEAD_DIM + d];
        }
        __syncthreads();
    }

    scalar_t* out_ptr = out + (seq_id * 16 + q_head) * HEAD_DIM;
    for (int d = lane; d < HEAD_DIM; d += 32)
        ((__nv_bfloat16*)out_ptr)[d] = __float2bfloat16(acc[d / 32] / l);
}

void launch_flash_decode_gqa8(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_kv_heads,  // num_kv_heads=2 for Qwen3.5-35B-A3B
    int head_dim, int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream
) {
    // 8 warps * 32 lanes = 256 threads; one block per (seq, kv_head)
    dim3 grid(num_seqs, num_kv_heads);
    dim3 block(256);
    size_t smem = 2 * block_size * head_dim * sizeof(float);

    flash_decode_gqa8_kernel<__nv_bfloat16, 128, 16>
        <<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool),
        block_table, seq_lens,
        reinterpret_cast<__nv_bfloat16*>(out),
        scale, max_blocks_per_seq
    );
}

} // namespace kernels
} // namespace blackwell
