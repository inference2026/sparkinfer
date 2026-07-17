#pragma once
#include <cuda_runtime.h>

// Weight-amortized (grouped-by-expert) MoE expert FFN for Qwen3.6 BATCHED prefill.
//
// forward_token runs the MoE FFN per token: each of a token's top_k experts reloads its whole
// gate/up/down weight from HBM, so a prompt of N tokens pays ~8N expert-weight loads. This path
// instead ingests all N prompt tokens at once: it routes them, PERMUTES them into per-expert
// contiguous groups (counting sort over the token->expert assignment), and runs ONE grouped GEMM
// where every expert's weight is loaded once and reused across all the tokens routed to it. Output
// is numerically the SwiGLU-weighted sum over the token's experts, matching the per-token path.
//
// Weights are pre-dequantized to bf16 [E,out,in] once per layer by the caller (Q4_K rows are
// independent, so a single launch_gguf_dequant over all 256 experts amortizes the weight read).

namespace sparkinfer { namespace kernels {

// Router logits (fp32): logits[t,e] = sum_h hn[t,h] * router_w[e,h].  hn/router_w bf16, native
// router_w [E,H] (row per expert).  fp32 output matches the per-token float router's top-k selection.
void launch_moe_prefill_router_logits(const void* hn, const void* router_w, float* logits,
                                      int N, int E, int H, cudaStream_t stream = nullptr);

// Build the expert permutation from the router's per-token top_k assignment.
//   expert_ids:     [N, top_k]   int32   (which expert each (token,slot) routes to)
//   expert_weights: [N, top_k]   float   (softmax routing weight)
// Outputs (device):
//   counts:   [E]      int32  histogram of tokens-per-expert (caller zeroes it)
//   offsets:  [E+1]    int32  prefix sum of counts (offsets[E] = P = N*top_k)
//   perm_src: [P]      int32  source token index for each permuted row
//   perm_w:   [P]      float  routing weight for each permuted row
// Single call; internally: histogram -> exclusive scan -> scatter.
void launch_moe_prefill_permute(
    const int* expert_ids, const float* expert_weights,
    int* counts, int* offsets, int* perm_src, float* perm_w,
    int N, int E, int top_k, cudaStream_t stream = nullptr);

// Gather permuted token rows: x_perm[p, :] = src[perm_src[p], :].  x/x_perm bf16 [.,dim].
void launch_moe_prefill_gather(const void* src, const int* perm_src, void* x_perm,
                               int P, int dim, cudaStream_t stream = nullptr);

// Upper bound on the number of BM-row tiles the grouped GEMM schedule needs, for P permuted rows over
// E experts. Caller allocates the schedule scratch (tile_expert[maxtiles], tile_row0[maxtiles], one
// d_ntiles int) once and reuses it across the gate/up/down GEMMs (same P/offsets).
int moe_prefill_grouped_maxtiles(int P, int E);

// Build the per-tile schedule (expert + start row for each BM-row tile) from `offsets`. Run once per
// MoE layer; the gate/up/down grouped GEMMs then share it.
void launch_moe_prefill_build_sched(const int* offsets, int* tile_expert, int* tile_row0,
                                    int* d_ntiles, int E, cudaStream_t stream = nullptr);

// Grouped GEMM: for each permuted row p (belonging to expert e=group(p)),
//   C[p, :] = A[p, :] @ W[e]^T,  W[e] native [Nout, K] bf16 (row r of C = dot over K).
// Uses the schedule built by launch_moe_prefill_build_sched (tile_expert/tile_row0/d_ntiles).
// A: [P,K] bf16, W: [E,Nout,K] bf16, C: [P,Nout] bf16.
void launch_moe_prefill_grouped_gemm(
    const void* A, const void* W, const int* offsets,
    const int* tile_expert, const int* tile_row0, const int* d_ntiles, void* C,
    int P, int E, int Nout, int K, cudaStream_t stream = nullptr);

// SwiGLU: h[p,f] = silu(gate[p,f]) * up[p,f].  gate/up/h bf16 [P,ffn].
void launch_moe_prefill_swiglu(const void* gate, const void* up, void* h,
                               long n, cudaStream_t stream = nullptr);

// Weighted un-permute (scatter-add): out[perm_src[p], :] += perm_w[p] * y[p, :].
// out is FP32 [N,H] (caller zeroes it), y bf16 [P,H].  A token's top_k slots land in different expert
// groups and collide on out[t], so this uses fp32 atomics; caller casts fp32 -> bf16 afterwards.
void launch_moe_prefill_scatter_weighted(
    const void* y, const int* perm_src, const float* perm_w, void* out,
    int P, int H, cudaStream_t stream = nullptr);

// Shared-expert scalar gate: dsw[t] = sigmoid( sum_h hn[t,h] * gate_inp[h] ).  hn/gate_inp bf16.
void launch_moe_shared_gate(const void* hn, const void* gate_inp, float* dsw,
                            int N, int H, cudaStream_t stream = nullptr);

// Finalize the MoE FFN output: out[t,h] = routed_f32[t,h] + gate_t * shared[t,h], cast to bf16.
//   shared may be nullptr (no shared expert) -> out = routed_f32.
//   dsw may be nullptr -> gate_t = 1 (shared present but ungated).
void launch_moe_prefill_finalize(const float* routed_f32, const void* shared, const float* dsw,
                                 void* out, int N, int H, cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
