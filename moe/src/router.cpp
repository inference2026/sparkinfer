// Router — thin host wrapper over the device top-k router kernel.
// In sync-free mode the per-expert counts stay on the GPU and are not copied
// back; last_expert_counts() returns empty so callers can't accidentally insert
// a host sync that would break CUDA-graph capture.

#include "sparkinfer/moe/router.h"
#include "sparkinfer/kernels/moe.h"

#include <memory>

namespace sparkinfer {
namespace moe {

struct Router::Impl {
    RouterConfig cfg;
};

Router::Router(const RouterConfig& cfg) : impl_(new Impl{cfg}) {}
Router::~Router() = default;

void Router::route(const float* router_logits, int* expert_ids, float* expert_weights,
                   int* tokens_per_expert, int num_tokens, cudaStream_t stream) {
    kernels::launch_moe_router(
        router_logits, expert_ids, expert_weights, tokens_per_expert,
        num_tokens, impl_->cfg.num_experts, impl_->cfg.top_k,
        impl_->cfg.normalize_weights ? 1 : 0, stream);
}

std::vector<int> Router::last_expert_counts() const {
    return {};   // sync-free: counts remain on device
}

}} // namespace sparkinfer::moe
