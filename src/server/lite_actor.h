// Dependency-free Actor inference for deployments without LibTorch.
//
// The trained Actor is a small MLP (see src/torch_nn/models.cpp):
//   joint = concat(state[270], actionFeat[18]) -> Linear(288, H) -> ReLU
//        -> Linear(H, H) -> ReLU -> Linear(H, H) -> ReLU -> Linear(H, 1)
// Weights are exported by server/tools/export_actor.py into actor.bin, so a
// plain C++20 build can serve the real neural policy on a box with no PyTorch.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "tetris/features.h"

namespace tsrv {

class LiteActor {
 public:
  // Loads actor.bin. Returns false with a human-readable reason on any
  // mismatch (magic, dims, truncated file) so the server can fall back to the
  // heuristic teacher instead of silently playing with wrong weights.
  bool load(const std::string& path, std::string& message);

  int hidden() const { return hidden_; }
  bool loaded() const { return !l1w_.empty(); }

  // One (state, action feature) pair -> scalar logit.
  float score(const std::array<float, tetris::kStateSize>& state,
              const std::array<float, tetris::kActionFeatSize>& action) const;

  // One shared state, n candidate action features -> n logits. The state
  // projection of the first layer is computed once and reused, mirroring the
  // decomposed addmm in ActorImpl::forward.
  void scoreBatch(const std::array<float, tetris::kStateSize>& state,
                  const float* actions, int n, float* out) const;

 private:
  int hidden_ = 0;
  std::vector<float> l1w_, l1b_, l2w_, l2b_, l3w_, l3b_, headw_;
  float headb_ = 0.f;
};

}  // namespace tsrv
