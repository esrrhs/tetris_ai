#pragma once

#include <string>
#include <torch/torch.h>

#include "tetris/features.h"

namespace torch_nn {

// Mirror feature dims as int64 for LibTorch APIs.
constexpr int64_t kStateInput = tetris::kStateSize;        // 270
constexpr int64_t kActionInput = tetris::kActionFeatSize;  // 18
// Max local options packed into a dense logit row (scatter by action id).
constexpr int64_t kMaxActions = 64;

struct ModelConfig {
  int64_t hidden = 128;
};

// Actor: joint = concat(state, actionFeat) → MLP → scalar logit;
// scores scattered into [B, kMaxActions] (illegal slots stay -1e9).
class ActorImpl : public torch::nn::Module {
 public:
  ActorImpl() : ActorImpl(ModelConfig{}) {}
  explicit ActorImpl(const ModelConfig& cfg);

  // state: [B, kStateInput]
  // actionFeat: [L, kActionInput]
  // actionSample: [L] batch index
  // actionId: [L] local option id in 0..kMaxActions-1
  // returns logits: [B, kMaxActions]
  torch::Tensor forward(const torch::Tensor& state,
                        const torch::Tensor& actionFeat,
                        const torch::Tensor& actionSample,
                        const torch::Tensor& actionId) const;

  void save(const std::string& path) const;
  void load(const std::string& path);

  ModelConfig config;
  mutable torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, head{nullptr};
};
TORCH_MODULE(Actor);

// Critic: state → MLP → value [B, 1]
class CriticImpl : public torch::nn::Module {
 public:
  CriticImpl() : CriticImpl(ModelConfig{}) {}
  explicit CriticImpl(const ModelConfig& cfg);

  torch::Tensor forward(const torch::Tensor& state) const;

  void save(const std::string& path) const;
  void load(const std::string& path);

  ModelConfig config;
  mutable torch::nn::Linear c1{nullptr}, c2{nullptr}, c3{nullptr}, out{nullptr};
};
TORCH_MODULE(Critic);

}  // namespace torch_nn
