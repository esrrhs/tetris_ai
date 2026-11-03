#pragma once

#include <memory>
#include <string>

#include "torch_nn/models.h"
#include <torch/torch.h>

namespace torch_nn {

struct PPOConfig {
  float lr = 3e-4f;
  float clip = 0.2f;
  float entCoef = 0.01f;
  float vfCoef = 0.5f;
  float maxGradNorm = 0.5f;
  float criticMaxGradNorm = 5.0f;
  int epochs = 4;
  int minibatch = 256;
  float targetKL = 0.03f;
};

struct BatchTensors {
  torch::Tensor state;         // [B, kStateInput] actor
  torch::Tensor criticState;   // [B, kStateInput]
  torch::Tensor actionFeat;    // [L, kActionInput]
  torch::Tensor actionSample;  // [L]
  torch::Tensor actionId;      // [L]
  torch::Tensor actions;       // [B] chosen local ids
  torch::Tensor oldLogp;       // [B]
  torch::Tensor adv;           // [B]
  torch::Tensor ret;           // [B]
  torch::Tensor oldValue;      // [B]
};

struct PPOStats {
  double pgLoss = 0;
  double vLoss = 0;
  double entropy = 0;
  double kl = 0;
  double clipFraction = 0;
  int epochsCompleted = 0;
};

class PPOLearner {
 public:
  PPOLearner(Actor actor, Critic critic, const PPOConfig& cfg,
             torch::Device device = torch::kCPU);

  void setLr(float lr);
  PPOStats update(BatchTensors& batch);

  void saveOptimizer(const std::string& path) const;
  void loadOptimizer(const std::string& path);

  Actor actor;
  Critic critic;
  PPOConfig config;
  torch::Device device;
  std::unique_ptr<torch::optim::Adam> actorOpt;
  std::unique_ptr<torch::optim::Adam> criticOpt;
};

}  // namespace torch_nn
