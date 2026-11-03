#pragma once

#include "algo/rollout.h"
#include "torch_nn/ppo_learner.h"
#include <torch/torch.h>

namespace torch_nn {

BatchTensors transitionsToBatch(const std::vector<algo::Transition>& tr,
                                float gamma = 0.99f, float lambda = 0.95f,
                                torch::Device device = torch::kCPU);

}  // namespace torch_nn
