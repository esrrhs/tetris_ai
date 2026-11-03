#pragma once

#include <cmath>
#include <vector>

#include "algo/rollout.h"

namespace algo {

struct PPOConfig {
  float clip = 0.2f;
  float entCoef = 0.01f;
  float vfCoef = 0.5f;
  int epochs = 4;
  int minibatch = 256;
  float maxGradNorm = 0.5f;
  float criticMaxGradNorm = 5.0f;
  float gamma = 0.99f;
  float lambda = 0.95f;
  float targetKL = 0.03f;  // 0 to disable early stop
};

struct PPOStats {
  double pgLoss = 0;
  double vLoss = 0;
  double entropy = 0;
  double meanRet = 0;
  double meanAdv = 0;
  double approxKL = 0;
  double lastEpochKL = 0;
  double clipFraction = 0;
  int epochsCompleted = 0;
};

// Running mean / variance (Welford) for optional return normalization.
struct RunningNormalizer {
  double count = 0.0;
  double mean = 0.0;
  double M2 = 0.0;

  void update(float val) {
    count += 1.0;
    double delta = double(val) - mean;
    mean += delta / count;
    double delta2 = double(val) - mean;
    M2 += delta * delta2;
  }

  float normalize(float val) const {
    double var = (count > 1.0) ? (M2 / (count - 1.0)) : 1.0;
    float std = float(std::sqrt(std::max(var, 1e-6)));
    return float((double(val) - mean) / std);
  }

  float denormalize(float val) const {
    double var = (count > 1.0) ? (M2 / (count - 1.0)) : 1.0;
    float std = float(std::sqrt(std::max(var, 1e-6)));
    return float(double(val) * std + mean);
  }
};

void assignEpisodeReturns(std::vector<Transition>& tr, float gamma = 0.99f,
                          float lambda = 0.95f);

void maskedSoftmax(const float* logits, int n, float* probs);

}  // namespace algo
