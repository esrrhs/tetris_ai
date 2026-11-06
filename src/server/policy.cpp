#include "server/policy.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>

#include "tetris/features.h"

#ifdef TETRIS_AI_SERVER_WITH_TORCH
#include <torch/cuda.h>
#include <torch/torch.h>

#include "torch_nn/models.h"
#endif

namespace tsrv {

#ifdef TETRIS_AI_SERVER_WITH_TORCH
struct ModelPolicy::Impl {
  std::shared_ptr<torch_nn::Actor> actor;
  torch::Device device{torch::kCPU};
  int hidden = 256;
  std::string stateFeatures = "zero";
  std::string dir;
};
#else
struct ModelPolicy::Impl {};
#endif

bool ModelPolicy::load(const Options& opt, std::string& message) {
#ifndef TETRIS_AI_SERVER_WITH_TORCH
  (void)opt;
  label_ = "heuristic";
  message = "本构建未链接 LibTorch，模型后端不可用（启发式教师可用）";
  return false;
#else
  if (opt.dir.empty()) {
    message = "未指定 --model 目录";
    return false;
  }
  const std::string weight = (std::filesystem::path(opt.dir) / "actor.pt").string();
  if (!std::filesystem::exists(weight)) {
    message = "找不到权重 " + weight;
    return false;
  }

  tetris::StateFeatureMode sf = tetris::StateFeatureMode::kZero;
  if (!tetris::ParseStateFeatureMode(opt.stateFeatures, sf)) {
    message = "无效的 --state-features: " + opt.stateFeatures;
    return false;
  }
  // Global by design: the checkpoint was trained with exactly one encoding.
  tetris::SetStateFeatureMode(sf);

  try {
    auto impl = std::make_shared<Impl>();
    impl->hidden = opt.hidden;
    impl->stateFeatures = opt.stateFeatures;
    impl->dir = opt.dir;
    if (opt.backend == "cuda") {
      if (torch::cuda::is_available()) {
        impl->device = torch::Device(torch::kCUDA);
        backendName_ = "cuda";
      } else {
        impl->device = torch::Device(torch::kCPU);
        backendName_ = "cpu (cuda unavailable)";
      }
    } else {
      impl->device = torch::Device(torch::kCPU);
      backendName_ = "cpu";
    }
    impl->actor = std::make_shared<torch_nn::Actor>(
        torch_nn::ModelConfig{static_cast<int64_t>(opt.hidden)});
    // ModuleHolder::operator-> returns ActorImpl*, so to()/eval()/load() are
    // torch::nn::Module methods on the implementation, not holder methods.
    (*impl->actor)->load(weight);
    (*impl->actor)->to(impl->device);
    (*impl->actor)->eval();
    impl_ = impl;
    available_ = true;
    label_ = "model:" + std::filesystem::path(opt.dir).filename().string();
    message = "已加载 " + weight + " (hidden=" + std::to_string(opt.hidden) +
              ", state-features=" + opt.stateFeatures + ", device=" + backendName_ + ")";
    return true;
  } catch (const std::exception& e) {
    impl_.reset();
    available_ = false;
    label_ = "heuristic";
    message = std::string("权重加载失败（hidden 是否匹配？）: ") + e.what();
    return false;
  }
#endif
}

int ModelPolicy::choose(const tetris::Env& env,
                        const std::vector<tetris::LegalOption>& options,
                        bool greedy, Decision& out) const {
#ifndef TETRIS_AI_SERVER_WITH_TORCH
  (void)env;
  (void)options;
  (void)greedy;
  (void)out;
  return -1;
#else
  if (!available_ || !impl_ || !impl_->actor || options.empty()) return -1;
  const int n = static_cast<int>(options.size());
  if (n > torch_nn::kMaxActions) return -1;

  auto t0 = std::chrono::steady_clock::now();
  torch::NoGradGuard noGrad;

  std::array<float, tetris::kStateSize> state{};
  env.EncodeActorState(state);
  auto stateT = torch::from_blob(state.data(), {1, tetris::kStateSize},
                                 torch::kFloat32)
                    .clone()
                    .to(impl_->device);
  auto actFeat = torch::empty({n, tetris::kActionFeatSize}, torch::kFloat32);
  float* af = actFeat.data_ptr<float>();
  for (int a = 0; a < n; ++a) {
    std::memcpy(af + static_cast<size_t>(a) * tetris::kActionFeatSize,
                options[static_cast<size_t>(a)].feature.data(),
                tetris::kActionFeatSize * sizeof(float));
  }
  auto sample = torch::zeros({n}, torch::kLong);
  auto actId = torch::arange(0, n, torch::kLong);
  torch::Tensor logits =
      (*impl_->actor)
          ->forward(stateT, actFeat.to(impl_->device), sample.to(impl_->device),
                    actId.to(impl_->device));
  auto row = logits[0].to(torch::kCPU);

  std::vector<float> vals(static_cast<size_t>(n));
  for (int a = 0; a < n; ++a) {
    vals[static_cast<size_t>(a)] = row[a].item<float>();
  }
  float mx = *std::max_element(vals.begin(), vals.end());
  std::vector<double> prob(static_cast<size_t>(n));
  double sum = 0.0;
  for (int a = 0; a < n; ++a) {
    prob[static_cast<size_t>(a)] =
        std::exp(static_cast<double>(vals[static_cast<size_t>(a)] - mx));
    sum += prob[static_cast<size_t>(a)];
  }
  for (double& p : prob) p = 100.0 * p / sum;

  int chosen = 0;
  if (greedy) {
    chosen = static_cast<int>(
        std::max_element(vals.begin(), vals.end()) - vals.begin());
  } else {
    static thread_local std::mt19937 rng{std::random_device{}()};
    double u = std::uniform_real_distribution<double>(0.0, 100.0)(rng);
    double cum = 0.0;
    chosen = n - 1;
    for (int a = 0; a < n; ++a) {
      cum += prob[static_cast<size_t>(a)];
      if (u <= cum) {
        chosen = a;
        break;
      }
    }
  }

  std::vector<int> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return prob[static_cast<size_t>(a)] > prob[static_cast<size_t>(b)];
  });
  out.source = "model";
  out.optionId = chosen;
  out.timeMs = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
  out.candidates.clear();
  for (int k = 0; k < std::min<int>(6, n); ++k) {
    int a = order[static_cast<size_t>(k)];
    const auto& o = options[static_cast<size_t>(a)];
    Candidate c;
    c.rot = o.rot;
    c.x = o.x;
    c.landingY = o.landing_y;
    c.linesCleared = o.lines_cleared;
    c.probability = prob[static_cast<size_t>(a)];
    c.score = vals[static_cast<size_t>(a)];
    out.candidates.push_back(c);
  }
  return chosen;
#endif
}

Json ModelPolicy::describe() const {
  Json j = Json::object();
  j.set("available", Json::boolean(available_));
  j.set("label", Json::str(label_));
  j.set("backend", Json::str(backendName_));
#ifdef TETRIS_AI_SERVER_WITH_TORCH
  if (impl_) {
    j.set("checkpoint", Json::str(impl_->dir));
    j.set("hidden", Json::integer(impl_->hidden));
    j.set("stateFeatures", Json::str(impl_->stateFeatures));
  }
#endif
  return j;
}

Json decisionToJson(const Decision& d) {
  Json j = Json::object();
  j.set("source", Json::str(d.source));
  j.set("optionId", Json::integer(d.optionId));
  j.set("timeMs", Json::real(d.timeMs));
  j.set("rot", Json::integer(d.rot));
  j.set("x", Json::integer(d.x));
  j.set("landingY", Json::integer(d.landingY));
  j.set("linesCleared", Json::integer(d.linesCleared));
  j.set("scoreGain", Json::integer(d.scoreGain));
  Json arr = Json::array();
  for (const auto& c : d.candidates) {
    Json item = Json::object();
    item.set("rot", Json::integer(c.rot));
    item.set("x", Json::integer(c.x));
    item.set("landingY", Json::integer(c.landingY));
    item.set("linesCleared", Json::integer(c.linesCleared));
    item.set("probability", Json::real(c.probability));
    item.set("score", Json::real(c.score));
    arr.push(std::move(item));
  }
  j.set("candidates", std::move(arr));
  return j;
}

}  // namespace tsrv
