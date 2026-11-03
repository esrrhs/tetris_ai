// Tetris AI — single-agent PPO training (PerfectDouCPP-style scaffold).
#include <algorithm>
//
// Example:
//   tetris_ai_train --updates 2 --games 8 --threads 2 --out /tmp/tetris_ckpt
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include "algo/ppo.h"
#include "algo/rollout.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"
#include <torch/version.h>

namespace {

struct Args {
  int updates = 100;
  int games = 64;
  int threads = std::clamp(static_cast<int>(std::thread::hardware_concurrency()),
                           1, 8);
  int hidden = 128;
  int epochs = 4;
  int minibatch = 256;
  float lr = 3e-4f;
  float clip = 0.2f;
  float ent = 0.01f;
  float gamma = 0.99f;
  float lambda = 0.95f;
  float targetKL = 0.03f;
  uint64_t seed = 1;
  std::string out = "ckpt";
  std::string backend = "cpu";
};

const char* argValue(int argc, char** argv, const char* key, const char* def) {
  std::string k = key;
  for (int i = 1; i + 1 < argc; ++i)
    if (k == argv[i]) return argv[i + 1];
  return def;
}

void parseArgs(int argc, char** argv, Args& a) {
  a.updates = std::atoi(argValue(argc, argv, "--updates",
                                 std::to_string(a.updates).c_str()));
  a.games = std::atoi(
      argValue(argc, argv, "--games", std::to_string(a.games).c_str()));
  a.threads = std::atoi(
      argValue(argc, argv, "--threads", std::to_string(a.threads).c_str()));
  a.hidden = std::atoi(
      argValue(argc, argv, "--hidden", std::to_string(a.hidden).c_str()));
  a.epochs = std::atoi(
      argValue(argc, argv, "--epochs", std::to_string(a.epochs).c_str()));
  a.minibatch = std::atoi(
      argValue(argc, argv, "--mb", std::to_string(a.minibatch).c_str()));
  a.lr = float(std::atof(argValue(argc, argv, "--lr", "3e-4")));
  a.clip = float(std::atof(argValue(argc, argv, "--clip", "0.2")));
  a.ent = float(std::atof(argValue(argc, argv, "--ent", "0.01")));
  a.gamma = float(std::atof(argValue(argc, argv, "--gamma", "0.99")));
  a.lambda = float(std::atof(argValue(argc, argv, "--lambda", "0.95")));
  a.targetKL = float(std::atof(argValue(argc, argv, "--target-kl", "0.03")));
  a.seed = std::atoll(argValue(argc, argv, "--seed", "1"));
  a.out = argValue(argc, argv, "--out", a.out.c_str());
  a.backend = argValue(argc, argv, "--backend", a.backend.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parseArgs(argc, argv, args);
  if (args.threads <= 0) args.threads = 1;

  torch::Device device = torch::kCPU;
  std::string deviceName = "CPU";
  if (args.backend == "cuda") {
    if (!torch::cuda::is_available()) {
      std::fprintf(stderr, "CUDA requested but not available.\n");
      return 1;
    }
    device = torch::kCUDA;
    deviceName = "GPU (CUDA)";
  } else if (args.backend != "cpu") {
    std::fprintf(stderr, "Unknown backend '%s' (use cpu|cuda).\n",
                 args.backend.c_str());
    return 1;
  }

  std::filesystem::create_directories(args.out);

  std::cout << "tetris_ai PPO training\n";
  std::cout << "  device=" << deviceName
            << "  state=" << tetris::kStateSize
            << "  actionFeat=" << tetris::kActionFeatSize << "\n";
  std::cout << "  updates=" << args.updates << " games=" << args.games
            << " threads=" << args.threads << " hidden=" << args.hidden
            << " lr=" << args.lr << "\n";

  torch_nn::ModelConfig mcfg;
  mcfg.hidden = args.hidden;
  torch_nn::Actor actor(mcfg);
  torch_nn::Critic critic(mcfg);

  torch_nn::PPOConfig pcfg;
  pcfg.lr = args.lr;
  pcfg.clip = args.clip;
  pcfg.entCoef = args.ent;
  pcfg.epochs = args.epochs;
  pcfg.minibatch = args.minibatch;
  pcfg.targetKL = args.targetKL;

  torch_nn::PPOLearner learner(actor, critic, pcfg, device);

  algo::RolloutConfig rcfg;
  rcfg.gamesPerUpdate = args.games;
  rcfg.threads = args.threads;
  rcfg.seed = args.seed;

  algo::ModelSet models;
  models.actor = &actor;
  models.critic = &critic;

  for (int u = 1; u <= args.updates; ++u) {
    auto t0 = std::chrono::steady_clock::now();
    rcfg.seed = args.seed + static_cast<uint64_t>(u) * 10007ULL;

    std::vector<algo::Transition> tr;
    algo::RolloutStats rs;
    algo::collectRollout(models, rcfg, tr, rs);

    auto batch =
        torch_nn::transitionsToBatch(tr, args.gamma, args.lambda, device);
    auto st = learner.update(batch);

    auto t1 = std::chrono::steady_clock::now();
    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();

    std::printf(
        "[update %d] games=%d transitions=%lld mean_lines=%.2f mean_score=%.1f "
        "ep_len=%.1f | pg=%.4f v=%.4f ent=%.4f kl=%.4f clip=%.3f epochs=%d | "
        "%.2fs\n",
        u, rs.games, rs.transitions, rs.meanLines, rs.meanScore,
        rs.meanEpisodeLen, st.pgLoss, st.vLoss, st.entropy, st.kl,
        st.clipFraction, st.epochsCompleted, sec);
    std::fflush(stdout);

    if (u == args.updates || u % 10 == 0) {
      actor->save(args.out + "/actor.pt");
      critic->save(args.out + "/critic.pt");
      learner.saveOptimizer(args.out + "/optim.pt");
    }
  }

  actor->save(args.out + "/actor.pt");
  critic->save(args.out + "/critic.pt");
  std::cout << "checkpoints written under " << args.out << "\n";
  return 0;
}
