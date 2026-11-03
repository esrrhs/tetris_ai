// Tetris AI — single-agent PPO training (PerfectDouCPP-style scaffold).
#include <algorithm>
//
// Example:
//   tetris_ai_train --updates 2 --games 8 --threads 2 --out /tmp/tetris_ckpt --backend cpu
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "algo/ppo.h"
#include "algo/rollout.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"
#include <torch/version.h>

namespace {

struct Args {
  int updates = 100;
  int startUpdate = 1;
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
  std::string resume;
  std::string backend = "cpu";  // cuda | mps | cpu (explicit; no auto)

  bool denseReward = true;
  float denseCoef = 0.01f;
  float lineCoef = 1.0f;
  float deathPenalty = -1.0f;
  float wH = 1.0f;
  float wHole = 1.0f;
  float wBump = 0.5f;

  int poolSize = 16;
  int archiveSize = 16;
  int poolEvery = 20;
  float histProb = 0.2f;
  int snapshotEvery = 10;
};

const char* argValue(int argc, char** argv, const char* key, const char* def) {
  std::string k = key;
  for (int i = 1; i + 1 < argc; ++i)
    if (k == argv[i]) return argv[i + 1];
  return def;
}

bool hasFlag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i)
    if (key == std::string(argv[i])) return true;
  return false;
}

void parseArgs(int argc, char** argv, Args& a) {
  a.updates = std::atoi(argValue(argc, argv, "--updates",
                                 std::to_string(a.updates).c_str()));
  a.startUpdate = std::atoi(argValue(argc, argv, "--start-update", "1"));
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
  const char* res = argValue(argc, argv, "--resume", "");
  if (*res) a.resume = res;

  if (hasFlag(argc, argv, "--dense-reward")) a.denseReward = true;
  if (hasFlag(argc, argv, "--no-dense-reward")) a.denseReward = false;
  a.denseCoef = float(std::atof(argValue(argc, argv, "--dense-coef", "0.01")));
  a.lineCoef = float(std::atof(argValue(argc, argv, "--line-coef", "1.0")));
  a.deathPenalty =
      float(std::atof(argValue(argc, argv, "--death-penalty", "-1.0")));
  a.wH = float(std::atof(argValue(argc, argv, "--w-height", "1.0")));
  a.wHole = float(std::atof(argValue(argc, argv, "--w-hole", "1.0")));
  a.wBump = float(std::atof(argValue(argc, argv, "--w-bump", "0.5")));

  a.poolSize = std::atoi(argValue(argc, argv, "--pool-size", "16"));
  a.archiveSize = std::atoi(argValue(argc, argv, "--archive-size", "16"));
  a.poolEvery = std::atoi(argValue(argc, argv, "--pool-every", "20"));
  a.histProb =
      float(std::atof(argValue(argc, argv, "--historical-prob", "0.2")));
  a.snapshotEvery = std::atoi(
      argValue(argc, argv, "--snapshot-every", std::to_string(a.snapshotEvery).c_str()));
}

void copyModuleWeights(torch::nn::Module& src, torch::nn::Module& dst) {
  torch::NoGradGuard no_grad;
  auto sp = src.parameters();
  auto dp = dst.parameters();
  for (size_t i = 0; i < sp.size() && i < dp.size(); ++i) {
    dp[i].copy_(sp[i].to(dp[i].device()));
  }
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
      std::fprintf(stderr,
                   "Backend 'cuda' requested but CUDA is not available in this "
                   "LibTorch build. Pass --backend mps or --backend cpu "
                   "explicitly.\n");
      return 1;
    }
    device = torch::kCUDA;
    deviceName = "GPU (CUDA)";
  } else if (args.backend == "mps") {
#if defined(__APPLE__) &&                                               \
    (TORCH_VERSION_MAJOR > 1 ||                                         \
     (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
    if (!torch::mps::is_available()) {
      std::fprintf(stderr,
                   "Backend 'mps' requested but MPS is not available (requires "
                   "macOS 12.3+ on Apple Silicon with an MPS-enabled LibTorch). "
                   "Pass --backend cpu explicitly.\n");
      return 1;
    }
    device = torch::Device(torch::kMPS);
    deviceName = "GPU (MPS)";
#else
    std::fprintf(stderr,
                 "Backend 'mps' requested but MPS is not supported on this "
                 "platform/LibTorch build (requires macOS 12.3+ and LibTorch "
                 ">= 1.12). Pass --backend cpu explicitly.\n");
    return 1;
#endif
  } else if (args.backend == "cpu") {
    device = torch::kCPU;
    deviceName = "CPU";
  } else {
    std::fprintf(stderr,
                 "Unknown or unsupported backend '%s'. Valid values: cuda | "
                 "mps | cpu (no auto-detection).\n",
                 args.backend.c_str());
    return 1;
  }

  if (!args.resume.empty() && !hasFlag(argc, argv, "--start-update")) {
    std::string metaPath = args.resume + "/meta.txt";
    std::ifstream metaFile(metaPath);
    if (metaFile.is_open()) {
      std::string key;
      int val = 0;
      while (metaFile >> key >> val) {
        if (key == "update" || key == "updates") {
          args.startUpdate = val + 1;
        }
      }
    }
  }

  std::filesystem::create_directories(args.out);

  std::cout << "tetris_ai PPO training\n";
  std::cout << "  Compute Device: " << deviceName
            << "  state=" << tetris::kStateSize
            << "  actionFeat=" << tetris::kActionFeatSize << "\n";
  std::cout << "  updates=" << args.updates
            << (args.startUpdate > 1
                    ? " (start=" + std::to_string(args.startUpdate) + ")"
                    : "")
            << " games=" << args.games << " threads=" << args.threads
            << " hidden=" << args.hidden << " lr=" << args.lr << "\n";
  std::cout << "  dense_reward="
            << (args.denseReward
                    ? ("yes(coef=" + std::to_string(args.denseCoef) +
                       ",line=" + std::to_string(args.lineCoef) +
                       ",death=" + std::to_string(args.deathPenalty) + ")")
                    : "no")
            << " league(recent=" << args.poolSize
            << ",archive=" << args.archiveSize << ",every=" << args.poolEvery
            << ",hist=" << args.histProb << ")\n";
  std::cout << "  note: rollout inference always on CPU actor/critic copies; "
               "PPO learner on " << deviceName << "\n";

  torch_nn::ModelConfig mcfg;
  mcfg.hidden = args.hidden;

  // Learner-side modules (may live on CUDA/MPS/CPU).
  torch_nn::Actor actor(mcfg);
  torch_nn::Critic critic(mcfg);

  // Rollout-side CPU copies (thread-safe inference; PerfectDou-style split).
  torch_nn::Actor cpuActor(mcfg);
  torch_nn::Critic cpuCritic(mcfg);

  auto syncWeightsToCpu = [&]() {
    torch::NoGradGuard no_grad;
    copyModuleWeights(*actor, *cpuActor);
    copyModuleWeights(*critic, *cpuCritic);
    cpuActor->to(torch::kCPU);
    cpuCritic->to(torch::kCPU);
    cpuActor->eval();
    cpuCritic->eval();
  };

  if (!args.resume.empty()) {
    std::string actPath = args.resume + "/actor.pt";
    std::string critPath = args.resume + "/critic.pt";
    if (std::filesystem::exists(actPath)) actor->load(actPath);
    if (std::filesystem::exists(critPath)) critic->load(critPath);
  }

  torch_nn::PPOConfig pcfg;
  pcfg.lr = args.lr;
  pcfg.clip = args.clip;
  pcfg.entCoef = args.ent;
  pcfg.epochs = args.epochs;
  pcfg.minibatch = args.minibatch;
  pcfg.targetKL = args.targetKL;

  torch_nn::PPOLearner learner(actor, critic, pcfg, device);
  if (!args.resume.empty()) {
    std::string optPath = args.resume + "/optim.pt";
    if (std::filesystem::exists(optPath)) learner.loadOptimizer(optPath);
  }

  syncWeightsToCpu();

  auto histPool = std::make_shared<algo::HistoricalPool>();

  auto pushHistPool = [&](torch_nn::Actor& srcActor, int currentUpdate) {
    auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
    snap->update = currentUpdate;
    snap->actor = torch_nn::Actor(mcfg);
    copyModuleWeights(*srcActor, *snap->actor);
    snap->actor->to(torch::kCPU);
    snap->actor->eval();

    int k = args.poolEvery > 0 ? (currentUpdate / args.poolEvery) : 0;
    bool isPowerOfTwo = (k > 0) && ((k & (k - 1)) == 0);
    bool isCentennial = (currentUpdate % 100 == 0);
    if (args.archiveSize > 0 &&
        (isPowerOfTwo || isCentennial || histPool->archive.empty())) {
      while ((int)histPool->archive.size() >= args.archiveSize &&
             !histPool->archive.empty()) {
        if (histPool->archive.size() > 2) {
          histPool->archive.erase(histPool->archive.begin() + 1);
        } else {
          histPool->archive.erase(histPool->archive.begin());
        }
      }
      // Archive keeps its own copy so recent ring eviction does not drop it.
      auto archSnap = std::make_shared<algo::HistoricalActorSnapshot>();
      archSnap->update = currentUpdate;
      archSnap->actor = torch_nn::Actor(mcfg);
      copyModuleWeights(*snap->actor, *archSnap->actor);
      archSnap->actor->eval();
      histPool->archive.push_back(std::move(archSnap));
    }

    if (args.poolSize > 0) {
      while ((int)histPool->recent.size() >= args.poolSize &&
             !histPool->recent.empty()) {
        histPool->recent.erase(histPool->recent.begin());
      }
      histPool->recent.push_back(std::move(snap));
    }
  };

  auto saveHistoricalPool = [&](const std::string& dir) {
    if (!histPool || histPool->empty()) return;
    std::string poolDir = dir + "/pool";
    std::filesystem::create_directories(poolDir);
    std::ofstream meta(poolDir + "/pool_meta.txt");
    if (!meta.is_open()) return;

    meta << "recent_count " << histPool->recent.size() << "\n";
    for (size_t i = 0; i < histPool->recent.size(); ++i) {
      const auto& snap = histPool->recent[i];
      meta << "recent " << i << " " << snap->update << "\n";
      snap->actor->save(poolDir + "/recent_" + std::to_string(i) + ".pt");
    }
    meta << "archive_count " << histPool->archive.size() << "\n";
    for (size_t i = 0; i < histPool->archive.size(); ++i) {
      const auto& snap = histPool->archive[i];
      meta << "archive " << i << " " << snap->update << "\n";
      snap->actor->save(poolDir + "/archive_" + std::to_string(i) + ".pt");
    }
  };

  auto loadHistoricalPool = [&](const std::string& dir) {
    std::string metaPath = dir + "/pool/pool_meta.txt";
    std::ifstream meta(metaPath);
    if (!meta.is_open()) return;

    std::string tag;
    while (meta >> tag) {
      if (tag == "recent") {
        size_t idx = 0;
        int snapUpd = 0;
        meta >> idx >> snapUpd;
        auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
        snap->update = snapUpd;
        snap->actor = torch_nn::Actor(mcfg);
        std::string actPath =
            dir + "/pool/recent_" + std::to_string(idx) + ".pt";
        if (std::filesystem::exists(actPath)) snap->actor->load(actPath);
        snap->actor->eval();
        histPool->recent.push_back(snap);
      } else if (tag == "archive") {
        size_t idx = 0;
        int snapUpd = 0;
        meta >> idx >> snapUpd;
        auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
        snap->update = snapUpd;
        snap->actor = torch_nn::Actor(mcfg);
        std::string actPath =
            dir + "/pool/archive_" + std::to_string(idx) + ".pt";
        if (std::filesystem::exists(actPath)) snap->actor->load(actPath);
        snap->actor->eval();
        histPool->archive.push_back(snap);
      } else {
        int dummy = 0;
        meta >> dummy;
      }
    }
    std::cout << "  Restored historical pool: recent=" << histPool->recent.size()
              << " archive=" << histPool->archive.size() << "\n";
  };

  if (!args.resume.empty()) {
    loadHistoricalPool(args.resume);
  }

  auto saveAll = [&](const std::string& dir, int currentUpd) {
    std::filesystem::create_directories(dir);
    actor->save(dir + "/actor.pt");
    critic->save(dir + "/critic.pt");
    learner.saveOptimizer(dir + "/optim.pt");
    if (currentUpd > 0) {
      std::ofstream metaFile(dir + "/meta.txt");
      if (metaFile.is_open()) {
        metaFile << "update " << currentUpd << "\n"
                 << "seed " << args.seed << "\n"
                 << "lr " << args.lr << "\n"
                 << "dense_reward " << (args.denseReward ? 1 : 0) << "\n"
                 << "dense_coef " << args.denseCoef << "\n";
      }
      saveHistoricalPool(dir);
    }
  };

  saveAll(args.out, 0);

  algo::RolloutConfig rcfg;
  rcfg.gamesPerUpdate = args.games;
  rcfg.threads = args.threads;
  rcfg.historicalProb = args.histProb;
  rcfg.historicalPool = histPool.get();
  rcfg.reward.denseReward = args.denseReward;
  rcfg.reward.denseCoef = args.denseCoef;
  rcfg.reward.lineCoef = args.lineCoef;
  rcfg.reward.deathPenalty = args.deathPenalty;
  rcfg.reward.wH = args.wH;
  rcfg.reward.wHole = args.wHole;
  rcfg.reward.wBump = args.wBump;

  algo::ModelSet models;
  models.actor = &cpuActor;
  models.critic = &cpuCritic;

  for (int u = args.startUpdate; u <= args.updates; ++u) {
    auto t0 = std::chrono::steady_clock::now();
    rcfg.seed = args.seed + static_cast<uint64_t>(u) * 10007ULL;

    std::vector<algo::Transition> tr;
    algo::RolloutStats rs;
    algo::collectRollout(models, rcfg, tr, rs);

    auto batch =
        torch_nn::transitionsToBatch(tr, args.gamma, args.lambda, device);
    auto st = learner.update(batch);

    // Sync trained weights → CPU rollout models.
    syncWeightsToCpu();

    auto t1 = std::chrono::steady_clock::now();
    double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();

    if (args.poolEvery > 0 && u % args.poolEvery == 0) {
      pushHistPool(cpuActor, u);
    }

    std::printf(
        "[update %d] games=%d transitions=%lld mean_lines=%.2f mean_score=%.1f "
        "ep_len=%.1f",
        u, rs.games, rs.transitions, rs.meanLines, rs.meanScore,
        rs.meanEpisodeLen);
    if (rs.leagueGames > 0) {
      std::printf(" | league_games=%d league_lines=%.2f league_score=%.1f "
                  "pool(recent=%zu,archive=%zu)",
                  rs.leagueGames, rs.leagueMeanLines, rs.leagueMeanScore,
                  histPool->recent.size(), histPool->archive.size());
    }
    std::printf(
        " | pg=%.4f v=%.4f ent=%.4f kl=%.4f clip=%.3f epochs=%d | %.2fs\n",
        st.pgLoss, st.vLoss, st.entropy, st.kl, st.clipFraction,
        st.epochsCompleted, sec);
    std::fflush(stdout);

    if (args.snapshotEvery > 0 &&
        (u % args.snapshotEvery == 0 || u == args.updates)) {
      saveAll(args.out, u);
    }
  }

  saveAll(args.out, args.updates);
  std::cout << "checkpoints written under " << args.out << "\n";
  return 0;
}
