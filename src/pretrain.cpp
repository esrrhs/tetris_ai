// Tetris AI — DAgger-style imitation learning from the classical board
// heuristic (src/tetris/heuristic.cpp).
//
// Why: PPO trained on 500-step rollouts optimises a short horizon and dies in
// long games. The heuristic teacher survives 50000+ pieces with zero deaths,
// so we distill it into the same Actor that tetris_ai_eval / tetris_ai_train
// use. States are collected on the *student's* own trajectories (DAgger) so
// the policy learns to recover from its own mistakes instead of only seeing
// flawless teacher boards.
//
// Output layout is the same as tetris_ai_train --out: actor.pt / critic.pt /
// meta.txt, so it can be evaluated directly or used as --resume for PPO.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "algo/eval.h"
#include "algo/ppo.h"
#include "tetris/env.h"
#include "tetris/features.h"
#include "tetris/heuristic.h"
#include "torch_nn/models.h"
#include <torch/torch.h>

namespace {

struct Args {
  int iters = 16;
  int games = 32;
  int steps = 2000;  // max decisions per collected game
  int threads = 8;
  int hidden = 256;
  int epochs = 3;
  int minibatch = 256;
  float lr = 5e-4f;
  uint64_t seed = 1;
  std::string out = "ckpt_bc";
  std::string resume;
  std::string backend = "cpu";
  std::string stateFeatures = "zero";  // stationary state by default
  float studentStart = 0.0f;           // behaviour policy mixing at iter 0
  float studentEnd = 0.95f;            // behaviour policy mixing at final iter
  int evalPieces = 2000;
  int evalGames = 2;
};

const char* argValue(int argc, char** argv, const char* key, const char* def) {
  std::string k = key;
  for (int i = 1; i + 1 < argc; ++i)
    if (k == argv[i]) return argv[i + 1];
  return def;
}

void parseArgs(int argc, char** argv, Args& a) {
  a.iters = std::atoi(argValue(argc, argv, "--iters", "16"));
  a.games = std::atoi(argValue(argc, argv, "--games", "32"));
  a.steps = std::atoi(argValue(argc, argv, "--steps", "2000"));
  a.threads = std::atoi(argValue(argc, argv, "--threads", "8"));
  a.hidden = std::atoi(argValue(argc, argv, "--hidden", "256"));
  a.epochs = std::atoi(argValue(argc, argv, "--epochs", "3"));
  a.minibatch = std::atoi(argValue(argc, argv, "--mb", "256"));
  a.lr = float(std::atof(argValue(argc, argv, "--lr", "5e-4")));
  a.seed = static_cast<uint64_t>(std::atoll(argValue(argc, argv, "--seed", "1")));
  a.out = argValue(argc, argv, "--out", a.out.c_str());
  a.backend = argValue(argc, argv, "--backend", "cpu");
  a.stateFeatures = argValue(argc, argv, "--state-features", "zero");
  a.studentStart = float(std::atof(
      argValue(argc, argv, "--student-prob-start", "0.0")));
  a.studentEnd = float(std::atof(
      argValue(argc, argv, "--student-prob-end", "0.95")));
  a.evalPieces = std::atoi(argValue(argc, argv, "--eval-pieces", "2000"));
  a.evalGames = std::atoi(argValue(argc, argv, "--eval-games", "2"));
  const char* res = argValue(argc, argv, "--resume", "");
  if (*res) a.resume = res;
}

struct ImitationSample {
  std::array<float, tetris::kStateSize> state{};
  // Indexed by local option id (0..n-1), matching Actor's scatter slots.
  std::vector<std::array<float, tetris::kActionFeatSize>> feats;
  int label = 0;  // teacher's local option id
};

struct CollectStats {
  long long agreement = 0;
  long long decisions = 0;
  long long deaths = 0;
};

void copyModuleWeights(torch::nn::Module& src, torch::nn::Module& dst) {
  torch::NoGradGuard no_grad;
  auto sp = src.parameters();
  auto dp = dst.parameters();
  for (size_t i = 0; i < sp.size() && i < dp.size(); ++i) {
    dp[i].copy_(sp[i].to(dp[i].device()));
  }
}

// Collect states on a mixture of teacher / student behaviour, always labelled
// with the teacher's preferred placement.
void collectImitation(torch_nn::Actor* student, float studentProb, int nGames,
                      int maxSteps, uint64_t seed,
                      std::vector<ImitationSample>& out, CollectStats& stats) {
  torch::NoGradGuard no_grad;
  tetris::EnvRewardConfig rcfg;
  rcfg.denseReward = false;
  rcfg.deathPenalty = 0.f;

  for (int gi = 0; gi < nGames; ++gi) {
    std::mt19937_64 rng(seed ^
                        (static_cast<uint64_t>(gi) * 0x9e3779b97f4a7c15ULL));
    tetris::Env env(rcfg);
    env.Reset(static_cast<int64_t>(rng()));

    for (int step = 0; step < maxSteps && !env.Done(); ++step) {
      std::vector<tetris::LegalOption> options = env.LegalOptions();
      if (options.empty()) break;
      const int n = static_cast<int>(options.size());
      if (n > torch_nn::kMaxActions) break;  // cannot be represented by Actor
      const int teacher = tetris::HeuristicBestOption(env.game(), options);
      if (teacher < 0 || teacher >= n) break;

      ImitationSample s;
      env.EncodeActorState(s.state);
      s.feats.resize(static_cast<size_t>(n));
      for (int a = 0; a < n; ++a) {
        s.feats[static_cast<size_t>(a)] = options[static_cast<size_t>(a)].feature;
      }
      s.label = teacher;

      // Student's greedy choice (also used as behaviour policy with probability
      // studentProb). Agreement on states visited by the student is the metric
      // that matters for compounding error.
      auto stateT =
          torch::from_blob(s.state.data(), {1, tetris::kStateSize},
                           torch::kFloat32)
              .clone();
      auto actFeat = torch::empty({n, tetris::kActionFeatSize}, torch::kFloat32);
      float* af = actFeat.data_ptr<float>();
      for (int a = 0; a < n; ++a) {
        std::memcpy(af + a * tetris::kActionFeatSize,
                    s.feats[static_cast<size_t>(a)].data(),
                    tetris::kActionFeatSize * sizeof(float));
      }
      auto sampleIdx = torch::zeros({n}, torch::kLong);
      auto actId = torch::arange(0, n, torch::kLong);
      torch::Tensor logits =
          (*student)->forward(stateT, actFeat, sampleIdx, actId);
      auto row = logits[0];
      int greedy = 0;
      float best = row[0].item<float>();
      for (int a = 1; a < n; ++a) {
        const float v = row[a].item<float>();
        if (v > best) {
          best = v;
          greedy = a;
        }
      }
      ++stats.decisions;
      if (greedy == teacher) ++stats.agreement;

      out.push_back(std::move(s));

      const float u = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
      const int chosen = (u < studentProb) ? greedy : teacher;
      env.Step(options[static_cast<size_t>(chosen)]);
      if (env.Done()) {
        ++stats.deaths;
        break;
      }
    }
  }
}

struct ImitationBatch {
  torch::Tensor state;
  torch::Tensor actionFeat;
  torch::Tensor actionSample;
  torch::Tensor actionId;
  torch::Tensor labels;
  int64_t B = 0;
};

ImitationBatch buildBatch(const std::vector<ImitationSample>& samples,
                          torch::Device device) {
  ImitationBatch b;
  b.B = static_cast<int64_t>(samples.size());
  if (b.B == 0) return b;

  int64_t L = 0;
  for (const auto& s : samples) L += static_cast<int64_t>(s.feats.size());

  b.state = torch::empty({b.B, torch_nn::kStateInput}, torch::kFloat32);
  b.labels = torch::empty({b.B}, torch::kLong);
  b.actionFeat = torch::empty({L, torch_nn::kActionInput}, torch::kFloat32);
  b.actionSample = torch::empty({L}, torch::kLong);
  b.actionId = torch::empty({L}, torch::kLong);

  float* ps = b.state.data_ptr<float>();
  int64_t* pl = b.labels.data_ptr<int64_t>();
  float* pf = b.actionFeat.data_ptr<float>();
  int64_t* psamp = b.actionSample.data_ptr<int64_t>();
  int64_t* pid = b.actionId.data_ptr<int64_t>();

  int64_t off = 0;
  for (int64_t i = 0; i < b.B; ++i) {
    const auto& s = samples[static_cast<size_t>(i)];
    std::memcpy(ps + i * torch_nn::kStateInput, s.state.data(),
                torch_nn::kStateInput * sizeof(float));
    pl[i] = s.label;
    for (size_t a = 0; a < s.feats.size(); ++a) {
      std::memcpy(pf + off * torch_nn::kActionInput, s.feats[a].data(),
                  torch_nn::kActionInput * sizeof(float));
      psamp[off] = i;
      pid[off] = static_cast<int64_t>(a);
      ++off;
    }
  }

  if (device != torch::kCPU) {
    b.state = b.state.to(device);
    b.labels = b.labels.to(device);
    b.actionFeat = b.actionFeat.to(device);
    b.actionSample = b.actionSample.to(device);
    b.actionId = b.actionId.to(device);
  }
  return b;
}

// Returns mean cross-entropy over the epoch.
double trainEpoch(torch_nn::Actor& actor, torch::optim::Adam& opt,
                  const ImitationBatch& batch, torch::Device device, int epochs,
                  int minibatch) {
  const int64_t N = batch.B;
  if (N == 0) return 0.0;
  const int64_t mb = std::min<int64_t>(minibatch, N);
  const int64_t nBatches = (N + mb - 1) / mb;
  double lossSum = 0.0;
  int64_t seen = 0;

  actor->train();
  for (int e = 0; e < epochs; ++e) {
    torch::Tensor indices =
        torch::randperm(N, torch::TensorOptions().dtype(torch::kLong));
    for (int64_t k = 0; k < nBatches; ++k) {
      const int64_t start = k * mb;
      const int64_t end = std::min(start + mb, N);
      const int64_t curB = end - start;
      torch::Tensor mbIdx = indices.slice(0, start, end).to(device);

      torch::Tensor stateMb = batch.state.index_select(0, mbIdx);
      torch::Tensor labelMb = batch.labels.index_select(0, mbIdx);

      // Map packed action rows to their minibatch-local sample index.
      auto lookup = torch::full(
          {N}, -1,
          torch::TensorOptions().dtype(torch::kLong).device(device));
      lookup.index_put_(
          {mbIdx},
          torch::arange(curB, torch::TensorOptions().dtype(torch::kLong).device(
                                  device)));
      torch::Tensor mapped = lookup.index_select(0, batch.actionSample);
      torch::Tensor actIdx = (mapped >= 0).nonzero().squeeze(-1);
      if (actIdx.numel() == 0) continue;

      torch::Tensor actFeatMb = batch.actionFeat.index_select(0, actIdx);
      torch::Tensor actSampleMb = mapped.index_select(0, actIdx);
      torch::Tensor actIdMb = batch.actionId.index_select(0, actIdx);

      torch::Tensor logits =
          actor->forward(stateMb, actFeatMb, actSampleMb, actIdMb);
      torch::Tensor logp = torch::log_softmax(logits, /*dim=*/1);
      torch::Tensor loss = torch::nll_loss(logp, labelMb);

      opt.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_norm_(actor->parameters(), 0.5);
      opt.step();

      const double lv = loss.item<double>();
      lossSum += lv * curB;
      seen += curB;
    }
  }
  actor->eval();
  return seen > 0 ? lossSum / seen : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parseArgs(argc, argv, args);
  if (args.iters < 1) args.iters = 1;
  if (args.games < 1) args.games = 1;
  if (args.steps < 1) args.steps = 1;
  if (args.threads < 1) args.threads = 1;

  tetris::StateFeatureMode sfMode{};
  if (!tetris::ParseStateFeatureMode(args.stateFeatures, sfMode)) {
    std::fprintf(stderr,
                 "Invalid --state-features '%s'. Use raw | clamp | zero.\n",
                 args.stateFeatures.c_str());
    return 1;
  }
  tetris::SetStateFeatureMode(sfMode);

  torch::Device device = torch::kCPU;
  std::string deviceName = "CPU";
  if (args.backend == "cuda") {
    if (!torch::cuda::is_available()) {
      std::fprintf(stderr, "Backend 'cuda' requested but CUDA is unavailable.\n");
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
  torch_nn::ModelConfig mcfg;
  mcfg.hidden = args.hidden;

  torch_nn::Actor actor(mcfg);
  torch_nn::Critic critic(mcfg);
  torch_nn::Actor cpuActor(mcfg);

  if (!args.resume.empty() &&
      std::filesystem::exists(args.resume + "/actor.pt")) {
    actor->load(args.resume + "/actor.pt");
  }
  actor->to(device);
  copyModuleWeights(*actor, *cpuActor);
  cpuActor->to(torch::kCPU);
  cpuActor->eval();

  torch::optim::AdamOptions aOpts(args.lr);
  aOpts.betas(std::make_tuple(0.9, 0.999)).eps(1e-5);
  torch::optim::Adam opt(actor->parameters(), aOpts);

  std::cout << "tetris_ai_pretrain (DAgger imitation of heuristic teacher)\n"
            << "  device=" << deviceName << " iters=" << args.iters
            << " games=" << args.games << " steps=" << args.steps
            << " threads=" << args.threads << " hidden=" << args.hidden
            << " lr=" << args.lr << "\n"
            << "  state_features=" << tetris::StateFeatureModeName(sfMode)
            << " student_prob=" << args.studentStart << "->"
            << args.studentEnd << " out=" << args.out << "\n";

  for (int it = 0; it < args.iters; ++it) {
    auto t0 = std::chrono::steady_clock::now();
    const float sFrac =
        args.iters > 1 ? static_cast<float>(it) / (args.iters - 1) : 1.0f;
    const float studentProb =
        args.studentStart + (args.studentEnd - args.studentStart) * sFrac;

    copyModuleWeights(*actor, *cpuActor);
    cpuActor->to(torch::kCPU);
    cpuActor->eval();

    const int threads = args.threads;
    const int per = args.games / threads;
    const int rem = args.games % threads;
    std::vector<std::vector<ImitationSample>> parts(
        static_cast<size_t>(threads));
    std::vector<CollectStats> cstats(static_cast<size_t>(threads));
    std::vector<std::thread> workers;
    int offset = 0;
    for (int t = 0; t < threads; ++t) {
      const int n = per + (t < rem ? 1 : 0);
      const uint64_t seed = args.seed +
                            static_cast<uint64_t>(it) * 1000003ULL +
                            static_cast<uint64_t>(offset) * 9973ULL +
                            static_cast<uint64_t>(t) * 0x9e3779b97f4a7c15ULL;
      offset += n;
      workers.emplace_back([&, t, n, seed, studentProb]() {
        collectImitation(&cpuActor, studentProb, n, args.steps, seed,
                         parts[static_cast<size_t>(t)],
                         cstats[static_cast<size_t>(t)]);
      });
    }
    for (auto& w : workers) w.join();

    std::vector<ImitationSample> samples;
    size_t totalSize = 0;
    for (const auto& p : parts) totalSize += p.size();
    samples.reserve(totalSize);
    for (auto& p : parts) {
      samples.insert(samples.end(), p.begin(), p.end());
    }

    CollectStats cs;
    for (const auto& s : cstats) {
      cs.agreement += s.agreement;
      cs.decisions += s.decisions;
      cs.deaths += s.deaths;
    }

    ImitationBatch batch = buildBatch(samples, device);
    const double ce =
        trainEpoch(actor, opt, batch, device, args.epochs, args.minibatch);

    // Greedy self-play sanity check on the CPU copy (same path as eval.cpp).
    copyModuleWeights(*actor, *cpuActor);
    cpuActor->to(torch::kCPU);
    cpuActor->eval();
    int survive = 0;
    long long totalPieces = 0;
    for (int g = 0; g < args.evalGames; ++g) {
      algo::EvalConfig cfg;
      cfg.maxPieces = args.evalPieces;
      cfg.greedy = true;
      cfg.seed = 700000ULL + static_cast<uint64_t>(it) * 1000ULL + g;
      cfg.reward.denseReward = false;
      cfg.reward.deathPenalty = 0.f;
      algo::EvalResult r = algo::evaluateGame(cpuActor, cfg);
      if (r.survived) ++survive;
      totalPieces += r.pieces_placed;
    }

    // Save after every iteration so a run can be stopped / resumed safely.
    actor->save(args.out + "/actor.pt");
    critic->save(args.out + "/critic.pt");
    {
      std::ofstream meta(args.out + "/meta.txt");
      if (meta.is_open()) {
        meta << "update 0\n"
             << "mode bc_dagger\n"
             << "iter " << (it + 1) << "\n"
             << "state_features " << args.stateFeatures << "\n"
             << "hidden " << args.hidden << "\n"
             << "seed " << args.seed << "\n";
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    const double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();
    const double agree =
        cs.decisions > 0 ? 100.0 * cs.agreement / cs.decisions : 0.0;
    std::printf(
        "[iter %d/%d] student_prob=%.2f samples=%zu agree=%.2f%% "
        "collect_deaths=%lld ce=%.4f eval_survive=%d/%d eval_pieces=%.0f "
        "| %.1fs\n",
        it + 1, args.iters, studentProb, samples.size(), agree, cs.deaths, ce,
        survive, args.evalGames,
        args.evalGames > 0 ? double(totalPieces) / args.evalGames : 0.0, sec);
    std::fflush(stdout);
  }

  std::cout << "imitation checkpoint written under " << args.out << "\n";
  return 0;
}
