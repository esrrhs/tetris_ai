// Tetris AI — external evaluation on the built-in Env (greedy or stochastic).
//
// Primary metric: score (and lines) after up to N pieces; die early → stop.
// Example:
//   tetris_ai_eval --ckpt /tmp/tetris_ckpt2 --max-pieces 1000 --games 5 --seed 1
//   tetris_ai_eval --ckpt DIR --max-pieces 10000 --games 3
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "algo/eval.h"
#include "torch_nn/models.h"
#include <torch/version.h>

namespace {

struct Args {
  std::string ckpt;
  int maxPieces = 1000;
  int games = 1;
  uint64_t seed = 1;
  int hidden = 128;
  std::string backend = "cpu";
  bool greedy = true;
  bool help = false;
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
  if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) a.help = true;
  a.ckpt = argValue(argc, argv, "--ckpt", "");
  a.maxPieces = std::atoi(argValue(argc, argv, "--max-pieces", "1000"));
  a.games = std::atoi(argValue(argc, argv, "--games", "1"));
  a.seed = static_cast<uint64_t>(std::atoll(argValue(argc, argv, "--seed", "1")));
  a.hidden = std::atoi(argValue(argc, argv, "--hidden", "128"));
  a.backend = argValue(argc, argv, "--backend", "cpu");
  if (hasFlag(argc, argv, "--stochastic")) a.greedy = false;
  if (hasFlag(argc, argv, "--greedy")) a.greedy = true;
}

void printHelp() {
  std::cout
      << "tetris_ai_eval — score after up to N pieces on built-in Env\n"
      << "  看最多 N 个积木的分数；死了就提前结束。\n"
      << "Usage:\n"
      << "  tetris_ai_eval --ckpt DIR [--max-pieces N] [--games G] [--seed S]\n"
      << "                [--hidden H] [--backend cpu|cuda|mps] [--greedy|--stochastic]\n"
      << "\n"
      << "Defaults: --max-pieces 1000. Typical: 1000 or 10000.\n"
      << "Stop when pieces_placed >= N OR spawn-fail death (Done).\n"
      << "Headline: score / lines. Also: pieces_placed, survived, death_piece.\n"
      << "\n"
      << "Per-game line (score first):\n"
      << "  score=... lines=... pieces_placed=... seed=... survived=... death_piece=...\n";
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parseArgs(argc, argv, args);
  if (args.help) {
    printHelp();
    return 0;
  }
  if (args.ckpt.empty()) {
    std::fprintf(stderr, "Missing --ckpt DIR (train --out layout: actor.pt).\n");
    printHelp();
    return 1;
  }
  if (args.maxPieces < 1) args.maxPieces = 1;
  if (args.games < 1) args.games = 1;

  torch::Device device = torch::kCPU;
  std::string deviceName = "CPU";
  if (args.backend == "cuda") {
    if (!torch::cuda::is_available()) {
      std::fprintf(stderr, "Backend 'cuda' requested but CUDA is not available.\n");
      return 1;
    }
    device = torch::kCUDA;
    deviceName = "GPU (CUDA)";
  } else if (args.backend == "mps") {
#if defined(__APPLE__) &&                                               \
    (TORCH_VERSION_MAJOR > 1 ||                                         \
     (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
    if (!torch::mps::is_available()) {
      std::fprintf(stderr, "Backend 'mps' requested but MPS is not available.\n");
      return 1;
    }
    device = torch::Device(torch::kMPS);
    deviceName = "GPU (MPS)";
#else
    std::fprintf(stderr, "Backend 'mps' not supported on this platform/LibTorch.\n");
    return 1;
#endif
  } else if (args.backend != "cpu") {
    std::fprintf(stderr, "Unknown backend '%s' (use cuda|mps|cpu).\n",
                 args.backend.c_str());
    return 1;
  }

  std::string actorPath = args.ckpt + "/actor.pt";
  if (!std::filesystem::exists(actorPath)) {
    std::fprintf(stderr, "Actor checkpoint not found: %s\n", actorPath.c_str());
    return 1;
  }

  torch_nn::ModelConfig mcfg;
  mcfg.hidden = args.hidden;
  torch_nn::Actor actor(mcfg);
  try {
    actor->load(actorPath);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Failed to load actor from %s: %s\n", actorPath.c_str(),
                 e.what());
    return 1;
  }
  actor->to(device);
  actor->eval();

  std::cout << "tetris_ai_eval  (score after up to N pieces; death → early stop)\n"
            << "  ckpt=" << args.ckpt << " device=" << deviceName
            << " hidden=" << args.hidden
            << " greedy=" << (args.greedy ? "true" : "false")
            << " max_pieces=" << args.maxPieces << " games=" << args.games
            << "\n";

  // Eval inference on CPU for the shared evaluateGame path (matches rollout).
  actor->to(torch::kCPU);

  long long sumPieces = 0;
  long long sumLines = 0;
  long long sumScore = 0;
  int deaths = 0;
  int caps = 0;

  for (int g = 0; g < args.games; ++g) {
    algo::EvalConfig cfg;
    cfg.maxPieces = args.maxPieces;
    cfg.greedy = args.greedy;
    cfg.seed = args.seed + static_cast<uint64_t>(g);
    cfg.reward.denseReward = false;
    cfg.reward.deathPenalty = 0.f;

    algo::EvalResult r = algo::evaluateGame(actor, cfg);
    std::cout << algo::formatEvalLine(r) << "\n";

    sumPieces += r.pieces_placed;
    sumLines += r.lines;
    sumScore += r.score;
    if (r.done) ++deaths;
    if (r.hit_cap) ++caps;
  }

  double n = static_cast<double>(args.games);
  std::cout << "summary mean_score=" << (static_cast<double>(sumScore) / n)
            << " mean_lines=" << (static_cast<double>(sumLines) / n)
            << " mean_pieces=" << (static_cast<double>(sumPieces) / n)
            << " games=" << args.games << " deaths=" << deaths
            << " hit_cap=" << caps << "\n";
  return 0;
}
