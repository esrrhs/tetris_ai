#include "algo/eval.h"
#include <array>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>

#include "algo/ppo.h"

namespace algo {

EvalResult evaluateGame(torch_nn::Actor& actor, const EvalConfig& cfg) {
  torch::NoGradGuard no_grad;
  actor->eval();

  EvalConfig local = cfg;
  if (local.maxPieces < 1) local.maxPieces = 1;

  tetris::Env env(local.reward);
  env.Reset(static_cast<int64_t>(local.seed));

  std::mt19937_64 rng(local.seed ^ 0xA5A5A5A5ULL);

  EvalResult out;
  out.seed = static_cast<int64_t>(local.seed);

  while (!env.Done() && env.game().PiecesPlaced() < local.maxPieces) {
    auto options = env.LegalOptions();
    if (options.empty()) {
      // No legal placement: hard-drop fallback via Env::Step(-1) path.
      auto step = env.Step(0);
      out.decisions += 1;
      if (step.done) break;
      if (env.game().PiecesPlaced() >= local.maxPieces) break;
      continue;
    }

    const int n = static_cast<int>(options.size());
    std::array<float, tetris::kStateSize> state{};
    env.EncodeActorState(state);

    auto stateT = torch::from_blob(state.data(), {1, tetris::kStateSize},
                                   torch::kFloat32)
                      .clone();
    auto actFeat = torch::empty({n, tetris::kActionFeatSize}, torch::kFloat32);
    float* af = actFeat.data_ptr<float>();
    for (int a = 0; a < n; ++a) {
      std::memcpy(af + a * tetris::kActionFeatSize, options[a].feature.data(),
                  tetris::kActionFeatSize * sizeof(float));
    }
    auto sample = torch::zeros({n}, torch::kLong);
    auto actId = torch::arange(0, n, torch::kLong);

    torch::Tensor logits = actor->forward(stateT, actFeat, sample, actId);
    auto row = logits[0];

    int chosen = 0;
    if (local.greedy) {
      float best = row[0].item<float>();
      for (int a = 1; a < n; ++a) {
        float v = row[a].item<float>();
        if (v > best) {
          best = v;
          chosen = a;
        }
      }
    } else {
      std::vector<float> logitBuf(static_cast<size_t>(n)),
          probBuf(static_cast<size_t>(n));
      for (int a = 0; a < n; ++a) logitBuf[static_cast<size_t>(a)] = row[a].item<float>();
      maskedSoftmax(logitBuf.data(), n, probBuf.data());
      float u = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
      float cum = 0.f;
      chosen = n - 1;
      for (int a = 0; a < n; ++a) {
        cum += probBuf[static_cast<size_t>(a)];
        if (u <= cum) {
          chosen = a;
          break;
        }
      }
    }

    auto step = env.Step(options[static_cast<size_t>(chosen)]);
    out.decisions += 1;
    (void)step;

    // Cap check after each lock/spawn step (PiecesPlaced is locks so far).
    if (env.Done()) break;
    if (env.game().PiecesPlaced() >= local.maxPieces) break;
  }

  out.pieces_placed = env.game().PiecesPlaced();
  out.lines = env.game().Lines();
  out.score = env.game().Score();
  out.done = env.Done();

  if (out.done) {
    // Died attempting spawn of piece (pieces_placed + 1).
    out.death_piece = out.pieces_placed + 1;
    out.survived = false;
    out.hit_cap = false;
  } else {
    // Stopped by maxPieces cap (or empty options without done — treat as cap).
    out.death_piece = -1;
    out.survived = (out.pieces_placed >= local.maxPieces);
    out.hit_cap = out.survived;
  }
  return out;
}

std::string formatEvalLine(const EvalResult& r) {
  // Headline metrics first: score / lines after up to N pieces (or death).
  std::ostringstream oss;
  oss << "score=" << r.score << " lines=" << r.lines
      << " pieces_placed=" << r.pieces_placed << " seed=" << r.seed
      << " survived=" << (r.survived ? "true" : "false")
      << " hit_cap=" << (r.hit_cap ? "true" : "false")
      << " done=" << (r.done ? "true" : "false")
      << " death_piece=" << r.death_piece
      << " decisions=" << r.decisions;
  return oss.str();
}

}  // namespace algo
