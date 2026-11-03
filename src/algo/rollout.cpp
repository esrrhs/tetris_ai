#include "algo/rollout.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>

#include "algo/ppo.h"
#include "tetris/env.h"

namespace algo {
namespace {

void runWorker(const ModelSet& models, const RolloutConfig& cfg, int nGames,
               uint64_t seed, std::vector<Transition>& out, RolloutStats& stats) {
  torch::NoGradGuard no_grad;
  std::mt19937_64 rng(seed);

  for (int gi = 0; gi < nGames; ++gi) {
    tetris::Env env;
    int64_t gameSeed = static_cast<int64_t>(rng());
    env.Reset(gameSeed);

    int steps = 0;
    int epLines = 0;
    int epScore = 0;
    int epLen = 0;

    while (!env.Done() && steps < cfg.maxStepsPerGame) {
      auto options = env.LegalOptions();
      if (options.empty()) break;

      Transition t;
      env.EncodeActorState(t.state);
      env.EncodeCriticState(t.criticState);
      t.gameId = static_cast<int>(seed % 1000003) * 100000 + gi;
      t.actions.clear();
      t.actions.reserve(options.size());
      for (const auto& o : options) {
        t.actions.emplace_back(o.id, o.feature);
      }

      const int n = static_cast<int>(options.size());
      // Actor: score each (state, actionFeat) jointly.
      auto stateT = torch::from_blob(t.state.data(), {1, tetris::kStateSize},
                                     torch::kFloat32)
                        .clone();
      auto actFeat = torch::empty({n, tetris::kActionFeatSize}, torch::kFloat32);
      float* af = actFeat.data_ptr<float>();
      for (int a = 0; a < n; ++a) {
        std::memcpy(af + a * tetris::kActionFeatSize, options[a].feature.data(),
                    tetris::kActionFeatSize * sizeof(float));
      }
      auto sample = torch::zeros({n}, torch::kLong);  // all belong to batch 0
      auto actId = torch::arange(0, n, torch::kLong);

      torch::Tensor logits =
          (*models.actor)->forward(stateT, actFeat, sample, actId);  // [1, maxA]
      // Use only first n logits (local options packed at ids 0..n-1).
      std::vector<float> logitBuf(n), probBuf(n);
      auto row = logits[0];
      for (int a = 0; a < n; ++a) {
        logitBuf[a] = row[a].item<float>();
      }
      maskedSoftmax(logitBuf.data(), n, probBuf.data());

      float u = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
      float cum = 0.f;
      int chosen = n - 1;
      for (int a = 0; a < n; ++a) {
        cum += probBuf[a];
        if (u <= cum) {
          chosen = a;
          break;
        }
      }
      float p = std::max(probBuf[chosen], 1e-8f);
      t.action = options[chosen].id;
      t.logp = std::log(p);

      auto critIn = torch::from_blob(t.criticState.data(),
                                     {1, tetris::kStateSize}, torch::kFloat32)
                        .clone();
      t.value = (*models.critic)->forward(critIn).squeeze().item<float>();

      auto step = env.Step(options[chosen]);
      t.reward = static_cast<float>(step.reward);
      t.terminal = step.done;
      out.push_back(std::move(t));

      epLines = step.lines_cleared > 0 ? env.game().Lines() : env.game().Lines();
      epScore = step.score;
      ++epLen;
      ++steps;
      if (step.done) break;
    }

    stats.games += 1;
    stats.totalLines += env.game().Lines();
    stats.totalScore += env.game().Score();
    stats.totalPieces += env.game().PiecesPlaced();
    stats.transitions += epLen;
    (void)epLines;
    (void)epScore;
  }
}

}  // namespace

void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::vector<Transition>& out, RolloutStats& stats) {
  out.clear();
  stats = RolloutStats{};
  if (!models.actor || !models.critic || cfg.gamesPerUpdate <= 0) return;

  (*models.actor)->eval();
  (*models.critic)->eval();

  int threads = std::max(1, cfg.threads);
  int per = cfg.gamesPerUpdate / threads;
  int rem = cfg.gamesPerUpdate % threads;

  std::vector<std::vector<Transition>> parts(threads);
  std::vector<RolloutStats> partStats(threads);
  std::vector<std::thread> workers;
  workers.reserve(threads);

  int offset = 0;
  for (int t = 0; t < threads; ++t) {
    int n = per + (t < rem ? 1 : 0);
    uint64_t seed = cfg.seed + static_cast<uint64_t>(offset) * 9973ULL +
                    static_cast<uint64_t>(t) * 0x9e3779b97f4a7c15ULL;
    offset += n;
    workers.emplace_back([&, t, n, seed]() {
      runWorker(models, cfg, n, seed, parts[t], partStats[t]);
    });
  }
  for (auto& w : workers) w.join();

  for (int t = 0; t < threads; ++t) {
    out.insert(out.end(), parts[t].begin(), parts[t].end());
    stats.games += partStats[t].games;
    stats.totalLines += partStats[t].totalLines;
    stats.totalScore += partStats[t].totalScore;
    stats.totalPieces += partStats[t].totalPieces;
    stats.transitions += partStats[t].transitions;
  }
  if (stats.games > 0) {
    stats.meanLines = double(stats.totalLines) / stats.games;
    stats.meanScore = double(stats.totalScore) / stats.games;
    stats.meanEpisodeLen = double(stats.transitions) / stats.games;
  }
}

}  // namespace algo
