// Batched / threaded Env episode collection (single-agent Tetris).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "tetris/env.h"
#include "tetris/features.h"
#include "torch_nn/models.h"

namespace algo {

struct Transition {
  std::array<float, tetris::kStateSize> state{};        // actor (no bag)
  std::array<float, tetris::kStateSize> criticState{};  // with bag peek
  // Local option id → action feature for every legal placement at decision time.
  std::vector<std::pair<int, std::array<float, tetris::kActionFeatSize>>> actions;
  int action = -1;  // chosen local option id
  int gameId = -1;
  bool terminal = false;
  float logp = 0.0f;
  float value = 0.0f;
  float reward = 0.0f;
  float adv = 0.0f;
  float ret = 0.0f;
};

struct RolloutStats {
  int games = 0;
  long long totalLines = 0;
  long long totalScore = 0;
  long long totalPieces = 0;
  long long transitions = 0;
  double meanLines = 0;
  double meanScore = 0;
  double meanEpisodeLen = 0;

  // League-eval games vs historical actors (not used for PPO learning).
  int leagueGames = 0;
  long long leagueTotalLines = 0;
  long long leagueTotalScore = 0;
  double leagueMeanLines = 0;
  double leagueMeanScore = 0;
};

// Single-agent historical actor snapshot (PerfectDou-style league pool).
struct HistoricalActorSnapshot {
  torch_nn::Actor actor;
  int update = 0;
};

// Hierarchical league pool: recent ring + geometrically spaced archive.
struct HistoricalPool {
  std::vector<std::shared_ptr<HistoricalActorSnapshot>> recent;
  std::vector<std::shared_ptr<HistoricalActorSnapshot>> archive;

  bool empty() const { return recent.empty() && archive.empty(); }
};

struct RolloutConfig {
  int gamesPerUpdate = 64;
  int threads = 2;
  uint64_t seed = 1;
  int maxStepsPerGame = 500;  // safety cap on agent decisions

  // Dense (potential) reward — forwarded to Env.
  tetris::EnvRewardConfig reward{};

  // League eval: with non-empty pool, also play
  //   round(gamesPerUpdate * historicalProb)
  // extra games with a sampled historical actor. Those transitions are NOT
  // written to the PPO buffer; only league* stats are updated.
  // Learning always collects exactly gamesPerUpdate games with the current actor.
  float historicalProb = 0.2f;
  const HistoricalPool* historicalPool = nullptr;
};

struct ModelSet {
  torch_nn::Actor* actor = nullptr;
  torch_nn::Critic* critic = nullptr;
};

void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::vector<Transition>& out, RolloutStats& stats);

}  // namespace algo
