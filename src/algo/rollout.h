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
  std::array<float, tetris::kStateSize> state{};        // actor (no bag peek)
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
  // V(s_T) for an episode cut off by maxStepsPerGame (not a real terminal).
  // Without this bootstrap the value target silently assumes 0 after the cap,
  // which biases the policy towards ending episodes near the cap.
  float bootstrapValue = 0.0f;
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
  int maxStepsPerGame = 500;  // episode cap; raise for long-horizon training

  // Dense (potential) reward — forwarded to Env.
  tetris::EnvRewardConfig reward{};

  // Terminal-scoring (survival-first, lexicographic) objective.
  // When enabled, per-step rewards are zeroed and each episode is scored once
  // at its end:
  //   real death   R = -deathBase * (1 - n / H)   (strictly negative, and
  //                  milder the longer the episode lasted)
  //   capped alive R = +surviveBase + scoreScale * S  (strictly positive;
  //                  S = score gained in the first scoreWindow pieces only)
  // H is maxStepsPerGame, n is pieces placed. Score is deliberately counted
  // only for the first part of the window so that "score high" and "survive
  // the whole window" are decoupled: the tail is a pure survival test.
  // A capped episode is NOT bootstrapped — R already is its whole return.
  // scoreWindow == 0 counts the full window. With terminalScoring=false the
  // legacy dense-reward path is bit-for-bit unchanged.
  bool terminalScoring = false;
  int scoreWindow = 0;
  float deathBase = 1000.f;
  float surviveBase = 100.f;
  float scoreScale = 0.0025f;

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
