// Single-agent RL Env wrapper over Game (PerfectDou-style legal options).
//
// Decision loop:
//   Reset(seed) → observation with an active piece
//   LegalOptions() → placement candidates
//   Step(placement_id) → apply ops+HardDrop in one Game::Step; on lock, if not
//     done, auto Step({}) to spawn the next piece so the agent always observes
//     a state with an active piece (or Done).
//
// Base reward = lineCoef * lines_cleared on the agent decision (lock step).
// Optional potential-based dense shaping (Ng et al.):
//   φ = wH * aggregate_height + wHole * holes + wBump * bumpiness
//   reward = lineCoef * lines + denseCoef * (φ_before - φ_after)
//   + deathPenalty if the step ends the game.
// This is Tetris board-metric shaping, not DouZero card-oracle reward.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "tetris/features.h"
#include "tetris/game.h"
#include "tetris/types.h"

namespace tetris {

struct EnvRewardConfig {
  bool denseReward = true;
  float denseCoef = 0.01f;
  float lineCoef = 1.0f;
  float wH = 1.0f;
  float wHole = 1.0f;
  float wBump = 0.5f;
  float deathPenalty = -1.0f;
};

// Potential φ(board) used by dense shaping.
inline float BoardPotential(const BoardMetrics& m, const EnvRewardConfig& cfg) {
  return cfg.wH * m.aggregate_height + cfg.wHole * m.holes +
         cfg.wBump * m.bumpiness;
}

struct EnvStepResult {
  std::array<float, kStateSize> state{};
  double reward = 0;
  bool done = false;
  int lines_cleared = 0;
  int score = 0;
  int pieces_placed = 0;
};

class Env {
 public:
  Env() : game_(0) {}
  explicit Env(const EnvRewardConfig& reward) : game_(0), reward_(reward) {}

  void setRewardConfig(const EnvRewardConfig& cfg) { reward_ = cfg; }
  const EnvRewardConfig& rewardConfig() const { return reward_; }

  // New game; Step({}) until an active piece exists (or Done).
  // Returns actor-facing state (no bag peek).
  EnvStepResult Reset(int64_t seed);

  // Apply LegalOptions()[id] (or the given option). Respawns after lock.
  EnvStepResult Step(int placement_id);
  EnvStepResult Step(const LegalOption& opt);

  std::vector<LegalOption> LegalOptions() const;
  void LegalOptions(std::vector<LegalOption>& out) const;

  // Actor state (bag zeros) / critic state (bag peek).
  void EncodeActorState(std::array<float, kStateSize>& out) const;
  void EncodeCriticState(std::array<float, kStateSize>& out) const;

  const Game& game() const { return game_; }
  Game& mutable_game() { return game_; }
  bool Done() const { return game_.Done(); }

 private:
  EnvStepResult MakeResult(double reward, int lines_cleared) const;
  void EnsureActivePiece();
  double ShapeReward(const BoardMetrics& before, const BoardMetrics& after,
                     int lines_cleared, bool done) const;

  Game game_;
  EnvRewardConfig reward_{};
};

}  // namespace tetris
