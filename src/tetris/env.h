// Single-agent RL Env wrapper over Game (PerfectDou-style legal options).
//
// Decision loop:
//   Reset(seed) → observation with an active piece
//   LegalOptions() → placement candidates
//   Step(placement_id) → apply ops+HardDrop in one Game::Step; on lock, if not
//     done, auto Step({}) to spawn the next piece so the agent always observes
//     a state with an active piece (or Done).
//
// Reward = lines cleared on the agent decision (lock step). Auto-spawn clears
// none, so reward equals the lock Result::reward.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "tetris/features.h"
#include "tetris/game.h"
#include "tetris/types.h"

namespace tetris {

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

  Game game_;
};

}  // namespace tetris
