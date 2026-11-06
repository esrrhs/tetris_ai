// Compact Tetris state / action features (PerfectDou-style option features,
// without DouZero card matrices).
//
// State (kStateSize = 270):
//   board 20×10 binary (200)
//   piece type one-hot(7), rot one-hot(4), x/y normalized (2)
//   column heights(10), holes, bumpiness, aggregate height
//   lines / score scalars (normalized)
//   bag remaining counts(7) — critic peek; actor path zeros these (imperfect)
//   next-5 preview one-hots (5×7) — public rule info, actor AND critic see it
//
// Action per placement (kActionFeatSize = 18):
//   resulting column heights(10), holes, bumpiness, aggregate height,
//   lines cleared, landing y (normalized),
//   delta holes / bumpiness / aggregate height vs pre-lock board
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "tetris/game.h"
#include "tetris/types.h"

namespace tetris {

constexpr int kStateSize = 235 + kNextPreview * 7;  // 270
constexpr int kActionFeatSize = 18;

// The two cumulative state features (lines/100, score/10000) are unbounded.
// A policy trained on short rollouts only ever sees ~2 / ~1.2 there, while a
// 10000-piece episode reaches ~40 / ~23. Making them stationary removes that
// train/eval distribution shift. kRaw keeps the legacy behaviour (default), so
// existing checkpoints and tests stay bit-for-bit compatible.
enum class StateFeatureMode {
  kRaw = 0,    // legacy, unbounded
  kClamp = 1,  // clamp to the short-rollout range
  kZero = 2,   // drop them entirely (recommended for long-episode training)
};

void SetStateFeatureMode(StateFeatureMode mode);
StateFeatureMode GetStateFeatureMode();
const char* StateFeatureModeName(StateFeatureMode mode);
bool ParseStateFeatureMode(const std::string& s, StateFeatureMode& out);

// Board metrics used by both state and action features.
struct BoardMetrics {
  std::array<float, kBoardWidth> heights{};
  float holes = 0.f;
  float bumpiness = 0.f;
  float aggregate_height = 0.f;
};

BoardMetrics ComputeBoardMetrics(const Board& board);

struct LegalOption {
  int id = -1;   // local slot 0..n-1
  int rot = 0;
  int x = 0;
  int landing_y = 0;
  int lines_cleared = 0;
  std::vector<Op> ops;  // rotate/translate + HardDrop for one Game::Step
  std::array<float, kActionFeatSize> feature{};
};

// Encode observation + scalars. When include_bag is false (actor / imperfect),
// the last 7 dims (bag remaining counts) are left at 0.
void EncodeState(const Game& g, std::array<float, kStateSize>& out,
                 bool include_bag = false);

// Enumerate legal final placements for the active piece. Empty if no piece
// or game over. Deduplicates O rotations and identical landings.
std::vector<LegalOption> LegalOptions(const Game& g);
void LegalOptions(const Game& g, std::vector<LegalOption>& out);

// Build action feature for a simulated post-lock board vs pre-lock metrics.
void EncodeActionFeature(const BoardMetrics& before, const BoardMetrics& after,
                         int lines_cleared, int landing_y,
                         std::array<float, kActionFeatSize>& out);

}  // namespace tetris
