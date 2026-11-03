// Compact Tetris state / action features (PerfectDou-style option features,
// without DouZero card matrices).
//
// State (kStateSize = 235):
//   board 20×10 binary (200)
//   piece type one-hot(7), rot one-hot(4), x/y normalized (2)
//   column heights(10), holes, bumpiness, aggregate height
//   lines / score scalars (normalized)
//   bag remaining counts(7) — critic peek; actor path zeros these (imperfect)
//
// Action per placement (kActionFeatSize = 18):
//   resulting column heights(10), holes, bumpiness, aggregate height,
//   lines cleared, landing y (normalized),
//   delta holes / bumpiness / aggregate height vs pre-lock board
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "tetris/game.h"
#include "tetris/types.h"

namespace tetris {

constexpr int kStateSize = 235;
constexpr int kActionFeatSize = 18;

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
