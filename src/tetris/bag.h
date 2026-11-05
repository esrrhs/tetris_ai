// 7-bag piece generator matching Go game/bag.go (same seed → same sequence).
//
// The upcoming pieces live in an explicit queue topped up with whole freshly
// shuffled bags, so the next N pieces can be previewed across bag boundaries
// without consuming RNG. Bags are shuffled in the same order from the same
// stream positions as the previous lazy-refill implementation, so the draw
// sequence for a given seed is unchanged (Parity.BagSequenceSeed42 guards it).
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <vector>

#include "tetris/go_rand.h"
#include "tetris/types.h"

namespace tetris {

class Bag7 {
 public:
  explicit Bag7(int64_t seed);

  PieceType Next();

  // Remaining pieces in the current bag (not yet drawn). Does not refill.
  // Empty when the bag is exhausted (next Next() will refill).
  // Used as optional critic bag-peek; actor features leave these zeros
  // (imperfect information).
  std::vector<PieceType> PeekRemaining() const;

  // Next n pieces in draw order, spanning bag boundaries. The queue is kept
  // at one full bag (7) after every draw, so PeekNext(0..7) is always full.
  // Public preview info (UI next-panel + state features).
  std::vector<PieceType> PeekNext(int n) const;

 private:
  void EnsureQueue();

  go_rand::Source rng_;
  std::deque<PieceType> queue_;  // upcoming pieces in exact draw order
  int curBag_ = 0;  // queue-front pieces belonging to the current bag
};

}  // namespace tetris
