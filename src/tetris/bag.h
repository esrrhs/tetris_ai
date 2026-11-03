// 7-bag piece generator matching Go game/bag.go (same seed → same sequence).
#pragma once

#include <array>
#include <cstdint>

#include "tetris/go_rand.h"
#include "tetris/types.h"

namespace tetris {

class Bag7 {
 public:
  explicit Bag7(int64_t seed);

  PieceType Next();

 private:
  void Refill();

  go_rand::Source rng_;
  std::array<PieceType, 7> bag_{};
  int next_ = 0;
};

}  // namespace tetris
