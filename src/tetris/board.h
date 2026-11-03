// Locked-cell board (active piece is NOT stored here).
#pragma once

#include <array>

#include "tetris/shapes.h"
#include "tetris/types.h"

namespace tetris {

struct Board {
  std::array<std::array<Cell, kBoardWidth>, kBoardHeight> cells{};

  void Clear();
  bool Occupied(int x, int y) const;
  bool Fits(PieceType t, int rot, int x, int y) const;
  void Lock(PieceType t, int rot, int x, int y);
  // Removes full rows; returns how many were cleared.
  int ClearLines();
};

int LineScore(int n);

}  // namespace tetris
