#include "tetris/board.h"

namespace tetris {

void Board::Clear() {
  for (int y = 0; y < kBoardHeight; ++y)
    for (int x = 0; x < kBoardWidth; ++x) cells[static_cast<size_t>(y)][static_cast<size_t>(x)] = 0;
}

bool Board::Occupied(int x, int y) const {
  if (x < 0 || x >= kBoardWidth || y >= kBoardHeight) return true;
  if (y < 0) return false;  // above visible board: allowed for kicks
  return cells[static_cast<size_t>(y)][static_cast<size_t>(x)] != 0;
}

bool Board::Fits(PieceType t, int rot, int x, int y) const {
  for (const Point& p : Cells(t, rot, x, y)) {
    if (Occupied(p.x, p.y)) return false;
  }
  return true;
}

void Board::Lock(PieceType t, int rot, int x, int y) {
  Cell id = static_cast<Cell>(static_cast<int>(t) + 1);
  for (const Point& p : Cells(t, rot, x, y)) {
    if (p.y >= 0 && p.y < kBoardHeight && p.x >= 0 && p.x < kBoardWidth) {
      cells[static_cast<size_t>(p.y)][static_cast<size_t>(p.x)] = id;
    }
  }
}

int Board::ClearLines() {
  int cleared = 0;
  int write = kBoardHeight - 1;
  for (int read = kBoardHeight - 1; read >= 0; --read) {
    bool full = true;
    for (int x = 0; x < kBoardWidth; ++x) {
      if (cells[static_cast<size_t>(read)][static_cast<size_t>(x)] == 0) {
        full = false;
        break;
      }
    }
    if (full) {
      ++cleared;
      continue;
    }
    if (write != read) {
      cells[static_cast<size_t>(write)] = cells[static_cast<size_t>(read)];
    }
    --write;
  }
  for (int y = write; y >= 0; --y) {
    for (int x = 0; x < kBoardWidth; ++x) {
      cells[static_cast<size_t>(y)][static_cast<size_t>(x)] = 0;
    }
  }
  return cleared;
}

int LineScore(int n) {
  switch (n) {
    case 1: return 40;
    case 2: return 100;
    case 3: return 300;
    case 4: return 1200;
    default: return 0;
  }
}

int ComboBonus(int combo_index) {
  if (combo_index <= 0) return 0;
  return 40 * combo_index;
}

}  // namespace tetris
