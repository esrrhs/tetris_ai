// Board dimensions and RL-facing observation / step result types.
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace tetris {

// Standard 10×20 visible playfield (no hidden buffer rows).
constexpr int kBoardWidth = 10;
constexpr int kBoardHeight = 20;

// Discrete action applied only inside Step.
enum class Op : int {
  Left = 0,
  Right,
  RotateCW,
  RotateCCW,
  HardDrop,  // soft gravity skipped; piece locks immediately after drop
};

inline const char* ToString(Op o) {
  switch (o) {
    case Op::Left: return "Left";
    case Op::Right: return "Right";
    case Op::RotateCW: return "RotateCW";
    case Op::RotateCCW: return "RotateCCW";
    case Op::HardDrop: return "HardDrop";
  }
  return "Op(?)";
}

// One of the seven tetrominoes.
enum class PieceType : int {
  I = 0,
  L,
  J,
  T,
  O,
  S,
  Z,
  Count
};

inline const char* ToString(PieceType p) {
  static constexpr const char* kNames[] = {"I", "L", "J", "T", "O", "S", "Z"};
  int i = static_cast<int>(p);
  if (i < 0 || i >= static_cast<int>(PieceType::Count)) return "?";
  return kNames[i];
}

// Locked board cell: 0 empty, or PieceType+1 for color/id.
using Cell = int8_t;

// Board / relative coordinate. Y increases downward.
struct Point {
  int x = 0;
  int y = 0;
};

// Currently falling piece (absent when waiting to spawn).
struct ActivePiece {
  PieceType type = PieceType::I;
  int rot = 0;  // 0..3
  int x = 0;    // pivot / origin on the board
  int y = 0;
};

// RL-facing state snapshot.
struct Observation {
  // Locked cells only (active piece not painted in). Rows[0] is the top.
  std::array<std::array<Cell, kBoardWidth>, kBoardHeight> board{};

  bool has_piece = false;
  PieceType piece_type = PieceType::I;
  int rot = 0;
  int x = 0;
  int y = 0;
};

// Step metadata for logging / RL wrappers.
struct Info {
  int lines_cleared = 0;
  int score = 0;  // cumulative
  PieceType piece_id = PieceType::I;
  bool has_piece = false;
  bool locked = false;
  bool spawned = false;
  bool hard_dropped = false;
};

// Returned by every Step call.
struct Result {
  Observation obs;
  double reward = 0;  // lines cleared this step (0 if none)
  bool done = false;  // true when spawn collides (game over)
  Info info;
};

}  // namespace tetris
