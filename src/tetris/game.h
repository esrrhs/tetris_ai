// Step-based Tetris client for RL (semantics match Go game/game.go).
//
// Step semantics (exact):
//  1. Time advances only via Step(ops...).
//  2. Piece RNG uses 7-bag (Go-compatible PRNG for the same seed).
//  3. When there is NO active piece, Step first SPAWNS the next piece at the
//     top, then applies ops to it, then tries gravity drop of 1 row.
//     If spawn collides with locked cells → Done=true (game over); ops/gravity skipped.
//  4. When there IS an active piece, Step applies ops in order, then drops
//     exactly one row (unless Op::HardDrop was used — then the piece falls to
//     lock immediately and gravity is skipped).
//  5. Ops take effect only inside Step. Invalid moves (wall/overlap) are ignored.
//  6. When the piece cannot drop (or hard-dropped), it LOCKS and completed
//     lines are cleared in THIS step. The NEXT Step (not the same one) spawns.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "tetris/bag.h"
#include "tetris/board.h"
#include "tetris/types.h"

namespace tetris {

class Game {
 public:
  explicit Game(int64_t seed);

  int64_t Seed() const { return seed_; }
  int Score() const { return score_; }
  int Lines() const { return lines_; }
  int PiecesPlaced() const { return pieces_; }
  // Consecutive clear-locks including the latest clear (0 if last lock cleared 0).
  int Combo() const { return combo_; }
  bool Done() const { return done_; }
  bool HasActivePiece() const { return piece_.has_value(); }

  Observation Observe() const;

  // Advance one discrete tick. ops may be empty.
  Result Step(std::span<const Op> ops);
  Result Step(std::initializer_list<Op> ops) {
    return Step(std::span<const Op>(ops.begin(), ops.size()));
  }
  Result Step() { return Step(std::span<const Op>{}); }
  Result Step(Op op) {
    Op arr[1] = {op};
    return Step(std::span<const Op>(arr, 1));
  }

  // Copy of locked cells.
  std::array<std::array<Cell, kBoardWidth>, kBoardHeight> BoardCells() const {
    return board_.cells;
  }

  // Copy of active piece, or nullopt.
  std::optional<ActivePiece> Active() const { return piece_; }

  // Mutable board access for tests / wrappers (matches Go package-level board).
  Board& mutable_board() { return board_; }
  const Board& board() const { return board_; }

  // Remaining pieces in the current 7-bag (critic peek). Empty near bag end.
  std::vector<PieceType> PeekBagRemaining() const { return bag_.PeekRemaining(); }

  // Next n upcoming pieces in draw order (spans bag boundaries). The standard
  // preview rule is kNextPreview=5; the next-panel UI and state features both
  // use this. Pure peek — does not affect the piece sequence.
  std::vector<PieceType> NextPieces(int n) const { return bag_.PeekNext(n); }

  // Force an active piece (tests).
  void ForcePiece(PieceType t, int rot, int x, int y) {
    piece_ = ActivePiece{t, rot, x, y};
  }

  // ASCII render (see render.cpp).
  std::string Render(bool paint_active = true) const;
  std::string String() const { return Render(true); }

 private:
  bool TryMove(int dx, int dy);
  bool TryRotate(int dir);
  int LockPiece();

  Board board_;
  Bag7 bag_;
  std::optional<ActivePiece> piece_;
  int score_ = 0;
  int lines_ = 0;
  int pieces_ = 0;
  int combo_ = 0;  // consecutive clear-locks; 0 when streak broken
  bool done_ = false;
  int64_t seed_ = 0;
};

}  // namespace tetris
