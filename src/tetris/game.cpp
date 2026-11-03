#include "tetris/game.h"

#include "tetris/shapes.h"

namespace tetris {

Game::Game(int64_t seed) : bag_(seed), seed_(seed) {}

Observation Game::Observe() const {
  Observation obs;
  obs.board = board_.cells;
  if (piece_) {
    obs.has_piece = true;
    obs.piece_type = piece_->type;
    obs.rot = piece_->rot;
    obs.x = piece_->x;
    obs.y = piece_->y;
  }
  return obs;
}

Result Game::Step(std::span<const Op> ops) {
  Result res;
  if (done_) {
    res.done = true;
    res.obs = Observe();
    res.info.score = score_;
    res.info.lines_cleared = 0;
    return res;
  }

  bool spawned = false;
  bool locked = false;
  bool hard_dropped = false;
  int lines_cleared = 0;

  // --- spawn if needed ---
  if (!piece_) {
    PieceType t = bag_.Next();
    Point origin = SpawnOrigin(t);
    if (!board_.Fits(t, 0, origin.x, origin.y)) {
      done_ = true;
      res.done = true;
      res.obs = Observe();
      res.info = Info{.score = score_, .locked = false, .spawned = false};
      return res;
    }
    piece_ = ActivePiece{t, 0, origin.x, origin.y};
    spawned = true;
    res.info.spawned = true;
    res.info.piece_id = t;
    res.info.has_piece = true;
  }

  // --- apply ops ---
  for (Op op : ops) {
    if (!piece_) break;
    switch (op) {
      case Op::Left:
        TryMove(-1, 0);
        break;
      case Op::Right:
        TryMove(1, 0);
        break;
      case Op::RotateCW:
        TryRotate(1);
        break;
      case Op::RotateCCW:
        TryRotate(-1);
        break;
      case Op::HardDrop:
        while (TryMove(0, 1)) {
        }
        hard_dropped = true;
        break;
    }
  }

  // --- gravity or lock after hard drop ---
  if (piece_) {
    if (hard_dropped) {
      lines_cleared = LockPiece();
      locked = true;
    } else {
      if (!TryMove(0, 1)) {
        lines_cleared = LockPiece();
        locked = true;
      }
    }
  }

  res.reward = static_cast<double>(lines_cleared);
  res.done = done_;
  res.obs = Observe();
  res.info = Info{
      .lines_cleared = lines_cleared,
      .score = score_,
      .has_piece = piece_.has_value(),
      .locked = locked,
      .spawned = spawned,
      .hard_dropped = hard_dropped,
  };
  if (piece_) res.info.piece_id = piece_->type;
  return res;
}

bool Game::TryMove(int dx, int dy) {
  if (!piece_) return false;
  int nx = piece_->x + dx;
  int ny = piece_->y + dy;
  if (!board_.Fits(piece_->type, piece_->rot, nx, ny)) return false;
  piece_->x = nx;
  piece_->y = ny;
  return true;
}

bool Game::TryRotate(int dir) {
  if (!piece_) return false;
  if (piece_->type == PieceType::O) return true;  // O is invariant
  int nrot = (piece_->rot + dir + 4) % 4;
  for (const Point& k : kKickOffsets) {
    int nx = piece_->x + k.x;
    int ny = piece_->y + k.y;
    if (board_.Fits(piece_->type, nrot, nx, ny)) {
      piece_->rot = nrot;
      piece_->x = nx;
      piece_->y = ny;
      return true;
    }
  }
  return false;
}

int Game::LockPiece() {
  ActivePiece p = *piece_;
  board_.Lock(p.type, p.rot, p.x, p.y);
  ++pieces_;
  int cleared = board_.ClearLines();
  lines_ += cleared;
  score_ += LineScore(cleared);
  piece_.reset();
  return cleared;
}

}  // namespace tetris
