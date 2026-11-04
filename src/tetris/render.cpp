#include "tetris/game.h"

#include <sstream>
#include <string>

#include "tetris/shapes.h"

namespace tetris {

std::string Game::Render(bool paint_active) const {
  char grid[kBoardHeight][kBoardWidth];
  for (int y = 0; y < kBoardHeight; ++y) {
    for (int x = 0; x < kBoardWidth; ++x) {
      grid[y][x] = board_.cells[static_cast<size_t>(y)][static_cast<size_t>(x)] != 0 ? '#' : '.';
    }
  }
  if (paint_active && piece_) {
    char ch = ::tetris::ToString(piece_->type)[0];
    for (const Point& p : Cells(piece_->type, piece_->rot, piece_->x, piece_->y)) {
      if (p.y >= 0 && p.y < kBoardHeight && p.x >= 0 && p.x < kBoardWidth) {
        grid[p.y][p.x] = ch;
      }
    }
  }

  std::ostringstream b;
  b << '+' << std::string(kBoardWidth, '-') << "+\n";
  for (int y = 0; y < kBoardHeight; ++y) {
    b << '|';
    for (int x = 0; x < kBoardWidth; ++x) b << grid[y][x];
    b << "|\n";
  }
  b << '+' << std::string(kBoardWidth, '-') << "+\n";
  b << "score=" << score_ << " lines=" << lines_ << " pieces=" << pieces_ << " combo=" << combo_;
  if (piece_) {
    b << " active=" << ::tetris::ToString(piece_->type) << " rot=" << piece_->rot << " @("
      << piece_->x << ',' << piece_->y << ')';
  } else {
    b << " active=none";
  }
  if (done_) b << " GAME_OVER";
  b << '\n';
  return b.str();
}

}  // namespace tetris
