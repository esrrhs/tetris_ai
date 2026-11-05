#include "tetris/game.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "tetris/shapes.h"

namespace tetris {
namespace {

// Mini glyphs for the next-piece panel: one 4×2 box per upcoming piece,
// drawn with the piece letter, top-left aligned to the shape's bounding box.
void AppendNextPanel(const Game& g, std::vector<std::string>& panel) {
  panel.clear();
  panel.push_back(" NEXT");
  std::vector<PieceType> nxt = g.NextPieces(kNextPreview);
  for (PieceType t : nxt) {
    char ch = ::tetris::ToString(t)[0];
    char box[2][5] = {{'.', '.', '.', '.', '\0'}, {'.', '.', '.', '.', '\0'}};
    int minX = kBoardWidth, maxX = -1, minY = kBoardHeight, maxY = -1;
    for (const Point& p : Cells(t, 0, 0, 0)) {
      minX = std::min(minX, p.x);
      maxX = std::max(maxX, p.x);
      minY = std::min(minY, p.y);
      maxY = std::max(maxY, p.y);
    }
    for (const Point& p : Cells(t, 0, 0, 0)) {
      box[p.y - minY][p.x - minX] = ch;
    }
    panel.emplace_back(box[0]);
    panel.emplace_back(box[1]);
    panel.emplace_back("    ");  // blank separator row
  }
}

}  // namespace

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

  std::vector<std::string> panel;
  AppendNextPanel(*this, panel);

  std::ostringstream b;
  b << '+' << std::string(kBoardWidth, '-') << "+\n";
  for (int y = 0; y < kBoardHeight; ++y) {
    b << '|';
    for (int x = 0; x < kBoardWidth; ++x) b << grid[y][x];
    b << '|';
    if (y < static_cast<int>(panel.size())) b << "  " << panel[y];
    b << '\n';
  }
  b << '+' << std::string(kBoardWidth, '-') << "+\n";
  b << "score=" << score_ << " lines=" << lines_ << " pieces=" << pieces_ << " combo=" << combo_;
  if (piece_) {
    b << " active=" << ::tetris::ToString(piece_->type) << " rot=" << piece_->rot << " @("
      << piece_->x << ',' << piece_->y << ')';
  } else {
    b << " active=none";
  }
  b << " next=";
  for (PieceType t : NextPieces(kNextPreview)) b << ::tetris::ToString(t) << ' ';
  if (done_) b << " GAME_OVER";
  b << '\n';
  return b.str();
}

}  // namespace tetris
