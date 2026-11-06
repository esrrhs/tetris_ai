#include "tetris/heuristic.h"

#include <algorithm>
#include <limits>

namespace tetris {
namespace {

struct Feats {
  double rowTrans = 0;
  double colTrans = 0;
  double holes = 0;
  double wells = 0;
  double maxHeight = 0;
};

Feats EvalBoard(const Board& b) {
  Feats f;
  const auto& c = b.cells;
  for (int y = 0; y < kBoardHeight; ++y) {
    int prev = 1;  // left border counts as filled
    for (int x = 0; x < kBoardWidth; ++x) {
      int cur = c[static_cast<size_t>(y)][static_cast<size_t>(x)] != 0 ? 1 : 0;
      if (cur != prev) ++f.rowTrans;
      prev = cur;
    }
    if (prev == 0) ++f.rowTrans;  // right border
  }
  for (int x = 0; x < kBoardWidth; ++x) {
    int prev = 1;  // top border counts as filled
    bool seen = false;
    for (int y = 0; y < kBoardHeight; ++y) {
      int cur = c[static_cast<size_t>(y)][static_cast<size_t>(x)] != 0 ? 1 : 0;
      if (cur != prev) ++f.colTrans;
      prev = cur;
      if (cur) {
        seen = true;
      } else if (seen) {
        ++f.holes;
      }
    }
    if (prev == 0) ++f.colTrans;  // bottom border

    // Well sums: consecutive empty cells with both neighbours filled.
    int depth = 0;
    for (int y = 0; y < kBoardHeight; ++y) {
      bool leftFilled =
          (x == 0) || (c[static_cast<size_t>(y)][static_cast<size_t>(x - 1)] != 0);
      bool rightFilled = (x == kBoardWidth - 1) ||
                         (c[static_cast<size_t>(y)][static_cast<size_t>(x + 1)] != 0);
      if (c[static_cast<size_t>(y)][static_cast<size_t>(x)] == 0 && leftFilled &&
          rightFilled) {
        ++depth;
        f.wells += depth;
      } else {
        depth = 0;
      }
    }
  }
  for (int y = 0; y < kBoardHeight; ++y) {
    for (int x = 0; x < kBoardWidth; ++x) {
      if (c[static_cast<size_t>(y)][static_cast<size_t>(x)] != 0) {
        f.maxHeight = std::max(f.maxHeight, double(kBoardHeight - y));
        break;
      }
    }
  }
  return f;
}

}  // namespace

double HeuristicOptionScore(const Game& g, const LegalOption& opt) {
  Game sim = g;
  Result r = sim.Step(opt.ops);
  if (r.done) return -1e9;

  Feats f = EvalBoard(sim.board());
  const double landingHeight = double(kBoardHeight - opt.landing_y);
  const double eroded = double(r.info.lines_cleared) * 4.0;
  double s = -4.500158825 * landingHeight + 3.418126810 * eroded -
             3.217888286 * f.rowTrans - 9.348695305 * f.colTrans -
             7.899265427 * f.holes - 3.385597225 * f.wells -
             0.30 * f.maxHeight;

  // One-step lookahead: if the next piece cannot spawn, this placement kills.
  Game next = sim;
  next.Step();
  if (next.Done()) s -= 1e8;
  return s;
}

int HeuristicBestOption(const Game& g,
                        const std::vector<LegalOption>& options) {
  if (options.empty()) return -1;
  int best = 0;
  double bestScore = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < options.size(); ++i) {
    const double s = HeuristicOptionScore(g, options[i]);
    if (s > bestScore) {
      bestScore = s;
      best = static_cast<int>(i);
    }
  }
  return best;
}

}  // namespace tetris
