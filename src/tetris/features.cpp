#include "tetris/features.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>

#include "tetris/shapes.h"

namespace tetris {
namespace {

bool TryMoveSim(ActivePiece& p, const Board& b, int dx, int dy) {
  int nx = p.x + dx;
  int ny = p.y + dy;
  if (!b.Fits(p.type, p.rot, nx, ny)) return false;
  p.x = nx;
  p.y = ny;
  return true;
}

bool TryRotateSim(ActivePiece& p, const Board& b, int dir) {
  if (p.type == PieceType::O) return true;
  int nrot = (p.rot + dir + 4) % 4;
  for (const Point& k : kKickOffsets) {
    int nx = p.x + k.x;
    int ny = p.y + k.y;
    if (b.Fits(p.type, nrot, nx, ny)) {
      p.rot = nrot;
      p.x = nx;
      p.y = ny;
      return true;
    }
  }
  return false;
}

// Ops: rotate toward target_rot, translate to target_x, HardDrop.
// Returns empty if the path cannot reach (target_rot, target_x) at current y.
std::vector<Op> BuildPlacementOps(ActivePiece start, int target_rot, int target_x,
                                  const Board& board) {
  std::vector<Op> ops;
  ActivePiece p = start;

  int cw = (target_rot - p.rot + 4) % 4;
  int ccw = (4 - cw) % 4;
  if (cw <= ccw) {
    for (int i = 0; i < cw; ++i) {
      if (!TryRotateSim(p, board, 1)) return {};
      ops.push_back(Op::RotateCW);
    }
  } else {
    for (int i = 0; i < ccw; ++i) {
      if (!TryRotateSim(p, board, -1)) return {};
      ops.push_back(Op::RotateCCW);
    }
  }

  if (p.type != PieceType::O && p.rot != target_rot) return {};

  while (p.x < target_x) {
    if (!TryMoveSim(p, board, 1, 0)) return {};
    ops.push_back(Op::Right);
  }
  while (p.x > target_x) {
    if (!TryMoveSim(p, board, -1, 0)) return {};
    ops.push_back(Op::Left);
  }
  if (p.x != target_x) return {};

  ops.push_back(Op::HardDrop);
  return ops;
}

}  // namespace

BoardMetrics ComputeBoardMetrics(const Board& board) {
  BoardMetrics m;
  for (int x = 0; x < kBoardWidth; ++x) {
    int h = 0;
    for (int y = 0; y < kBoardHeight; ++y) {
      if (board.cells[static_cast<size_t>(y)][static_cast<size_t>(x)] != 0) {
        h = kBoardHeight - y;
        break;
      }
    }
    m.heights[static_cast<size_t>(x)] = static_cast<float>(h);
    m.aggregate_height += static_cast<float>(h);

    bool seen = false;
    for (int y = 0; y < kBoardHeight; ++y) {
      if (board.cells[static_cast<size_t>(y)][static_cast<size_t>(x)] != 0) {
        seen = true;
      } else if (seen) {
        m.holes += 1.f;
      }
    }
  }
  for (int x = 0; x + 1 < kBoardWidth; ++x) {
    m.bumpiness += std::fabs(m.heights[static_cast<size_t>(x)] -
                             m.heights[static_cast<size_t>(x + 1)]);
  }
  return m;
}

void EncodeActionFeature(const BoardMetrics& before, const BoardMetrics& after,
                         int lines_cleared, int landing_y,
                         std::array<float, kActionFeatSize>& out) {
  out.fill(0.f);
  int i = 0;
  for (int x = 0; x < kBoardWidth; ++x) {
    out[static_cast<size_t>(i++)] = after.heights[static_cast<size_t>(x)] /
                                    static_cast<float>(kBoardHeight);
  }
  out[static_cast<size_t>(i++)] = after.holes / 100.f;
  out[static_cast<size_t>(i++)] = after.bumpiness / 50.f;
  out[static_cast<size_t>(i++)] = after.aggregate_height / 200.f;
  out[static_cast<size_t>(i++)] = static_cast<float>(lines_cleared) / 4.f;
  out[static_cast<size_t>(i++)] = static_cast<float>(landing_y) /
                                  static_cast<float>(kBoardHeight);
  out[static_cast<size_t>(i++)] = (after.holes - before.holes) / 40.f;
  out[static_cast<size_t>(i++)] = (after.bumpiness - before.bumpiness) / 40.f;
  out[static_cast<size_t>(i++)] =
      (after.aggregate_height - before.aggregate_height) / 40.f;
}

void EncodeState(const Game& g, std::array<float, kStateSize>& out,
                 bool include_bag) {
  out.fill(0.f);
  int i = 0;
  const auto cells = g.BoardCells();
  for (int y = 0; y < kBoardHeight; ++y) {
    for (int x = 0; x < kBoardWidth; ++x) {
      out[static_cast<size_t>(i++)] =
          cells[static_cast<size_t>(y)][static_cast<size_t>(x)] != 0 ? 1.f : 0.f;
    }
  }

  auto obs = g.Observe();
  if (obs.has_piece) {
    int pt = static_cast<int>(obs.piece_type);
    if (pt >= 0 && pt < 7) out[static_cast<size_t>(i + pt)] = 1.f;
  }
  i += 7;
  if (obs.has_piece) {
    int r = obs.rot & 3;
    out[static_cast<size_t>(i + r)] = 1.f;
  }
  i += 4;
  if (obs.has_piece) {
    out[static_cast<size_t>(i++)] =
        static_cast<float>(obs.x) / static_cast<float>(kBoardWidth);
    out[static_cast<size_t>(i++)] =
        static_cast<float>(obs.y) / static_cast<float>(kBoardHeight);
  } else {
    i += 2;
  }

  BoardMetrics m = ComputeBoardMetrics(g.board());
  for (int x = 0; x < kBoardWidth; ++x) {
    out[static_cast<size_t>(i++)] =
        m.heights[static_cast<size_t>(x)] / static_cast<float>(kBoardHeight);
  }
  out[static_cast<size_t>(i++)] = m.holes / 100.f;
  out[static_cast<size_t>(i++)] = m.bumpiness / 50.f;
  out[static_cast<size_t>(i++)] = m.aggregate_height / 200.f;
  out[static_cast<size_t>(i++)] = static_cast<float>(g.Lines()) / 100.f;
  out[static_cast<size_t>(i++)] = static_cast<float>(g.Score()) / 10000.f;

  // Bag remaining counts (7) — imperfect actor leaves zeros.
  if (include_bag) {
    for (PieceType t : g.PeekBagRemaining()) {
      int idx = static_cast<int>(t);
      if (idx >= 0 && idx < 7) {
        out[static_cast<size_t>(i + idx)] += 1.f / 7.f;
      }
    }
  }
  i += 7;

  // Next-5 preview (5 × 7 one-hot) — public rule info shown in the UI panel.
  // Written for BOTH actor and critic paths (only the bag peek above is
  // critic-only), so the policy can plan across the upcoming pieces.
  std::vector<PieceType> nxt = g.NextPieces(kNextPreview);
  for (size_t slot = 0; slot < nxt.size() && slot < kNextPreview; ++slot) {
    int pt = static_cast<int>(nxt[slot]);
    if (pt >= 0 && pt < 7) {
      out[static_cast<size_t>(i + slot * 7 + pt)] = 1.f;
    }
  }
  i += kNextPreview * 7;  // end of state
  (void)i;
}

void LegalOptions(const Game& g, std::vector<LegalOption>& out) {
  out.clear();
  if (g.Done() || !g.HasActivePiece()) return;

  ActivePiece cur = *g.Active();
  BoardMetrics before = ComputeBoardMetrics(g.board());
  std::set<std::tuple<int, int, int>> seen;  // rot, lock_x, landing_y

  const int rot_limit = (cur.type == PieceType::O) ? 1 : 4;
  for (int rot = 0; rot < rot_limit; ++rot) {
    for (int tx = -2; tx <= kBoardWidth + 1; ++tx) {
      std::vector<Op> ops = BuildPlacementOps(cur, rot, tx, g.board());
      if (ops.empty()) continue;

      // Simulate landing without full Game copy when possible: drop from
      // post-rotate/translate pose.
      ActivePiece p = cur;
      bool ok = true;
      for (Op op : ops) {
        if (op == Op::HardDrop) break;
        if (op == Op::RotateCW) {
          if (!TryRotateSim(p, g.board(), 1)) {
            ok = false;
            break;
          }
        } else if (op == Op::RotateCCW) {
          if (!TryRotateSim(p, g.board(), -1)) {
            ok = false;
            break;
          }
        } else if (op == Op::Left) {
          if (!TryMoveSim(p, g.board(), -1, 0)) {
            ok = false;
            break;
          }
        } else if (op == Op::Right) {
          if (!TryMoveSim(p, g.board(), 1, 0)) {
            ok = false;
            break;
          }
        }
      }
      if (!ok) continue;
      if (p.x != tx) continue;
      if (cur.type != PieceType::O && p.rot != rot) continue;

      int ly = p.y;
      while (g.board().Fits(p.type, p.rot, p.x, ly + 1)) ++ly;
      if (!g.board().Fits(p.type, p.rot, p.x, ly)) continue;

      auto key = std::make_tuple(p.rot, p.x, ly);
      if (!seen.insert(key).second) continue;

      // Simulate lock + line clear for action features / lines_cleared.
      Board after = g.board();
      after.Lock(p.type, p.rot, p.x, ly);
      int cleared = after.ClearLines();
      BoardMetrics after_m = ComputeBoardMetrics(after);

      LegalOption opt;
      opt.id = static_cast<int>(out.size());
      opt.rot = p.rot;
      opt.x = p.x;
      opt.landing_y = ly;
      opt.lines_cleared = cleared;
      opt.ops = std::move(ops);
      EncodeActionFeature(before, after_m, cleared, ly, opt.feature);
      out.push_back(std::move(opt));
    }
  }
}

std::vector<LegalOption> LegalOptions(const Game& g) {
  std::vector<LegalOption> out;
  LegalOptions(g, out);
  return out;
}

}  // namespace tetris
