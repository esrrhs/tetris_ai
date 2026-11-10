#include "server/session.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <span>
#include <vector>

#include "tetris/features.h"
#include "tetris/heuristic.h"
#include "tetris/shapes.h"

namespace tsrv {
namespace {

// Actor scatters logits into a dense [kMaxActions] row. Keep the option list
// within that bound; with a 10-wide board there are at most ~34 placements.
constexpr int kMaxOptions = 64;

Json cellsJson(const tetris::ActivePiece& p) {
  Json arr = Json::array();
  for (const auto& c : tetris::Cells(p.type, p.rot, p.x, p.y)) {
    Json pair = Json::array();
    pair.push(Json::integer(c.x));
    pair.push(Json::integer(c.y));
    arr.push(std::move(pair));
  }
  return arr;
}

Json boardJson(const tetris::Board& b) {
  Json rows = Json::array();
  for (int y = 0; y < tetris::kBoardHeight; ++y) {
    Json row = Json::array();
    for (int x = 0; x < tetris::kBoardWidth; ++x) {
      row.push(Json::integer(b.cells[static_cast<size_t>(y)][static_cast<size_t>(x)]));
    }
    rows.push(std::move(row));
  }
  return rows;
}

Json poseJson(const tetris::ActivePiece& p) {
  Json j = Json::object();
  j.set("type", Json::str(tetris::ToString(p.type)));
  j.set("typeId", Json::integer(static_cast<int>(p.type)));
  j.set("rot", Json::integer(p.rot));
  j.set("x", Json::integer(p.x));
  j.set("y", Json::integer(p.y));
  j.set("cells", cellsJson(p));
  return j;
}

// Replays the placement ops on a *copy* of the pre-step board to obtain the
// exact frame sequence the browser animates (rotate → shift → row-by-row drop).
Json buildAnimation(const tetris::Board& beforeBoard,
                    const tetris::ActivePiece& start,
                    const tetris::LegalOption& opt,
                    const tetris::EnvStepResult& result) {
  tetris::ActivePiece p = start;
  Json frames = Json::array();
  frames.push(poseJson(p));

  auto pushPose = [&]() { frames.push(poseJson(p)); };
  for (tetris::Op op : opt.ops) {
    switch (op) {
      case tetris::Op::Left:
      case tetris::Op::Right: {
        int dx = op == tetris::Op::Left ? -1 : 1;
        if (beforeBoard.Fits(p.type, p.rot, p.x + dx, p.y)) {
          p.x += dx;
          pushPose();
        }
        break;
      }
      case tetris::Op::RotateCW:
      case tetris::Op::RotateCCW: {
        if (p.type == tetris::PieceType::O) break;
        int dir = op == tetris::Op::RotateCW ? 1 : -1;
        int nrot = (p.rot + dir + 4) % 4;
        for (const auto& k : tetris::kKickOffsets) {
          if (beforeBoard.Fits(p.type, nrot, p.x + k.x, p.y + k.y)) {
            p.rot = nrot;
            p.x += k.x;
            p.y += k.y;
            pushPose();
            break;
          }
        }
        break;
      }
      case tetris::Op::HardDrop: {
        while (beforeBoard.Fits(p.type, p.rot, p.x, p.y + 1)) {
          ++p.y;
          pushPose();
        }
        break;
      }
    }
  }

  tetris::Board sim = beforeBoard;
  sim.Lock(p.type, p.rot, p.x, p.y);
  Json cleared = Json::array();
  for (int y = 0; y < tetris::kBoardHeight; ++y) {
    bool full = true;
    for (int x = 0; x < tetris::kBoardWidth; ++x) {
      if (sim.cells[static_cast<size_t>(y)][static_cast<size_t>(x)] == 0) {
        full = false;
        break;
      }
    }
    if (full) cleared.push(Json::integer(y));
  }

  Json a = Json::object();
  a.set("type", Json::str(tetris::ToString(p.type)));
  a.set("typeId", Json::integer(static_cast<int>(p.type)));
  a.set("rot", Json::integer(p.rot));
  a.set("x", Json::integer(p.x));
  a.set("fromY", Json::integer(start.y));
  a.set("toY", Json::integer(p.y));
  a.set("frames", std::move(frames));
  a.set("boardBefore", boardJson(beforeBoard));
  a.set("lockedCells", cellsJson(p));
  a.set("clearedRows", std::move(cleared));
  a.set("linesCleared", Json::integer(result.lines_cleared));
  a.set("scoreGain", Json::integer(result.score_gain));
  a.set("combo", Json::integer(result.combo));
  a.set("gameOver", Json::boolean(result.done));
  return a;
}

Decision teacherDecision(const tetris::Env& env,
                         const std::vector<tetris::LegalOption>& options) {
  Decision d;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::pair<double, int>> ranked;
  ranked.reserve(options.size());
  for (size_t i = 0; i < options.size(); ++i) {
    ranked.emplace_back(tetris::HeuristicOptionScore(env.game(), options[i]),
                        static_cast<int>(i));
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });
  d.source = "heuristic";
  d.optionId = ranked.empty() ? -1 : ranked[0].second;
  d.timeMs = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0)
                 .count();
  for (int k = 0; k < std::min<int>(6, static_cast<int>(ranked.size())); ++k) {
    const auto& o = options[static_cast<size_t>(ranked[static_cast<size_t>(k)].second)];
    Candidate c;
    c.rot = o.rot;
    c.x = o.x;
    c.landingY = o.landing_y;
    c.linesCleared = o.lines_cleared;
    c.score = ranked[static_cast<size_t>(k)].first;
    d.candidates.push_back(c);
  }
  return d;
}

}  // namespace

int64_t nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::vector<tetris::Op> parseOps(const Json& body) {
  std::vector<tetris::Op> ops;
  const Json* arr = body.find("ops");
  if (!arr || !arr->isArray()) {
    // No ops → gravity-only tick.
    return ops;
  }
  for (const auto& v : arr->items()) {
    const std::string s = v.asString();
    if (s == "Left") ops.push_back(tetris::Op::Left);
    else if (s == "Right") ops.push_back(tetris::Op::Right);
    else if (s == "RotateCW") ops.push_back(tetris::Op::RotateCW);
    else if (s == "RotateCCW") ops.push_back(tetris::Op::RotateCCW);
    else if (s == "HardDrop") ops.push_back(tetris::Op::HardDrop);
  }
  return ops;
}

void GameSession::reset(int64_t seed, const std::string& mode,
                        const ModelPolicy& model) {
  seed_ = seed;
  mode_ = (mode == "model" && model.available()) ? "model" : "heuristic";
  steps_ = 0;
  lastDecision_ = Json::null();
  lastAnimation_ = Json::null();
  env_.Reset(seed);
  touch();
}

Json GameSession::state(const std::string& sessionId, const ModelPolicy& model,
                        const std::string& error) const {
  const auto& g = env_.game();
  const auto& board = g.board();
  Json j = Json::object();
  j.set("sessionId", Json::str(sessionId));
  j.set("step", Json::integer(steps_));
  j.set("seed", Json::integer(seed_));
  j.set("mode", Json::str(mode_));
  j.set("backend", Json::str(mode_ == "model" ? "model" : "heuristic"));
  j.set("modelAvailable", Json::boolean(model.available()));
  j.set("width", Json::integer(tetris::kBoardWidth));
  j.set("height", Json::integer(tetris::kBoardHeight));
  j.set("board", boardJson(board));
  j.set("score", Json::integer(g.Score()));
  j.set("lines", Json::integer(g.Lines()));
  j.set("pieces", Json::integer(g.PiecesPlaced()));
  j.set("combo", Json::integer(g.Combo()));
  j.set("gameOver", Json::boolean(g.Done()));

  auto active = g.Active();
  if (active && !g.Done()) {
    j.set("active", poseJson(*active));
    tetris::ActivePiece ghost = *active;
    while (board.Fits(ghost.type, ghost.rot, ghost.x, ghost.y + 1)) ++ghost.y;
    Json gj = Json::object();
    gj.set("y", Json::integer(ghost.y));
    gj.set("cells", cellsJson(ghost));
    j.set("ghost", std::move(gj));
  } else {
    j.set("active", Json::null());
    j.set("ghost", Json::null());
  }

  Json next = Json::array();
  for (auto t : g.NextPieces(tetris::kNextPreview)) {
    next.push(Json::str(tetris::ToString(t)));
  }
  j.set("next", std::move(next));

  auto m = tetris::ComputeBoardMetrics(board);
  Json metrics = Json::object();
  Json heights = Json::array();
  float maxH = 0.f;
  for (int x = 0; x < tetris::kBoardWidth; ++x) {
    heights.push(Json::real(m.heights[static_cast<size_t>(x)]));
    maxH = std::max(maxH, m.heights[static_cast<size_t>(x)]);
  }
  metrics.set("heights", std::move(heights));
  metrics.set("holes", Json::real(m.holes));
  metrics.set("bumpiness", Json::real(m.bumpiness));
  metrics.set("aggregateHeight", Json::real(m.aggregate_height));
  metrics.set("maxHeight", Json::real(maxH));
  j.set("metrics", std::move(metrics));

  if (!lastDecision_.isNull()) j.set("decision", lastDecision_);
  if (!lastAnimation_.isNull()) j.set("animation", lastAnimation_);
  if (!error.empty()) j.set("error", Json::str(error));
  touch();
  return j;
}

Json GameSession::step(const std::string& sessionId, const ModelPolicy& model,
                       bool greedy) {
  if (env_.Done()) {
    return state(sessionId, model);
  }
  auto beforeBoard = env_.game().board();
  auto beforeActive = env_.game().Active();
  if (!beforeActive) {
    // Should not happen (Env always keeps an active piece), but stay safe.
    env_.mutable_game().Step();
    return state(sessionId, model);
  }

  auto options = env_.LegalOptions();
  if (options.size() > kMaxOptions) options.resize(kMaxOptions);
  if (options.empty()) {
    auto r = env_.Step(-1);  // hard-drop fallback
    (void)r;
    Decision d;
    d.source = "harddrop";
    d.optionId = -1;
    lastDecision_ = decisionToJson(d);
    lastAnimation_ = Json::null();
    ++steps_;
    return state(sessionId, model);
  }

  Decision d;
  int chosen = -1;
  if (mode_ == "model") {
    chosen = model.choose(env_, options, greedy, d);
  }
  if (chosen < 0) {
    d = teacherDecision(env_, options);
    chosen = d.optionId;
  }
  if (chosen < 0 || chosen >= static_cast<int>(options.size())) {
    chosen = 0;
    d.optionId = 0;
    d.source = "harddrop";
  }

  const tetris::LegalOption& opt = options[static_cast<size_t>(chosen)];
  d.rot = opt.rot;
  d.x = opt.x;
  d.landingY = opt.landing_y;
  d.linesCleared = opt.lines_cleared;

  auto r = env_.Step(opt);
  d.scoreGain = r.score_gain;
  lastDecision_ = decisionToJson(d);
  lastAnimation_ = buildAnimation(beforeBoard, *beforeActive, opt, r);
  ++steps_;
  return state(sessionId, model);
}

Json GameSession::tick(const std::string& sessionId, const ModelPolicy& model,
                       const std::vector<tetris::Op>& ops) {
  lastAnimation_ = Json::null();
  if (env_.Done()) {
    return state(sessionId, model, "游戏已结束，请新开一局");
  }
  auto& g = env_.mutable_game();
  g.Step(std::span<const tetris::Op>(ops.data(), ops.size()));
  // Match Env::Step: after a lock, the next empty step spawns the next piece.
  if (!g.Done() && !g.HasActivePiece()) g.Step();
  ++steps_;
  return state(sessionId, model);
}

}  // namespace tsrv
