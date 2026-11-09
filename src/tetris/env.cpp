#include "tetris/env.h"

namespace tetris {

void Env::EnsureActivePiece() {
  // After a lock, the next empty Step spawns. If spawn collides → Done.
  while (!game_.Done() && !game_.HasActivePiece()) {
    game_.Step();
  }
}

EnvStepResult Env::MakeResult(double reward, int lines_cleared, int score_gain,
                              int combo) const {
  EnvStepResult r;
  EncodeState(game_, r.state, /*include_bag=*/false);
  r.reward = reward;
  r.done = game_.Done();
  r.lines_cleared = lines_cleared;
  r.score_gain = score_gain;
  r.combo = combo;
  r.score = game_.Score();
  r.pieces_placed = game_.PiecesPlaced();
  return r;
}

double Env::ShapeReward(const BoardMetrics& before, const BoardMetrics& after,
                        int score_gain, bool done) const {
  double reward = 0.0;
  if (done) {
    // When done, survival is broken. Even if this step had line clears,
    // death should never be rewarded as a net positive action.
    double lineReward = static_cast<double>(reward_.lineCoef) *
                        (static_cast<double>(score_gain) / 40.0);
    reward = lineReward + static_cast<double>(reward_.deathPenalty);
    if (reward_.deathPenalty < 0.f && reward > static_cast<double>(reward_.deathPenalty)) {
      reward = static_cast<double>(reward_.deathPenalty);
    }
  } else {
    // Scale by /40 so a single clear (40 pts, no combo) ≈ lineCoef * 1.0.
    reward = static_cast<double>(reward_.lineCoef) *
             (static_cast<double>(score_gain) / 40.0);
    if (reward_.survivalBonus != 0.f) {
      reward += static_cast<double>(reward_.survivalBonus);
    }
  }

  if (reward_.denseReward) {
    float phiBefore = BoardPotential(before, reward_);
    float phiAfter = BoardPotential(after, reward_);
    reward += static_cast<double>(reward_.denseCoef) *
              static_cast<double>(phiBefore - reward_.gamma * phiAfter);
  }
  return reward;
}

EnvStepResult Env::Reset(int64_t seed) {
  game_ = Game(seed);
  EnsureActivePiece();
  return MakeResult(0.0, 0, 0, 0);
}

void Env::EncodeActorState(std::array<float, kStateSize>& out) const {
  EncodeState(game_, out, /*include_bag=*/false);
}

void Env::EncodeCriticState(std::array<float, kStateSize>& out) const {
  EncodeState(game_, out, /*include_bag=*/true);
}

void Env::LegalOptions(std::vector<LegalOption>& out) const {
  tetris::LegalOptions(game_, out);
}

std::vector<LegalOption> Env::LegalOptions() const {
  return tetris::LegalOptions(game_);
}

EnvStepResult Env::Step(const LegalOption& opt) {
  if (game_.Done()) {
    return MakeResult(0.0, 0, 0, 0);
  }
  if (!game_.HasActivePiece()) {
    EnsureActivePiece();
    if (game_.Done()) return MakeResult(0.0, 0, 0, 0);
  }

  BoardMetrics before = ComputeBoardMetrics(game_.board());

  Result lock = game_.Step(opt.ops);
  int lines = lock.info.lines_cleared;
  int score_gain = lock.info.score_gain;
  int combo = lock.info.combo;

  // Auto-spawn so the agent always sees an active piece (or Done).
  if (!game_.Done() && !game_.HasActivePiece()) {
    Result spawn = game_.Step();
    (void)spawn;
  }

  BoardMetrics after = ComputeBoardMetrics(game_.board());
  bool done = game_.Done();
  double reward = ShapeReward(before, after, score_gain, done);
  return MakeResult(reward, lines, score_gain, combo);
}

EnvStepResult Env::Step(int placement_id) {
  auto opts = LegalOptions();
  if (placement_id < 0 || placement_id >= static_cast<int>(opts.size())) {
    if (!opts.empty()) {
      return Step(opts.front());
    }
    BoardMetrics before = ComputeBoardMetrics(game_.board());
    Result r = game_.Step(Op::HardDrop);
    if (!game_.Done() && !game_.HasActivePiece()) game_.Step();
    BoardMetrics after = ComputeBoardMetrics(game_.board());
    bool done = game_.Done();
    double reward = ShapeReward(before, after, r.info.score_gain, done);
    return MakeResult(reward, r.info.lines_cleared, r.info.score_gain,
                      r.info.combo);
  }
  return Step(opts[static_cast<size_t>(placement_id)]);
}

}  // namespace tetris
