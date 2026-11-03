#include "tetris/env.h"

namespace tetris {

void Env::EnsureActivePiece() {
  // After a lock, the next empty Step spawns. If spawn collides → Done.
  while (!game_.Done() && !game_.HasActivePiece()) {
    game_.Step();
  }
}

EnvStepResult Env::MakeResult(double reward, int lines_cleared) const {
  EnvStepResult r;
  EncodeState(game_, r.state, /*include_bag=*/false);
  r.reward = reward;
  r.done = game_.Done();
  r.lines_cleared = lines_cleared;
  r.score = game_.Score();
  r.pieces_placed = game_.PiecesPlaced();
  return r;
}

EnvStepResult Env::Reset(int64_t seed) {
  game_ = Game(seed);
  EnsureActivePiece();
  return MakeResult(0.0, 0);
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
    return MakeResult(0.0, 0);
  }
  if (!game_.HasActivePiece()) {
    EnsureActivePiece();
    if (game_.Done()) return MakeResult(0.0, 0);
  }

  Result lock = game_.Step(opt.ops);
  double reward = lock.reward;
  int lines = lock.info.lines_cleared;

  // Auto-spawn so the agent always sees an active piece (or Done).
  if (!game_.Done() && !game_.HasActivePiece()) {
    Result spawn = game_.Step();
    // Spawn never clears lines; reward unchanged.
    (void)spawn;
  }
  return MakeResult(reward, lines);
}

EnvStepResult Env::Step(int placement_id) {
  auto opts = LegalOptions();
  if (placement_id < 0 || placement_id >= static_cast<int>(opts.size())) {
    // Invalid id: treat as no-op hard-drop of current pose if possible.
    if (!opts.empty()) {
      return Step(opts.front());
    }
    // No legal moves → force a hard drop tick then respawn path.
    Result r = game_.Step(Op::HardDrop);
    if (!game_.Done() && !game_.HasActivePiece()) game_.Step();
    return MakeResult(r.reward, r.info.lines_cleared);
  }
  return Step(opts[static_cast<size_t>(placement_id)]);
}

}  // namespace tetris
