// External evaluation on the built-in Env (no separate game server).
//
// Primary metric: score (and lines) after playing up to maxPieces locks
// (e.g. 1000 / 10000). If the agent dies earlier, stop and report score at death.
//
// Secondary fields:
//   death_piece = K + 1 when spawn fails after K locks; -1 if survived to cap.
//   survived / hit_cap = true iff pieces_placed reached maxPieces without Done.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tetris/env.h"
#include "torch_nn/models.h"

namespace algo {

struct EvalConfig {
  int maxPieces = 1000;     // hard cap on successful locks (horizon)
  bool greedy = true;       // true: argmax over legal logits; false: sample
  uint64_t seed = 1;        // Env Reset seed (also seeds stochastic RNG)
  tetris::EnvRewardConfig reward{};  // shaping unused for metrics; lines/score from Game
};

struct EvalResult {
  int64_t seed = 0;
  int score = 0;            // headline: Game::Score at end (death or cap)
  int lines = 0;            // headline: Game::Lines at end
  int pieces_placed = 0;    // successful locks (Game::PiecesPlaced)
  int death_piece = -1;     // K+1 on death; -1 if survived to maxPieces
  bool survived = false;    // true iff hit maxPieces without Done
  bool hit_cap = false;     // alias of survived
  bool done = false;        // Env/Game Done at end
  int decisions = 0;        // agent placement steps taken
};

// Play one game with actor on built-in Env. Cap enforced inside the loop:
// stop when Done OR pieces_placed >= maxPieces (after a Step).
EvalResult evaluateGame(torch_nn::Actor& actor, const EvalConfig& cfg);

// Machine-readable one-liner; leads with score= / lines=.
std::string formatEvalLine(const EvalResult& r);

}  // namespace algo
