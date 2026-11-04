// External evaluation: run a loaded actor on the built-in Env until death or
// a hard piece cap (no separate game server).
//
// death_piece convention (「第几个方块会死掉」):
//   After K successful locks, Env auto-spawns the next piece. If that spawn
//   collides → Done and Game::PiecesPlaced() == K.
//   death_piece = K + 1  (died attempting the (K+1)-th piece spawn).
//   If the agent reaches maxPieces successful locks without Done → survived=true,
//   death_piece = -1, pieces_placed = maxPieces (cap hit inside the play loop).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tetris/env.h"
#include "torch_nn/models.h"

namespace algo {

struct EvalConfig {
  int maxPieces = 1000;     // hard cap on successful locks
  bool greedy = true;       // true: argmax over legal logits; false: sample
  uint64_t seed = 1;        // Env Reset seed (also seeds stochastic RNG)
  tetris::EnvRewardConfig reward{};  // reward shaping unused for metrics; lines/score from Game
};

struct EvalResult {
  int64_t seed = 0;
  int pieces_placed = 0;  // successful locks (Game::PiecesPlaced)
  int death_piece = -1;   // K+1 on death; -1 if survived to maxPieces
  bool survived = false;  // true iff hit maxPieces without Done
  bool hit_cap = false;   // alias of survived (machine-readable)
  bool done = false;      // Env/Game Done at end
  int lines = 0;
  int score = 0;
  int decisions = 0;      // agent placement steps taken
};

// Play one game with actor on built-in Env. Cap enforced inside the loop:
// stop when Done OR pieces_placed >= maxPieces (after a Step).
EvalResult evaluateGame(torch_nn::Actor& actor, const EvalConfig& cfg);

// Format one game as machine-readable key=value tokens (one line).
std::string formatEvalLine(const EvalResult& r);

}  // namespace algo
