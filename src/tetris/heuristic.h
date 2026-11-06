// Classical board-evaluation controller (Dellacherie / El-Tetris style).
//
// Used as an expert teacher for imitation learning and as a sanity baseline:
// it plays indefinitely in this environment, while a PPO policy trained only
// on 500-step rollouts does not. Kept in the core (no Torch dependency) so it
// can be unit-tested and reused by any training/eval entry point.
#pragma once

#include <vector>

#include "tetris/features.h"
#include "tetris/game.h"

namespace tetris {

// Score one candidate placement on the post-lock board. Higher is better.
// Includes a one-step lookahead penalty if the following spawn would kill.
double HeuristicOptionScore(const Game& g, const LegalOption& opt);

// Index of the best option according to HeuristicOptionScore.
// Returns -1 for an empty option list.
int HeuristicBestOption(const Game& g,
                        const std::vector<LegalOption>& options);

}  // namespace tetris
