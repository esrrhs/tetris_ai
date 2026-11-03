// Demo: runs a short step-based Tetris game with scripted / random ops.
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tetris/game.h"

int main(int argc, char** argv) {
  int64_t seed = 42;
  int steps = 40;
  if (argc > 1) {
    try {
      seed = std::stoll(argv[1]);
    } catch (...) {
    }
  }
  if (argc > 2) {
    try {
      steps = std::stoi(argv[2]);
    } catch (...) {
    }
  }

  // Match Go demo: ops RNG seeded with seed+1 (stdlib mt19937, not Go-identical).
  std::mt19937 rng(static_cast<uint32_t>(seed + 1));
  std::uniform_int_distribution<int> dist_n(0, 3);
  std::uniform_int_distribution<int> dist_op(0, 4);
  std::uniform_int_distribution<int> dist_hard(0, 7);

  tetris::Game g(seed);

  std::cout << "tetris_ai demo  seed=" << seed << "  maxSteps=" << steps
            << "  (7-bag)\n";
  std::cout << "Step semantics: spawn(if needed) → apply ops → gravity drop 1 "
               "(or lock on hard-drop)\n\n";

  for (int i = 0; i < steps; ++i) {
    std::vector<tetris::Op> ops;
    int n = dist_n(rng);
    for (int j = 0; j < n; ++j) {
      switch (dist_op(rng)) {
        case 0: ops.push_back(tetris::Op::Left); break;
        case 1: ops.push_back(tetris::Op::Right); break;
        case 2: ops.push_back(tetris::Op::RotateCW); break;
        case 3: ops.push_back(tetris::Op::RotateCCW); break;
        case 4:
          if (dist_hard(rng) == 0) ops.push_back(tetris::Op::HardDrop);
          break;
      }
    }

    tetris::Result r = g.Step(ops);
    std::cout << "=== step " << (i + 1) << " ops=[";
    for (size_t k = 0; k < ops.size(); ++k) {
      if (k) std::cout << ' ';
      std::cout << tetris::ToString(ops[k]);
    }
    std::cout << "] spawned=" << (r.info.spawned ? "true" : "false")
              << " locked=" << (r.info.locked ? "true" : "false")
              << " lines=" << r.info.lines_cleared << " reward=" << r.reward
              << " done=" << (r.done ? "true" : "false") << " ===\n";
    std::cout << g.Render(true) << '\n';
    if (r.done) {
      std::cout << "game over\n";
      break;
    }
  }
  return 0;
}
