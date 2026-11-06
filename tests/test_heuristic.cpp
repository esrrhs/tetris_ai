#include <gtest/gtest.h>

#include <array>

#include "tetris/env.h"
#include "tetris/features.h"
#include "tetris/heuristic.h"

using namespace tetris;

TEST(Heuristic, BestOptionIsLegal) {
  EnvRewardConfig cfg;
  cfg.denseReward = false;
  cfg.deathPenalty = 0.f;
  Env env(cfg);
  env.Reset(42);

  std::vector<LegalOption> opts = env.LegalOptions();
  ASSERT_FALSE(opts.empty());
  const int best = HeuristicBestOption(env.game(), opts);
  EXPECT_GE(best, 0);
  EXPECT_LT(best, static_cast<int>(opts.size()));
}

TEST(Heuristic, SurvivesLongGame) {
  // The teacher is the reference for "can this environment be played forever?".
  EnvRewardConfig cfg;
  cfg.denseReward = false;
  cfg.deathPenalty = 0.f;
  constexpr int kPieces = 3000;
  for (int s = 0; s < 3; ++s) {
    Env env(cfg);
    env.Reset(1000 + s);
    while (!env.Done() && env.game().PiecesPlaced() < kPieces) {
      std::vector<LegalOption> opts = env.LegalOptions();
      ASSERT_FALSE(opts.empty());
      const int best = HeuristicBestOption(env.game(), opts);
      ASSERT_GE(best, 0);
      env.Step(opts[static_cast<size_t>(best)]);
    }
    EXPECT_FALSE(env.Done()) << "teacher died on seed " << (1000 + s);
    EXPECT_EQ(env.game().PiecesPlaced(), kPieces);
  }
}

TEST(Features, CumulativeFeatureModes) {
  Game g(7);
  g.Step();
  std::array<float, kStateSize> raw{}, clamp{}, zero{};

  SetStateFeatureMode(StateFeatureMode::kRaw);
  EncodeState(g, raw, false);
  SetStateFeatureMode(StateFeatureMode::kClamp);
  EncodeState(g, clamp, false);
  SetStateFeatureMode(StateFeatureMode::kZero);
  EncodeState(g, zero, false);
  SetStateFeatureMode(StateFeatureMode::kRaw);  // restore global default

  // Dims 226/227 are lines/100 and score/10000.
  EXPECT_EQ(zero[226], 0.f);
  EXPECT_EQ(zero[227], 0.f);
  EXPECT_LE(clamp[226], 2.f);
  EXPECT_LE(clamp[227], 1.2f);
  for (int i = 0; i < kStateSize; ++i) {
    if (i == 226 || i == 227) continue;
    EXPECT_EQ(raw[static_cast<size_t>(i)], zero[static_cast<size_t>(i)])
        << "dim " << i;
  }
}
