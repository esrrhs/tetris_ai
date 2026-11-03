#include <gtest/gtest.h>

#include "tetris/env.h"
#include "tetris/features.h"

using namespace tetris;

TEST(Features, StateDimConstant) {
  EXPECT_EQ(kStateSize, 235);
  EXPECT_EQ(kActionFeatSize, 18);
}

TEST(Features, EncodeStateSize) {
  Game g(1);
  g.Step();
  std::array<float, kStateSize> s{};
  EncodeState(g, s, false);
  float sum = 0;
  for (float v : s) sum += v;
  EXPECT_GT(sum, 0.f);
}

TEST(Env, ResetHasActivePieceAndLegalOptions) {
  Env env;
  auto r = env.Reset(42);
  EXPECT_FALSE(r.done);
  EXPECT_TRUE(env.game().HasActivePiece());
  auto opts = env.LegalOptions();
  EXPECT_FALSE(opts.empty());
  for (const auto& o : opts) {
    EXPECT_FALSE(o.ops.empty());
    EXPECT_EQ(o.ops.back(), Op::HardDrop);
  }
}

TEST(Env, StepLocksAndRespawns) {
  EnvRewardConfig cfg;
  cfg.denseReward = false;
  cfg.deathPenalty = 0.f;
  Env env(cfg);
  env.Reset(7);
  auto opts = env.LegalOptions();
  ASSERT_FALSE(opts.empty());
  auto r = env.Step(opts.front());
  EXPECT_GE(env.game().PiecesPlaced(), 1);
  if (!r.done) {
    EXPECT_TRUE(env.game().HasActivePiece());
  }
}

TEST(Env, LineReward) {
  EnvRewardConfig cfg;
  cfg.denseReward = false;
  cfg.deathPenalty = 0.f;
  cfg.lineCoef = 1.f;
  Env env(cfg);
  env.Reset(1);
  Game& g = env.mutable_game();
  for (int x = 0; x < kBoardWidth; ++x) {
    if (x == 4 || x == 5) continue;
    g.mutable_board().cells[kBoardHeight - 1][static_cast<size_t>(x)] = 1;
    g.mutable_board().cells[kBoardHeight - 2][static_cast<size_t>(x)] = 1;
  }
  g.ForcePiece(PieceType::O, 0, 4, kBoardHeight - 3);

  LegalOption opt;
  opt.ops = {Op::HardDrop};
  opt.id = 0;
  auto r = env.Step(opt);
  EXPECT_EQ(r.lines_cleared, 2);
  EXPECT_DOUBLE_EQ(r.reward, 2.0);
  if (!r.done) {
    EXPECT_TRUE(env.game().HasActivePiece());
  }
}

TEST(Env, DenseRewardHolesWorseThanLineClear) {
  // Potential φ = wH*aggH + wHole*holes + wBump*bump.
  // Placement that increases holes (0 lines) yields lower reward than a line clear.
  EnvRewardConfig cfg;
  cfg.denseReward = true;
  cfg.denseCoef = 0.01f;
  cfg.lineCoef = 1.0f;
  cfg.wH = 1.0f;
  cfg.wHole = 1.0f;
  cfg.wBump = 0.5f;
  cfg.deathPenalty = 0.f;

  // --- Scenario A: clear a nearly-full bottom row with a vertical I ---
  Env envClear(cfg);
  envClear.Reset(11);
  {
    Game& g = envClear.mutable_game();
    for (int x = 0; x < kBoardWidth - 1; ++x) {
      g.mutable_board().cells[kBoardHeight - 1][static_cast<size_t>(x)] = 1;
    }
    g.ForcePiece(PieceType::I, 1, 9, kBoardHeight - 4);
  }
  LegalOption clearOpt;
  clearOpt.ops = {Op::HardDrop};
  clearOpt.id = 0;
  BoardMetrics beforeClear = ComputeBoardMetrics(envClear.game().board());
  auto rClear = envClear.Step(clearOpt);
  BoardMetrics afterClear = ComputeBoardMetrics(envClear.game().board());
  EXPECT_GE(rClear.lines_cleared, 1);
  float phiClearBefore = BoardPotential(beforeClear, cfg);
  float phiClearAfter = BoardPotential(afterClear, cfg);
  double expectedClear =
      cfg.lineCoef * rClear.lines_cleared +
      cfg.denseCoef * (phiClearBefore - phiClearAfter);
  EXPECT_NEAR(rClear.reward, expectedClear, 1e-4);

  // --- Scenario B: horizontal I bridges two bottom pillars → holes, 0 lines ---
  Env envHole(cfg);
  envHole.Reset(12);
  {
    Game& g = envHole.mutable_game();
    // Pillars at x=0 and x=3 on the bottom row; I (rot0, 4-wide) lands on top
    // and covers x=1,2 → two holes, no line clear.
    g.mutable_board().cells[kBoardHeight - 1][0] = 1;
    g.mutable_board().cells[kBoardHeight - 1][3] = 1;
    g.ForcePiece(PieceType::I, 0, 0, kBoardHeight - 4);
  }
  LegalOption holeOpt;
  holeOpt.ops = {Op::HardDrop};
  holeOpt.id = 0;
  BoardMetrics beforeHole = ComputeBoardMetrics(envHole.game().board());
  auto rHole = envHole.Step(holeOpt);
  BoardMetrics afterHole = ComputeBoardMetrics(envHole.game().board());
  EXPECT_EQ(rHole.lines_cleared, 0);
  EXPECT_GT(afterHole.holes, beforeHole.holes);

  float phiHoleBefore = BoardPotential(beforeHole, cfg);
  float phiHoleAfter = BoardPotential(afterHole, cfg);
  double expectedHole = cfg.lineCoef * rHole.lines_cleared +
                        cfg.denseCoef * (phiHoleBefore - phiHoleAfter);
  EXPECT_NEAR(rHole.reward, expectedHole, 1e-4);

  EXPECT_LT(rHole.reward, rClear.reward);
}

TEST(Bag, PeekRemaining) {
  Bag7 bag(123);
  auto peek = bag.PeekRemaining();
  EXPECT_EQ(peek.size(), 7u);
  bag.Next();
  EXPECT_EQ(bag.PeekRemaining().size(), 6u);
}
