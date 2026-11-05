#include <gtest/gtest.h>

#include "tetris/env.h"
#include "tetris/features.h"

using namespace tetris;

TEST(Features, StateDimConstant) {
  EXPECT_EQ(kNextPreview, 5);
  EXPECT_EQ(kStateSize, 270);
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

TEST(Features, NextFiveVisibleToActorBagHidden) {
  Game g(42);
  g.Step();
  std::array<float, kStateSize> actor{}, critic{};
  EncodeState(g, actor, /*include_bag=*/false);
  EncodeState(g, critic, /*include_bag=*/true);

  // Bag block (dims 228..234): zeros for actor, non-zero for critic early game.
  float actorBag = 0, criticBag = 0;
  for (int i = 228; i < 235; ++i) {
    actorBag += actor[static_cast<size_t>(i)];
    criticBag += critic[static_cast<size_t>(i)];
  }
  EXPECT_EQ(actorBag, 0.f);
  EXPECT_GT(criticBag, 0.f);

  // Next-5 block (dims 235..269): exactly 5 one-hot entries for both paths.
  auto checkNext = [&](const std::array<float, kStateSize>& s) {
    float sum = 0;
    int hot = 0;
    for (int i = 235; i < 270; ++i) {
      sum += s[static_cast<size_t>(i)];
      if (s[static_cast<size_t>(i)] == 1.f) ++hot;
    }
    EXPECT_FLOAT_EQ(sum, 5.f);
    EXPECT_EQ(hot, 5);
  };
  checkNext(actor);
  checkNext(critic);

  // Encoded one-hots match Game::NextPieces order.
  std::vector<PieceType> nxt = g.NextPieces(kNextPreview);
  ASSERT_EQ(nxt.size(), 5u);
  for (int slot = 0; slot < 5; ++slot) {
    int pt = static_cast<int>(nxt[static_cast<size_t>(slot)]);
    for (int j = 0; j < 7; ++j) {
      float want = (j == pt) ? 1.f : 0.f;
      EXPECT_EQ(actor[static_cast<size_t>(235 + slot * 7 + j)], want);
    }
  }
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
  EXPECT_EQ(r.score_gain, 100);  // LineScore(2), first clear
  // lineCoef * score_gain / 40 = 100/40 = 2.5
  EXPECT_DOUBLE_EQ(r.reward, 2.5);
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
      cfg.lineCoef * (rClear.score_gain / 40.0) +
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
  double expectedHole = cfg.lineCoef * (rHole.score_gain / 40.0) +
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

TEST(Env, SpawnDeathPiecesPlacedConvention) {
  // After K locks, a failed auto-spawn → Done with PiecesPlaced()==K.
  // External eval reports death_piece = K + 1 (died on the (K+1)-th spawn).
  EnvRewardConfig cfg;
  cfg.denseReward = false;
  cfg.deathPenalty = 0.f;
  Env env(cfg);
  env.Reset(1);
  auto opts = env.LegalOptions();
  ASSERT_FALSE(opts.empty());
  env.Step(opts.front());
  ASSERT_FALSE(env.Done());
  const int k_after_first = env.game().PiecesPlaced();
  EXPECT_GE(k_after_first, 1);

  Game& g = env.mutable_game();
  // Block spawn region (all pieces spawn at (4,0)) without completing lines:
  // fill x=1..9 on rows 0..3, leave x=0 empty so ClearLines will not wipe them.
  for (int y = 0; y < 4; ++y) {
    for (int x = 1; x < kBoardWidth; ++x) {
      g.mutable_board().cells[static_cast<size_t>(y)][static_cast<size_t>(x)] = 1;
    }
  }
  // Lock one more piece in the empty column so Env auto-spawns → collision.
  g.ForcePiece(PieceType::O, 0, 0, kBoardHeight - 3);
  LegalOption opt;
  opt.ops = {Op::HardDrop};
  opt.id = 0;
  auto r = env.Step(opt);
  EXPECT_TRUE(r.done);
  EXPECT_TRUE(env.Done());
  const int k = env.game().PiecesPlaced();
  EXPECT_EQ(k, k_after_first + 1);
  const int death_piece = k + 1;  // convention used by tetris_ai_eval
  EXPECT_EQ(death_piece, k + 1);
  EXPECT_GT(death_piece, k);
}

TEST(Env, ComboStreakBeatsIsolatedSingles) {
  // Two consecutive single clears (combo) should reward more than one isolated
  // single of the same line count, encouraging "刀口舔血" stacking.
  EnvRewardConfig cfg;
  cfg.denseReward = false;
  cfg.deathPenalty = 0.f;
  cfg.lineCoef = 1.f;

  auto prep_gap = [](Game& g) {
    for (int y = kBoardHeight - 4; y < kBoardHeight; ++y) {
      for (int x = 0; x < kBoardWidth; ++x) {
        g.mutable_board().cells[static_cast<size_t>(y)][static_cast<size_t>(x)] = 0;
      }
    }
    for (int x = 0; x < kBoardWidth; ++x) {
      if (x == 4 || x == 5) continue;
      g.mutable_board().cells[kBoardHeight - 1][static_cast<size_t>(x)] = 1;
    }
  };

  Env env(cfg);
  env.Reset(1);
  Game& g = env.mutable_game();

  prep_gap(g);
  g.ForcePiece(PieceType::O, 0, 4, kBoardHeight - 3);
  LegalOption opt;
  opt.ops = {Op::HardDrop};
  opt.id = 0;
  auto r1 = env.Step(opt);
  EXPECT_EQ(r1.lines_cleared, 1);
  EXPECT_DOUBLE_EQ(r1.reward, 1.0);  // 40/40

  prep_gap(g);
  g.ForcePiece(PieceType::O, 0, 4, kBoardHeight - 3);
  auto r2 = env.Step(opt);
  EXPECT_EQ(r2.lines_cleared, 1);
  EXPECT_EQ(r2.score_gain, 80);       // 40 + ComboBonus(1)
  EXPECT_EQ(r2.combo, 2);
  EXPECT_DOUBLE_EQ(r2.reward, 2.0);   // 80/40

  // Combo follow-up single > isolated single of 1 line.
  EXPECT_GT(r2.reward, r1.reward);
  // Two-line clear in one lock (100) still beats a lone single (40).
  EXPECT_GT(100.0 / 40.0, r1.reward);
}
