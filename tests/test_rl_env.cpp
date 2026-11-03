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
  Env env;
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
  Env env;
  env.Reset(1);
  // Fill bottom row except under a placement path is hard; use board mutate +
  // ForcePiece on underlying game, then Step HardDrop via option sim.
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

TEST(Bag, PeekRemaining) {
  Bag7 bag(123);
  auto peek = bag.PeekRemaining();
  EXPECT_EQ(peek.size(), 7u);
  bag.Next();
  EXPECT_EQ(bag.PeekRemaining().size(), 6u);
}
