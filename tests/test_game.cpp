#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "tetris/bag.h"
#include "tetris/board.h"
#include "tetris/game.h"
#include "tetris/go_rand.h"
#include "tetris/shapes.h"
#include "tetris/types.h"

using namespace tetris;

TEST(Game, SpawnOnFirstStep) {
  Game g(1);
  EXPECT_FALSE(g.HasActivePiece());
  Result r = g.Step();
  EXPECT_TRUE(r.info.spawned);
  EXPECT_TRUE(r.obs.has_piece);
  EXPECT_FALSE(r.done);
  int spawn_y = 0;
  if (!r.info.locked) {
    EXPECT_EQ(r.obs.y, spawn_y + 1);
  }
}

TEST(Game, GravityOneRowPerStep) {
  Game g(42);
  Result r1 = g.Step();
  int y1 = r1.obs.y;
  Result r2 = g.Step();
  EXPECT_FALSE(r2.info.spawned);
  if (r2.info.locked) return;
  EXPECT_EQ(r2.obs.y, y1 + 1);
}

TEST(Game, MultiOpsInOneStep) {
  Game g(7);
  Result r = g.Step({Op::Left, Op::Left, Op::Right});
  EXPECT_TRUE(r.info.spawned);
  EXPECT_EQ(r.obs.x, 3);
}

TEST(Game, InvalidMovesIgnored) {
  Game g(3);
  g.Step();
  for (int i = 0; i < 20; ++i) {
    g.Step({Op::Left, Op::Left, Op::Left, Op::Left, Op::Left});
    if (g.Done()) break;
    auto a = g.Active();
    if (!a) break;
    EXPECT_GE(a->x, 0);
  }
}

TEST(Game, LockThenNextStepSpawns) {
  Game g(99);
  Result r = g.Step(Op::HardDrop);
  EXPECT_TRUE(r.info.locked);
  EXPECT_FALSE(g.HasActivePiece());
  Result r2 = g.Step();
  EXPECT_TRUE(r2.info.spawned);
}

TEST(Game, LineClear) {
  Game g(1);
  for (int x = 0; x < kBoardWidth; ++x) {
    g.mutable_board().cells[kBoardHeight - 1][static_cast<size_t>(x)] = 1;
  }
  EXPECT_EQ(g.mutable_board().ClearLines(), 1);

  g = Game(1);
  for (int x = 0; x < kBoardWidth; ++x) {
    if (x == 4 || x == 5) continue;
    g.mutable_board().cells[kBoardHeight - 1][static_cast<size_t>(x)] = 1;
    g.mutable_board().cells[kBoardHeight - 2][static_cast<size_t>(x)] = 1;
  }
  g.ForcePiece(PieceType::O, 0, 4, kBoardHeight - 3);
  Result r = g.Step(Op::HardDrop);
  EXPECT_GE(r.info.lines_cleared, 1);
  EXPECT_EQ(r.reward, static_cast<double>(r.info.lines_cleared));
}

TEST(Game, GameOverOnSpawnCollision) {
  Game g(5);
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < kBoardWidth; ++x)
      g.mutable_board().cells[static_cast<size_t>(y)][static_cast<size_t>(x)] = 1;
  Result r = g.Step();
  EXPECT_TRUE(r.done);
  EXPECT_FALSE(r.info.spawned);
}

TEST(Bag, SevenBagFairness) {
  Bag7 b(12345);
  std::map<PieceType, int> seen;
  for (int i = 0; i < 7; ++i) seen[b.Next()]++;
  EXPECT_EQ(seen.size(), 7u);
  for (auto& [_, c] : seen) EXPECT_EQ(c, 1);

  std::map<PieceType, int> seen2;
  for (int i = 0; i < 7; ++i) seen2[b.Next()]++;
  EXPECT_EQ(seen2.size(), 7u);
}

TEST(Bag, PeekNextMatchesDraws) {
  // PeekNext must return exactly the upcoming draws, including across bag
  // boundaries, and must not consume anything.
  Bag7 b(777);
  for (int draw = 0; draw < 40; ++draw) {
    std::vector<PieceType> peek = b.PeekNext(5);
    ASSERT_EQ(peek.size(), 5u);
    for (int k = 0; k < 5; ++k) {
      EXPECT_EQ(static_cast<int>(peek[static_cast<size_t>(k)]),
                static_cast<int>(b.Next()))
          << "draw=" << draw << " k=" << k;
    }
  }
}

TEST(Bag, PeekRemainingSemanticsPreserved) {
  Bag7 b(123);
  EXPECT_EQ(b.PeekRemaining().size(), 7u);
  for (int i = 6; i >= 1; --i) {
    b.Next();
    EXPECT_EQ(b.PeekRemaining().size(), static_cast<size_t>(i));
  }
  b.Next();  // bag exhausted
  EXPECT_TRUE(b.PeekRemaining().empty());
  b.Next();  // refills into the next bag; first piece of it drawn
  EXPECT_EQ(b.PeekRemaining().size(), 6u);
  // PeekRemaining must be a prefix of PeekNext.
  std::vector<PieceType> rem = b.PeekRemaining();
  std::vector<PieceType> nxt = b.PeekNext(7);
  ASSERT_GE(nxt.size(), rem.size());
  for (size_t i = 0; i < rem.size(); ++i) EXPECT_EQ(rem[i], nxt[i]);
}

TEST(Game, NextPiecesPredictsSpawns) {
  Game g(99);
  g.Step();  // spawn piece 0
  for (int i = 0; i < 30; ++i) {
    std::vector<PieceType> nxt = g.NextPieces(kNextPreview);
    ASSERT_EQ(nxt.size(), 5u);
    // Hard-drop until the next spawn; the spawned piece must be nxt[0].
    PieceType spawned = nxt[0];
    while (g.HasActivePiece() && !g.Done()) g.Step(Op::HardDrop);
    if (g.Done()) return;
    Result r = g.Step();  // spawn (may collide → done; piece still consumed)
    if (r.done) return;
    ASSERT_TRUE(r.info.spawned);
    EXPECT_EQ(r.info.piece_id, spawned);
  }
}

TEST(Game, RenderShowsNextPanel) {
  Game g(1);
  g.Step();
  std::string s = g.Render(true);
  EXPECT_NE(s.find("NEXT"), std::string::npos);
  EXPECT_NE(s.find("next="), std::string::npos);
  // Panel lists kNextPreview glyphs (one letter per upcoming piece).
  size_t pos = s.find("next=");
  ASSERT_NE(pos, std::string::npos);
  int letters = 0;
  for (size_t i = pos + 5; i < s.size() && s[i] != '\n'; ++i) {
    if (s[i] != ' ') ++letters;
  }
  EXPECT_EQ(letters, kNextPreview);
}

TEST(Game, RotateCW) {
  Game g(11);
  Result r = g.Step(Op::RotateCW);
  EXPECT_TRUE(r.obs.has_piece);
  if (r.obs.piece_type != PieceType::O) {
    EXPECT_EQ(r.obs.rot, 1);
  }
}

TEST(Game, StepAfterDoneIsNoOp) {
  Game g(5);
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < kBoardWidth; ++x)
      g.mutable_board().cells[static_cast<size_t>(y)][static_cast<size_t>(x)] = 1;
  g.Step();
  EXPECT_TRUE(g.Done());
  Result r = g.Step(Op::Left);
  EXPECT_TRUE(r.done);
}

TEST(Game, ManyStepsNoPanic) {
  Game g(2024);
  for (int i = 0; i < 500; ++i) {
    std::vector<Op> ops;
    switch (i % 5) {
      case 0: ops = {Op::Left, Op::RotateCW}; break;
      case 1: ops = {Op::Right, Op::Right}; break;
      case 2: ops = {Op::RotateCCW}; break;
      case 3: ops = {Op::HardDrop}; break;
      default: break;
    }
    Result r = g.Step(ops);
    if (r.done) return;
  }
}

TEST(Parity, GoRandInt63Seed42) {
  go_rand::Source src(42);
  const int64_t expect[] = {
      3440579354231278675LL, 608747136543856411LL, 5571782338101878760LL,
      1926012586526624009LL, 404153945743547657LL, 3534334367214237261LL,
      7497468244883513247LL, 3545887102062614208LL, 3532963341805492868LL,
      5961769557461764184LL,
  };
  for (int64_t e : expect) {
    EXPECT_EQ(src.Int63(), e);
  }
}

TEST(Parity, BagSequenceSeed42) {
  // Matches Go NewBag7(42) first 21 pieces (PieceType int values).
  Bag7 b(42);
  const int expect[] = {1, 6, 4, 5, 3, 0, 2, 4, 0, 3, 6, 1, 2, 5, 5, 4, 6, 1, 3, 0, 2};
  for (int e : expect) {
    EXPECT_EQ(static_cast<int>(b.Next()), e);
  }
}

TEST(Parity, GravityPathSeed42) {
  Game g(42);
  for (int i = 0; i < 15; ++i) {
    Result r = g.Step();
    ASSERT_TRUE(g.HasActivePiece());
    auto a = g.Active();
    EXPECT_EQ(static_cast<int>(a->type), 1);  // L
    EXPECT_EQ(a->rot, 0);
    EXPECT_EQ(a->x, 4);
    EXPECT_EQ(a->y, i + 1);
    EXPECT_EQ(r.info.spawned, i == 0);
    EXPECT_FALSE(r.info.locked);
  }
}

TEST(Board, LineScore) {
  EXPECT_EQ(LineScore(0), 0);
  EXPECT_EQ(LineScore(1), 40);
  EXPECT_EQ(LineScore(2), 100);
  EXPECT_EQ(LineScore(3), 300);
  EXPECT_EQ(LineScore(4), 1200);
}

TEST(Board, ComboBonus) {
  EXPECT_EQ(ComboBonus(0), 0);
  EXPECT_EQ(ComboBonus(-1), 0);
  EXPECT_EQ(ComboBonus(1), 40);
  EXPECT_EQ(ComboBonus(2), 80);
  EXPECT_EQ(ComboBonus(3), 120);
}

// Zero a few bottom rows, then fill all but x=4,5 so an O clears one line.
static void PrepSingleClearGap(Game& g) {
  for (int y = kBoardHeight - 4; y < kBoardHeight; ++y) {
    for (int x = 0; x < kBoardWidth; ++x) {
      g.mutable_board().cells[static_cast<size_t>(y)][static_cast<size_t>(x)] = 0;
    }
  }
  for (int x = 0; x < kBoardWidth; ++x) {
    if (x == 4 || x == 5) continue;
    g.mutable_board().cells[kBoardHeight - 1][static_cast<size_t>(x)] = 1;
  }
}

TEST(Game, ComboScoreAndReset) {
  Game g(1);
  // First clear: LineScore(1)=40, combo index 0 → no bonus; combo becomes 1.
  PrepSingleClearGap(g);
  g.ForcePiece(PieceType::O, 0, 4, kBoardHeight - 3);
  Result r1 = g.Step(Op::HardDrop);
  EXPECT_EQ(r1.info.lines_cleared, 1);
  EXPECT_EQ(r1.info.score_gain, 40);
  EXPECT_EQ(r1.info.combo, 1);
  EXPECT_EQ(g.Combo(), 1);
  EXPECT_EQ(g.Score(), 40);

  // Second consecutive clear (no intermediate non-clear lock):
  // +40 LineScore + ComboBonus(1)=40.
  PrepSingleClearGap(g);
  g.ForcePiece(PieceType::O, 0, 4, kBoardHeight - 3);
  Result r2 = g.Step(Op::HardDrop);
  EXPECT_EQ(r2.info.lines_cleared, 1);
  EXPECT_EQ(r2.info.score_gain, 80);  // 40 + 40
  EXPECT_EQ(r2.info.combo, 2);
  EXPECT_EQ(g.Score(), 120);

  // Lock with 0 lines → combo resets.
  for (int y = kBoardHeight - 4; y < kBoardHeight; ++y) {
    for (int x = 0; x < kBoardWidth; ++x) {
      g.mutable_board().cells[static_cast<size_t>(y)][static_cast<size_t>(x)] = 0;
    }
  }
  g.ForcePiece(PieceType::O, 0, 0, kBoardHeight - 3);
  Result r3 = g.Step(Op::HardDrop);
  EXPECT_EQ(r3.info.lines_cleared, 0);
  EXPECT_EQ(r3.info.score_gain, 0);
  EXPECT_EQ(r3.info.combo, 0);
  EXPECT_EQ(g.Combo(), 0);
  EXPECT_EQ(g.Score(), 120);

  // New streak first clear again: no combo bonus.
  PrepSingleClearGap(g);
  g.ForcePiece(PieceType::O, 0, 4, kBoardHeight - 3);
  Result r4 = g.Step(Op::HardDrop);
  EXPECT_EQ(r4.info.lines_cleared, 1);
  EXPECT_EQ(r4.info.score_gain, 40);
  EXPECT_EQ(r4.info.combo, 1);
  EXPECT_EQ(g.Score(), 160);
}

TEST(Game, DoubleClearScoresMoreThanSingle) {
  Game g(1);
  for (int x = 0; x < kBoardWidth; ++x) {
    if (x == 4 || x == 5) continue;
    g.mutable_board().cells[kBoardHeight - 1][static_cast<size_t>(x)] = 1;
    g.mutable_board().cells[kBoardHeight - 2][static_cast<size_t>(x)] = 1;
  }
  g.ForcePiece(PieceType::O, 0, 4, kBoardHeight - 3);
  Result r = g.Step(Op::HardDrop);
  EXPECT_EQ(r.info.lines_cleared, 2);
  EXPECT_EQ(r.info.score_gain, 100);  // LineScore(2), first clear no combo
  EXPECT_EQ(g.Score(), 100);
}
