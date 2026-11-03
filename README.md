# tetris_ai

俄罗斯方块 **强化学习** 环境：按 `Step` 推进的可玩客户端核心。**C++ 为核心**（训练将基于此）；Go 包保留作参考实现。

Tetris **reinforcement-learning** play client: the clock advances only via `Step`. **C++ is the canonical core** for training (styled like [PerfectDouCPP](https://github.com/esrrhs/PerfectDouCPP) game layer). The Go package under `game/` remains as a secondary reference. Self-contained; does **not** depend on LibTorch yet, nor on `github.com/esrrhs/tetris`.

仓库对应专属开发 Bot：`tetris_ai`。

## Step 语义 / Step semantics

1. 游戏时间 **只** 在调用 `Step(ops...)` 时前进。
2. 方块生成使用 **7-bag**（每 7 个为一袋，袋内各出现一次，打乱后依次取出）。C++ 使用与 Go `math/rand` 兼容的 PRNG，**同一 seed 方块序列一致**。
3. **无** 活动方块时：本次 `Step` 先在顶部 **生成** 下一个方块；若生成位置与已锁定格子重叠 → `Done=true`（游戏结束）。否则：对新生方块应用 `ops`，再尝试 **下落 1 行**。
4. **有** 活动方块时：按顺序应用任意多个 `ops`，然后重力下落 **恰好 1 行**（若本次 ops 含 `HardDrop`，则一插到底并锁定，跳过普通重力）。
5. 操作只在 `Step` 内生效；非法移动（撞墙/重叠）**忽略**，不报错。
6. 方块无法再下落时 **锁定** 并消行；**下一次** `Step`（不是同一次）才生成下一个方块。

允许操作：`Left` / `Right` / `RotateCW` / `RotateCCW` /（可选）`HardDrop`。

## 盘面与旋转 / Board & rotation

- 标准 **10×20** 可见盘面，**无** 隐藏行（Y=0 为顶，向下增大）。
- 七种方块 I,L,J,T,O,S,Z；固定 4 态形状表 + **基础墙踢**（试偏移 `(0,0), (±1,0), (0,-1), (±2,0)`），非完整 SRS。
- 奖励 `Reward` = 本步消除行数；另有累计 `Score`（1/2/3/4 行 → 40/100/300/1200）。

## C++ API（主） / C++ API (primary)

```cpp
#include "tetris/game.h"

tetris::Game g(seed);  // 7-bag RNG seeded (Go-compatible)
tetris::Result r = g.Step({tetris::Op::Left, tetris::Op::RotateCW});
// r.obs     Observation (board + active piece)
// r.reward  double      lines cleared this step
// r.done    bool        game over
// r.info    locked / spawned / lines_cleared / score / ...

auto obs = g.Observe();
std::cout << g.Render(true);  // ASCII
```

### RL 循环示例 / RL loop sketch

```cpp
tetris::Game g(0);
while (!g.Done()) {
  auto obs = g.Observe();
  std::vector<tetris::Op> ops = policy(obs);  // 0..N ops for this tick
  tetris::Result r = g.Step(ops);
  // learn from r.reward, r.done, r.obs
}
```

### 构建与测试 / Build & test

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tetris_ai_demo          # seed=42, 40 steps
./build/tetris_ai_demo 123 60   # seed=123, 60 steps
```

CMake targets: static lib `tetris_ai_core`, demo `tetris_ai_demo`, gtest `test_game`. **No LibTorch** in this stage (training hooks later).

## 布局 / Layout

```
tetris_ai/
  CMakeLists.txt
  README.md
  src/tetris/     # C++ core: types, shapes, bag, board, game, render
  src/demo/       # ASCII demo
  tests/          # gtest
  game/           # Go reference (kept)
  cmd/demo/       # Go demo (kept)
  go.mod
```

## Go API（次） / Go API (secondary)

```go
import "github.com/esrrhs/tetris_ai/game"

g := game.NewGame(seed)
r := g.Step(game.OpLeft, game.OpRotateCW)
fmt.Print(g)
```

```bash
go test ./...
go run ./cmd/demo
```

## 设计说明 / Notes

- C++ 核心风格对齐 PerfectDouCPP 的 `src/<domain>/` + CMake + gtest；训练侧 Torch 依赖后续再加。
- 旋转系统刻意保持简单，便于 RL；若需完整 SRS / 20+2 缓冲行，可后续扩展。
