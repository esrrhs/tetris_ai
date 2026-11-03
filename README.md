# tetris_ai

俄罗斯方块 **强化学习** 环境：按 `Step` 推进的可玩客户端核心 + PerfectDouCPP 风格的单智能体 PPO 训练脚手架。**C++ 为核心**。

Tetris **reinforcement-learning** play client and PPO scaffold. The clock advances only via `Step`. **C++ is the canonical core**, styled like [PerfectDouCPP](https://github.com/esrrhs/PerfectDouCPP). The Go package under `game/` remains as a secondary reference.

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

### 构建核心与测试 / Build core & test（无需 LibTorch）

```bash
export PATH="/home/box/.local/cmake/bin:$PATH"
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tetris_ai_demo          # seed=42, 40 steps
./build/tetris_ai_demo 123 60   # seed=123, 60 steps
```

CMake targets (always): static lib `tetris_ai_core`, demo `tetris_ai_demo`, gtests `test_game` + `test_rl_env`.

## RL scaffold（PerfectDou 风格）

单智能体 self-play vs environment（v1 **无** 3-seat / league）。

### Action space — placement options

Agent 每步在当前活动方块上选择一个 **最终落点** `(rot, x)`：旋转/平移后 `HardDrop`，在 **一次** `Game::Step(ops)` 内执行。`LegalOptions()` 枚举所有可到达且能硬降锁定的选项（O 去重）。

`Env`（`src/tetris/env.*`）：

- `Reset(seed)`：`Step({})` 直到有活动方块，返回 observation。
- `Step(placement_id | LegalOption)`：应用 ops+HardDrop；锁定后若未结束，自动 `Step({})` **重生下一块**，保证 agent 始终看到带活动方块的状态（或 Done）。Reward = `lineCoef * lines` + optional potential dense shaping（见下）。
- `LegalOptions()`：当前状态合法落点。

### Feature dims

| Constant | Value | Contents |
|----------|------:|----------|
| `kStateSize` | **235** | board 20×10 binary (200) + piece one-hot(7) + rot one-hot(4) + x/y (2) + heights(10) + holes/bump/agg (3) + lines/score (2) + bag remaining counts(7) |
| `kActionFeatSize` | **18** | post-lock heights(10) + holes/bump/agg (3) + lines cleared + landing y + Δholes/Δbump/Δagg (3) |

- **Actor** 使用 imperfect state：bag 维全 0（不偷看 7-bag）。
- **Critic** 可带 `Bag7::PeekRemaining()` 计数（当前袋剩余，袋末为空直至下次 `Next` refill）。

### Models / Algo

- `Actor`: `joint = concat(state, actionFeat)` → MLP → scalar logit；对合法 option 做 masked softmax（见 `algo/ppo.cpp`）。
- `Critic`: state → MLP → value。
- `PPOLearner::update`: clip、entropy、value loss、grad clip、可选 KL early stop。
- `collectRollout`: 同步线程池收集 Env episode + GAE（`assignEpisodeReturns`）；可选 historical league eval games。

### Dense (potential) reward

Tetris analogue of PerfectDou’s oracle shaping — **potential-based shaping on board metrics**, not DouZero card-oracle:

```
φ = wH * aggregate_height + wHole * holes + wBump * bumpiness
reward = lineCoef * lines_cleared + denseCoef * (φ_before - φ_after)
       + (done ? deathPenalty : 0)
```

Defaults: `wH=1`, `wHole=1`, `wBump=0.5`, `lineCoef=1`, `denseCoef=0.01`, `deathPenalty=-1`, `denseReward=true`.

CLI: `--dense-reward` / `--no-dense-reward`, `--dense-coef`, `--line-coef`, `--death-penalty`, `--w-height`, `--w-hole`, `--w-bump`.

### Historical league pool (single-agent)

PerfectDou-style `HistoricalPool` with `recent` (ring) + `archive` (geometric / centennial promotion).

- Each update collects exactly `--games` **learning** games with the **current** CPU actor → PPO buffer.
- When the pool is non-empty, also plays `round(games * historicalProb)` **extra league-eval** games with a randomly sampled historical actor (50/50 archive vs recent). Those transitions are **not** written to the learning buffer; only `leagueGames` / `leagueMeanLines` / `leagueMeanScore` are logged.
- After each update (every `--pool-every`), push the current CPU actor into `recent`; periodically also copy into `archive`.
- Saved under `--out/pool/` as `recent_*.pt`, `archive_*.pt`, and `pool_meta.txt`. `--resume DIR` reloads actor/critic/optim + pool.

CLI: `--historical-prob`, `--pool-size`, `--archive-size`, `--pool-every`, `--resume`.

### Backend: CUDA / MPS / CPU

`--backend cuda|mps|cpu` is **explicit** (no silent auto-detect):

- `cuda`: requires `torch::cuda::is_available()`, else fatal.
- `mps`: requires Apple + MPS-enabled LibTorch (`torch::mps::is_available()`), else fatal.
- `cpu`: always available.

PPO learner + batch tensors live on the selected device. **Rollout inference always uses CPU actor/critic copies** (thread safety / no GPU contention). After each `learner.update`, trained weights are synced back to the CPU rollout modules.

This environment’s packaged LibTorch is **CPU-only**; `--backend cuda` needs a CUDA LibTorch build + GPU. The code path still compiles and selects correctly.

### 构建训练（LibTorch）

需要 LibTorch（如 2.5.1 cxx11 ABI）并设置 `CMAKE_PREFIX_PATH`：

```bash
export PATH="/home/box/.local/cmake/bin:$PATH"
cmake -S . -B build -DCMAKE_PREFIX_PATH=/workspace/deps/libtorch
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tetris_ai_train --updates 3 --games 8 --threads 2 \
  --pool-every 1 --pool-size 4 --archive-size 2 --historical-prob 0.5 \
  --dense-reward --out /tmp/tetris_ckpt2 --backend cpu
```

CLI (main): `--updates` `--games` `--threads` `--hidden` `--lr` `--epochs` `--mb` `--gamma` `--lambda` `--seed` `--out` `--backend` `--resume` plus dense-reward and league-pool flags above.

CMake：`TETRIS_AI_WITH_TORCH`（找到 Torch 时 ON）构建 `tetris_ai_rl`、`tetris_ai_train`、可选 `test_rl`。未找到 Torch 时核心与 `test_game` / `test_rl_env` 仍可构建。

## 布局 / Layout

```
tetris_ai/
  CMakeLists.txt
  README.md
  src/tetris/     # core + features + env
  src/algo/       # PPO GAE / rollout (Torch)
  src/torch_nn/   # Actor Critic PPOLearner tensor_bridge
  src/train.cpp   # training entry
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

- C++ 核心风格对齐 PerfectDouCPP 的 `src/<domain>/` + `algo/` + `torch_nn/` + CMake + gtest。
- 旋转系统刻意保持简单，便于 RL；若需完整 SRS / 20+2 缓冲行，可后续扩展。
