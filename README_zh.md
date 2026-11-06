# tetris_ai

[![CI](https://github.com/esrrhs/tetris_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/esrrhs/tetris_ai/actions/workflows/ci.yml)
[![Language](https://img.shields.io/badge/language-C%2B%2B20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](https://github.com/esrrhs/tetris_ai)

[**English**](README.md) | [**中文说明**](README_zh.md)

俄罗斯方块 **强化学习** 环境：按 `Step` 推进的可玩客户端核心 + PerfectDouCPP 风格的单智能体 PPO 训练脚手架。**C++ 为唯一核心**，布局对齐 [PerfectDouCPP](https://github.com/esrrhs/PerfectDouCPP)。


## Step 语义

1. 游戏时间 **只** 在调用 `Step(ops...)` 时前进。
2. 方块生成使用 **7-bag**（每 7 个为一袋，袋内各出现一次，打乱后依次取出）。使用确定性种子 PRNG，**同一 seed 方块序列一致**。
3. **下 5 块预览规则**：后续 **5 个方块**（可跨袋边界）为公开信息——在 `Render()` ASCII 侧栏（`NEXT`）与状态特征中均可见。预览不消耗 RNG、不影响出块序列。
4. **无** 活动方块时：本次 `Step` 先在顶部 **生成** 下一个方块；若生成位置与已锁定格子重叠 → `Done=true`（游戏结束）。否则：对新生方块应用 `ops`，再尝试 **下落 1 行**。
5. **有** 活动方块时：按顺序应用任意多个 `ops`，然后重力下落 **恰好 1 行**（若本次 ops 含 `HardDrop`，则一插到底并锁定，跳过普通重力）。
6. 操作只在 `Step` 内生效；非法移动（撞墙/重叠）**忽略**，不报错。
7. 方块无法再下落时 **锁定** 并消行；**下一次** `Step`（不是同一次）才生成下一个方块。

允许操作：`Left` / `Right` / `RotateCW` / `RotateCCW` /（可选）`HardDrop`。

## 盘面与旋转

- 标准 **10×20** 可见盘面，**无** 隐藏行（Y=0 为顶，向下增大）。
- 七种方块 I,L,J,T,O,S,Z；固定 4 态形状表 + **基础墙踢**（试偏移 `(0,0), (±1,0), (0,-1), (±2,0)`），非完整 SRS。
- 底层 `Game::Step` 的 `Reward` = 本步消除行数（原始 API）；累计 **`Score`** 与真实计分一致：
  - 消行基础分 `LineScore`：1/2/3/4 行 → **40 / 100 / 300 / 1200**
  - **连消加成** `ComboBonus`：连续「有消行的锁定」计 streak。第 1 次消行 combo index=0（无加成）；之后每次连续消行 index=1,2,…，加成 **40 × index**（即 +40, +80, …）。无消行的锁定将 combo 清零。
  - 本锁得分增量 = `LineScore(cleared) + ComboBonus(prev_combo)`。
- RL `Env` 的消行奖励跟上述得分增量对齐（鼓励「刀口舔血」堆连消 / 多消）：`lineCoef * (score_gain / 40)`，使单消 ≈ `lineCoef * 1.0`；dense shaping 与 `deathPenalty` 仍叠加上去。

## C++ API

```cpp
#include "tetris/game.h"

tetris::Game g(seed);  // 7-bag RNG seeded
tetris::Result r = g.Step({tetris::Op::Left, tetris::Op::RotateCW});
// r.obs     Observation (board + active piece)
// r.reward  double      lines cleared this step (raw Game API)
// r.done    bool        game over
// r.info    locked / spawned / lines_cleared / score / score_gain / combo / ...

auto obs = g.Observe();
std::cout << g.Render(true);  // ASCII
```

### 构建核心与测试（无需 LibTorch）

```bash
export PATH="/home/box/.local/cmake/bin:$PATH"
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tetris_ai_demo          # seed=42, 40 steps
./build/tetris_ai_demo 123 60   # seed=123, 60 steps
```

CMake 目标（始终构建）：静态库 `tetris_ai_core`、demo `tetris_ai_demo`、gtest `test_game` + `test_rl_env`。

## RL 脚手架（PerfectDou 风格）

单智能体 self-play vs environment（v1 **无** 3-seat / 多智能体联赛）。

### 动作空间 — 落点选项

Agent 每步在当前活动方块上选择一个 **最终落点** `(rot, x)`：旋转/平移后 `HardDrop`，在 **一次** `Game::Step(ops)` 内执行。`LegalOptions()` 枚举所有可到达且能硬降锁定的选项（O 去重）。

`Env`（`src/tetris/env.*`）：

- `Reset(seed)`：`Step({})` 直到有活动方块，返回 observation。
- `Step(placement_id | LegalOption)`：应用 ops+HardDrop；锁定后若未结束，自动 `Step({})` **重生下一块**，保证 agent 始终看到带活动方块的状态（或 Done）。Reward = `lineCoef * (score_gain/40)` + 可选 potential dense shaping（见下）。
- `LegalOptions()`：当前状态合法落点。

### 特征维度

| 常量 | 值 | 内容 |
|------|---:|------|
| `kStateSize` | **270** | board 20×10 二值 (200) + piece one-hot(7) + rot one-hot(4) + x/y (2) + heights(10) + holes/bump/agg (3) + lines/score (2) + bag remaining counts(7) + 下5块预览 one-hot (5×7=35) |
| `kActionFeatSize` | **18** | 锁定后 heights(10) + holes/bump/agg (3) + lines cleared + landing y + Δholes/Δbump/Δagg (3) |

- **Actor** 使用 imperfect state：bag 维全 0（不偷看 7-bag）；**下 5 块预览维度对 actor 可见**（公开规则信息）。
- **Critic** 在同样的下 5 块预览之外，还可带 `Bag7::PeekRemaining()` 计数（当前袋剩余，袋末为空直至下次 `Next` refill）。
- 下 5 块规则引入前训练的旧检查点（状态维度 235）与 270 维新模型**不兼容**，需重新训练/评测。

**累计特征与长局分布漂移**：`lines/100`（维度 226）与 `score/10000`（维度 227）是无上界累计量。500 步训练局里最大约 `2 / 1.2`，而 10000 块长局会到 `40 / 23`，策略进入分布外。可用 `--state-features raw|clamp|zero` 控制：

| 模式 | 行为 | 用途 |
|---|---|---|
| `raw` | 原样、无上界（默认） | 兼容旧检查点 |
| `clamp` | 裁剪到训练范围 `2.0 / 1.2` | 诊断 / 平滑迁移 |
| `zero` | 两维恒为 0，状态变平稳 | **长局训练推荐** |

`train` / `eval` / `pretrain` 三个入口都接受该参数，**评测时必须与训练时一致**。默认 `raw` 保证旧检查点结果不变。

### 模型 / 算法

- `Actor`: `joint = concat(state, actionFeat)` → MLP → scalar logit；对合法 option 做 masked softmax（见 `algo/ppo.cpp`）。
- `Critic`: state → MLP → value。
- `PPOLearner::update`: clip、entropy、value loss、grad clip、可选 KL early stop。
- `collectRollout`: 同步线程池收集 Env episode + GAE（`assignEpisodeReturns`）；可选 historical league eval games。

### Dense（potential）奖励

对应 PerfectDou 的 oracle shaping —— **基于盘面指标的 potential shaping**，不是 DouZero 牌局 oracle：

```
φ = wH * aggregate_height + wHole * holes + wBump * bumpiness
score_gain = LineScore(lines) + ComboBonus(combo_index)   # 与 Game 得分一致
reward = lineCoef * (score_gain / 40.0) + denseCoef * (φ_before - φ_after)
       + (done ? deathPenalty : survivalBonus)
```

`survivalBonus`（`--survival-bonus`，默认 0）是每存活一个决策的奖励；配合较长的折扣视野与较大的 `--death-penalty`，让"不死"成为显式目标而非消行之间的奖励空白。

示例（`lineCoef=1`，无 dense）：单消=1.0，双消=2.5，三消=7.5，四消=30.0；连消后续单消=2.0 (40+40)/40。

默认：`wH=1`，`wHole=1`，`wBump=0.5`，`lineCoef=1`，`denseCoef=0.01`，`deathPenalty=-1`，`denseReward=true`。

CLI：`--dense-reward` / `--no-dense-reward`，`--dense-coef`，`--line-coef`，`--death-penalty`，`--survival-bonus`，`--w-height`，`--w-hole`，`--w-bump`。

### 长局训练（避免 500 步截断）

`collectRollout` 默认 `maxStepsPerGame = 500`。只优化到 500 步的策略在 10000 块长局里会大比例死亡（见下）。相关改动：

- `--max-steps N`：提高每局训练上限；建议配合 `--games` 反向缩小以维持相近的 transition 预算（例如 `500×64` → `2000×16` → `8000×4`）。
- **截断 bootstrap**：被步数上限截断（而非真正死亡）的局，GAE 末端现在用 `V(s_T)` 引导；旧实现默认末端价值为 0，会把"接近上限"压成低价值，反而鼓励早点结束。
- `gameId` 现在按 worker/game 唯一分配，不再有溢出与碰撞风险。
- `--gamma`、`--ent`、`--target-kl` 可直接用于延长视野与保熵。

### 模仿学习教师（`tetris_ai_pretrain`）

`src/tetris/heuristic.*` 提供一个 Dellacherie / El-Tetris 风格的经典盘面评估控制器（含下一步生成失败的一步前瞻）。在本环境下它能稳定打到 50000 块且零死亡，因此作为专家教师用于 **DAgger 行为克隆**：

- 每轮在 **student 自己会访问到的盘面**上采样（行为策略按 `--student-prob-start` → `--student-prob-end` 从教师逐渐切到学生），标签始终是教师的最优落点；
- 用交叉熵训练与 PPO 完全相同的 `Actor`（合法落点 masked softmax），输出 `actor.pt` / `critic.pt` / `meta.txt`，可直接评估或作为 `--resume` 做 PPO 微调；
- 每轮打印 `agree`（学生贪心与教师一致率）、`ce`、采样死亡数、贪心自评存活数，并保存检查点。

```bash
./build/tetris_ai_pretrain --iters 20 --games 32 --steps 2000 --threads 8 \
  --hidden 256 --lr 5e-4 --backend cuda --state-features zero \
  --out ckpt_bc
./build/tetris_ai_eval --ckpt ckpt_bc --hidden 256 --state-features zero \
  --max-pieces 10000 --games 20 --seed 4001 --backend cpu
# 可选：从模仿模型出发做长局 PPO 微调
./build/tetris_ai_train --resume ckpt_bc --hidden 256 --max-steps 4000 \
  --games 8 --state-features zero --survival-bonus 0.01 \
  --death-penalty -20 --gamma 0.997 --ent 0.02 --backend cuda --out ckpt_bc_ppo
```

CLI：`--iters` `--games` `--steps` `--threads` `--hidden` `--epochs` `--mb` `--lr` `--seed` `--backend` `--state-features` `--student-prob-start` / `--student-prob-end` `--eval-pieces` `--eval-games` `--out` `--resume`。

### 历史联赛池（单智能体）

PerfectDou 风格的 `HistoricalPool`：`recent`（环形）+ `archive`（几何 / 百次晋升）。

- 每次 update 用 **当前** CPU actor 收集恰好 `--games` 局 **学习** 对局 → PPO buffer。
- 池非空时，额外打 `round(games * historicalProb)` 局 **联赛评估**，随机采样历史 actor（archive / recent 各 50%）。这些 transition **不**写入学习 buffer；只记录 `leagueGames` / `leagueMeanLines` / `leagueMeanScore`。
- 每次 update 后（每 `--pool-every`），把当前 CPU actor 推入 `recent`；周期性再拷入 `archive`。
- 保存在 `--out/pool/`：`recent_*.pt`、`archive_*.pt`、`pool_meta.txt`。`--resume DIR` 可重载 actor/critic/optim + pool。

CLI：`--historical-prob`，`--pool-size`，`--archive-size`，`--pool-every`，`--resume`。

### Backend：CUDA / MPS / CPU

`--backend cuda|mps|cpu` 为 **显式** 选择（无静默自动探测）：

- `cuda`：需要 `torch::cuda::is_available()`，否则 fatal。
- `mps`：需要 Apple + 启用 MPS 的 LibTorch（`torch::mps::is_available()`），否则 fatal。
- `cpu`：始终可用。

PPO learner + batch 张量落在所选设备上。**Rollout 推理始终使用 CPU actor/critic 副本**（线程安全 / 避免 GPU 争用）。每次 `learner.update` 后，训练权重同步回 CPU rollout 模块。

本环境打包的 LibTorch 为 **CPU-only**；`--backend cuda` 需要 CUDA 版 LibTorch + GPU。代码路径仍可正确编译与选择。

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

CLI（主要）：`--updates` `--games` `--threads` `--hidden` `--lr` `--epochs` `--mb` `--gamma` `--lambda` `--seed` `--out` `--backend` `--resume`，以及上面的 dense-reward / league-pool 参数。

CMake：`TETRIS_AI_WITH_TORCH`（找到 Torch 时 ON）构建 `tetris_ai_rl`、`tetris_ai_train`、`tetris_ai_eval`、`tetris_ai_pretrain`、可选 `test_rl`。未找到 Torch 时核心与 `test_game` / `test_rl_env` / `test_heuristic` 仍可构建。

### 外部评估（自带 Env 客户端）

**目的：** 看最多 **N** 个积木的 **分数**（以及消行数）；**死了就提前结束**。加载 `actor.pt`，在内置 `Env`/`Game` 上贪心（默认）对局，不依赖外部游戏服务。典型 N：`1000` / `10000`。

- 打到 `pieces_placed >= N` 且未死 → 报告该局 `score` / `lines`，`survived=true`。
- 中途死亡 → 提前停，报告死亡时的 `score` / `lines` / `pieces_placed`，`survived=false`，`death_piece=K+1`（锁定 K 块后第 K+1 块生成失败）。

```bash
export PATH="/home/box/.local/cmake/bin:$PATH"
cmake -S . -B build -DCMAKE_PREFIX_PATH=/workspace/deps/libtorch
cmake --build build -j --target tetris_ai_eval
./build/tetris_ai_eval --ckpt /tmp/tetris_ckpt2 --max-pieces 1000 --games 5 --seed 1 --backend cpu
./build/tetris_ai_eval --ckpt /tmp/tetris_ckpt2 --max-pieces 10000 --games 3 --seed 1 --backend cpu
# optional: --stochastic  --hidden 256  --state-features zero
```

每局一行（分数在前）：`score=... lines=... pieces_placed=... seed=... survived=... death_piece=...`  
末尾 `summary mean_score=... mean_lines=... mean_pieces=...`。

CMake（需 LibTorch）：额外目标 `tetris_ai_eval`。

## 布局

```
tetris_ai/
  CMakeLists.txt
  README.md
  README_zh.md
  src/tetris/     # core + features + env + heuristic teacher
  src/algo/       # PPO GAE / rollout (Torch)
  src/torch_nn/   # Actor Critic PPOLearner tensor_bridge
  src/train.cpp   # PPO training entry
  src/pretrain.cpp# DAgger imitation entry
  src/eval.cpp    # external eval entry
  src/demo/       # ASCII demo
  tests/          # gtest
```

## 设计说明

- C++ 核心风格对齐 PerfectDouCPP 的 `src/<domain>/` + `algo/` + `torch_nn/` + CMake + gtest。
- 旋转系统刻意保持简单，便于 RL；若需完整 SRS / 20+2 缓冲行，可后续扩展。
