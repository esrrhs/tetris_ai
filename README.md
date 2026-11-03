# tetris_ai

俄罗斯方块 **强化学习** 环境基础库：按 `Step` 推进的可玩客户端（不是 esrrhs/tetris 竞赛求解器）。

Tetris **reinforcement-learning** play client: the clock advances only via `Step`. Self-contained; does **not** depend on `github.com/esrrhs/tetris`.

仓库对应专属开发 Bot：`tetris_ai`。

## Step 语义 / Step semantics

1. 游戏时间 **只** 在调用 `Step(ops...)` 时前进。
2. 方块生成使用 **7-bag**（每 7 个为一袋，袋内各出现一次，打乱后依次取出）。
3. **无** 活动方块时：本次 `Step` 先在顶部 **生成** 下一个方块；若生成位置与已锁定格子重叠 → `Done=true`（游戏结束）。否则：对新生方块应用 `ops`，再尝试 **下落 1 行**。
4. **有** 活动方块时：按顺序应用任意多个 `ops`，然后重力下落 **恰好 1 行**（若本次 ops 含 `OpHardDrop`，则一插到底并锁定，跳过普通重力）。
5. 操作只在 `Step` 内生效；非法移动（撞墙/重叠）**忽略**，不报错。
6. 方块无法再下落时 **锁定** 并消行；**下一次** `Step`（不是同一次）才生成下一个方块。

允许操作：`Left` / `Right` / `RotateCW` / `RotateCCW` /（可选）`HardDrop`。

## 盘面与旋转 / Board & rotation

- 标准 **10×20** 可见盘面，**无** 隐藏行（Y=0 为顶，向下增大）。
- 七种方块 I,L,J,T,O,S,Z；固定 4 态形状表 + **基础墙踢**（试偏移 `(0,0), (±1,0), (0,-1), (±2,0)`），非完整 SRS。
- 奖励 `Reward` = 本步消除行数；另有累计 `Score`（1/2/3/4 行 → 40/100/300/1200）。

## API

```go
import "github.com/esrrhs/tetris_ai/game"

g := game.NewGame(seed)          // 7-bag RNG seeded
r := g.Step(game.OpLeft, game.OpRotateCW)
// r.Obs     Observation (board + active piece)
// r.Reward  float64     lines cleared this step
// r.Done    bool        game over
// r.Info    Locked / Spawned / LinesCleared / Score / ...

obs := g.Observation()
fmt.Print(g)                     // ASCII render
```

### RL 循环示例 / RL loop sketch

```go
g := game.NewGame(0)
for !g.Done() {
    obs := g.Observation()
    ops := policy(obs)           // your agent: 0..N ops for this tick
    r := g.Step(ops...)
    // learn from r.Reward, r.Done, r.Obs
}
```

## 布局 / Layout

```
tetris_ai/
  go.mod
  README.md
  game/           # board, piece, 7-bag, Step engine, tests
  cmd/demo/       # terminal ASCII demo
```

## 运行 / Run

```bash
go test ./...
go run ./cmd/demo              # seed=42, 40 steps
go run ./cmd/demo 123 60       # seed=123, 60 steps
```

## 设计说明 / Notes

- 不引入竞赛仓库的加权 RNG；本仓库自包含。
- 旋转系统刻意保持简单，便于 RL；若需完整 SRS / 20+2 缓冲行，可后续扩展。
