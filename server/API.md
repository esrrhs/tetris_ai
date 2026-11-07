# 网页服务接口契约

`server/` 的对外协议，分层与同级目录 `doudizhu_ai/server` 一致：**引擎与推理留在 C++，
HTTP 层只做会话、路由与状态 DTO**。线上由 fakelua 提供 HTTP server 与 JSON，Lua 负责
路由 / 会话 / 参数编排，C++ 薄宿主注册 `tetris.*` 原生函数；另保留一套零依赖 C++ server
供 Windows / MinGW / MSVC 使用。两套共用同一份 `session.cpp` 状态 DTO，契约完全相同。

- 线上实现：`host/main.cpp`（原生函数 + 生命周期）、`lua/main.lua`（init/update/fini）、
  `lua/http_server.lua`（路由 / CORS / 静态资源）、`lua/game.lua`（会话元数据 / 回收）
- 引擎与 DTO：`src/server/session.cpp`、`src/server/policy.cpp`、
  `src/server/lite_actor.cpp`（`actor.bin` 的零依赖 MLP 推理）
- 零依赖回退：`src/server/http.cpp` + `src/server/main.cpp`
- 前端：`static/`（纯 HTML/CSS/Canvas，不依赖构建工具）
- 用例：`tests/api_test.py`
- 会话：`X-Session-Id` 头，缺失或非法时服务端现造并通过响应头回传；同一浏览器固定一个 id

---

## 1. 驱动模型

**一次请求 = 一次决策 = 一块方块的完整运动。** 前端不需要知道规则、不模拟重力：

```
POST /api/game/step  →  后端在「当前盘面 + 活动块」上枚举全部合法落点
                       →  策略挑一个（神经网络 或 启发式教师）
                       →  应用旋转/平移/直落，锁定并消行，自动生成下一块
                       →  返回新状态 + 这一块的完整帧序列
                     →  前端按帧播放（旋转 → 横移 → 逐行下落 → 锁定 → 消行闪白）
```

`animation.frames` 是服务端在**落子前盘面的副本**上重放 placement ops 得到的真实轨迹，
所以前端看到的每一步都和引擎里发生的一致，不会两边各算一套。手动模式走
`POST /api/game/tick`，一次请求推进一格重力，用于键盘逐格驱动。

---

## 2. HTTP 接口

请求体一律 JSON；成功 `200`。业务失败 `400` 且响应体仍是状态 DTO 加 `error` 字段
（前端据此弹 toast，不需要额外分支）。所有响应带 CORS 头与 `X-Session-Id`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 存活探针：服务状态、模型是否加载、会话数、累计步数 |
| GET | `/api/game/state` | 当前会话的完整状态 DTO（不推进游戏） |
| POST | `/api/game/new` | 新开一局：`{"seed":123,"mode":"model\|heuristic"}`，`seed=0` 随机 |
| POST | `/api/game/step` | AI 推进一步：`{"count":1,"stochastic":false}`，`count` 上限 200 |
| POST | `/api/game/tick` | 手动推进一格：`{"ops":["Left","RotateCW"]}`，`ops` 省略即仅重力 |
| GET | `/*` | `static/` 下的前端资源，`/` 即 `index.html` |
| OPTIONS | 任意 | `204` + CORS |

`ops` 取值：`Left` / `Right` / `RotateCW` / `RotateCCW` / `HardDrop`。

`--mode heuristic` 或构建时没有 LibTorch 时，模型后端不可用，`/api/status` 的
`modelAvailable=false`，所有决策自动回退到启发式教师；`mode=model` 请求会被静默降级，
状态里的 `backend` 始终是**实际**使用的后端。

---

## 3. 状态 DTO

```jsonc
{
  "sessionId": "sess_...", "step": 12, "seed": 8001,
  "mode": "model", "backend": "model", "modelAvailable": true,
  "width": 10, "height": 20,
  "board": [[0,0,...], ...],          // 20 行 × 10 列，0 空 / 1..7 = I,L,J,T,O,S,Z
  "active": { "type":"T","typeId":3,"rot":1,"x":4,"y":0,
              "cells":[[x,y],[x,y],[x,y],[x,y]] },
  "ghost":  { "y":17, "cells":[[x,y],...] },
  "next": ["T","L","J","O","S"],       // 下 5 块，公开预览
  "score": 1200, "lines": 12, "pieces": 12, "combo": 2, "gameOver": false,
  "metrics": { "heights":[..10..], "holes":0, "bumpiness":4,
               "aggregateHeight":32, "maxHeight":5 },
  "decision": { ... },                 // 可选，第一次推进前缺席
  "animation": { ... },                // 可选，tick / 新局后缺席
  "error": "..."                       // 可选，仅出错时出现
}
```

`decision`：

```jsonc
{
  "source": "model",             // model | heuristic | harddrop
  "optionId": 7, "timeMs": 0.42,
  "rot":1, "x":4, "landingY":17, "linesCleared":2, "scoreGain":100,
  "candidates": [                // 最多 6 项，按优先级降序
    {"rot":1,"x":4,"landingY":17,"linesCleared":2,
     "probability":37.9,"score":2.31}
  ]
}
```

`probability` 是模型 softmax 后的**策略概率**（百分数、一位小数），不是胜率；
启发式教师没有概率，`probability=0`，`score` 是盘面效用值（越高越好）。

`animation`：

```jsonc
{
  "type":"T", "typeId":3, "rot":1, "x":4, "fromY":0, "toY":17,
  "frames":[ {"rot":0,"x":4,"y":0,"cells":[[x,y],...]}, ... ],
  "boardBefore":[[0,...], ...],  // 落子前盘面，动画期间的绘制背景
  "lockedCells":[[x,y],...],     // 最终锁定位置
  "clearedRows":[17,18], "linesCleared":2, "scoreGain":100,
  "combo":1, "gameOver":false
}
```

`frames` 按时间顺序：出生 → 每次旋转/平移 → 每下落一行 → 最后锁定前的位置。
`cells` 是绝对棋盘坐标，可能含 `y<0`（出生时超出可见区），前端需裁剪。

---

## 4. 边界与不变量

1. **状态与动画统一**：`board` 是消行后、且已生成下一块的状态；动画要用
   `boardBefore` 做背景，播放完再切到 `board`。
2. `pieces` 是成功锁定数，`step` 是决策/推进次数；两者在 tick 模式下不相同。
3. `Env::Step` 锁定后会自动 spawn 下一块；若 spawn 碰撞，`gameOver=true`、
   `active`/`ghost` 为 `null`，此时 `step` 返回带 `error` 的状态 DTO。
4. 会话在服务端常驻，重开一局复用同一个 id（不泄漏）；闲置超过 1 小时的会话由
   每 64 次请求一次的 `Sweep` 回收。
5. 静态资源按字面拼路径，含 `..` 或反斜杠一律 404。
6. 单线程阻塞事件循环，`Connection: close`，一次请求一个连接；loopback 下
   单步延迟主要由动画播放决定，而非 HTTP。

---

## 5. 示例

```bash
curl -s -H "X-Session-Id: demo1" http://127.0.0.1:8080/api/status
curl -s -X POST -H "X-Session-Id: demo1" -H "Content-Type: application/json" \
     -d '{"seed":8001,"mode":"model"}' http://127.0.0.1:8080/api/game/new
curl -s -X POST -H "X-Session-Id: demo1" -H "Content-Type: application/json" \
     -d '{"count":1}' http://127.0.0.1:8080/api/game/step > one_step.json
```
