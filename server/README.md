# Tetris AI 网页演示服务

把 C++ 引擎和训练出来的策略包成一个能看的网页：**前端按 step 请求驱动，服务端每步返回
一块方块的完整运动帧，浏览器把它画出来**。分层与契约参考同级目录 `doudizhu_ai/server`，
但没有引入 Lua / fakelua：这个服务只依赖一个 C++20 编译器，MinGW 与 MSVC 都能编。

- 引擎与推理：`src/tetris/`、`src/torch_nn/`（原有代码，未改动）
- HTTP / 会话：`src/server/`（`http.cpp` 零依赖 socket HTTP、`json.h` 极简 JSON、
  `session.cpp` 状态与动画帧、`policy.cpp` 决策后端、`main.cpp` 路由）
- 前端：`static/`（HTML + CSS + Canvas，无框架、无构建步骤）
- 契约：[API.md](API.md)；回归：[tests/api_test.py](tests/api_test.py)

## 构建

```bash
# MinGW（无 LibTorch，只有启发式教师，最轻）
cmake -S . -B build_gcc -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTETRIS_AI_WITH_TORCH=OFF -DCMAKE_CXX_COMPILER=g++
cmake --build build_gcc --target tetris_ai_server -j 8

# MSVC + CUDA LibTorch（可加载训练好的 Actor）
cmake -S . -B build_msvc -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=<python>/site-packages/torch \
      -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build_msvc --target tetris_ai_server -j 8
```

Windows 上运行 MSVC 版需要把 `torch/lib` 加进 `PATH`。静态资源目录在编译期写进可执行文件，
所以从任何目录启动都能找到；也可以用 `--static` 覆盖。

## 运行

```bash
# 项目根目录启动（模型权重在 ckpt_bc_ppo/ 时会自动探测）
./build_msvc/tetris_ai_server --port 8080
# 浏览器打开 http://127.0.0.1:8080
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--port` | 8080 | 监听端口 |
| `--model` | 自动探测 | 含 `actor.pt` 的 checkpoint 目录 |
| `--hidden` | 256 | Actor 隐层宽度，必须与权重一致（旧 PPO 权重是 128） |
| `--state-features` | zero | `raw` / `clamp` / `zero`，必须与训练时一致 |
| `--backend` | cpu | `cpu` / `cuda`；单步推理很小，CPU 通常更快 |
| `--mode` | model | `heuristic` 可强制只用启发式教师 |
| `--seed` | 0 | 新会话默认种子，0 为随机 |
| `--static` | 编译期路径 | 静态资源目录 |
| `--stochastic` | 关 | 模型按概率采样而非取 argmax |

默认探测顺序：`ckpt_bc_ppo`、`ckpt_bc`、`../ckpt_bc_ppo`、`../ckpt_bc`。
本项目最终权重来自 `DAgger 行为克隆 + 长局 PPO 微调`，训练时用的是
`hidden=256`、`state-features=zero`；用错参数会加载失败并自动回退到启发式教师。

## 前端表现

- 10×20 棋盘为 Canvas：格子带渐变高光与发光、幽灵落点用虚线框标出；
  下 5 块预览用 4×2 迷你网格，与 `src/tetris/shapes.cpp` 的 rot-0 形状一致。
- 「单步推进」发一次 `POST /api/game/step`；服务端返回的 `animation.frames`
  被逐帧播放：出生 → 旋转 → 平移 → 逐行下落 → 锁定 → 满行闪白 → 显示新盘面。
- 「自动播放」按速度滑块循环发同一个请求；「快进 10 步」一次请求推进 10 块（不播动画）。
- 「手动模式」把键盘映射成 `/api/game/tick` 的一格推进：← → 平移、↑ 旋转、
  ↓ 下移、空格直落、Z 反转。
- 侧栏展示本步决策来源、耗时、落点、消行与候选落点排序（模型显示 softmax 概率，
  教师显示盘面效用值），以及分数、消行、连击、空洞/起伏/最高堆高。

## 接口回归

```bash
python server/tests/api_test.py                              # 自己拉起服务（随机端口）
python server/tests/api_test.py --base-url http://127.0.0.1:8080
python server/tests/api_test.py --exe build_msvc/tetris_ai_server.exe
```

覆盖存活探针、静态资源 MIME、CORS、会话隔离、DTO 形状、单步/快进守恒、
手动 tick、错误路径与非法 body 不崩溃。

## 已知边界

- 单线程阻塞事件循环，`Connection: close`，一次请求一个连接；适合本机演示与单人会话。
- 会话常驻内存，闲置 1 小时回收；不做持久化、不做鉴权，勿直接暴露到公网。
- 动画是「已经发生的事实」的重放，不是预测：引擎先算完，前端再播。
