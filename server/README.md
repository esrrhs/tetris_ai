# Tetris AI 网页演示服务

把 C++ 引擎和训练出来的策略包成一个能看的网页：**前端按 step 请求驱动，服务端每步返回
一块方块的完整运动帧，浏览器把它画出来**。

**线上运行的是 fakelua 版**，分层与同级目录 `doudizhu_ai/server` 一致：规则与推理留在
C++，Lua 只做 HTTP 路由、会话隔离与参数编排；HTTP server / JSON 由 fakelua 提供。
另保留一套零依赖 C++ server，供 Windows / MinGW / MSVC 等没有 fakelua 的环境使用。
两套共用同一份引擎、同一个状态 DTO 与同一套 HTTP 契约，前端 `static/` 一行未改。

- 引擎与推理：`src/tetris/`（规则 / 特征 / Env / 启发式教师）+ `src/server/lite_actor.cpp`
  （`actor.bin` 的零依赖 MLP 推理，权重由 `tools/export_actor.py` 从 `actor.pt` 导出）
- **线上 fakelua 版**：`host/main.cpp` 薄宿主把引擎注册成 `tetris.*` 原生函数；
  `lua/main.lua`（生命周期 / 节拍）、`lua/http_server.lua`（路由 / CORS / 静态资源）、
  `lua/game.lua`（会话元数据 / 闲置回收）
- 零依赖 C++ 回退：`src/server/http.cpp` + `main.cpp`（自写 socket HTTP），见文末
- 前端：`static/`（HTML + CSS + Canvas，无框架、无构建步骤）
- 契约：[API.md](API.md)；回归：[tests/api_test.py](tests/api_test.py)

## 构建（线上 fakelua 版）

需要 [fakelua](https://github.com/esrrhs/fakelua)（`libfakelua` + 头文件），以及
`ckpt_bc_ppo/actor.bin`。部署机没有 LibTorch、也访问不到 GitHub，所以模型走
「导出成裸 float32 + C++ 轻量推理」，构建完全离线。

```bash
# 1) 导出权重（仓库里已带一份；换权重时重跑）
python3 server/tools/export_actor.py        # ckpt_bc_ppo/actor.pt -> ckpt_bc_ppo/actor.bin

# 2) 编译宿主（独立 CMake 工程，不引入 googletest / LibTorch）
cd server
cmake -S host -B host/build -DCMAKE_BUILD_TYPE=Release -DFAKELUA_PREFIX=/usr/local
cmake --build host/build -j2

# 3) 自检：不建 HTTP 服务，跑完 init 就退出，非 0 即失败
LD_LIBRARY_PATH=/usr/local/lib TETRIS_SMOKE=1 ./host/build/tetris_server lua --once
```

宿主用 C++23；`FAKELUA_PREFIX` 默认 `/opt/homebrew`，Linux 上装的是 `/usr/local`。
找不到已安装版时可用 `-DFAKELUA_DIR=<fakelua 源码>` 走源码内编。自检覆盖模型加载、
20×10 棋盘形状、神经网络决策来源、手动 tick、状态接口与会话显式释放。

## 运行

**工作目录必须是 `server/`**：静态资源按 `static/` 相对路径找，权重默认探测
`../ckpt_bc_ppo/actor.bin`。端口取环境变量 `TETRIS_PORT`（Lua 侧 `os.getenv`）。

```bash
cd server
TETRIS_PORT=8080 LD_LIBRARY_PATH=/usr/local/lib ./host/build/tetris_server lua
# 浏览器打开 http://127.0.0.1:8080
```

宿主只读三个开关：`<lua 目录>`、`--model_dir=...`、`--once`、`--jit_type=0|1|2`。
模型参数固定 `hidden=256` / `state-features=zero`，与 `ckpt_bc_ppo` 训练时一致
（该权重来自 DAgger 行为克隆 + 长局 PPO 微调，10 万块 × 10 局零死亡）。

生命周期沿用 flua 约定：`main.init` 建 HTTP 服务，`main.update` 每帧泵 IO 并分发请求，
`main.fini` 收尾。每次调用都是独立的顶层 Call，帧 arena 在入口 Reset —— 所以可变状态
一律放 `container.map`，会话元数据先 `serialize.encode` 再存；宿主的 C++ 会话对象没有
GC，`lua/game.lua` 的 `Sweep` 会显式 `tetris.free(sid)` 回收闲置 1 小时的会话。

跨 C++/Lua 边界只传 int64 与字符串：状态 DTO 是 `session.cpp` 生成的 JSON 文本，
Lua 原样回包（避免深层嵌套数组经 `json.encode` 重编码出现数组/对象歧义），只
`json.decode` 一次用来判断 `error` 字段决定 200 / 400。

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
python3 server/tests/api_test.py --base-url http://127.0.0.1:8080
python3 server/tests/api_test.py --base-url https://tetris.esrrhs.xyz
python3 server/tests/api_test.py --exe build_gcc/tetris_ai_server   # 零依赖 C++ 回退
```

36 个用例：存活探针、静态资源 MIME、CORS、会话隔离、DTO 形状、单步/快进守恒、
手动 tick、错误路径与非法 body 不崩溃。路径穿越一项接受 400 / 404 —— 直连服务返回
404，Nginx 在 URI 解析阶段就会以 400 拒绝 `%2e%2e`。

## 线上部署

<https://tetris.esrrhs.xyz> —— nginx 反代到 `127.0.0.1:8892`，进程由 systemd 管
（`tetris.service`，`User=ubuntu`、`WorkingDirectory=<repo>/server`、
`Environment=TETRIS_PORT=8892`、`Environment=LD_LIBRARY_PATH=/usr/local/lib`、
`Restart=always`），与 `doudizhu.service` / `majiang.service` 同一套模式。
代码在 `ubuntu@gq.esrrhs.xyz:/home/ubuntu/project/tetris_ai`。

**那台机器访问不到 GitHub**，所以发布走 rsync 而不是 `git pull`：

```bash
# 本地：同步代码（不含 .git 与各类构建目录）
rsync -az --delete --exclude='.git/' --exclude='.workbuddy/' \
    --exclude='build/' --exclude='build_*/' \
    tetris_ai/ ubuntu@gq.esrrhs.xyz:/home/ubuntu/project/tetris_ai/

# 远端：编译 + 自检 + 重启
ssh ubuntu@gq.esrrhs.xyz 'cd /home/ubuntu/project/tetris_ai/server && \
    cmake -S host -B host/build -DCMAKE_BUILD_TYPE=Release -DFAKELUA_PREFIX=/usr/local && \
    cmake --build host/build -j2 && \
    LD_LIBRARY_PATH=/usr/local/lib TETRIS_SMOKE=1 ./host/build/tetris_server lua --once && \
    sudo systemctl restart tetris'

# 打线上跑一遍接口契约
python3 server/tests/api_test.py --base-url https://tetris.esrrhs.xyz
```

宿主机上已装 `/usr/local` 版 fakelua（与 `doudizhu` / `majiang` 用的是同一份），
所以只需编本仓库的宿主，不用重编 boost。gq 是 2 核 2G 共享机，服务与
majiang(:8888)、doudizhu(:8889)、paodekuai(:8891)、fakefish(:8080) 并存，端口别撞。
TLS 证书用 `certbot certonly --webroot -w /var/www/certbot -d tetris.esrrhs.xyz` 申请
（这台机器 snapd 被禁用，`/usr/bin/certbot` 0.27 可用；snap 版 5.8 会因缺 augeas 报错）。

## 零依赖 C++ 回退

没有 fakelua 时（Windows / MinGW / MSVC），顶层 CMake 仍提供 `tetris_ai_server`：
`http.cpp` 自写 socket HTTP、`json.h` 极简 JSON，缺 LibTorch 时加载 `actor.bin`
而非回退启发式教师。

```bash
cmake -S . -B build_gcc -DCMAKE_BUILD_TYPE=Release \
      -DTETRIS_AI_WITH_TORCH=OFF -DTETRIS_AI_BUILD_TESTS=OFF
cmake --build build_gcc --target tetris_ai_server -j 8
./build_gcc/tetris_ai_server --port 8080
```

参数：`--port` / `--model`（含 `actor.pt` 或 `actor.bin` 的目录）/ `--hidden` /
`--state-features` / `--backend` / `--mode` / `--seed` / `--static` / `--stochastic`。
静态资源目录在编译期写进可执行文件，也可用 `--static` 覆盖。

## 已知边界

- 单线程事件循环，一次请求跨三拍（收 → 派发 → 回包），尾延迟由节拍决定；
  空闲 20ms / 忙碌 1ms 自适应。
- 会话常驻内存，闲置 1 小时回收；不做持久化、不做鉴权，勿直接暴露到公网。
- 动画是「已经发生的事实」的重放，不是预测：引擎先算完，前端再播。
- fakelua 不用协程、元表与 `require`；文件级 `local` 是常量，可变状态一律走
  `container.map`。
