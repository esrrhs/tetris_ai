package "main"

-- 标准生命周期。宿主固定调用 package "main" 下的这三个函数, 每次调用都是独立的
-- 顶层 Call (帧 arena 在入口 Reset, 请求级临时内存随之回收)。切勿在 Lua 侧写
-- while true 把三者套进一个 Call。
--
-- 节拍直接决定 HTTP 延迟: 一个请求要跨三拍 (收请求 -> 派发 -> 把响应泵出去),
-- 空闲时退避到 IDLE_TICK_INTERVAL_MS 不空转 CPU, 有请求的那几拍回到 1ms。
local BUSY_TICK_INTERVAL_MS = 1
local IDLE_TICK_INTERVAL_MS = 20
local BUSY_TICKS = 16
local SWEEP_TICKS = 512

-- 可变状态一律落 C++ 堆: 文件级 local 落在常量区不可写
local pace = container.map()

-- 自检: 模型已加载 + 状态 DTO 形状 + 神经网络决策 + tick + 会话释放。
-- TETRIS_SMOKE=1 ./tetris_server lua --once 触发, init 返回非 0 即失败。
local function self_check()
    local ver = tetris.version()
    print("smoke version=", ver)
    if not string.contains(ver, "policy=loaded") then
        return 1
    end

    local st = json.decode(tetris.newgame("smoke", 8001, "model"))
    if st == nil or st.modelAvailable ~= true or st.backend ~= "model" then
        print("smoke model backend not active")
        return 1
    end
    if #st.board ~= 20 or #st.board[1] ~= 10 then
        print("smoke board shape broken")
        return 1
    end

    local stepped = json.decode(tetris.step("smoke", 10, 1))
    if stepped == nil or stepped.pieces < 10 or stepped.decision == nil
            or stepped.decision.source ~= "model" then
        print("smoke model step broken")
        return 1
    end

    local ticked = json.decode(tetris.tick("smoke", "Left,RotateCW"))
    if ticked == nil or ticked.board == nil then
        print("smoke tick broken")
        return 1
    end

    local status = json.decode(tetris.status())
    if status == nil or status.status ~= "ok" or status.modelAvailable ~= true then
        print("smoke status broken")
        return 1
    end

    if tetris.free("smoke") ~= "1" or tetris.sessions() ~= 0 then
        print("smoke free broken")
        return 1
    end
    print("smoke OK: model=loaded board=20x10 step=10 backend=model")
    return 0
end

function init()
    local smoke = os.getenv("TETRIS_SMOKE")
    if smoke ~= nil and #smoke > 0 then
        return self_check()
    end

    print("=================================================")
    print("  Tetris AI 网页演示 (FakeLua + C++ 引擎版)")
    print("=================================================")
    print("推理后端: " .. tetris.version())

    local port = 8080
    local env_port = os.getenv("TETRIS_PORT")
    if env_port ~= nil and #env_port > 0 then
        port = math.floor(tonumber(env_port) or 8080)
    end

    local cfg = {}
    cfg["ip"] = "0.0.0.0"
    cfg["port"] = port
    local srv = http.server(cfg)
    srv:dispatch("TetrisHttp.OnRequest")
    print("HTTP 监听 :" .. tostring(port))
    print("浏览器打开 http://127.0.0.1:" .. tostring(port))
    print("-------------------------------------------------")
    return 0
end

function update()
    runtime.tick()

    local seen = TetrisHttp.ServedCount()
    if seen ~= pace:get("last_seen") then
        pace:set("last_seen", seen)
        pace:set("busy_left", BUSY_TICKS)
        pace:set("ticks", 0)
    end

    local ticks = (pace:get("ticks") or 0) + 1
    pace:set("ticks", ticks)
    if math.floor(ticks / SWEEP_TICKS) * SWEEP_TICKS == ticks then
        TetrisGame.Sweep(os.time())
    end

    local ms = IDLE_TICK_INTERVAL_MS
    local busy_left = pace:get("busy_left")
    if busy_left ~= nil and busy_left > 0 then
        ms = BUSY_TICK_INTERVAL_MS
        pace:set("busy_left", busy_left - 1)
    end
    os.sleep(ms)
    return 0
end

function fini()
    print("TetrisServer: shutdown")
    return 0
end
