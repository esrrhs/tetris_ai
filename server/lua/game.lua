package "TetrisGame"

-- 会话流程编排。牌局本体与状态 DTO 全在宿主的 C++ 里 (src/server/session.cpp),
-- 这里只持有「谁在玩、最后一次访问时间」并负责闲置回收。
--
-- 与 doudizhu_ai/server/lua/game.lua 同一套约束: fakelua 文件级 local 是常量,
-- container.map 又只能存标量, 所以会话元数据整块 serialize 成字符串挂在 map 上。

local store = container.map()
local EXPIRE_S = 3600

local function load_session(sid)
    local bin = store:get(sid)
    if bin == nil then
        return nil
    end
    return serialize.decode(bin)
end

local function save_session(sid, s)
    s.seen = os.time()
    store:set(sid, serialize.encode(s))
end

local function ensure(sid)
    local s = load_session(sid)
    if s == nil then
        s = { gameId = sid, mode = "model", seen = 0 }
        save_session(sid, s)
    end
    return s
end

local function touch(sid)
    save_session(sid, ensure(sid))
end

function View(sid)
    touch(sid)
    return tetris.state(sid)
end

function New(sid, seed, mode)
    local s = ensure(sid)
    s.mode = mode
    save_session(sid, s)
    return tetris.newgame(sid, seed, mode)
end

function Step(sid, count, greedy)
    touch(sid)
    return tetris.step(sid, count, greedy)
end

function Tick(sid, ops_csv)
    touch(sid)
    return tetris.tick(sid, ops_csv)
end

function Count()
    return #store:keys()
end

-- 宿主没有 GC: 会话对象必须显式 free, 否则一局一局堆在 C++ 的 map 里。
function Sweep(now)
    local keys = store:keys()
    local i = 1
    while i <= #keys do
        local sid = keys[i]
        local s = load_session(sid)
        if s ~= nil and now - (s.seen or 0) > EXPIRE_S then
            tetris.free(s.gameId)
            store:erase(sid)
        end
        i = i + 1
    end
end
