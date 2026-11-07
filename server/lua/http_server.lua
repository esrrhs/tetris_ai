package "TetrisHttp"

-- HTTP 路由与静态资源。所有牌局动作都转给 TetrisGame, 这里只管收发与错误码。
--
-- 接口与旧 C++ server 完全一致 (前端 static/ 就是那一版的, 一行未改):
--   GET  /api/status             存活探针 / 模型信息
--   GET  /api/game/state         当前会话的完整状态
--   POST /api/game/new           {"seed":123,"mode":"model|heuristic"}
--   POST /api/game/step          {"count":1,"stochastic":false}
--   POST /api/game/tick          {"ops":["Left","RotateCW"]}
-- 会话用 X-Session-Id 头隔离, 同一浏览器固定一个 id。

local static_root = "static"

-- 已派发的请求数。文件级 local 落在常量区不可写, 计数放 container.map(C++ 堆)。
local served = container.map()
local SESSION_MAX_LEN = 64

function ServedCount()
    return served:get("n") or 0
end

local function cors_headers()
    local h = {}
    h["Access-Control-Allow-Origin"] = "*"
    h["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    h["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
    h["Content-Type"] = "application/json; charset=UTF-8"
    return h
end

local function respond(status, body, extra)
    local resp = {}
    resp["status"] = status
    local h = cors_headers()
    if extra ~= nil then
        local k = 1
        while k <= #extra do
            h[extra[k][1]] = extra[k][2]
            k = k + 1
        end
    end
    resp["headers"] = h
    resp["body"] = body
    return resp
end

local function json_raw(status, text)
    return respond(status, text, nil)
end

-- DTO 是 C++ 生成的 JSON 文本, 原样回包 —— 状态里有 board(20x10)、animation.frames
-- 这类深层嵌套数组, 不经 fakelua json.encode 重编码就不会出现数组/对象歧义。
-- 只 decode 一次, 用来判断有没有 error 字段决定 200 / 400。
local function reply(text, sid)
    if text == nil or text == "" then
        return respond(500, "{\"error\":\"internal error\"}", { { "X-Session-Id", sid } })
    end
    local status = 200
    local ok, dto = pcall(function()
        return json.decode(text)
    end)
    if ok and type(dto) == "table" and dto.error ~= nil then
        status = 400
    end
    return respond(status, text, { { "X-Session-Id", sid } })
end

local function parse_body(body)
    if body == nil or body == "" then
        return {}
    end
    local ok, t = pcall(function()
        return json.decode(body)
    end)
    if ok and type(t) == "table" then
        return t
    end
    return {}
end

-- 只接受 [A-Za-z0-9_-] 的会话号且限长: 宿主的会话 map 常驻内存, 不能被路人撑爆。
local function valid_session(sid)
    if sid == nil then
        return false
    end
    if #sid < 1 or #sid > SESSION_MAX_LEN then
        return false
    end
    return string.match(sid, "^[A-Za-z0-9_%-]+$") ~= nil
end

local function session_id(req)
    local sid = nil
    if req.headers ~= nil then
        sid = req.headers["X-Session-Id"]
        if sid == nil then
            sid = req.headers["x-session-id"]
        end
    end
    if not valid_session(sid) then
        sid = "sess" .. tostring(os.time()) .. tostring(math.random(100000, 999999))
            .. tostring(math.random(100000, 999999))
    end
    return sid
end

local function mime(path)
    if string.ends_with(path, ".html") then return "text/html; charset=UTF-8" end
    if string.ends_with(path, ".css") then return "text/css; charset=UTF-8" end
    if string.ends_with(path, ".js") then return "application/javascript; charset=UTF-8" end
    if string.ends_with(path, ".svg") then return "image/svg+xml" end
    if string.ends_with(path, ".png") then return "image/png" end
    if string.ends_with(path, ".ico") then return "image/x-icon" end
    if string.ends_with(path, ".json") then return "application/json; charset=UTF-8" end
    return "application/octet-stream"
end

local function serve_static(path)
    if path == "/" or path == "" then
        path = "/index.html"
    end
    -- 只按字面拼路径, 带 .. 的一律拒绝 (fakelua 没有 realpath)
    if string.find(path, "..", 1, true) ~= nil then
        return json_raw(404, "404 Not Found")
    end
    local f = io.open(static_root .. path, "rb")
    if f == nil then
        return json_raw(404, "404 Not Found")
    end
    local body = f:read("*a")
    f:close()
    local resp = {}
    resp["status"] = 200
    local h = {}
    h["Content-Type"] = mime(path)
    h["Access-Control-Allow-Origin"] = "*"
    resp["headers"] = h
    resp["body"] = body
    return resp
end

local function bad_method()
    return json_raw(405, "405 Method Not Allowed")
end

function OnRequest(typ, connid, req)
    if typ ~= "request" then
        return nil
    end
    served:set("n", ServedCount() + 1)

    local method = req.method
    local path = req.path
    if method == "OPTIONS" then
        return respond(204, "", nil)
    end

    if path == "/api/status" then
        return respond(200, tetris.status(), { { "X-Session-Id", "" } })
    end

    if string.starts_with(path, "/api/game") then
        local sid = session_id(req)
        local body = parse_body(req.body)

        if path == "/api/game/state" then
            if method ~= "GET" then return bad_method() end
            return reply(TetrisGame.View(sid), sid)
        end

        if path == "/api/game/new" then
            if method ~= "POST" then return bad_method() end
            local seed = math.floor(tonumber(body["seed"]) or 0)
            local mode = "model"
            if type(body["mode"]) == "string" then mode = body["mode"] end
            return reply(TetrisGame.New(sid, seed, mode), sid)
        end

        if path == "/api/game/step" then
            if method ~= "POST" then return bad_method() end
            local count = math.floor(tonumber(body["count"]) or 1)
            local greedy = 1
            if body["stochastic"] == true then greedy = 0 end
            return reply(TetrisGame.Step(sid, count, greedy), sid)
        end

        if path == "/api/game/tick" then
            if method ~= "POST" then return bad_method() end
            local csv = ""
            if type(body["ops"]) == "table" then
                local ok, joined = pcall(function()
                    return table.concat(body["ops"], ",")
                end)
                if ok then csv = joined end
            end
            return reply(TetrisGame.Tick(sid, csv), sid)
        end

        return json_raw(404, "404 Not Found")
    end

    if string.starts_with(path, "/api/") then
        return json_raw(404, "404 Not Found")
    end

    return serve_static(path)
end
