// tetris_server -- Tetris AI 网页服务的 fakelua 宿主。
//
// 分层与 doudizhu_ai/server/host 一致: 规则与推理留在 C++ 引擎里, Lua 负责 HTTP
// 路由、会话隔离、参数编排与前端数据收发。跨 C++/Lua 边界只传 int64 与字符串,
// 状态 DTO 由 session.cpp 生成的 JSON 文本原样交给 Lua, 因此前端契约与旧版
// C++ server 逐字一致 (server/static 一行未改)。
//
// 生命周期沿用 flua 约定: init 一次 / update 每帧一次 / fini 一次。每次调用都是
// 独立的顶层 Call, 帧 arena 在 Call 入口 Reset —— 所以不能在 Lua 里写 while true
// 把请求包进单个 Call。会话对象由本宿主持有, fakelua 没有 GC, Lua 必须在会话
// 闲置超时后调用 tetris.free(sid)。
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "fakelua.h"

#include "server/json.h"
#include "server/policy.h"
#include "server/session.h"
#include "tetris/types.h"

using namespace fakelua;

// nowMs() 由 session.cpp 提供, 宿主不需要链接旧 C++ server 的 http.cpp
// (那里面有 Winsock / socket 代码)。

namespace {

struct Options {
  bool once = false;
  int jitType = 0;
  std::string modelDir;
  std::string script;

  bool Parse(int argc, char** argv, std::string* error) {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--once") {
        once = true;
      } else if (a.rfind("--jit_type=", 0) == 0) {
        jitType = std::atoi(a.c_str() + 11);
        if (jitType < 0 || jitType > 2) {
          *error = "--jit_type 只能是 0(tcc) 1(gcc) 2(interp)";
          return false;
        }
      } else if (a.rfind("--model_dir=", 0) == 0) {
        modelDir = a.substr(12);
      } else if (a == "--help" || a == "-h") {
        *error = "help";
        return false;
      } else if (!a.empty() && a[0] == '-') {
        *error = "未知选项: " + a;
        return false;
      } else if (script.empty()) {
        script = a;
      } else {
        *error = "只接受一个 lua 路径, 多出来的是: " + a;
        return false;
      }
    }
    if (script.empty()) *error = "缺少 lua 目录或文件";
    return error->empty();
  }
};

constexpr const char* kUsage =
    "用法: ./tetris_server <lua 目录或文件> [--model_dir=...] [--once] [--jit_type=n]\n"
    "\n"
    "入口固定为 package \"main\" 的 init / update / fini。监听端口取 TETRIS_PORT。\n"
    "工作目录必须是 server/ (static/ 与 ../ckpt_bc_ppo 都按相对路径找)。\n"
    "自检: TETRIS_SMOKE=1 ./tetris_server lua --once\n";

constexpr const char* kInitEntry = "main.init";
constexpr const char* kUpdateEntry = "main.update";
constexpr const char* kFiniEntry = "main.fini";

std::atomic<bool> g_exit_requested{false};
std::atomic<int> g_exit_code{0};

// 模型只加载一次, 全部会话共用。无 LibTorch 构建走 LiteActor (actor.bin)。
tsrv::ModelPolicy g_model;
std::string g_modelMessage;
std::map<std::string, std::unique_ptr<tsrv::GameSession>> g_sessions;
uint64_t g_requests = 0;
const int64_t g_startedMs = tsrv::nowMs();
std::mt19937_64 g_rng{std::random_device{}()};

int64_t NewSeed() {
  return static_cast<int64_t>(g_rng() % 900000000LL + 1);
}

tsrv::GameSession* Ensure(const std::string& sid) {
  auto it = g_sessions.find(sid);
  if (it == g_sessions.end()) {
    auto s = std::make_unique<tsrv::GameSession>();
    s->reset(NewSeed(), "model", g_model);
    it = g_sessions.emplace(sid, std::move(s)).first;
  }
  it->second->touch();
  return it->second.get();
}

std::string AutoModelDir() {
  // 从 server/ 启动时权重在上一级; 也兼容直接在仓库根目录启动。
  for (const char* d : {"../ckpt_bc_ppo", "ckpt_bc_ppo", "../ckpt_bc", "ckpt_bc"}) {
    if (std::filesystem::exists(std::filesystem::path(d) / "actor.bin")) return d;
  }
  return {};
}

// 一个坏请求不该让整个进程退出: fakelua 里错误逃出 update 等于服务终止。
template <typename Fn>
std::string Guard(Fn&& fn) {
  try {
    return fn();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tetris_server: 原生调用失败: %s\n", e.what());
    tsrv::Json j = tsrv::Json::object();
    j.set("error", tsrv::Json::str(std::string("引擎异常: ") + e.what()));
    return j.dump();
  }
}

std::vector<tetris::Op> ParseOps(const std::string& csv) {
  std::vector<tetris::Op> ops;
  size_t start = 0;
  while (start <= csv.size()) {
    const size_t end = csv.find(',', start);
    const std::string tok =
        csv.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (tok == "Left") ops.push_back(tetris::Op::Left);
    else if (tok == "Right") ops.push_back(tetris::Op::Right);
    else if (tok == "RotateCW") ops.push_back(tetris::Op::RotateCW);
    else if (tok == "RotateCCW") ops.push_back(tetris::Op::RotateCCW);
    else if (tok == "HardDrop") ops.push_back(tetris::Op::HardDrop);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return ops;
}

// ---- 原生回调 --------------------------------------------------------------
// 参数只有 int64 / std::string, 返回值只有 string / int64, 取参转值的样板全在
// 下面的 Native 模板里 (与 doudizhu_ai/server/host/main.cpp 同一套)。

std::string NativeVersion() {
  return std::string("engine=cpp policy=") +
         (g_model.available() ? "loaded" : "missing") +
         " backend=" + g_model.backend() + " label=" + g_model.label();
}

std::string NativeStatus() {
  tsrv::Json j = tsrv::Json::object();
  j.set("status", tsrv::Json::str("ok"));
  const char* envPort = std::getenv("TETRIS_PORT");
  j.set("port", tsrv::Json::integer(envPort ? std::atoi(envPort) : 8080));
  j.set("backend", tsrv::Json::str(g_model.label()));
  j.set("modelAvailable", tsrv::Json::boolean(g_model.available()));
  j.set("model", g_model.describe());
  j.set("modelMessage", tsrv::Json::str(g_modelMessage));
  j.set("sessions", tsrv::Json::integer(static_cast<long long>(g_sessions.size())));
  int64_t totalSteps = 0;
  for (const auto& kv : g_sessions) totalSteps += kv.second->steps();
  j.set("steps", tsrv::Json::integer(static_cast<long long>(totalSteps)));
  j.set("requests", tsrv::Json::integer(static_cast<long long>(g_requests)));
  j.set("uptimeMs", tsrv::Json::integer(static_cast<long long>(tsrv::nowMs() - g_startedMs)));
  j.set("staticDir", tsrv::Json::str("static"));
  return j.dump();
}

std::string NativeState(std::string sid) {
  ++g_requests;
  tsrv::GameSession* s = Ensure(sid);
  return s->state(sid, g_model).dump();
}

std::string NativeNewGame(std::string sid, int64_t seed, std::string mode) {
  ++g_requests;
  tsrv::GameSession* s = Ensure(sid);
  const int64_t sd = seed != 0 ? seed : NewSeed();
  if (mode != "model" && mode != "heuristic") mode = "model";
  if (mode == "model" && !g_model.available()) mode = "heuristic";
  s->reset(sd, mode, g_model);
  return s->state(sid, g_model).dump();
}

std::string NativeStep(std::string sid, int64_t count, int64_t greedy) {
  ++g_requests;
  tsrv::GameSession* s = Ensure(sid);
  const int c = std::clamp<int>(static_cast<int>(count), 1, 200);
  tsrv::Json dto;
  for (int i = 0; i < c; ++i) {
    dto = s->step(sid, g_model, greedy != 0);
    if (s->done()) break;
  }
  return dto.dump();
}

std::string NativeTick(std::string sid, std::string opsCsv) {
  ++g_requests;
  tsrv::GameSession* s = Ensure(sid);
  return s->tick(sid, g_model, ParseOps(opsCsv)).dump();
}

int64_t NativeSessions() {
  return static_cast<int64_t>(g_sessions.size());
}

std::string NativeFree(std::string sid) {
  return g_sessions.erase(sid) > 0 ? "1" : "0";
}

int64_t ArgInt(State* s, CVar* args, int n, int idx) {
  return GetNativeIntArg(s, args, n, idx, 0);
}

std::string ArgStr(State* s, CVar* args, int n, int idx) {
  if (idx >= n) return std::string();
  return inter::FakeluaToNativeString(s, inter::GetNativeArg(s, args, n, idx));
}

template <typename Arg>
Arg OneArg(State* s, CVar* args, int n, int idx) {
  if constexpr (std::is_same_v<Arg, std::string>) return ArgStr(s, args, n, idx);
  else return ArgInt(s, args, n, idx);
}

template <typename Ret, typename... Args, std::size_t... I>
Ret CallPack(State* s, CVar* args, int n, Ret (*fn)(Args...), std::index_sequence<I...>) {
  return fn(OneArg<Args>(s, args, n, static_cast<int>(I))...);
}

template <typename Ret, typename... Args>
void Native(State* s, const std::string& name, Ret (*fn)(Args...)) {
  RegisterNativeFunction(s, name, static_cast<int>(sizeof...(Args)), false,
      [fn](State* state, CVar* args, int n) -> CVar {
        using Seq = std::index_sequence_for<Args...>;
        if constexpr (std::is_same_v<Ret, std::string>) {
          return inter::NativeToFakeluaString(state, Guard([&] { return CallPack(state, args, n, fn, Seq{}); }));
        } else {
          return inter::NativeToFakeluaLonglong(
              state, static_cast<int64_t>(CallPack(state, args, n, fn, Seq{})));
        }
      });
}

void RegisterEngineApi(State* s) {
  Native(s, "tetris.version", NativeVersion);
  Native(s, "tetris.status", NativeStatus);
  Native(s, "tetris.state", NativeState);
  Native(s, "tetris.newgame", NativeNewGame);
  Native(s, "tetris.step", NativeStep);
  Native(s, "tetris.tick", NativeTick);
  Native(s, "tetris.sessions", NativeSessions);
  Native(s, "tetris.free", NativeFree);

  RegisterNativeFunction(s, "tetris.exit", 1, true,
      [](State* state, CVar* args, int n) -> CVar {
        g_exit_code.store(static_cast<int>(GetNativeIntArg(state, args, n, 0, 0)),
                          std::memory_order_relaxed);
        g_exit_requested.store(true, std::memory_order_relaxed);
        return NativeNil(state);
      });
}

std::vector<std::string> CollectLuaFiles(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
      if (entry.is_regular_file() && entry.path().extension() == ".lua") {
        files.push_back(entry.path().string());
      }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) std::cerr << "tetris_server: " << path << " 下没有 .lua 文件\n";
    return files;
  }
  return {path};
}

[[noreturn]] void ReportMissingEntry(const char* name) {
  std::cerr << "tetris_server: 找不到入口 '" << name << "'\n"
            << "        脚本必须提供 package \"main\" 的 init / update / fini\n";
  std::exit(1);
}

int RunLifecycle(State* s, const Options& opts) {
  const JITType jit = static_cast<JITType>(opts.jitType);
  if (!HasEntryFunction(s, jit, kInitEntry)) ReportMissingEntry(kInitEntry);
  if (!opts.once && !HasEntryFunction(s, jit, kUpdateEntry)) ReportMissingEntry(kUpdateEntry);
  if (!HasEntryFunction(s, jit, kFiniEntry)) ReportMissingEntry(kFiniEntry);

  bool initOk = false;
  int exitCode = 0;
  try {
    int code = 0;
    Call(s, jit, kInitEntry, code);
    if (code != 0) {
      exitCode = code;
      std::fprintf(stderr, "tetris_server: init 返回 %d\n", code);
    } else {
      initOk = true;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tetris_server: init 失败: %s\n", e.what());
    return 1;
  }

  if (initOk && !opts.once) {
    while (!g_exit_requested.load(std::memory_order_relaxed)) {
      int code = 0;
      try {
        Call(s, jit, kUpdateEntry, code);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "tetris_server: update 失败: %s\n", e.what());
        exitCode = 1;
        break;
      }
      if (code != 0) {
        g_exit_code.store(code, std::memory_order_relaxed);
        break;
      }
    }
  }

  if (initOk) {
    try {
      int code = 0;
      Call(s, jit, kFiniEntry, code);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "tetris_server: fini 失败: %s\n", e.what());
      exitCode = 1;
    }
  }
  return exitCode != 0 ? exitCode : g_exit_code.load(std::memory_order_relaxed);
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  std::string error;
  if (!opts.Parse(argc, argv, &error)) {
    if (error != "help") std::fprintf(stderr, "tetris_server: %s\n\n", error.c_str());
    std::cout << kUsage;
    return error == "help" ? 0 : 1;
  }

  tsrv::ModelPolicy::Options mo;
  mo.dir = opts.modelDir.empty() ? AutoModelDir() : opts.modelDir;
  mo.hidden = 256;
  mo.backend = "cpu";
  mo.stateFeatures = "zero";
  if (mo.dir.empty() || !g_model.load(mo, g_modelMessage)) {
    std::fprintf(stderr, "tetris_server: 加载权重失败 (%s): %s\n",
                 mo.dir.c_str(), g_modelMessage.c_str());
    return 1;
  }
  std::fprintf(stdout, "tetris_server: %s\n", g_modelMessage.c_str());

  const FakeluaStateGuard guard;
  State* s = guard.GetState();
  SetDebugLogLevel(s, 3);  // 3 = Warn, 常驻服务只需要看得见错误
  try {
    RegisterEngineApi(s);
    CompileConfig cfg;
    for (const auto& file : CollectLuaFiles(opts.script)) {
      CompileFile(s, file, cfg);
    }
    return RunLifecycle(s, opts);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tetris_server: %s\n", e.what());
    return 1;
  }
}
