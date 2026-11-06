// tetris_ai_server — HTTP entry point.
//
// Layering follows doudizhu_ai/server: the engine and inference stay in C++,
// this process only does HTTP, sessions and request/response shaping. The
// browser drives the game one step at a time and renders the returned frames.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "server/http.h"
#include "server/json.h"
#include "server/policy.h"
#include "server/session.h"

#ifndef TETRIS_AI_STATIC_DIR
#define TETRIS_AI_STATIC_DIR "static"
#endif

namespace fs = std::filesystem;
using tsrv::GameSession;
using tsrv::HttpResponse;
using tsrv::Json;

namespace {

const char* argValue(int argc, char** argv, const std::string& key,
                     const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) return argv[i + 1];
  }
  return fallback;
}

bool hasFlag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

std::string mime(const std::string& path) {
  if (path.size() >= 5 && path.substr(path.size() - 5) == ".html")
    return "text/html; charset=utf-8";
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".css")
    return "text/css; charset=utf-8";
  if (path.size() >= 3 && path.substr(path.size() - 3) == ".js")
    return "application/javascript; charset=utf-8";
  if (path.size() >= 5 && path.substr(path.size() - 5) == ".json")
    return "application/json; charset=utf-8";
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".svg")
    return "image/svg+xml";
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".png")
    return "image/png";
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".ico")
    return "image/x-icon";
  return "application/octet-stream";
}

bool readFile(const fs::path& p, std::string& out) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

bool validSession(const std::string& s) {
  if (s.empty() || s.size() > 64) return false;
  for (char c : s) {
    bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

struct Context {
  tsrv::ModelPolicy model;
  std::map<std::string, std::unique_ptr<GameSession>> sessions;
  std::string staticDir = TETRIS_AI_STATIC_DIR;
  int64_t startedMs = tsrv::nowMs();
  uint64_t requests = 0;
  std::string modelMessage;

  GameSession* session(const std::string& sid, int64_t defaultSeed) {
    auto it = sessions.find(sid);
    if (it != sessions.end()) return it->second.get();
    auto s = std::make_unique<GameSession>();
    s->reset(defaultSeed, "model", model);
    GameSession* raw = s.get();
    sessions.emplace(sid, std::move(s));
    return raw;
  }

  void sweep() {
    const int64_t now = tsrv::nowMs();
    for (auto it = sessions.begin(); it != sessions.end();) {
      if (now - it->second->lastAccessMs() > 3600LL * 1000LL) {
        it = sessions.erase(it);
      } else {
        ++it;
      }
    }
  }
};

HttpResponse reply(const Json& dto, const std::string& sid, int status = 200) {
  HttpResponse r;
  r.status = dto.find("error") ? 400 : status;
  r.body = dto.dump();
  r.extraHeaders.emplace_back("X-Session-Id", sid);
  return r;
}

HttpResponse errorReply(const std::string& message, const std::string& sid,
                        int status = 400) {
  HttpResponse r;
  r.status = status;
  Json j = Json::object();
  j.set("error", Json::str(message));
  r.body = j.dump();
  r.extraHeaders.emplace_back("X-Session-Id", sid);
  return r;
}

HttpResponse serveStatic(Context& ctx, const std::string& path) {
  std::string rel = path.empty() || path == "/" ? "/index.html" : path;
  if (rel.find("..") != std::string::npos || rel.find('\\') != std::string::npos ||
      rel.find('\0') != std::string::npos) {
    return errorReply("invalid path", "", 404);
  }
  while (!rel.empty() && rel.front() == '/') rel.erase(rel.begin());
  fs::path full = fs::path(ctx.staticDir) / rel;
  std::string body;
  if (!readFile(full, body)) {
    return errorReply("404 Not Found: " + rel, "", 404);
  }
  HttpResponse r;
  r.status = 200;
  r.contentType = mime(rel);
  r.body = std::move(body);
  return r;
}

std::string autoModelDir() {
  const std::vector<std::string> candidates = {
      "ckpt_bc_ppo", "ckpt_bc", "../ckpt_bc_ppo", "../ckpt_bc",
      "server/ckpt_bc_ppo", "server/ckpt_bc"};
  for (const auto& d : candidates) {
    if (fs::exists(fs::path(d) / "actor.pt")) return d;
  }
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  Context ctx;
  const int port = std::atoi(argValue(argc, argv, "--port", "8080"));
  const std::string staticArg = argValue(argc, argv, "--static", "");
  if (!staticArg.empty()) ctx.staticDir = staticArg;

  tsrv::ModelPolicy::Options mo;
  mo.dir = argValue(argc, argv, "--model", "");
  if (mo.dir.empty()) mo.dir = autoModelDir();
  mo.hidden = std::atoi(argValue(argc, argv, "--hidden", "256"));
  mo.backend = argValue(argc, argv, "--backend", "cpu");
  mo.stateFeatures = argValue(argc, argv, "--state-features", "zero");
  const std::string requestedMode = argValue(argc, argv, "--mode", "model");
  const int64_t defaultSeed = std::strtoll(
      argValue(argc, argv, "--seed", "0"), nullptr, 10);

  if (requestedMode != "heuristic" && !mo.dir.empty()) {
    std::string msg;
    if (!ctx.model.load(mo, msg)) {
      std::cerr << "[warn] " << msg << "\n";
    }
    ctx.modelMessage = msg;
  } else if (mo.dir.empty()) {
    ctx.modelMessage = "no checkpoint found; using heuristic teacher";
  } else {
    ctx.modelMessage = "--mode heuristic: model disabled";
  }

  std::random_device rd;
  std::mt19937_64 rng((static_cast<uint64_t>(rd()) << 32) ^ rd());
  auto newSeed = [&]() -> int64_t {
    return defaultSeed != 0 ? defaultSeed
                            : static_cast<int64_t>(rng() % 900000000LL + 1);
  };

  tsrv::HttpServer server;
  if (!server.start(port)) {
    std::cerr << "failed to listen on port " << port << "\n";
    return 1;
  }

  std::cout << "tetris_ai_server listening on http://127.0.0.1:" << port << "\n"
            << "  static     : " << ctx.staticDir << "\n"
            << "  decision   : " << ctx.model.label() << "\n"
            << "  model info : " << ctx.modelMessage << "\n"
            << "  drive      : POST /api/game/step (one placement per request)\n"
            << std::flush;

  auto handler = [&](const tsrv::HttpRequest& req) -> HttpResponse {
    ++ctx.requests;
    if ((ctx.requests & 63) == 0) ctx.sweep();

    std::string sid = req.header("x-session-id");
    if (!validSession(sid)) {
      sid = "sess" + std::to_string(tsrv::nowMs()) + "_" +
            std::to_string(rng() % 1000000);
    }
    if (req.method == "OPTIONS") {
      HttpResponse r;
      r.status = 204;
      r.contentType = "text/plain; charset=utf-8";
      r.extraHeaders.emplace_back("X-Session-Id", sid);
      return r;
    }

    const std::string& path = req.path;
    if (path.rfind("/api/", 0) != 0) {
      if (req.method != "GET") return errorReply("method not allowed", sid, 405);
      return serveStatic(ctx, path);
    }

    if (path == "/api/status" && req.method == "GET") {
      Json j = Json::object();
      j.set("status", Json::str("ok"));
      j.set("port", Json::integer(port));
      j.set("backend", Json::str(ctx.model.label()));
      j.set("modelAvailable", Json::boolean(ctx.model.available()));
      j.set("model", ctx.model.describe());
      j.set("modelMessage", Json::str(ctx.modelMessage));
      j.set("sessions", Json::integer(static_cast<long long>(ctx.sessions.size())));
      int64_t totalSteps = 0;
      for (const auto& kv : ctx.sessions) totalSteps += kv.second->steps();
      j.set("steps", Json::integer(totalSteps));
      j.set("requests", Json::integer(static_cast<long long>(ctx.requests)));
      j.set("uptimeMs", Json::integer(tsrv::nowMs() - ctx.startedMs));
      j.set("staticDir", Json::str(ctx.staticDir));
      return reply(j, sid);
    }

    if (path == "/api/game/state" && req.method == "GET") {
      return reply(ctx.session(sid, newSeed())->state(sid, ctx.model), sid);
    }

    if (path == "/api/game/new" && req.method == "POST") {
      Json body = Json::parse(req.body);
      int64_t seed = body.find("seed") ? body.find("seed")->asInt(0) : 0;
      if (seed == 0) seed = newSeed();
      std::string mode = body.find("mode") ? body.find("mode")->asString("model")
                                           : "model";
      if (mode != "model" && mode != "heuristic") mode = "model";
      if (mode == "model" && !ctx.model.available()) {
        // Still create the session; state() reports the actual backend.
        mode = "heuristic";
      }
      GameSession* s = ctx.session(sid, seed);
      s->reset(seed, mode, ctx.model);
      return reply(s->state(sid, ctx.model), sid);
    }

    if (path == "/api/game/step" && req.method == "POST") {
      Json body = Json::parse(req.body);
      int count = body.find("count") ? static_cast<int>(body.find("count")->asInt(1)) : 1;
      count = std::clamp(count, 1, 200);
      bool greedy = !hasFlag(argc, argv, "--stochastic");
      if (body.find("stochastic")) greedy = !body.find("stochastic")->asBool(false);
      GameSession* s = ctx.session(sid, newSeed());
      Json dto;
      for (int i = 0; i < count; ++i) {
        dto = s->step(sid, ctx.model, greedy);
        if (s->done()) break;
      }
      return reply(dto, sid);
    }

    if (path == "/api/game/tick" && req.method == "POST") {
      Json body = Json::parse(req.body);
      auto ops = tsrv::parseOps(body);
      GameSession* s = ctx.session(sid, newSeed());
      return reply(s->tick(sid, ctx.model, ops), sid);
    }

    return errorReply("unknown endpoint or method: " + req.method + " " + path,
                      sid, 404);
  };

  server.serve(handler);
  return 0;
}
