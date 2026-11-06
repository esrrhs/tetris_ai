// One browser session = one Tetris Env plus the last decision/animation.
// The frontend drives everything through POST /api/game/step (one AI placement
// per request) or POST /api/game/tick (one manual gravity tick per request).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "server/json.h"
#include "server/policy.h"
#include "tetris/env.h"
#include "tetris/types.h"

namespace tsrv {

int64_t nowMs();

class GameSession {
 public:
  void reset(int64_t seed, const std::string& mode, const ModelPolicy& model);

  Json state(const std::string& sessionId, const ModelPolicy& model,
             const std::string& error = "") const;
  Json step(const std::string& sessionId, const ModelPolicy& model,
            bool greedy);
  Json tick(const std::string& sessionId, const ModelPolicy& model,
            const std::vector<tetris::Op>& ops);

  bool done() const { return env_.Done(); }
  int64_t lastAccessMs() const { return lastAccessMs_; }
  void touch() const { lastAccessMs_ = nowMs(); }
  int64_t seed() const { return seed_; }
  const std::string& mode() const { return mode_; }
  int steps() const { return steps_; }

 private:
  tetris::Env env_;
  int64_t seed_ = 1;
  std::string mode_ = "heuristic";
  int steps_ = 0;
  mutable int64_t lastAccessMs_ = 0;
  Json lastDecision_ = Json::null();
  Json lastAnimation_ = Json::null();
};

// Parse the "ops" string array of a tick request. Unknown names are ignored.
std::vector<tetris::Op> parseOps(const Json& body);

}  // namespace tsrv
