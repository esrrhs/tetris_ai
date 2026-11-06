// Decision backends for one placement. The neural policy is compiled only when
// LibTorch is available; the classical heuristic teacher is always present.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "server/json.h"
#include "tetris/env.h"

namespace tsrv {

struct Candidate {
  int rot = 0;
  int x = 0;
  int landingY = 0;
  int linesCleared = 0;
  double probability = 0.0;  // 0..100, softmax over legal options (model only)
  double score = 0.0;        // raw logit (model) or heuristic utility (teacher)
};

struct Decision {
  int optionId = -1;
  std::string source = "heuristic";
  double timeMs = 0.0;
  int rot = 0;
  int x = 0;
  int landingY = 0;
  int linesCleared = 0;
  int scoreGain = 0;
  std::vector<Candidate> candidates;
};

class ModelPolicy {
 public:
  struct Options {
    std::string dir;                 // checkpoint directory containing actor.pt
    int hidden = 256;                // must match the trained Actor
    std::string backend = "cpu";     // cpu | cuda
    std::string stateFeatures = "zero";  // raw | clamp | zero
  };

  // Returns true when actor.pt was loaded. message carries the human-readable
  // reason otherwise (missing file, dim mismatch, no LibTorch build...).
  bool load(const Options& opt, std::string& message);
  bool available() const { return available_; }
  const std::string& label() const { return label_; }
  const std::string& backend() const { return backendName_; }
  Json describe() const;

  // Chooses among the first N legal options (N <= kMaxActions). Fills `out`
  // with timing and the ranked candidate list shown in the web UI.
  // Returns -1 when no model is loaded.
  int choose(const tetris::Env& env,
             const std::vector<tetris::LegalOption>& options, bool greedy,
             Decision& out) const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  bool available_ = false;
  std::string label_ = "heuristic";
  std::string backendName_ = "cpu";
};

Json decisionToJson(const Decision& d);

}  // namespace tsrv
