#include "algo/rollout.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>

#include "algo/ppo.h"
#include "tetris/env.h"

namespace algo {
namespace {

// Play one game with the given actor (+ optional critic for learning).
// When collectTransitions is false (league eval), only episode stats are kept.
void playGame(torch_nn::Actor* actor, torch_nn::Critic* critic,
              const RolloutConfig& cfg, uint64_t seed, int gameIdBase, int gi,
              bool collectTransitions, std::vector<Transition>& out,
              int& epLines, int& epScore, int& epPieces, int& epLen) {
  torch::NoGradGuard no_grad;
  std::mt19937_64 rng(seed ^ (static_cast<uint64_t>(gi) * 0x9e3779b97f4a7c15ULL));

  tetris::Env env(cfg.reward);
  int64_t gameSeed = static_cast<int64_t>(rng());
  env.Reset(gameSeed);

  int steps = 0;
  double windowScore = 0.0;
  epLines = 0;
  epScore = 0;
  epPieces = 0;
  epLen = 0;
  size_t lastTransition = 0;
  bool hasTransition = false;

  while (!env.Done() && steps < cfg.maxStepsPerGame) {
    auto options = env.LegalOptions();
    if (options.empty()) break;

    Transition t;
    env.EncodeActorState(t.state);
    env.EncodeCriticState(t.criticState);
    // Unique across workers and games: no overflow, no hash collisions.
    t.gameId = gameIdBase + gi;
    t.actions.clear();
    t.actions.reserve(options.size());
    for (const auto& o : options) {
      t.actions.emplace_back(o.id, o.feature);
    }

    const int n = static_cast<int>(options.size());
    auto stateT = torch::from_blob(t.state.data(), {1, tetris::kStateSize},
                                   torch::kFloat32)
                      .clone();
    auto actFeat = torch::empty({n, tetris::kActionFeatSize}, torch::kFloat32);
    float* af = actFeat.data_ptr<float>();
    for (int a = 0; a < n; ++a) {
      std::memcpy(af + a * tetris::kActionFeatSize, options[a].feature.data(),
                  tetris::kActionFeatSize * sizeof(float));
    }
    auto sample = torch::zeros({n}, torch::kLong);
    auto actId = torch::arange(0, n, torch::kLong);

    torch::Tensor logits = (*actor)->forward(stateT, actFeat, sample, actId);
    std::vector<float> logitBuf(n), probBuf(n);
    auto row = logits[0];
    for (int a = 0; a < n; ++a) {
      logitBuf[a] = row[a].item<float>();
    }
    maskedSoftmax(logitBuf.data(), n, probBuf.data());

    float u = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
    float cum = 0.f;
    int chosen = n - 1;
    for (int a = 0; a < n; ++a) {
      cum += probBuf[a];
      if (u <= cum) {
        chosen = a;
        break;
      }
    }
    float p = std::max(probBuf[chosen], 1e-8f);
    t.action = options[chosen].id;
    t.logp = std::log(p);

    if (collectTransitions && critic) {
      auto critIn = torch::from_blob(t.criticState.data(),
                                     {1, tetris::kStateSize}, torch::kFloat32)
                        .clone();
      t.value = (*critic)->forward(critIn).squeeze().item<float>();
    }

    auto step = env.Step(options[chosen]);
    if (cfg.terminalScoring) {
      if (cfg.scoreWindow <= 0 ||
          step.pieces_placed <= cfg.scoreWindow) {
        windowScore += static_cast<double>(step.score_gain);
      }
      // The whole point of terminal scoring: no per-step score, otherwise a
      // high-scoring death could end up with a positive return.
      t.reward = 0.0f;
      t.terminal = step.done;
    } else {
      t.reward = static_cast<float>(step.reward);
      t.terminal = step.done;
    }
    if (collectTransitions) {
      lastTransition = out.size();
      out.push_back(std::move(t));
      hasTransition = true;
    }

    ++epLen;
    ++steps;
    if (step.done) break;
  }

  // Episode ended only because it hit maxStepsPerGame: bootstrap the tail with
  // V(s_T) instead of pretending the future is worth zero. In terminal-scoring
  // mode the capped episode instead receives its full positive return below,
  // so bootstrapping here would double-count its future.
  if (!cfg.terminalScoring && collectTransitions && critic && hasTransition &&
      !env.Done() && steps >= cfg.maxStepsPerGame) {
    std::array<float, tetris::kStateSize> tailState{};
    env.EncodeCriticState(tailState);
    auto tailT = torch::from_blob(tailState.data(),
                                  {1, tetris::kStateSize}, torch::kFloat32)
                     .clone();
    out[lastTransition].bootstrapValue =
        (*critic)->forward(tailT).squeeze().item<float>();
  }

  epLines = env.game().Lines();
  epScore = env.game().Score();
  epPieces = env.game().PiecesPlaced();

  if (cfg.terminalScoring && collectTransitions && hasTransition) {
    const double horizon =
        static_cast<double>(std::max(1, cfg.maxStepsPerGame));
    const double survivedFrac =
        std::clamp(static_cast<double>(epPieces) / horizon, 0.0, 1.0);
    auto& last = out[lastTransition];
    if (!env.Done() && steps >= cfg.maxStepsPerGame) {
      // Survived the entire window: strictly positive, and only now does
      // score enter the objective.
      last.reward += static_cast<float>(cfg.surviveBase +
                                        cfg.scoreScale * windowScore);
    } else {
      // Real death (or no legal placement left): strictly negative, but the
      // longer it survived the milder the penalty.
      last.reward -= static_cast<float>(cfg.deathBase * (1.0 - survivedFrac));
      last.terminal = true;
    }
    last.bootstrapValue = 0.0f;
  }
}

void runLearningWorker(const ModelSet& models, const RolloutConfig& cfg,
                       int nGames, uint64_t seed, int gameIdBase,
                       std::vector<Transition>& out, RolloutStats& stats) {
  for (int gi = 0; gi < nGames; ++gi) {
    int epLines = 0, epScore = 0, epPieces = 0, epLen = 0;
    playGame(models.actor, models.critic, cfg, seed, gameIdBase, gi,
             /*collect=*/true, out, epLines, epScore, epPieces, epLen);
    stats.games += 1;
    stats.totalLines += epLines;
    stats.totalScore += epScore;
    stats.totalPieces += epPieces;
    stats.transitions += epLen;
  }
}

// Sample a historical actor pointer (50/50 archive vs recent when both nonempty).
torch_nn::Actor* sampleHistoricalActor(const HistoricalPool& pool,
                                       std::mt19937_64& rng) {
  bool hasRecent = !pool.recent.empty();
  bool hasArchive = !pool.archive.empty();
  if (!hasRecent && !hasArchive) return nullptr;

  bool useArchive = false;
  if (hasRecent && hasArchive) {
    useArchive = std::uniform_real_distribution<float>(0.f, 1.f)(rng) < 0.5f;
  } else if (hasArchive) {
    useArchive = true;
  }

  if (useArchive) {
    int n = static_cast<int>(pool.archive.size());
    int idx = std::uniform_int_distribution<int>(0, n - 1)(rng);
    return &pool.archive[static_cast<size_t>(idx)]->actor;
  }
  int n = static_cast<int>(pool.recent.size());
  int idx = std::uniform_int_distribution<int>(0, n - 1)(rng);
  return &pool.recent[static_cast<size_t>(idx)]->actor;
}

void runLeagueWorker(const RolloutConfig& cfg, int nGames, uint64_t seed,
                     int gameIdBase, RolloutStats& stats) {
  if (!cfg.historicalPool || cfg.historicalPool->empty() || nGames <= 0) return;
  std::mt19937_64 rng(seed);
  std::vector<Transition> discard;
  for (int gi = 0; gi < nGames; ++gi) {
    torch_nn::Actor* hist = sampleHistoricalActor(*cfg.historicalPool, rng);
    if (!hist) continue;
    (*hist)->eval();
    int epLines = 0, epScore = 0, epPieces = 0, epLen = 0;
    playGame(hist, nullptr, cfg, seed + 17ULL, gameIdBase, gi,
             /*collect=*/false, discard, epLines, epScore, epPieces, epLen);
    stats.leagueGames += 1;
    stats.leagueTotalLines += epLines;
    stats.leagueTotalScore += epScore;
    (void)epPieces;
    (void)epLen;
  }
}

}  // namespace

void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::vector<Transition>& out, RolloutStats& stats) {
  out.clear();
  stats = RolloutStats{};
  if (!models.actor || !models.critic || cfg.gamesPerUpdate <= 0) return;

  (*models.actor)->eval();
  (*models.critic)->eval();

  int threads = std::max(1, cfg.threads);
  int per = cfg.gamesPerUpdate / threads;
  int rem = cfg.gamesPerUpdate % threads;

  std::vector<std::vector<Transition>> parts(threads);
  std::vector<RolloutStats> partStats(threads);
  std::vector<std::thread> workers;
  workers.reserve(threads);

  int offset = 0;
  for (int t = 0; t < threads; ++t) {
    int n = per + (t < rem ? 1 : 0);
    uint64_t seed = cfg.seed + static_cast<uint64_t>(offset) * 9973ULL +
                    static_cast<uint64_t>(t) * 0x9e3779b97f4a7c15ULL;
    offset += n;
    int gameIdBase = t * (cfg.gamesPerUpdate + 1);
    workers.emplace_back([&, t, n, seed, gameIdBase]() {
      runLearningWorker(models, cfg, n, seed, gameIdBase, parts[t],
                        partStats[t]);
    });
  }
  for (auto& w : workers) w.join();

  for (int t = 0; t < threads; ++t) {
    out.insert(out.end(), parts[t].begin(), parts[t].end());
    stats.games += partStats[t].games;
    stats.totalLines += partStats[t].totalLines;
    stats.totalScore += partStats[t].totalScore;
    stats.totalPieces += partStats[t].totalPieces;
    stats.transitions += partStats[t].transitions;
  }
  if (stats.games > 0) {
    stats.meanLines = double(stats.totalLines) / stats.games;
    stats.meanScore = double(stats.totalScore) / stats.games;
    stats.meanEpisodeLen = double(stats.transitions) / stats.games;
  }

  // Extra league-eval games (not learning buffer).
  int leagueN = 0;
  if (cfg.historicalPool && !cfg.historicalPool->empty() &&
      cfg.historicalProb > 0.f) {
    leagueN = static_cast<int>(
        std::lround(static_cast<double>(cfg.gamesPerUpdate) *
                    static_cast<double>(cfg.historicalProb)));
  }
  if (leagueN > 0) {
    int lPer = leagueN / threads;
    int lRem = leagueN % threads;
    std::vector<RolloutStats> leagueParts(threads);
    std::vector<std::thread> leagueWorkers;
    leagueWorkers.reserve(threads);
    int loff = 0;
    for (int t = 0; t < threads; ++t) {
      int n = lPer + (t < lRem ? 1 : 0);
      uint64_t seed = cfg.seed + 0xC0FFEEULL +
                      static_cast<uint64_t>(loff) * 7919ULL +
                      static_cast<uint64_t>(t) * 0x85ebca6bULL;
      loff += n;
      int leagueIdBase = 100000000 + t * (leagueN + 1);
      leagueWorkers.emplace_back([&, t, n, seed, leagueIdBase]() {
        runLeagueWorker(cfg, n, seed, leagueIdBase, leagueParts[t]);
      });
    }
    for (auto& w : leagueWorkers) w.join();
    for (int t = 0; t < threads; ++t) {
      stats.leagueGames += leagueParts[t].leagueGames;
      stats.leagueTotalLines += leagueParts[t].leagueTotalLines;
      stats.leagueTotalScore += leagueParts[t].leagueTotalScore;
    }
    if (stats.leagueGames > 0) {
      stats.leagueMeanLines =
          double(stats.leagueTotalLines) / stats.leagueGames;
      stats.leagueMeanScore =
          double(stats.leagueTotalScore) / stats.leagueGames;
    }
  }

}

}  // namespace algo
