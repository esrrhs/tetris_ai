# tetris_ai

[![Language](https://img.shields.io/badge/language-C%2B%2B20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/esrrhs/tetris_ai)

[**English**](README.md) | [**中文说明**](README_zh.md)

Tetris **reinforcement-learning** play client and PPO scaffold. The clock advances only via `Step`. **C++ is the sole core**, styled like [PerfectDouCPP](https://github.com/esrrhs/PerfectDouCPP).

Dedicated repo bot: `tetris_ai`.

## Step semantics

1. Game time advances **only** when `Step(ops...)` is called.
2. Piece generation uses a **7-bag** (each bag holds one of each of the 7 pieces, shuffled, then drawn in order). A deterministic seeded PRNG is used, so the **same seed yields the same piece sequence**.
3. When there is **no** active piece: this `Step` first **spawns** the next piece at the top; if the spawn overlaps locked cells → `Done=true` (game over). Otherwise: apply `ops` to the new piece, then try to **drop 1 row**.
4. When there **is** an active piece: apply any number of `ops` in order, then gravity drops **exactly 1 row** (if the ops include `HardDrop`, the piece locks at the bottom and normal gravity is skipped).
5. Ops only take effect inside `Step`; illegal moves (wall / overlap) are **ignored**, not errors.
6. When a piece can no longer fall it **locks** and lines clear; the **next** `Step` (not the same one) spawns the following piece.

Allowed ops: `Left` / `Right` / `RotateCW` / `RotateCCW` / (optional) `HardDrop`.

## Board & rotation

- Standard **10×20** visible board, **no** hidden rows (Y=0 is the top, increasing downward).
- Seven pieces I,L,J,T,O,S,Z; fixed 4-state shape tables + **basic wall kicks** (try offsets `(0,0), (±1,0), (0,-1), (±2,0)`), not full SRS.
- Low-level `Game::Step` `Reward` = lines cleared this step (raw API). Accumulated **`Score`** matches classic scoring:
  - Base line score `LineScore`: 1/2/3/4 lines → **40 / 100 / 300 / 1200**
  - **Combo bonus** `ComboBonus`: consecutive locks that clear lines keep a streak. First clear has combo index=0 (no bonus); later consecutive clears use index=1,2,… with bonus **40 × index** (+40, +80, …). A lock with no clears resets the combo.
  - Score gain on this lock = `LineScore(cleared) + ComboBonus(prev_combo)`.
- RL `Env` clear reward aligns with that score gain (encourages combo / multi-line clears): `lineCoef * (score_gain / 40)`, so a single clear ≈ `lineCoef * 1.0`; dense shaping and `deathPenalty` still stack on top.

## C++ API

```cpp
#include "tetris/game.h"

tetris::Game g(seed);  // 7-bag RNG seeded
tetris::Result r = g.Step({tetris::Op::Left, tetris::Op::RotateCW});
// r.obs     Observation (board + active piece)
// r.reward  double      lines cleared this step (raw Game API)
// r.done    bool        game over
// r.info    locked / spawned / lines_cleared / score / score_gain / combo / ...

auto obs = g.Observe();
std::cout << g.Render(true);  // ASCII
```

### Build core & tests (no LibTorch required)

```bash
export PATH="/home/box/.local/cmake/bin:$PATH"
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tetris_ai_demo          # seed=42, 40 steps
./build/tetris_ai_demo 123 60   # seed=123, 60 steps
```

CMake targets (always): static lib `tetris_ai_core`, demo `tetris_ai_demo`, gtests `test_game` + `test_rl_env`.

## RL scaffold (PerfectDou-style)

Single-agent self-play vs environment (v1: **no** 3-seat / multi-agent league).

### Action space — placement options

Each agent step picks a **final placement** `(rot, x)` for the active piece: rotate/translate then `HardDrop`, executed inside **one** `Game::Step(ops)`. `LegalOptions()` enumerates all reachable options that can hard-drop and lock (O placements are deduplicated).

`Env` (`src/tetris/env.*`):

- `Reset(seed)`: `Step({})` until an active piece exists, then return the observation.
- `Step(placement_id | LegalOption)`: apply ops+HardDrop; if not done after lock, automatically `Step({})` to **respawn the next piece**, so the agent always sees a state with an active piece (or Done). Reward = `lineCoef * (score_gain/40)` + optional potential dense shaping (below).
- `LegalOptions()`: legal placements in the current state.

### Feature dims

| Constant | Value | Contents |
|----------|------:|----------|
| `kStateSize` | **235** | board 20×10 binary (200) + piece one-hot(7) + rot one-hot(4) + x/y (2) + heights(10) + holes/bump/agg (3) + lines/score (2) + bag remaining counts(7) |
| `kActionFeatSize` | **18** | post-lock heights(10) + holes/bump/agg (3) + lines cleared + landing y + Δholes/Δbump/Δagg (3) |

- **Actor** uses imperfect state: bag dims are all 0 (no peeking into the 7-bag).
- **Critic** may include `Bag7::PeekRemaining()` counts (remaining in the current bag; empty at bag end until the next `Next` refill).

### Models / Algo

- `Actor`: `joint = concat(state, actionFeat)` → MLP → scalar logit; masked softmax over legal options (see `algo/ppo.cpp`).
- `Critic`: state → MLP → value.
- `PPOLearner::update`: clip, entropy, value loss, grad clip, optional KL early stop.
- `collectRollout`: sync thread pool collects Env episodes + GAE (`assignEpisodeReturns`); optional historical league eval games.

### Dense (potential) reward

Tetris analogue of PerfectDou’s oracle shaping — **potential-based shaping on board metrics**, not DouZero card-oracle:

```
φ = wH * aggregate_height + wHole * holes + wBump * bumpiness
score_gain = LineScore(lines) + ComboBonus(combo_index)   # same as Game score
reward = lineCoef * (score_gain / 40.0) + denseCoef * (φ_before - φ_after)
       + (done ? deathPenalty : 0)
```

Examples (`lineCoef=1`, no dense): single=1.0, double=2.5, triple=7.5, tetris=30.0; combo follow-up single=2.0 (40+40)/40.

Defaults: `wH=1`, `wHole=1`, `wBump=0.5`, `lineCoef=1`, `denseCoef=0.01`, `deathPenalty=-1`, `denseReward=true`.

CLI: `--dense-reward` / `--no-dense-reward`, `--dense-coef`, `--line-coef`, `--death-penalty`, `--w-height`, `--w-hole`, `--w-bump`.

### Historical league pool (single-agent)

PerfectDou-style `HistoricalPool` with `recent` (ring) + `archive` (geometric / centennial promotion).

- Each update collects exactly `--games` **learning** games with the **current** CPU actor → PPO buffer.
- When the pool is non-empty, also plays `round(games * historicalProb)` **extra league-eval** games with a randomly sampled historical actor (50/50 archive vs recent). Those transitions are **not** written to the learning buffer; only `leagueGames` / `leagueMeanLines` / `leagueMeanScore` are logged.
- After each update (every `--pool-every`), push the current CPU actor into `recent`; periodically also copy into `archive`.
- Saved under `--out/pool/` as `recent_*.pt`, `archive_*.pt`, and `pool_meta.txt`. `--resume DIR` reloads actor/critic/optim + pool.

CLI: `--historical-prob`, `--pool-size`, `--archive-size`, `--pool-every`, `--resume`.

### Backend: CUDA / MPS / CPU

`--backend cuda|mps|cpu` is **explicit** (no silent auto-detect):

- `cuda`: requires `torch::cuda::is_available()`, else fatal.
- `mps`: requires Apple + MPS-enabled LibTorch (`torch::mps::is_available()`), else fatal.
- `cpu`: always available.

PPO learner + batch tensors live on the selected device. **Rollout inference always uses CPU actor/critic copies** (thread safety / no GPU contention). After each `learner.update`, trained weights are synced back to the CPU rollout modules.

This environment’s packaged LibTorch is **CPU-only**; `--backend cuda` needs a CUDA LibTorch build + GPU. The code path still compiles and selects correctly.

### Build training (LibTorch)

Requires LibTorch (e.g. 2.5.1 cxx11 ABI) and `CMAKE_PREFIX_PATH`:

```bash
export PATH="/home/box/.local/cmake/bin:$PATH"
cmake -S . -B build -DCMAKE_PREFIX_PATH=/workspace/deps/libtorch
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tetris_ai_train --updates 3 --games 8 --threads 2 \
  --pool-every 1 --pool-size 4 --archive-size 2 --historical-prob 0.5 \
  --dense-reward --out /tmp/tetris_ckpt2 --backend cpu
```

CLI (main): `--updates` `--games` `--threads` `--hidden` `--lr` `--epochs` `--mb` `--gamma` `--lambda` `--seed` `--out` `--backend` `--resume` plus dense-reward and league-pool flags above.

CMake: `TETRIS_AI_WITH_TORCH` (ON when Torch is found) builds `tetris_ai_rl`, `tetris_ai_train`, `tetris_ai_eval`, and optional `test_rl`. Without Torch, the core plus `test_game` / `test_rl_env` still build.

### External eval (built-in Env client)

**Purpose:** measure **score** (and lines) after at most **N** pieces; **stop early on death**. Loads `actor.pt` and plays greedy (default) games on the built-in `Env`/`Game` — no external game service. Typical N: `1000` / `10000`.

- Reach `pieces_placed >= N` without dying → report that game’s `score` / `lines`, `survived=true`.
- Die earlier → stop and report score / lines / `pieces_placed` at death, `survived=false`, `death_piece=K+1` (spawn of piece K+1 failed after locking K pieces).

```bash
export PATH="/home/box/.local/cmake/bin:$PATH"
cmake -S . -B build -DCMAKE_PREFIX_PATH=/workspace/deps/libtorch
cmake --build build -j --target tetris_ai_eval
./build/tetris_ai_eval --ckpt /tmp/tetris_ckpt2 --max-pieces 1000 --games 5 --seed 1 --backend cpu
./build/tetris_ai_eval --ckpt /tmp/tetris_ckpt2 --max-pieces 10000 --games 3 --seed 1 --backend cpu
# optional: --stochastic  --hidden 128
```

One line per game (score first): `score=... lines=... pieces_placed=... seed=... survived=... death_piece=...`  
Ends with `summary mean_score=... mean_lines=... mean_pieces=...`.

CMake (needs LibTorch): extra target `tetris_ai_eval`.

## Layout

```
tetris_ai/
  CMakeLists.txt
  README.md
  README_zh.md
  src/tetris/     # core + features + env
  src/algo/       # PPO GAE / rollout (Torch)
  src/torch_nn/   # Actor Critic PPOLearner tensor_bridge
  src/train.cpp   # training entry
  src/eval.cpp    # external eval entry
  src/demo/       # ASCII demo
  tests/          # gtest
```

## Notes

- C++ core layout mirrors PerfectDouCPP: `src/<domain>/` + `algo/` + `torch_nn/` + CMake + gtest.
- Rotation is intentionally simple for RL; full SRS / 20+2 buffer rows can be added later if needed.
