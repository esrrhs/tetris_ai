// Go math/rand–compatible PRNG (Mitchell–Reeds lagged Fibonacci) so 7-bag
// sequences match github.com/esrrhs/tetris_ai/game for the same seed.
#pragma once

#include <cstdint>

#include "tetris/rng_cooked.h"

namespace tetris {
namespace go_rand {

constexpr int kRngTap = 273;
constexpr uint64_t kRngMask = (uint64_t{1} << 63) - 1;
constexpr int32_t kInt32Max = 0x7fffffff;

inline int32_t seedrand(int32_t x) {
  constexpr int32_t A = 48271;
  constexpr int32_t Q = 44488;
  constexpr int32_t R = 3399;
  int32_t hi = x / Q;
  int32_t lo = x % Q;
  x = A * lo - R * hi;
  if (x < 0) x += kInt32Max;
  return x;
}

// Mirrors math/rand.rngSource + Rand.Int63 / Uint32 / int31n / Shuffle.
class Source {
 public:
  explicit Source(int64_t seed) { Seed(seed); }

  void Seed(int64_t seed) {
    tap_ = 0;
    feed_ = kRngLen - kRngTap;

    seed = seed % kInt32Max;
    if (seed < 0) seed += kInt32Max;
    if (seed == 0) seed = 89482311;

    int32_t x = static_cast<int32_t>(seed);
    for (int i = -20; i < kRngLen; ++i) {
      x = seedrand(x);
      if (i >= 0) {
        int64_t u = static_cast<int64_t>(x) << 40;
        x = seedrand(x);
        u ^= static_cast<int64_t>(x) << 20;
        x = seedrand(x);
        u ^= static_cast<int64_t>(x);
        u ^= kRngCooked[i];
        vec_[i] = u;
      }
    }
  }

  uint64_t Uint64() {
    --tap_;
    if (tap_ < 0) tap_ += kRngLen;
    --feed_;
    if (feed_ < 0) feed_ += kRngLen;
    int64_t x = vec_[feed_] + vec_[tap_];
    vec_[feed_] = x;
    return static_cast<uint64_t>(x);
  }

  int64_t Int63() { return static_cast<int64_t>(Uint64() & kRngMask); }

  uint32_t Uint32() { return static_cast<uint32_t>(Int63() >> 31); }

  // Fast int31n used by Shuffle for n < 2^31 (Go math/rand.int31n).
  int32_t Int31nFast(int32_t n) {
    uint32_t v = Uint32();
    uint64_t prod = static_cast<uint64_t>(v) * static_cast<uint64_t>(n);
    uint32_t low = static_cast<uint32_t>(prod);
    if (low < static_cast<uint32_t>(n)) {
      uint32_t thresh = static_cast<uint32_t>(-n) % static_cast<uint32_t>(n);
      while (low < thresh) {
        v = Uint32();
        prod = static_cast<uint64_t>(v) * static_cast<uint64_t>(n);
        low = static_cast<uint32_t>(prod);
      }
    }
    return static_cast<int32_t>(prod >> 32);
  }

  // Fisher–Yates as in math/rand.Rand.Shuffle for small n (n <= 2^31).
  template <typename SwapFn>
  void Shuffle(int n, SwapFn swap) {
    for (int i = n - 1; i > 0; --i) {
      int j = static_cast<int>(Int31nFast(static_cast<int32_t>(i + 1)));
      swap(i, j);
    }
  }

 private:
  int tap_ = 0;
  int feed_ = 0;
  int64_t vec_[kRngLen]{};
};

}  // namespace go_rand
}  // namespace tetris
