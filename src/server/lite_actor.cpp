#include "server/lite_actor.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tsrv {
namespace {

constexpr char kMagic[8] = {'T', 'E', 'T', 'R', 'A', 'C', 'T', '1'};
constexpr int kHeaderInts = 6 + 8;  // 6 scalars + 8 tensor element counts

bool readExact(std::ifstream& f, void* dst, std::size_t bytes) {
  f.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  return f.gcount() == static_cast<std::streamsize>(bytes);
}

inline float relu(float x) { return x > 0.f ? x : 0.f; }

// Dense layer: y[h] = bias[h] + dot(w[h * cols, ...], x[0..cols))
inline void dense(const std::vector<float>& w, const std::vector<float>& bias,
                  int rows, int cols, const float* x, float* y) {
  for (int h = 0; h < rows; ++h) {
    const float* row = w.data() + static_cast<std::size_t>(h) * cols;
    float sum = bias[static_cast<std::size_t>(h)];
    for (int j = 0; j < cols; ++j) sum += row[j] * x[j];
    y[h] = sum;
  }
}

}  // namespace

bool LiteActor::load(const std::string& path, std::string& message) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    message = "无法打开 " + path;
    return false;
  }

  char magic[8];
  int32_t hdr[kHeaderInts];
  if (!readExact(f, magic, sizeof(magic)) ||
      !readExact(f, hdr, sizeof(hdr))) {
    message = "权重头不完整: " + path;
    return false;
  }
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    message = "权重格式不符（magic 错误）: " + path;
    return false;
  }

  const int32_t hidden = hdr[0];
  const int32_t stateSize = hdr[1];
  const int32_t actionSize = hdr[2];
  const int32_t inDim = hdr[4];
  const int32_t nTensors = hdr[5];
  if (hidden <= 0 || hidden > 4096 ||
      stateSize != tetris::kStateSize ||
      actionSize != tetris::kActionFeatSize ||
      inDim != stateSize + actionSize || nTensors != 8) {
    message = "权重维度与引擎不匹配: " + path;
    return false;
  }

  const int32_t h = hidden;
  const int32_t expected[8] = {h * inDim, h,     h * h, h,
                               h * h,     h,     h,     1};
  int64_t total = 0;
  for (int i = 0; i < 8; ++i) {
    if (hdr[6 + i] != expected[i]) {
      message = "权重张量 " + std::to_string(i) + " 长度不符: " + path;
      return false;
    }
    total += hdr[6 + i];
  }

  std::vector<float> data(static_cast<std::size_t>(total));
  if (!readExact(f, data.data(), data.size() * sizeof(float))) {
    message = "权重数据截断: " + path;
    return false;
  }

  // Slice the flat buffer into the eight named tensors (row-major).
  const float* p = data.data();
  auto take = [&p](int64_t count) {
    std::vector<float> v(p, p + count);
    p += count;
    return v;
  };
  l1w_ = take(expected[0]);
  l1b_ = take(expected[1]);
  l2w_ = take(expected[2]);
  l2b_ = take(expected[3]);
  l3w_ = take(expected[4]);
  l3b_ = take(expected[5]);
  headw_ = take(expected[6]);
  headb_ = data.back();  // expected[7] == 1
  hidden_ = h;

  std::ostringstream oss;
  oss << "已加载 " << path << " (lightweight Actor, hidden=" << h
      << ", " << total << " float32)";
  message = oss.str();
  return true;
}

float LiteActor::score(
    const std::array<float, tetris::kStateSize>& state,
    const std::array<float, tetris::kActionFeatSize>& action) const {
  float out = 0.f;
  if (loaded()) scoreBatch(state, action.data(), 1, &out);
  return out;
}

void LiteActor::scoreBatch(
    const std::array<float, tetris::kStateSize>& state,
    const float* actions, int n, float* out) const {
  if (!loaded() || n <= 0) return;
  const int h = hidden_;
  const int inDim = tetris::kStateSize + tetris::kActionFeatSize;

  // l1 is decomposed: the state part depends only on the board, so compute it
  // once and add each candidate's 18-dim action part per option.
  std::vector<float> stateProj(h);
  for (int i = 0; i < h; ++i) {
    const float* row = l1w_.data() + static_cast<std::size_t>(i) * inDim;
    float sum = l1b_[static_cast<std::size_t>(i)];
    for (int j = 0; j < tetris::kStateSize; ++j) sum += row[j] * state[j];
    stateProj[i] = sum;
  }

  std::vector<float> f1(h), f2(h), f3(h);
  for (int a = 0; a < n; ++a) {
    const float* act = actions + static_cast<std::size_t>(a) *
                                     tetris::kActionFeatSize;
    for (int i = 0; i < h; ++i) {
      const float* row = l1w_.data() + static_cast<std::size_t>(i) * inDim +
                         tetris::kStateSize;
      float sum = stateProj[i];
      for (int j = 0; j < tetris::kActionFeatSize; ++j) sum += row[j] * act[j];
      f1[i] = relu(sum);
    }
    dense(l2w_, l2b_, h, h, f1.data(), f2.data());
    for (int i = 0; i < h; ++i) f2[i] = relu(f2[i]);
    dense(l3w_, l3b_, h, h, f2.data(), f3.data());
    for (int i = 0; i < h; ++i) f3[i] = relu(f3[i]);
    float logit = headb_;
    for (int i = 0; i < h; ++i) logit += headw_[i] * f3[i];
    out[a] = logit;
  }
}

}  // namespace tsrv
