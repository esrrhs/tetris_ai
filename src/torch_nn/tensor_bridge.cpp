#include "torch_nn/tensor_bridge.h"

#include "algo/ppo.h"

#include <cmath>
#include <cstring>

namespace torch_nn {

BatchTensors transitionsToBatch(const std::vector<algo::Transition>& trInput,
                                float gamma, float lambda,
                                torch::Device device) {
  int64_t B = static_cast<int64_t>(trInput.size());
  BatchTensors bt;
  if (B == 0) return bt;

  std::vector<algo::Transition> tr = trInput;
  algo::assignEpisodeReturns(tr, gamma, lambda);

  double meanAdv = 0.0, varAdv = 0.0;
  for (const auto& t : tr) meanAdv += t.adv;
  meanAdv /= B;
  for (const auto& t : tr)
    varAdv += (t.adv - float(meanAdv)) * (t.adv - float(meanAdv));
  float stdAdv = float(std::sqrt(varAdv / B + 1e-8));
  for (auto& t : tr) t.adv = (t.adv - float(meanAdv)) / stdAdv;

  int64_t L = 0;
  for (const auto& t : tr) L += static_cast<int64_t>(t.actions.size());

  bt.state = torch::empty({B, kStateInput}, torch::kFloat32);
  bt.criticState = torch::empty({B, kStateInput}, torch::kFloat32);
  bt.actions = torch::empty({B}, torch::kLong);
  bt.oldLogp = torch::empty({B}, torch::kFloat32);
  bt.adv = torch::empty({B}, torch::kFloat32);
  bt.ret = torch::empty({B}, torch::kFloat32);
  bt.oldValue = torch::empty({B}, torch::kFloat32);
  bt.actionFeat = torch::empty({L, kActionInput}, torch::kFloat32);
  bt.actionSample = torch::empty({L}, torch::kLong);
  bt.actionId = torch::empty({L}, torch::kLong);

  float* pState = bt.state.data_ptr<float>();
  float* pCrit = bt.criticState.data_ptr<float>();
  int64_t* pActions = bt.actions.data_ptr<int64_t>();
  float* pOldLogp = bt.oldLogp.data_ptr<float>();
  float* pAdv = bt.adv.data_ptr<float>();
  float* pRet = bt.ret.data_ptr<float>();
  float* pOldValue = bt.oldValue.data_ptr<float>();
  float* pActFeat = bt.actionFeat.data_ptr<float>();
  int64_t* pActSample = bt.actionSample.data_ptr<int64_t>();
  int64_t* pActId = bt.actionId.data_ptr<int64_t>();

  int64_t actOffset = 0;
  for (int64_t i = 0; i < B; ++i) {
    const auto& t = tr[i];
    std::memcpy(pState + i * kStateInput, t.state.data(),
                kStateInput * sizeof(float));
    std::memcpy(pCrit + i * kStateInput, t.criticState.data(),
                kStateInput * sizeof(float));
    pActions[i] = t.action;
    pOldLogp[i] = t.logp;
    pAdv[i] = t.adv;
    pRet[i] = t.ret;
    pOldValue[i] = t.value;

    for (const auto& [aid, afeat] : t.actions) {
      std::memcpy(pActFeat + actOffset * kActionInput, afeat.data(),
                  kActionInput * sizeof(float));
      pActSample[actOffset] = i;
      pActId[actOffset] = aid;
      ++actOffset;
    }
  }

  if (device != torch::kCPU) {
    // Blocking H2D copies: pageable-memory non_blocking copies to MPS can race
    // (staging buffer reuse) and deliver garbage tensors on some LibTorch builds.
    bt.state = bt.state.to(device);
    bt.criticState = bt.criticState.to(device);
    bt.actions = bt.actions.to(device);
    bt.oldLogp = bt.oldLogp.to(device);
    bt.adv = bt.adv.to(device);
    bt.ret = bt.ret.to(device);
    bt.oldValue = bt.oldValue.to(device);
    bt.actionFeat = bt.actionFeat.to(device);
    bt.actionSample = bt.actionSample.to(device);
    bt.actionId = bt.actionId.to(device);
  }
  return bt;
}

}  // namespace torch_nn
