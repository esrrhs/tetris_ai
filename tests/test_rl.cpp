#include <gtest/gtest.h>

#include "algo/ppo.h"
#include "algo/rollout.h"
#include "tetris/env.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"

TEST(TorchRL, ActorCriticForward) {
  torch_nn::ModelConfig cfg;
  cfg.hidden = 32;
  torch_nn::Actor actor(cfg);
  torch_nn::Critic critic(cfg);

  int64_t B = 2;
  int64_t L = 6;
  auto state = torch::randn({B, torch_nn::kStateInput});
  auto actFeat = torch::randn({L, torch_nn::kActionInput});
  auto sample = torch::tensor({0, 0, 0, 1, 1, 1}, torch::kLong);
  auto actId = torch::tensor({0, 1, 2, 0, 1, 2}, torch::kLong);

  auto logits = actor->forward(state, actFeat, sample, actId);
  ASSERT_EQ(logits.size(0), B);
  ASSERT_EQ(logits.size(1), torch_nn::kMaxActions);

  auto val = critic->forward(state);
  ASSERT_EQ(val.size(0), B);
  ASSERT_EQ(val.size(1), 1);

  logits.sum().backward();
  ASSERT_TRUE(actor->l1->weight.grad().defined());
}

TEST(TorchRL, TinyPPOUpdate) {
  torch_nn::ModelConfig cfg;
  cfg.hidden = 32;
  torch_nn::Actor actor(cfg);
  torch_nn::Critic critic(cfg);

  torch_nn::PPOConfig pcfg;
  pcfg.epochs = 1;
  pcfg.minibatch = 4;
  pcfg.targetKL = 0;
  torch_nn::PPOLearner learner(actor, critic, pcfg);

  // Collect a few real env transitions with random policy via collectRollout.
  algo::ModelSet models{&actor, &critic};
  algo::RolloutConfig rcfg;
  rcfg.gamesPerUpdate = 2;
  rcfg.threads = 1;
  rcfg.seed = 3;
  rcfg.maxStepsPerGame = 20;

  std::vector<algo::Transition> tr;
  algo::RolloutStats stats;
  algo::collectRollout(models, rcfg, tr, stats);
  ASSERT_FALSE(tr.empty());

  auto batch = torch_nn::transitionsToBatch(tr, 0.99f, 0.95f);
  auto st = learner.update(batch);
  ASSERT_GE(st.epochsCompleted, 1);
  ASSERT_TRUE(std::isfinite(st.pgLoss));
  ASSERT_TRUE(std::isfinite(st.vLoss));
}

TEST(Algo, MaskedSoftmaxAndGAE) {
  float logits[3] = {1.f, 2.f, -1e9f};
  float probs[3];
  algo::maskedSoftmax(logits, 3, probs);
  EXPECT_NEAR(probs[0] + probs[1] + probs[2], 1.0, 1e-5);
  EXPECT_NEAR(probs[2], 0.0, 1e-6);

  std::vector<algo::Transition> tr(2);
  tr[0].gameId = 1;
  tr[0].reward = 1.f;
  tr[0].value = 0.5f;
  tr[0].terminal = false;
  tr[1].gameId = 1;
  tr[1].reward = 0.f;
  tr[1].value = 0.2f;
  tr[1].terminal = true;
  algo::assignEpisodeReturns(tr, 0.99f, 0.95f);
  EXPECT_TRUE(std::isfinite(tr[0].adv));
  EXPECT_TRUE(std::isfinite(tr[1].ret));
}
