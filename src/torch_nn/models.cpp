#include "torch_nn/models.h"

namespace torch_nn {

ActorImpl::ActorImpl(const ModelConfig& cfg) : config(cfg) {
  int64_t jointDim = kStateInput + kActionInput;
  l1 = register_module("l1", torch::nn::Linear(jointDim, cfg.hidden));
  l2 = register_module("l2", torch::nn::Linear(cfg.hidden, cfg.hidden));
  l3 = register_module("l3", torch::nn::Linear(cfg.hidden, cfg.hidden));
  head = register_module("head", torch::nn::Linear(cfg.hidden, 1));

  torch::nn::init::zeros_(head->weight);
  torch::nn::init::zeros_(head->bias);
}

torch::Tensor ActorImpl::forward(const torch::Tensor& state,
                                 const torch::Tensor& actionFeat,
                                 const torch::Tensor& actionSample,
                                 const torch::Tensor& actionId) const {
  // Decomposed first layer: avoid replicating state L times.
  torch::Tensor W_state = l1->weight.slice(1, 0, kStateInput);
  torch::Tensor W_act = l1->weight.slice(1, kStateInput);

  torch::Tensor stateProj = torch::addmm(l1->bias, state, W_state.t());
  torch::Tensor actProj = torch::mm(actionFeat, W_act.t());

  torch::Tensor sampleIndices = actionSample.to(torch::kLong);
  torch::Tensor f1 =
      torch::relu(stateProj.index_select(0, sampleIndices) + actProj);
  torch::Tensor f2 = torch::relu(l2->forward(f1));
  torch::Tensor f3 = torch::relu(l3->forward(f2));
  torch::Tensor scores = head->forward(f3).squeeze(-1);  // [L]

  int64_t B = state.size(0);
  torch::Tensor logits = torch::full(
      {B, kMaxActions}, -1e9f,
      torch::TensorOptions().dtype(torch::kFloat32).device(state.device()));
  torch::Tensor aIndices = actionId.to(torch::kLong);
  logits.index_put_({sampleIndices, aIndices}, scores);
  return logits;
}

void ActorImpl::save(const std::string& path) const {
  torch::serialize::OutputArchive archive;
  torch::nn::Module::save(archive);
  archive.save_to(path);
}

void ActorImpl::load(const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  torch::nn::Module::load(archive);
}

CriticImpl::CriticImpl(const ModelConfig& cfg) : config(cfg) {
  c1 = register_module("c1", torch::nn::Linear(kStateInput, cfg.hidden));
  c2 = register_module("c2", torch::nn::Linear(cfg.hidden, cfg.hidden));
  c3 = register_module("c3", torch::nn::Linear(cfg.hidden, cfg.hidden));
  out = register_module("out", torch::nn::Linear(cfg.hidden, 1));

  torch::nn::init::zeros_(out->weight);
  torch::nn::init::zeros_(out->bias);
}

torch::Tensor CriticImpl::forward(const torch::Tensor& state) const {
  torch::Tensor q1 = torch::relu(c1->forward(state));
  torch::Tensor q2 = torch::relu(c2->forward(q1));
  torch::Tensor q3 = torch::relu(c3->forward(q2));
  return out->forward(q3);  // [B, 1]
}

void CriticImpl::save(const std::string& path) const {
  torch::serialize::OutputArchive archive;
  torch::nn::Module::save(archive);
  archive.save_to(path);
}

void CriticImpl::load(const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  torch::nn::Module::load(archive);
}

}  // namespace torch_nn
