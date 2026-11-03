#include "tetris/bag.h"

namespace tetris {

Bag7::Bag7(int64_t seed) : rng_(seed) { Refill(); }

void Bag7::Refill() {
  bag_ = {PieceType::I, PieceType::L, PieceType::J, PieceType::T,
          PieceType::O, PieceType::S, PieceType::Z};
  rng_.Shuffle(7, [this](int i, int j) {
    PieceType tmp = bag_[static_cast<size_t>(i)];
    bag_[static_cast<size_t>(i)] = bag_[static_cast<size_t>(j)];
    bag_[static_cast<size_t>(j)] = tmp;
  });
  next_ = 0;
}

PieceType Bag7::Next() {
  if (next_ >= 7) Refill();
  return bag_[static_cast<size_t>(next_++)];
}

std::vector<PieceType> Bag7::PeekRemaining() const {
  std::vector<PieceType> out;
  for (int i = next_; i < 7; ++i) {
    out.push_back(bag_[static_cast<size_t>(i)]);
  }
  return out;
}

}  // namespace tetris
