#include "tetris/bag.h"

namespace tetris {

Bag7::Bag7(int64_t seed) : rng_(seed) {
  EnsureQueue();
  curBag_ = 7;  // first queued bag is the current one
}

void Bag7::EnsureQueue() {
  // Keep one full upcoming bag queued (≥7) so PeekNext(7) is always valid.
  // Appending a bag never reorders or consumes existing queue entries.
  while (static_cast<int>(queue_.size()) < 7) {
    std::array<PieceType, 7> bag = {PieceType::I, PieceType::L, PieceType::J,
                                    PieceType::T, PieceType::O, PieceType::S,
                                    PieceType::Z};
    rng_.Shuffle(7, [&bag](int i, int j) {
      PieceType tmp = bag[static_cast<size_t>(i)];
      bag[static_cast<size_t>(i)] = bag[static_cast<size_t>(j)];
      bag[static_cast<size_t>(j)] = tmp;
    });
    for (PieceType t : bag) queue_.push_back(t);
  }
}

PieceType Bag7::Next() {
  EnsureQueue();
  PieceType t = queue_.front();
  queue_.pop_front();
  if (curBag_ > 0) {
    --curBag_;
  } else {
    // Popped the first piece of the next bag → it becomes the current bag.
    // Invariant: with ≥7 queued pre-pop, the front starts a full bag, so
    // exactly 6 pieces of it remain queued behind the popped one.
    curBag_ = 6;
  }
  return t;
}

std::vector<PieceType> Bag7::PeekRemaining() const {
  std::vector<PieceType> out;
  int n = std::min(curBag_, static_cast<int>(queue_.size()));
  for (int i = 0; i < n; ++i) {
    out.push_back(queue_[static_cast<size_t>(i)]);
  }
  return out;
}

std::vector<PieceType> Bag7::PeekNext(int n) const {
  std::vector<PieceType> out;
  int count = std::min(n, static_cast<int>(queue_.size()));
  for (int i = 0; i < count; ++i) {
    out.push_back(queue_[static_cast<size_t>(i)]);
  }
  return out;
}

}  // namespace tetris
