package game

import "math/rand"

// Bag7 is a 7-bag (random bag) piece generator: each bag is a random
// permutation of all seven tetrominoes; bags are reshuffled when empty.
type Bag7 struct {
	rng  *rand.Rand
	bag  []PieceType
	next int
}

func NewBag7(seed int64) *Bag7 {
	b := &Bag7{rng: rand.New(rand.NewSource(seed))}
	b.refill()
	return b
}

func (b *Bag7) refill() {
	b.bag = []PieceType{PieceI, PieceL, PieceJ, PieceT, PieceO, PieceS, PieceZ}
	b.rng.Shuffle(len(b.bag), func(i, j int) {
		b.bag[i], b.bag[j] = b.bag[j], b.bag[i]
	})
	b.next = 0
}

// Next returns the next piece type.
func (b *Bag7) Next() PieceType {
	if b.next >= len(b.bag) {
		b.refill()
	}
	t := b.bag[b.next]
	b.next++
	return t
}
