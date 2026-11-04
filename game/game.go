package game

// Game is a step-based Tetris client for RL.
//
// Step semantics (exact):
//  1. Time advances only via Step(ops...).
//  2. Piece RNG uses 7-bag (documented).
//  3. When there is NO active piece, Step first SPAWNS the next piece at the
//     top, then applies ops to it, then tries gravity drop of 1 row.
//     If spawn collides with locked cells → Done=true (game over); ops/gravity skipped.
//  4. When there IS an active piece, Step applies ops in order, then drops
//     exactly one row (unless OpHardDrop was used — then the piece falls to
//     lock immediately and gravity is skipped).
//  5. Ops take effect only inside Step. Invalid moves (wall/overlap) are ignored.
//  6. When the piece cannot drop (or hard-dropped), it LOCKS and completed
//     lines are cleared in THIS step. The NEXT Step (not the same one) spawns.
type Game struct {
	board  Board
	bag    *Bag7
	piece  *ActivePiece // nil ⇒ waiting to spawn
	score  int
	lines  int
	pieces int
	combo  int // consecutive clear-locks; 0 when streak broken
	done   bool
	seed   int64
}

// NewGame creates a fresh game. seed controls the 7-bag RNG.
func NewGame(seed int64) *Game {
	return &Game{
		bag:  NewBag7(seed),
		seed: seed,
	}
}

// Seed returns the RNG seed used at construction.
func (g *Game) Seed() int64 { return g.seed }

// Score / Lines / PiecesPlaced are cumulative counters.
func (g *Game) Score() int         { return g.score }
func (g *Game) Lines() int         { return g.lines }
func (g *Game) PiecesPlaced() int  { return g.pieces }
// Combo is consecutive clear-locks including the latest clear (0 if last lock cleared 0).
func (g *Game) Combo() int         { return g.combo }
func (g *Game) Done() bool         { return g.done }
func (g *Game) HasActivePiece() bool { return g.piece != nil }

// Observation returns the current state (locked board + active piece if any).
func (g *Game) Observation() Observation {
	obs := Observation{Board: g.board.Cells}
	if g.piece != nil {
		obs.HasPiece = true
		obs.PieceType = g.piece.Type
		obs.Rot = g.piece.Rot
		obs.X = g.piece.X
		obs.Y = g.piece.Y
	}
	return obs
}

// Step advances the game by one discrete tick. See package/Game docs for ordering.
func (g *Game) Step(ops ...Op) Result {
	res := Result{}
	if g.done {
		res.Done = true
		res.Obs = g.Observation()
		res.Info.Score = g.score
		res.Info.LinesCleared = 0
		return res
	}

	spawned := false
	locked := false
	hardDropped := false
	linesCleared := 0
	scoreGain := 0

	// --- spawn if needed ---
	if g.piece == nil {
		t := g.bag.Next()
		sx, sy := SpawnOrigin(t)
		if !g.board.Fits(t, 0, sx, sy) {
			g.done = true
			res.Done = true
			res.Obs = g.Observation()
			res.Info = Info{Score: g.score, Spawned: false, Locked: false}
			return res
		}
		g.piece = &ActivePiece{Type: t, Rot: 0, X: sx, Y: sy}
		spawned = true
		res.Info.Spawned = true
		res.Info.PieceID = t
		res.Info.HasPiece = true
	}

	// --- apply ops ---
	for _, op := range ops {
		if g.piece == nil {
			break
		}
		switch op {
		case OpLeft:
			g.tryMove(-1, 0)
		case OpRight:
			g.tryMove(1, 0)
		case OpRotateCW:
			g.tryRotate(1)
		case OpRotateCCW:
			g.tryRotate(-1)
		case OpHardDrop:
			for g.tryMove(0, 1) {
			}
			hardDropped = true
		}
	}

	// --- gravity or lock after hard drop ---
	if g.piece != nil {
		if hardDropped {
			before := g.score
			linesCleared = g.lockPiece()
			scoreGain = g.score - before
			locked = true
		} else {
			if !g.tryMove(0, 1) {
				before := g.score
				linesCleared = g.lockPiece()
				scoreGain = g.score - before
				locked = true
			}
		}
	}

	res.Reward = float64(linesCleared)
	res.Done = g.done
	res.Obs = g.Observation()
	res.Info = Info{
		LinesCleared: linesCleared,
		Score:        g.score,
		ScoreGain:    scoreGain,
		Combo:        g.combo,
		Locked:       locked,
		Spawned:      spawned,
		HardDropped:  hardDropped,
		HasPiece:     g.piece != nil,
	}
	if g.piece != nil {
		res.Info.PieceID = g.piece.Type
	}
	_ = spawned // already in Info
	return res
}

func (g *Game) tryMove(dx, dy int) bool {
	p := g.piece
	if p == nil {
		return false
	}
	nx, ny := p.X+dx, p.Y+dy
	if !g.board.Fits(p.Type, p.Rot, nx, ny) {
		return false
	}
	p.X, p.Y = nx, ny
	return true
}

func (g *Game) tryRotate(dir int) bool {
	p := g.piece
	if p == nil {
		return false
	}
	if p.Type == PieceO {
		return true // O is invariant
	}
	nrot := (p.Rot + dir + 4) % 4
	for _, k := range kickOffsets {
		nx, ny := p.X+k.X, p.Y+k.Y
		if g.board.Fits(p.Type, nrot, nx, ny) {
			p.Rot = nrot
			p.X, p.Y = nx, ny
			return true
		}
	}
	return false
}

func (g *Game) lockPiece() int {
	p := g.piece
	g.board.Lock(p.Type, p.Rot, p.X, p.Y)
	g.pieces++
	cleared := g.board.ClearLines()
	g.lines += cleared
	if cleared > 0 {
		// g.combo is previous streak length (0 = first clear → no combo bonus).
		g.score += lineScore(cleared) + comboBonus(g.combo)
		g.combo++
	} else {
		g.combo = 0
	}
	g.piece = nil
	return cleared
}

// BoardCells returns a copy of locked cells (for tests / wrappers).
func (g *Game) BoardCells() [BoardHeight][BoardWidth]Cell {
	return g.board.Cells
}

// Active returns a copy of the active piece, or nil.
func (g *Game) Active() *ActivePiece {
	if g.piece == nil {
		return nil
	}
	cp := *g.piece
	return &cp
}
