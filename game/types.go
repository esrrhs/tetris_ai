package game

// Board dimensions: standard 10×20 visible playfield (no hidden buffer rows).
const (
	BoardWidth  = 10
	BoardHeight = 20
)

// Op is a discrete action applied only inside Step.
type Op int

const (
	OpLeft Op = iota
	OpRight
	OpRotateCW
	OpRotateCCW
	OpHardDrop // optional: soft gravity skipped; piece locks immediately after drop
)

func (o Op) String() string {
	switch o {
	case OpLeft:
		return "Left"
	case OpRight:
		return "Right"
	case OpRotateCW:
		return "RotateCW"
	case OpRotateCCW:
		return "RotateCCW"
	case OpHardDrop:
		return "HardDrop"
	default:
		return "Op(?)"
	}
}

// PieceType is one of the seven tetrominoes.
type PieceType int

const (
	PieceI PieceType = iota
	PieceL
	PieceJ
	PieceT
	PieceO
	PieceS
	PieceZ
	PieceCount
)

func (p PieceType) String() string {
	names := [...]string{"I", "L", "J", "T", "O", "S", "Z"}
	if p < 0 || int(p) >= len(names) {
		return "?"
	}
	return names[p]
}

// Cell is a locked board cell: 0 empty, or PieceType+1 for color/id.
type Cell int8

// Point is a board / relative coordinate. Y increases downward.
type Point struct {
	X, Y int
}

// ActivePiece is the currently falling piece (nil when waiting to spawn).
type ActivePiece struct {
	Type PieceType
	Rot  int // 0..3
	X, Y int // pivot / origin on the board
}

// Observation is the RL-facing state snapshot.
type Observation struct {
	// Board is locked cells only (active piece not painted in).
	// Rows[0] is the top of the playfield.
	Board [BoardHeight][BoardWidth]Cell

	HasPiece  bool
	PieceType PieceType
	Rot       int
	X, Y      int
}

// Info carries step metadata for logging / RL wrappers.
type Info struct {
	LinesCleared int
	Score        int // cumulative
	PieceID      PieceType
	HasPiece     bool
	Locked       bool
	Spawned      bool
	HardDropped  bool
}

// Result is returned by every Step call.
type Result struct {
	Obs    Observation
	Reward float64 // lines cleared this step (0 if none)
	Done   bool    // true when spawn collides (game over)
	Info   Info
}
