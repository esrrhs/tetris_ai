package game

// Rotation system: fixed 4-state tables with basic wall kicks (not full SRS).
// Coordinates are relative to the piece origin; Y increases downward.
// Spawn orientation (rot 0) keeps all cells at Y >= 0 so pieces fit at the top.
//
// Kick offsets tried on rotate (in order): (0,0), (-1,0), (1,0), (0,-1), (-2,0), (2,0).
// Invalid placements (wall / overlap) are ignored.

var kickOffsets = []Point{
	{0, 0},
	{-1, 0}, {1, 0},
	{0, -1},
	{-2, 0}, {2, 0},
}

// shapes[type][rot] = 4 relative cells.
var shapes = [PieceCount][4][4]Point{
	// I
	{
		{{-1, 0}, {0, 0}, {1, 0}, {2, 0}},
		{{0, -1}, {0, 0}, {0, 1}, {0, 2}},
		{{-1, 0}, {0, 0}, {1, 0}, {2, 0}},
		{{0, -1}, {0, 0}, {0, 1}, {0, 2}},
	},
	// L
	{
		{{-1, 0}, {0, 0}, {1, 0}, {-1, 1}},
		{{0, -1}, {0, 0}, {0, 1}, {1, 1}},
		{{-1, 0}, {0, 0}, {1, 0}, {1, -1}},
		{{0, -1}, {0, 0}, {0, 1}, {-1, -1}},
	},
	// J
	{
		{{-1, 0}, {0, 0}, {1, 0}, {1, 1}},
		{{0, -1}, {0, 0}, {0, 1}, {1, -1}},
		{{-1, 0}, {0, 0}, {1, 0}, {-1, -1}},
		{{0, -1}, {0, 0}, {0, 1}, {-1, 1}},
	},
	// T
	{
		{{-1, 0}, {0, 0}, {1, 0}, {0, 1}},
		{{0, -1}, {0, 0}, {0, 1}, {1, 0}},
		{{-1, 0}, {0, 0}, {1, 0}, {0, -1}},
		{{0, -1}, {0, 0}, {0, 1}, {-1, 0}},
	},
	// O
	{
		{{0, 0}, {1, 0}, {0, 1}, {1, 1}},
		{{0, 0}, {1, 0}, {0, 1}, {1, 1}},
		{{0, 0}, {1, 0}, {0, 1}, {1, 1}},
		{{0, 0}, {1, 0}, {0, 1}, {1, 1}},
	},
	// S
	{
		{{0, 0}, {1, 0}, {-1, 1}, {0, 1}},
		{{0, -1}, {0, 0}, {1, 0}, {1, 1}},
		{{0, 0}, {1, 0}, {-1, 1}, {0, 1}},
		{{0, -1}, {0, 0}, {1, 0}, {1, 1}},
	},
	// Z
	{
		{{-1, 0}, {0, 0}, {0, 1}, {1, 1}},
		{{1, -1}, {0, 0}, {1, 0}, {0, 1}},
		{{-1, 0}, {0, 0}, {0, 1}, {1, 1}},
		{{1, -1}, {0, 0}, {1, 0}, {0, 1}},
	},
}

// Cells returns absolute board cells for a piece placement.
func Cells(t PieceType, rot, x, y int) [4]Point {
	r := rot & 3
	var out [4]Point
	for i, p := range shapes[t][r] {
		out[i] = Point{X: x + p.X, Y: y + p.Y}
	}
	return out
}

// SpawnOrigin is the default top-center spawn for each piece type.
func SpawnOrigin(t PieceType) (x, y int) {
	// Center-ish column; Y=1 gives I-vertical / T-up room without leaving the top.
	// Horizontal I at y=0 with origin x=4 covers columns 3..6.
	switch t {
	case PieceI, PieceO:
		return 4, 0
	default:
		return 4, 0
	}
}
