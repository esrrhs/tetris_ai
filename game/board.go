package game

// Board holds locked cells. The active piece is NOT stored here.
type Board struct {
	Cells [BoardHeight][BoardWidth]Cell
}

func (b *Board) Clear() {
	for y := 0; y < BoardHeight; y++ {
		for x := 0; x < BoardWidth; x++ {
			b.Cells[y][x] = 0
		}
	}
}

func (b *Board) Occupied(x, y int) bool {
	if x < 0 || x >= BoardWidth || y >= BoardHeight {
		return true
	}
	if y < 0 {
		// Above the visible board: allowed for kicks / vertical I, treated as empty.
		return false
	}
	return b.Cells[y][x] != 0
}

func (b *Board) Fits(t PieceType, rot, x, y int) bool {
	for _, p := range Cells(t, rot, x, y) {
		if b.Occupied(p.X, p.Y) {
			return false
		}
	}
	return true
}

func (b *Board) Lock(t PieceType, rot, x, y int) {
	id := Cell(t) + 1
	for _, p := range Cells(t, rot, x, y) {
		if p.Y >= 0 && p.Y < BoardHeight && p.X >= 0 && p.X < BoardWidth {
			b.Cells[p.Y][p.X] = id
		}
	}
}

// ClearLines removes full rows and returns how many were cleared.
func (b *Board) ClearLines() int {
	cleared := 0
	write := BoardHeight - 1
	for read := BoardHeight - 1; read >= 0; read-- {
		full := true
		for x := 0; x < BoardWidth; x++ {
			if b.Cells[read][x] == 0 {
				full = false
				break
			}
		}
		if full {
			cleared++
			continue
		}
		if write != read {
			b.Cells[write] = b.Cells[read]
		}
		write--
	}
	for y := write; y >= 0; y-- {
		for x := 0; x < BoardWidth; x++ {
			b.Cells[y][x] = 0
		}
	}
	return cleared
}

func lineScore(n int) int {
	switch n {
	case 1:
		return 40
	case 2:
		return 100
	case 3:
		return 300
	case 4:
		return 1200
	default:
		return 0
	}
}

// comboBonus for consecutive clear-locks. comboIndex 0 = first clear (no bonus),
// 1 = first follow-up (+40), 2 → +80, etc. Mirrors C++ ComboBonus.
func comboBonus(comboIndex int) int {
	if comboIndex <= 0 {
		return 0
	}
	return 40 * comboIndex
}
