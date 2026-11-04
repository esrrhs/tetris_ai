package game

import (
	"fmt"
	"strings"
)

// String renders an ASCII board with the active piece painted on.
func (g *Game) String() string {
	return g.Render(true)
}

// Render returns an ASCII playfield. If paintActive, the falling piece is shown.
func (g *Game) Render(paintActive bool) string {
	var grid [BoardHeight][BoardWidth]rune
	for y := 0; y < BoardHeight; y++ {
		for x := 0; x < BoardWidth; x++ {
			if g.board.Cells[y][x] != 0 {
				grid[y][x] = '#'
			} else {
				grid[y][x] = '.'
			}
		}
	}
	if paintActive && g.piece != nil {
		ch := []rune(g.piece.Type.String())[0]
		for _, p := range Cells(g.piece.Type, g.piece.Rot, g.piece.X, g.piece.Y) {
			if p.Y >= 0 && p.Y < BoardHeight && p.X >= 0 && p.X < BoardWidth {
				grid[p.Y][p.X] = ch
			}
		}
	}

	var b strings.Builder
	b.WriteString("+" + strings.Repeat("-", BoardWidth) + "+\n")
	for y := 0; y < BoardHeight; y++ {
		b.WriteByte('|')
		for x := 0; x < BoardWidth; x++ {
			b.WriteRune(grid[y][x])
		}
		b.WriteString("|\n")
	}
	b.WriteString("+" + strings.Repeat("-", BoardWidth) + "+\n")
	status := fmt.Sprintf("score=%d lines=%d pieces=%d combo=%d", g.score, g.lines, g.pieces, g.combo)
	if g.piece != nil {
		status += fmt.Sprintf(" active=%s rot=%d @(%d,%d)", g.piece.Type, g.piece.Rot, g.piece.X, g.piece.Y)
	} else {
		status += " active=none"
	}
	if g.done {
		status += " GAME_OVER"
	}
	b.WriteString(status)
	b.WriteByte('\n')
	return b.String()
}
