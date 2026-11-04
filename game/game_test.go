package game

import (
	"testing"
)

func TestSpawnOnFirstStep(t *testing.T) {
	g := NewGame(1)
	if g.HasActivePiece() {
		t.Fatal("expected no piece before first Step")
	}
	r := g.Step()
	if !r.Info.Spawned {
		t.Fatal("expected Spawned=true")
	}
	if !r.Obs.HasPiece {
		t.Fatal("expected active piece after spawn Step")
	}
	if r.Done {
		t.Fatal("unexpected game over on empty board spawn")
	}
	// spawn → gravity: piece should have dropped 1 row from spawn Y
	spawnY := 0
	if r.Obs.Y != spawnY+1 && !r.Info.Locked {
		// either dropped one, or locked if somehow couldn't (shouldn't on empty)
		t.Fatalf("after spawn+gravity expected Y=%d, got Y=%d locked=%v", spawnY+1, r.Obs.Y, r.Info.Locked)
	}
}

func TestGravityOneRowPerStep(t *testing.T) {
	g := NewGame(42)
	r1 := g.Step()
	y1 := r1.Obs.Y
	r2 := g.Step()
	if r2.Info.Spawned {
		t.Fatal("should not re-spawn while piece is active")
	}
	if r2.Info.Locked {
		// possible if near bottom; skip assert
		return
	}
	if r2.Obs.Y != y1+1 {
		t.Fatalf("gravity should drop exactly 1: y1=%d y2=%d", y1, r2.Obs.Y)
	}
}

func TestMultiOpsInOneStep(t *testing.T) {
	g := NewGame(7)
	r := g.Step(OpLeft, OpLeft, OpRight) // net Left once, then gravity
	if !r.Info.Spawned {
		t.Fatal("expected spawn")
	}
	// spawn X is 4; after L,L,R → X=3, then gravity Y+=1
	if r.Obs.X != 3 {
		t.Fatalf("expected X=3 after L,L,R; got %d", r.Obs.X)
	}
}

func TestInvalidMovesIgnored(t *testing.T) {
	g := NewGame(3)
	g.Step() // spawn + drop
	// slam into left wall
	for i := 0; i < 20; i++ {
		g.Step(OpLeft, OpLeft, OpLeft, OpLeft, OpLeft)
		if g.Done() {
			break
		}
		a := g.Active()
		if a == nil {
			break
		}
		if a.X < 0 {
			t.Fatalf("piece went out of bounds X=%d", a.X)
		}
	}
}

func TestLockThenNextStepSpawns(t *testing.T) {
	g := NewGame(99)
	// Hard-drop to lock quickly
	r := g.Step(OpHardDrop)
	if !r.Info.Locked {
		t.Fatal("expected lock after hard drop")
	}
	if g.HasActivePiece() {
		t.Fatal("no active piece after lock; next Step should spawn")
	}
	r2 := g.Step()
	if !r2.Info.Spawned {
		t.Fatal("next Step after lock must spawn")
	}
	if r2.Info.Locked && r2.Info.Spawned {
		// spawn+gravity can also lock on a nearly full board; that's ok
	}
}

func TestLineClear(t *testing.T) {
	g := NewGame(1)
	// Fill bottom row except where we will drop an O or horizontal piece.
	// Manually fill rows via board for a controlled clear.
	for x := 0; x < BoardWidth; x++ {
		g.board.Cells[BoardHeight-1][x] = 1
	}
	// Clear the middle 2 cells so a piece can complete — actually fill all
	// and use ClearLines directly first to verify helper.
	n := g.board.ClearLines()
	if n != 1 {
		t.Fatalf("ClearLines want 1 got %d", n)
	}
	// Rebuild: leave a gap, spawn O via hard drop into place — simpler unit test:
	g = NewGame(1)
	for x := 0; x < BoardWidth; x++ {
		if x == 4 || x == 5 {
			continue
		}
		g.board.Cells[BoardHeight-1][x] = 1
		g.board.Cells[BoardHeight-2][x] = 1
	}
	// Force an O piece by exhausting bag until O, or set piece manually.
	g.piece = &ActivePiece{Type: PieceO, Rot: 0, X: 4, Y: BoardHeight - 3}
	r := g.Step(OpHardDrop)
	if r.Info.LinesCleared < 1 {
		t.Fatalf("expected line clear, got %d; board bottom=%v", r.Info.LinesCleared, g.board.Cells[BoardHeight-1])
	}
	if r.Reward != float64(r.Info.LinesCleared) {
		t.Fatalf("reward should equal lines cleared")
	}
}

func TestGameOverOnSpawnCollision(t *testing.T) {
	g := NewGame(5)
	// Fill the entire top area so any spawn fails.
	for y := 0; y < 4; y++ {
		for x := 0; x < BoardWidth; x++ {
			g.board.Cells[y][x] = 1
		}
	}
	r := g.Step()
	if !r.Done {
		t.Fatal("expected Done on spawn collision")
	}
	if r.Info.Spawned {
		t.Fatal("should not mark Spawned when spawn fails")
	}
}

func TestSevenBagFairness(t *testing.T) {
	b := NewBag7(12345)
	seen := map[PieceType]int{}
	for i := 0; i < 7; i++ {
		seen[b.Next()]++
	}
	if len(seen) != 7 {
		t.Fatalf("first bag must contain all 7 unique pieces, got %v", seen)
	}
	for _, c := range seen {
		if c != 1 {
			t.Fatalf("each piece once per bag: %v", seen)
		}
	}
	// Second bag also complete
	seen2 := map[PieceType]int{}
	for i := 0; i < 7; i++ {
		seen2[b.Next()]++
	}
	if len(seen2) != 7 {
		t.Fatalf("second bag incomplete: %v", seen2)
	}
}

func TestRotateCW(t *testing.T) {
	g := NewGame(11)
	r := g.Step(OpRotateCW)
	if !r.Obs.HasPiece {
		t.Fatal("expected piece")
	}
	// After CW rotate from 0 → rot 1 (unless O)
	if r.Obs.PieceType != PieceO && r.Obs.Rot != 1 {
		t.Fatalf("expected rot=1 after CW, got %d type=%s", r.Obs.Rot, r.Obs.PieceType)
	}
}

func TestStepAfterDoneIsNoOp(t *testing.T) {
	g := NewGame(5)
	for y := 0; y < 4; y++ {
		for x := 0; x < BoardWidth; x++ {
			g.board.Cells[y][x] = 1
		}
	}
	g.Step()
	if !g.Done() {
		t.Fatal("expected done")
	}
	r := g.Step(OpLeft)
	if !r.Done {
		t.Fatal("still done")
	}
}

func TestManyStepsNoPanic(t *testing.T) {
	g := NewGame(2024)
	for i := 0; i < 500; i++ {
		ops := []Op{}
		switch i % 5 {
		case 0:
			ops = []Op{OpLeft, OpRotateCW}
		case 1:
			ops = []Op{OpRight, OpRight}
		case 2:
			ops = []Op{OpRotateCCW}
		case 3:
			ops = []Op{OpHardDrop}
		default:
			ops = nil
		}
		r := g.Step(ops...)
		if r.Done {
			return
		}
	}
}

func TestLineScoreAndComboBonus(t *testing.T) {
	if lineScore(0) != 0 || lineScore(1) != 40 || lineScore(2) != 100 || lineScore(3) != 300 || lineScore(4) != 1200 {
		t.Fatalf("unexpected lineScore table")
	}
	if comboBonus(0) != 0 || comboBonus(1) != 40 || comboBonus(2) != 80 {
		t.Fatalf("unexpected comboBonus")
	}
}

func prepSingleClearGap(g *Game) {
	for y := BoardHeight - 4; y < BoardHeight; y++ {
		for x := 0; x < BoardWidth; x++ {
			g.board.Cells[y][x] = 0
		}
	}
	for x := 0; x < BoardWidth; x++ {
		if x == 4 || x == 5 {
			continue
		}
		g.board.Cells[BoardHeight-1][x] = 1
	}
}

func TestComboScoreAndReset(t *testing.T) {
	g := NewGame(1)
	prepSingleClearGap(g)
	g.piece = &ActivePiece{Type: PieceO, Rot: 0, X: 4, Y: BoardHeight - 3}
	r1 := g.Step(OpHardDrop)
	if r1.Info.LinesCleared != 1 || r1.Info.ScoreGain != 40 || r1.Info.Combo != 1 || g.Score() != 40 {
		t.Fatalf("first clear: got lines=%d gain=%d combo=%d score=%d", r1.Info.LinesCleared, r1.Info.ScoreGain, r1.Info.Combo, g.Score())
	}

	prepSingleClearGap(g)
	g.piece = &ActivePiece{Type: PieceO, Rot: 0, X: 4, Y: BoardHeight - 3}
	r2 := g.Step(OpHardDrop)
	if r2.Info.LinesCleared != 1 || r2.Info.ScoreGain != 80 || r2.Info.Combo != 2 || g.Score() != 120 {
		t.Fatalf("combo clear: got lines=%d gain=%d combo=%d score=%d", r2.Info.LinesCleared, r2.Info.ScoreGain, r2.Info.Combo, g.Score())
	}

	for y := BoardHeight - 4; y < BoardHeight; y++ {
		for x := 0; x < BoardWidth; x++ {
			g.board.Cells[y][x] = 0
		}
	}
	g.piece = &ActivePiece{Type: PieceO, Rot: 0, X: 0, Y: BoardHeight - 3}
	r3 := g.Step(OpHardDrop)
	if r3.Info.LinesCleared != 0 || r3.Info.ScoreGain != 0 || r3.Info.Combo != 0 || g.Combo() != 0 {
		t.Fatalf("reset: got lines=%d gain=%d combo=%d", r3.Info.LinesCleared, r3.Info.ScoreGain, r3.Info.Combo)
	}
}
