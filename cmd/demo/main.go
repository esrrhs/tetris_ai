// Demo: runs a short step-based Tetris game with scripted / random ops.
package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"

	"github.com/esrrhs/tetris_ai/game"
)

func main() {
	seed := int64(42)
	steps := 40
	if len(os.Args) > 1 {
		if v, err := strconv.ParseInt(os.Args[1], 10, 64); err == nil {
			seed = v
		}
	}
	if len(os.Args) > 2 {
		if v, err := strconv.Atoi(os.Args[2]); err == nil {
			steps = v
		}
	}

	rng := rand.New(rand.NewSource(seed + 1))
	g := game.NewGame(seed)

	fmt.Printf("tetris_ai demo  seed=%d  maxSteps=%d  (7-bag)\n", seed, steps)
	fmt.Println("Step semantics: spawn(if needed) → apply ops → gravity drop 1 (or lock on hard-drop)")
	fmt.Println()

	for i := 0; i < steps; i++ {
		var ops []game.Op
		n := rng.Intn(4) // 0..3 ops
		for j := 0; j < n; j++ {
			switch rng.Intn(5) {
			case 0:
				ops = append(ops, game.OpLeft)
			case 1:
				ops = append(ops, game.OpRight)
			case 2:
				ops = append(ops, game.OpRotateCW)
			case 3:
				ops = append(ops, game.OpRotateCCW)
			case 4:
				if rng.Intn(8) == 0 {
					ops = append(ops, game.OpHardDrop)
				}
			}
		}

		r := g.Step(ops...)
		fmt.Printf("=== step %d ops=%v spawned=%v locked=%v lines=%d reward=%.0f done=%v ===\n",
			i+1, ops, r.Info.Spawned, r.Info.Locked, r.Info.LinesCleared, r.Reward, r.Done)
		fmt.Print(g.Render(true))
		fmt.Println()
		if r.Done {
			fmt.Println("game over")
			break
		}
	}
}
