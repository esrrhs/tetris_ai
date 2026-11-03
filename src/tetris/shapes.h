// Fixed 4-state shape tables + basic wall kicks (matches Go game/shapes.go).
#pragma once

#include <array>

#include "tetris/types.h"

namespace tetris {

// Kick offsets tried on rotate (in order).
inline constexpr std::array<Point, 6> kKickOffsets = {{
    {0, 0},
    {-1, 0},
    {1, 0},
    {0, -1},
    {-2, 0},
    {2, 0},
}};

// shapes[type][rot] = 4 relative cells.
const std::array<std::array<std::array<Point, 4>, 4>,
                 static_cast<int>(PieceType::Count)>& ShapeTable();

// Absolute board cells for a piece placement.
std::array<Point, 4> Cells(PieceType t, int rot, int x, int y);

// Default top-center spawn for each piece type.
Point SpawnOrigin(PieceType t);

}  // namespace tetris
