# Python Game Engine Port — Design Spec

## Goal

Port the Block Blast game engine from TypeScript (`lib/engine/`) to Python (`training/env/`) to serve as the foundation for RL training. The Python engine must produce identical results to the TypeScript engine given the same inputs.

## Project Setup

- **Package manager:** uv
- **Python version:** 3.11+
- **Dependencies:** numpy, pytest
- **Directory:** `/training/` at project root

## Architecture

Each Python module mirrors a TypeScript file:

| TypeScript | Python | Purpose |
|---|---|---|
| `lib/engine/types.ts` | `training/env/types.py` | Dataclasses + constants |
| `lib/engine/pieces.ts` | `training/env/pieces.py` | Piece catalog, random selection |
| `lib/engine/board.ts` | `training/env/board.py` | Placement, line detection, clearing |
| `lib/engine/scoring.ts` | `training/env/scoring.py` | Score calculation |
| `lib/engine/game.ts` | `training/env/game.py` | Game state orchestrator |

### Key Differences from TypeScript

- Board: `numpy.ndarray` shape `(8, 8)`, dtype `int8` (0=empty, 1=filled) instead of `boolean[][]`
- Pieces: `numpy.ndarray` shape `(h, w)`, dtype `int8` instead of `boolean[][]`
- All functions are pure — no side effects, no localStorage
- RNG: `numpy.random.Generator` passed explicitly for reproducibility
- No smart difficulty scaling — training uses uniform random piece selection
- No `color` field on Piece — training doesn't render
- No `highScore` on GameState — no persistence concerns

## Data Types

### `types.py`

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Board = NDArray[np.int8]  # shape (8, 8)
BOARD_SIZE = 8

@dataclass
class Piece:
    shape: NDArray[np.int8]  # shape (h, w), 1=filled 0=empty
    id: str

@dataclass
class GameState:
    board: Board
    current_pieces: list[Piece | None]  # 3 slots
    score: int
    combo: int
    placements_since_last_clear: int
    is_game_over: bool
    turn_number: int
    total_lines_cleared: int
    highest_combo: int
```

## Function Signatures

### `board.py`

- `create_empty_board() -> Board`
- `can_place_piece(board: Board, piece: Piece, row: int, col: int) -> bool`
- `place_piece(board: Board, piece: Piece, row: int, col: int) -> Board` — returns new array, no mutation
- `find_completed_lines(board: Board) -> tuple[list[int], list[int]]` — returns (rows, cols)
- `clear_lines(board: Board, rows: list[int], cols: list[int]) -> Board` — returns new array
- `has_valid_placement(board: Board, pieces: list[Piece | None]) -> bool`

### `pieces.py`

- `PIECE_CATALOG: list[Piece]` — all 28 pieces matching TypeScript catalog (same IDs, same shapes, no colors)
- `get_random_pieces(count: int, rng: numpy.random.Generator) -> list[Piece]` — uniform random selection with deep clone

### `scoring.py`

- `calculate_placement_score(cells_placed: int) -> int` — 1 point per cell
- `calculate_clear_score(lines_cleared: int, cells_cleared: int, combo: int) -> int` — 10pts/cell x combo x multi-line bonus

### `game.py`

- `init_game(rng: numpy.random.Generator) -> GameState`
- `handle_placement(state: GameState, piece_index: int, row: int, col: int, rng: numpy.random.Generator) -> GameState | None` — returns None if invalid

Combo logic (identical to TypeScript):
- Line clear: combo increments, `placements_since_last_clear` resets to 0
- No clear: `placements_since_last_clear` increments; at 3, combo resets to 0
- `calculate_clear_score` uses combo value BEFORE incrementing

## Piece Catalog

All 28 pieces from TypeScript, same IDs, same shapes:

- dot, h2, v2, h3, v3, h4, v4, h5, v5
- sq2, sq3, r2x3, r3x2
- L0, L1, L2, L3
- T0, T1, T2, T3
- S0, S1, Z0, Z1

Each defined as a numpy int8 array. Shapes match the TypeScript `p()` helper output exactly.

## Testing

`training/tests/test_parity.py` — 22 test cases with hardcoded expected outputs derived from the TypeScript engine.

### Board basics (5 cases)
1. Empty board is all zeros, shape (8,8)
2. Place a 2x2 piece at (0,0) — cells (0,0), (0,1), (1,0), (1,1) are 1
3. Reject placement overlapping existing piece
4. Place piece at board edge — e.g., h5 at (0,3) fills (0,3)-(0,7)
5. Reject piece that extends out of bounds — h3 at (0,6) would need col 8

### Line detection (5 cases)
6. Full row 0 → detected
7. Full column 3 → detected
8. Full row 2 AND full column 5 simultaneously → both detected
9. Row with 7/8 filled → not detected
10. Empty board → no lines

### Line clearing (3 cases)
11. Clear a full row → that row becomes all zeros, rest unchanged
12. Clear a full column → that column becomes all zeros
13. Clear row 3 + column 4 → intersection cell (3,4) cleared once, both row 3 and col 4 are zeros

### Scoring (4 cases)
14. Place 4-cell piece → placement score = 4
15. Clear 1 row (8 cells), combo=0 → clear score = 80
16. Clear 1 row, combo=2 → clear score = 80 * (1+2) = 240
17. Clear 2 lines at once (16 cells minus intersection), combo=0 → base * 1.5 multi-line bonus

### Game orchestration (5 cases)
18. Place all 3 pieces → new pieces dealt, turn increments
19. Clear a line → combo increments to 1, placements_since_last_clear = 0
20. 3 placements without clear → combo resets to 0
21. Board full with no valid placements → is_game_over = True
22. Seeded RNG produces reproducible piece sequence (same seed → same pieces twice)

## File Structure

```
training/
  pyproject.toml
  env/
    __init__.py
    types.py
    pieces.py
    board.py
    scoring.py
    game.py
  tests/
    __init__.py
    test_parity.py
```
