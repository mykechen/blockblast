# Python Game Engine Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Block Blast game engine from TypeScript to Python with full parity, to serve as the foundation for RL training.

**Architecture:** Mirror the TypeScript engine file-by-file. Each Python module is a pure-function library operating on numpy arrays. RNG is passed explicitly for reproducibility. A comprehensive test suite validates parity with hardcoded expected outputs derived from the TypeScript engine.

**Tech Stack:** Python 3.11+, numpy, pytest, uv (package manager)

---

## File Structure

```
training/
  pyproject.toml            — Project config, dependencies (numpy, pytest)
  env/
    __init__.py             — Package init, re-exports
    types.py                — Piece dataclass, Board type alias, BOARD_SIZE
    pieces.py               — PIECE_CATALOG (28 pieces), get_random_pieces()
    board.py                — create_empty_board, can_place_piece, place_piece,
                              find_completed_lines, clear_lines, has_valid_placement
    scoring.py              — calculate_placement_score, calculate_clear_score
    game.py                 — init_game, handle_placement
  tests/
    __init__.py
    test_parity.py          — 22 parity test cases
```

---

### Task 1: Project Setup

**Files:**
- Create: `training/pyproject.toml`
- Create: `training/env/__init__.py`
- Create: `training/tests/__init__.py`

- [ ] **Step 1: Initialize the training directory with uv**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
mkdir -p training/env training/tests
```

- [ ] **Step 2: Create pyproject.toml**

Create `training/pyproject.toml`:

```toml
[project]
name = "blockblast-training"
version = "0.1.0"
description = "Block Blast RL training environment"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create package init files**

Create `training/env/__init__.py`:

```python
from .types import Piece, Board, GameState, BOARD_SIZE
from .board import (
    create_empty_board,
    can_place_piece,
    place_piece,
    find_completed_lines,
    clear_lines,
    has_valid_placement,
)
from .pieces import PIECE_CATALOG, get_random_pieces
from .scoring import calculate_placement_score, calculate_clear_score
from .game import init_game, handle_placement
```

Create `training/tests/__init__.py` (empty file).

- [ ] **Step 4: Install dependencies**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv venv
uv pip install -e ".[dev]"
```

- [ ] **Step 5: Commit**

```bash
git add training/pyproject.toml training/env/__init__.py training/tests/__init__.py
git commit -m "feat: initialize Python training package with uv"
```

---

### Task 2: Types

**Files:**
- Create: `training/env/types.py`

- [ ] **Step 1: Create types module**

```python
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

Board = NDArray[np.int8]  # shape (8, 8), 0=empty 1=filled
BOARD_SIZE = 8


@dataclass
class Piece:
    shape: NDArray[np.int8]  # shape (h, w), 0=empty 1=filled
    id: str


@dataclass
class GameState:
    board: Board
    current_pieces: list[Piece | None]
    score: int
    combo: int
    placements_since_last_clear: int
    is_game_over: bool
    turn_number: int
    total_lines_cleared: int
    highest_combo: int
```

- [ ] **Step 2: Commit**

```bash
git add training/env/types.py
git commit -m "feat: add Python game engine types"
```

---

### Task 3: Piece Catalog

**Files:**
- Create: `training/env/pieces.py`
- Test: `training/tests/test_parity.py` (partial — piece tests)

- [ ] **Step 1: Write piece catalog test**

Add to `training/tests/test_parity.py`:

```python
import numpy as np
from env.pieces import PIECE_CATALOG, get_random_pieces


def test_piece_catalog_has_28_pieces():
    assert len(PIECE_CATALOG) == 28


def test_piece_ids_match_typescript():
    expected_ids = [
        "dot", "h2", "v2", "h3", "v3", "h4", "v4", "h5", "v5",
        "sq2", "sq3", "r2x3", "r3x2",
        "L0", "L1", "L2", "L3",
        "T0", "T1", "T2", "T3",
        "S0", "S1", "Z0", "Z1",
    ]
    catalog_ids = [p.id for p in PIECE_CATALOG]
    for eid in expected_ids:
        assert eid in catalog_ids, f"Missing piece: {eid}"


def test_dot_shape():
    dot = next(p for p in PIECE_CATALOG if p.id == "dot")
    expected = np.array([[1]], dtype=np.int8)
    np.testing.assert_array_equal(dot.shape, expected)


def test_sq2_shape():
    sq2 = next(p for p in PIECE_CATALOG if p.id == "sq2")
    expected = np.array([[1, 1], [1, 1]], dtype=np.int8)
    np.testing.assert_array_equal(sq2.shape, expected)


def test_L0_shape():
    L0 = next(p for p in PIECE_CATALOG if p.id == "L0")
    expected = np.array([[1, 0], [1, 0], [1, 1]], dtype=np.int8)
    np.testing.assert_array_equal(L0.shape, expected)


def test_T2_shape():
    T2 = next(p for p in PIECE_CATALOG if p.id == "T2")
    expected = np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8)
    np.testing.assert_array_equal(T2.shape, expected)


def test_S0_shape():
    S0 = next(p for p in PIECE_CATALOG if p.id == "S0")
    expected = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8)
    np.testing.assert_array_equal(S0.shape, expected)


def test_get_random_pieces_seeded_reproducibility():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    pieces1 = get_random_pieces(3, rng1)
    pieces2 = get_random_pieces(3, rng2)
    assert [p.id for p in pieces1] == [p.id for p in pieces2]


def test_get_random_pieces_returns_clones():
    rng = np.random.default_rng(99)
    pieces = get_random_pieces(3, rng)
    # Mutating the returned piece should not affect the catalog
    pieces[0].shape[0, 0] = 99
    catalog_piece = next(p for p in PIECE_CATALOG if p.id == pieces[0].id)
    assert catalog_piece.shape[0, 0] != 99
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v -k "piece or dot or sq2 or L0 or T2 or S0 or random" 2>&1 | head -30
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement pieces module**

Create `training/env/pieces.py`:

```python
import numpy as np
from .types import Piece


def _p(id: str, *rows: str) -> Piece:
    shape = np.array(
        [[1 if c == "X" else 0 for c in row] for row in rows],
        dtype=np.int8,
    )
    return Piece(shape=shape, id=id)


PIECE_CATALOG: list[Piece] = [
    # Dot
    _p("dot", "X"),
    # Lines
    _p("h2", "XX"),
    _p("v2", "X", "X"),
    _p("h3", "XXX"),
    _p("v3", "X", "X", "X"),
    _p("h4", "XXXX"),
    _p("v4", "X", "X", "X", "X"),
    _p("h5", "XXXXX"),
    _p("v5", "X", "X", "X", "X", "X"),
    # Squares and rectangles
    _p("sq2", "XX", "XX"),
    _p("sq3", "XXX", "XXX", "XXX"),
    _p("r2x3", "XXX", "XXX"),
    _p("r3x2", "XX", "XX", "XX"),
    # L-pieces (4 rotations)
    _p("L0", "X.", "X.", "XX"),
    _p("L1", "XXX", "X.."),
    _p("L2", "XX", ".X", ".X"),
    _p("L3", "..X", "XXX"),
    # T-pieces (4 rotations)
    _p("T0", "XXX", ".X."),
    _p("T1", ".X", "XX", ".X"),
    _p("T2", ".X.", "XXX"),
    _p("T3", "X.", "XX", "X."),
    # S-pieces (2 orientations)
    _p("S0", ".XX", "XX."),
    _p("S1", "X.", "XX", ".X"),
    # Z-pieces (2 orientations)
    _p("Z0", "XX.", ".XX"),
    _p("Z1", ".X", "XX", "X."),
]


def get_random_pieces(count: int, rng: np.random.Generator) -> list[Piece]:
    pieces: list[Piece] = []
    for _ in range(count):
        idx = rng.integers(0, len(PIECE_CATALOG))
        source = PIECE_CATALOG[idx]
        pieces.append(Piece(shape=source.shape.copy(), id=source.id))
    return pieces
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v -k "piece or dot or sq2 or L0 or T2 or S0 or random" 2>&1
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add training/env/pieces.py training/tests/test_parity.py
git commit -m "feat: add Python piece catalog with parity tests"
```

---

### Task 4: Board Logic

**Files:**
- Create: `training/env/board.py`
- Modify: `training/tests/test_parity.py` (add board tests)

- [ ] **Step 1: Write board tests**

Append to `training/tests/test_parity.py`:

```python
from env.board import (
    create_empty_board,
    can_place_piece,
    place_piece,
    find_completed_lines,
    clear_lines,
    has_valid_placement,
)
from env.types import BOARD_SIZE


def test_empty_board():
    board = create_empty_board()
    assert board.shape == (8, 8)
    assert board.dtype == np.int8
    assert board.sum() == 0


def test_place_sq2_at_origin():
    board = create_empty_board()
    sq2 = next(p for p in PIECE_CATALOG if p.id == "sq2")
    result = place_piece(board, sq2, 0, 0)
    assert result[0, 0] == 1
    assert result[0, 1] == 1
    assert result[1, 0] == 1
    assert result[1, 1] == 1
    assert result.sum() == 4
    # Original board unchanged
    assert board.sum() == 0


def test_reject_overlap():
    board = create_empty_board()
    sq2 = next(p for p in PIECE_CATALOG if p.id == "sq2")
    board = place_piece(board, sq2, 0, 0)
    assert not can_place_piece(board, sq2, 0, 0)
    assert not can_place_piece(board, sq2, 0, 1)
    assert not can_place_piece(board, sq2, 1, 0)


def test_place_h5_at_edge():
    board = create_empty_board()
    h5 = next(p for p in PIECE_CATALOG if p.id == "h5")
    assert can_place_piece(board, h5, 0, 3)  # cols 3-7, fits
    result = place_piece(board, h5, 0, 3)
    for c in range(3, 8):
        assert result[0, c] == 1
    assert result.sum() == 5


def test_reject_out_of_bounds():
    board = create_empty_board()
    h3 = next(p for p in PIECE_CATALOG if p.id == "h3")
    assert not can_place_piece(board, h3, 0, 6)  # needs cols 6,7,8 — 8 is OOB
    assert not can_place_piece(board, h3, -1, 0)  # negative row


def test_find_completed_row():
    board = create_empty_board()
    board[0, :] = 1  # fill row 0
    rows, cols = find_completed_lines(board)
    assert rows == [0]
    assert cols == []


def test_find_completed_col():
    board = create_empty_board()
    board[:, 3] = 1  # fill column 3
    rows, cols = find_completed_lines(board)
    assert rows == []
    assert cols == [3]


def test_find_row_and_col_simultaneously():
    board = create_empty_board()
    board[2, :] = 1  # fill row 2
    board[:, 5] = 1  # fill column 5
    rows, cols = find_completed_lines(board)
    assert rows == [2]
    assert cols == [5]


def test_incomplete_row_not_detected():
    board = create_empty_board()
    board[0, :7] = 1  # 7 of 8 filled
    rows, cols = find_completed_lines(board)
    assert rows == []
    assert cols == []


def test_empty_board_no_lines():
    board = create_empty_board()
    rows, cols = find_completed_lines(board)
    assert rows == []
    assert cols == []


def test_clear_row():
    board = create_empty_board()
    board[0, :] = 1
    board[1, 0] = 1  # extra cell on row 1
    result = clear_lines(board, [0], [])
    assert result[0, :].sum() == 0  # row 0 cleared
    assert result[1, 0] == 1  # row 1 untouched


def test_clear_col():
    board = create_empty_board()
    board[:, 4] = 1
    board[0, 0] = 1  # extra cell
    result = clear_lines(board, [], [4])
    assert result[:, 4].sum() == 0  # col 4 cleared
    assert result[0, 0] == 1  # untouched


def test_clear_row_and_col_intersection():
    board = create_empty_board()
    board[3, :] = 1  # fill row 3
    board[:, 4] = 1  # fill col 4
    result = clear_lines(board, [3], [4])
    assert result[3, :].sum() == 0
    assert result[:, 4].sum() == 0
    assert result[3, 4] == 0  # intersection cleared


def test_has_valid_placement_empty_board():
    board = create_empty_board()
    dot = next(p for p in PIECE_CATALOG if p.id == "dot")
    assert has_valid_placement(board, [dot, None, None])


def test_has_valid_placement_full_board():
    board = np.ones((8, 8), dtype=np.int8)
    dot = next(p for p in PIECE_CATALOG if p.id == "dot")
    assert not has_valid_placement(board, [dot])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v -k "board or place or reject or find or clear or valid" 2>&1 | head -30
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement board module**

Create `training/env/board.py`:

```python
import numpy as np
from .types import Board, Piece, BOARD_SIZE


def create_empty_board() -> Board:
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


def can_place_piece(board: Board, piece: Piece, row: int, col: int) -> bool:
    h, w = piece.shape.shape
    for r in range(h):
        for c in range(w):
            if piece.shape[r, c] == 0:
                continue
            br = row + r
            bc = col + c
            if br < 0 or br >= BOARD_SIZE or bc < 0 or bc >= BOARD_SIZE:
                return False
            if board[br, bc] != 0:
                return False
    return True


def place_piece(board: Board, piece: Piece, row: int, col: int) -> Board:
    new_board = board.copy()
    h, w = piece.shape.shape
    for r in range(h):
        for c in range(w):
            if piece.shape[r, c] == 1:
                new_board[row + r, col + c] = 1
    return new_board


def find_completed_lines(board: Board) -> tuple[list[int], list[int]]:
    rows: list[int] = []
    cols: list[int] = []
    for r in range(BOARD_SIZE):
        if board[r, :].all():
            rows.append(r)
    for c in range(BOARD_SIZE):
        if board[:, c].all():
            cols.append(c)
    return rows, cols


def clear_lines(board: Board, rows: list[int], cols: list[int]) -> Board:
    new_board = board.copy()
    for r in rows:
        new_board[r, :] = 0
    for c in cols:
        new_board[:, c] = 0
    return new_board


def has_valid_placement(board: Board, pieces: list[Piece | None]) -> bool:
    for piece in pieces:
        if piece is None:
            continue
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if can_place_piece(board, piece, r, c):
                    return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v 2>&1
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add training/env/board.py training/tests/test_parity.py
git commit -m "feat: add Python board logic with parity tests"
```

---

### Task 5: Scoring

**Files:**
- Create: `training/env/scoring.py`
- Modify: `training/tests/test_parity.py` (add scoring tests)

- [ ] **Step 1: Write scoring tests**

Append to `training/tests/test_parity.py`:

```python
from env.scoring import calculate_placement_score, calculate_clear_score


def test_placement_score_4_cells():
    assert calculate_placement_score(4) == 4


def test_placement_score_1_cell():
    assert calculate_placement_score(1) == 1


def test_clear_score_one_row_no_combo():
    # 1 line, 8 cells, combo=0 → 8*10 * 1.0 * (1+0) = 80
    assert calculate_clear_score(1, 8, 0) == 80


def test_clear_score_one_row_combo_2():
    # 1 line, 8 cells, combo=2 → 8*10 * 1.0 * (1+2) = 240
    assert calculate_clear_score(1, 8, 2) == 240


def test_clear_score_two_lines_no_combo():
    # 2 lines, 15 cells (8+8-1 intersection), combo=0
    # 15*10 * 1.5 * 1 = 225
    assert calculate_clear_score(2, 15, 0) == 225


def test_clear_score_zero_cells():
    assert calculate_clear_score(0, 0, 5) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v -k "score" 2>&1 | head -20
```

Expected: FAIL

- [ ] **Step 3: Implement scoring module**

Create `training/env/scoring.py`:

```python
BASE_POINTS_PER_CELL = 1
CLEAR_POINTS_PER_CELL = 10


def calculate_placement_score(cells_placed: int) -> int:
    return cells_placed * BASE_POINTS_PER_CELL


def calculate_clear_score(lines_cleared: int, cells_cleared: int, combo: int) -> int:
    if cells_cleared == 0:
        return 0

    base_points = cells_cleared * CLEAR_POINTS_PER_CELL
    line_bonus_multiplier = 1 + (lines_cleared - 1) * 0.5 if lines_cleared >= 2 else 1.0
    combo_multiplier = 1 + combo

    return round(base_points * line_bonus_multiplier * combo_multiplier)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v -k "score" 2>&1
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add training/env/scoring.py training/tests/test_parity.py
git commit -m "feat: add Python scoring with parity tests"
```

---

### Task 6: Game Orchestrator

**Files:**
- Create: `training/env/game.py`
- Modify: `training/tests/test_parity.py` (add game tests)

- [ ] **Step 1: Write game orchestration tests**

Append to `training/tests/test_parity.py`:

```python
from env.game import init_game, handle_placement


def test_init_game():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    assert state.board.shape == (8, 8)
    assert state.board.sum() == 0
    assert len(state.current_pieces) == 3
    assert all(p is not None for p in state.current_pieces)
    assert state.score == 0
    assert state.combo == 0
    assert state.placements_since_last_clear == 0
    assert state.is_game_over is False
    assert state.turn_number == 1


def test_place_all_three_deals_new():
    rng = np.random.default_rng(10)
    state = init_game(rng)
    # Place all 3 pieces at separate locations
    for i in range(3):
        piece = state.current_pieces[i]
        # Find a valid position
        placed = False
        for r in range(8):
            for c in range(8):
                if can_place_piece(state.board, piece, r, c):
                    result = handle_placement(state, i, r, c, rng)
                    assert result is not None
                    state = result
                    placed = True
                    break
            if placed:
                break
        assert placed, f"Could not place piece {i}"

    # After placing all 3, should have 3 new pieces and turn incremented
    assert all(p is not None for p in state.current_pieces)
    assert state.turn_number == 2


def test_combo_increments_on_clear():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    # Manually set up a board with row 0 nearly full (7/8)
    state.board[0, :7] = 1
    # Create a dot piece and place at (0,7) to complete the row
    dot = Piece(shape=np.array([[1]], dtype=np.int8), id="dot")
    state.current_pieces[0] = dot
    result = handle_placement(state, 0, 0, 7, rng)
    assert result is not None
    assert result.combo == 1
    assert result.placements_since_last_clear == 0
    assert result.total_lines_cleared == 1


def test_combo_resets_after_3_misses():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    state.combo = 3  # pretend we had a combo going

    # Place 3 pieces with no line clears
    dot = Piece(shape=np.array([[1]], dtype=np.int8), id="dot")
    for i in range(3):
        state.current_pieces[i] = dot
        result = handle_placement(state, i, i, 0, rng)
        assert result is not None
        state = result

    # After 3 placements without a clear, combo should reset
    assert state.combo == 0


def test_game_over_on_full_board():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    # Fill the entire board except one cell
    state.board[:] = 1
    state.board[7, 7] = 0
    # Give only a sq2 piece which needs 2x2 space — won't fit
    sq2 = Piece(shape=np.array([[1, 1], [1, 1]], dtype=np.int8), id="sq2")
    dot = Piece(shape=np.array([[1]], dtype=np.int8), id="dot")
    state.current_pieces = [dot, None, None]
    # Place the dot at (7,7) — fills the last cell
    result = handle_placement(state, 0, 7, 7, rng)
    assert result is not None
    # Board is now completely full, should be game over
    # (new pieces dealt but none can fit)
    assert result.is_game_over is True


def test_invalid_placement_returns_none():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    # Place a piece, then try to place at same spot
    dot = Piece(shape=np.array([[1]], dtype=np.int8), id="dot")
    state.current_pieces[0] = dot
    state = handle_placement(state, 0, 0, 0, rng)
    state.current_pieces[1] = dot
    result = handle_placement(state, 1, 0, 0, rng)
    assert result is None  # overlap — invalid


def test_seeded_rng_reproducible_game():
    rng1 = np.random.default_rng(777)
    rng2 = np.random.default_rng(777)
    state1 = init_game(rng1)
    state2 = init_game(rng2)
    ids1 = [p.id for p in state1.current_pieces]
    ids2 = [p.id for p in state2.current_pieces]
    assert ids1 == ids2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v -k "init_game or place_all or combo or game_over or invalid or seeded_rng" 2>&1 | head -20
```

Expected: FAIL

- [ ] **Step 3: Implement game module**

Create `training/env/game.py`:

```python
import numpy as np
from .types import GameState, Piece, BOARD_SIZE
from .board import (
    create_empty_board,
    can_place_piece,
    place_piece,
    find_completed_lines,
    clear_lines,
    has_valid_placement,
)
from .pieces import get_random_pieces
from .scoring import calculate_placement_score, calculate_clear_score


def init_game(rng: np.random.Generator) -> GameState:
    board = create_empty_board()
    pieces = get_random_pieces(3, rng)
    return GameState(
        board=board,
        current_pieces=pieces,
        score=0,
        combo=0,
        placements_since_last_clear=0,
        is_game_over=False,
        turn_number=1,
        total_lines_cleared=0,
        highest_combo=0,
    )


def _count_piece_cells(shape: np.ndarray) -> int:
    return int(shape.sum())


def handle_placement(
    state: GameState,
    piece_index: int,
    row: int,
    col: int,
    rng: np.random.Generator,
) -> GameState | None:
    if piece_index < 0 or piece_index >= len(state.current_pieces):
        return None
    piece = state.current_pieces[piece_index]
    if piece is None:
        return None
    if not can_place_piece(state.board, piece, row, col):
        return None

    new_board = place_piece(state.board, piece, row, col)

    # Base points
    cells_placed = _count_piece_cells(piece.shape)
    placement_points = calculate_placement_score(cells_placed)

    # Line clears
    cleared_rows, cleared_cols = find_completed_lines(new_board)
    lines_cleared = len(cleared_rows) + len(cleared_cols)
    cells_cleared = (
        len(cleared_rows) * BOARD_SIZE
        + len(cleared_cols) * BOARD_SIZE
        - len(cleared_rows) * len(cleared_cols)
    )

    if lines_cleared > 0:
        new_board = clear_lines(new_board, cleared_rows, cleared_cols)

    # Combo logic
    new_combo = state.combo
    new_placements_since = state.placements_since_last_clear

    if lines_cleared > 0:
        new_combo = state.combo + 1
        new_placements_since = 0
    else:
        new_placements_since = state.placements_since_last_clear + 1
        if new_placements_since >= 3:
            new_combo = 0
            new_placements_since = 0

    # Clear score uses combo BEFORE incrementing
    clear_points = calculate_clear_score(lines_cleared, cells_cleared, state.combo)

    total_lines_cleared = state.total_lines_cleared + lines_cleared
    highest_combo = max(state.highest_combo, new_combo)

    # Remove piece from tray
    new_pieces = list(state.current_pieces)
    new_pieces[piece_index] = None

    # Deal new pieces if all 3 placed
    all_placed = all(p is None for p in new_pieces)
    new_turn_number = state.turn_number

    total_points = placement_points + clear_points
    new_score = state.score + total_points

    if all_placed:
        new_pieces = get_random_pieces(3, rng)
        new_turn_number = state.turn_number + 1

    is_game_over = not has_valid_placement(new_board, new_pieces)

    return GameState(
        board=new_board,
        current_pieces=new_pieces,
        score=new_score,
        combo=new_combo,
        placements_since_last_clear=new_placements_since,
        is_game_over=is_game_over,
        turn_number=new_turn_number,
        total_lines_cleared=total_lines_cleared,
        highest_combo=highest_combo,
    )
```

- [ ] **Step 4: Run all tests**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v 2>&1
```

Expected: all 22 tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/env/game.py training/tests/test_parity.py
git commit -m "feat: add Python game orchestrator with parity tests"
```

---

### Task 7: Final Parity Verification and Package Init

**Files:**
- Modify: `training/env/__init__.py` (verify re-exports work)

- [ ] **Step 1: Write an integration test using the package imports**

Append to `training/tests/test_parity.py`:

```python
def test_full_game_sequence():
    """Play a deterministic game and verify final state."""
    rng = np.random.default_rng(42)
    state = init_game(rng)

    moves = 0
    while not state.is_game_over and moves < 200:
        placed = False
        for i, piece in enumerate(state.current_pieces):
            if piece is None:
                continue
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if can_place_piece(state.board, piece, r, c):
                        result = handle_placement(state, i, r, c, rng)
                        if result is not None:
                            state = result
                            placed = True
                            break
                if placed:
                    break
            if placed:
                break
        if not placed:
            break
        moves += 1

    # Should have played some moves and accumulated score
    assert moves > 10, f"Only played {moves} moves"
    assert state.score > 0
    assert state.total_lines_cleared >= 0
    assert state.highest_combo >= 0
```

- [ ] **Step 2: Run the full test suite**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_parity.py -v 2>&1
```

Expected: all tests PASS (23 total)

- [ ] **Step 3: Verify package imports work**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run python -c "from env import PIECE_CATALOG, init_game, BOARD_SIZE; import numpy as np; rng = np.random.default_rng(1); g = init_game(rng); print(f'Board: {g.board.shape}, Pieces: {len(g.current_pieces)}, Catalog: {len(PIECE_CATALOG)}')"
```

Expected: `Board: (8, 8), Pieces: 3, Catalog: 28`

- [ ] **Step 4: Commit**

```bash
git add training/tests/test_parity.py
git commit -m "feat: add full game integration test, verify package exports"
```

---

## Summary

| Task | Component | Test Count | Depends On |
|------|-----------|------------|------------|
| 1 | Project setup | 0 | — |
| 2 | Types | 0 | 1 |
| 3 | Piece catalog | 9 | 2 |
| 4 | Board logic | 14 | 2, 3 |
| 5 | Scoring | 6 | 2 |
| 6 | Game orchestrator | 7 | 3, 4, 5 |
| 7 | Integration test | 1 | 6 |

Tasks 1-2 are sequential setup. Tasks 3, 4, 5 can be parallelized. Task 6 depends on all three. Task 7 is the final verification.

Total: 37 test cases across 7 tasks.
