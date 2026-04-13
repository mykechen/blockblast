import numpy as np
from env.pieces import PIECE_CATALOG, get_random_pieces
from env.board import can_place_piece, create_empty_board, place_piece, find_completed_lines, clear_lines, has_valid_placement
from env.scoring import calculate_placement_score, calculate_clear_score
from env.game import init_game, handle_placement
from env.types import Piece, BOARD_SIZE


def test_piece_catalog_has_25_pieces():
    assert len(PIECE_CATALOG) == 25


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
    pieces[0].shape[0, 0] = 99
    catalog_piece = next(p for p in PIECE_CATALOG if p.id == pieces[0].id)
    assert catalog_piece.shape[0, 0] != 99


# --- Board tests ---

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
    assert can_place_piece(board, h5, 0, 3)
    result = place_piece(board, h5, 0, 3)
    for c in range(3, 8):
        assert result[0, c] == 1
    assert result.sum() == 5


def test_reject_out_of_bounds():
    board = create_empty_board()
    h3 = next(p for p in PIECE_CATALOG if p.id == "h3")
    assert not can_place_piece(board, h3, 0, 6)
    assert not can_place_piece(board, h3, -1, 0)


def test_find_completed_row():
    board = create_empty_board()
    board[0, :] = 1
    rows, cols = find_completed_lines(board)
    assert rows == [0]
    assert cols == []


def test_find_completed_col():
    board = create_empty_board()
    board[:, 3] = 1
    rows, cols = find_completed_lines(board)
    assert rows == []
    assert cols == [3]


def test_find_row_and_col_simultaneously():
    board = create_empty_board()
    board[2, :] = 1
    board[:, 5] = 1
    rows, cols = find_completed_lines(board)
    assert rows == [2]
    assert cols == [5]


def test_incomplete_row_not_detected():
    board = create_empty_board()
    board[0, :7] = 1
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
    board[1, 0] = 1
    result = clear_lines(board, [0], [])
    assert result[0, :].sum() == 0
    assert result[1, 0] == 1


def test_clear_col():
    board = create_empty_board()
    board[:, 4] = 1
    board[0, 0] = 1
    result = clear_lines(board, [], [4])
    assert result[:, 4].sum() == 0
    assert result[0, 0] == 1


def test_clear_row_and_col_intersection():
    board = create_empty_board()
    board[3, :] = 1
    board[:, 4] = 1
    result = clear_lines(board, [3], [4])
    assert result[3, :].sum() == 0
    assert result[:, 4].sum() == 0
    assert result[3, 4] == 0


def test_has_valid_placement_empty_board():
    board = create_empty_board()
    dot = next(p for p in PIECE_CATALOG if p.id == "dot")
    assert has_valid_placement(board, [dot, None, None])


def test_has_valid_placement_full_board():
    board = np.ones((8, 8), dtype=np.int8)
    dot = next(p for p in PIECE_CATALOG if p.id == "dot")
    assert not has_valid_placement(board, [dot])


# --- Scoring tests ---

def test_placement_score_4_cells():
    assert calculate_placement_score(4) == 4


def test_placement_score_1_cell():
    assert calculate_placement_score(1) == 1


def test_clear_score_one_row_no_combo():
    assert calculate_clear_score(1, 8, 0) == 80


def test_clear_score_one_row_combo_2():
    assert calculate_clear_score(1, 8, 2) == 240


def test_clear_score_two_lines_no_combo():
    assert calculate_clear_score(2, 15, 0) == 225


def test_clear_score_zero_cells():
    assert calculate_clear_score(0, 0, 5) == 0


# --- Game orchestrator tests ---

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
    for i in range(3):
        piece = state.current_pieces[i]
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
    assert all(p is not None for p in state.current_pieces)
    assert state.turn_number == 2


def test_combo_increments_on_clear():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    state.board[0, :7] = 1
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
    state.combo = 3
    dot = Piece(shape=np.array([[1]], dtype=np.int8), id="dot")
    for i in range(3):
        state.current_pieces[i] = dot
        result = handle_placement(state, i, i, 0, rng)
        assert result is not None
        state = result
    assert state.combo == 0


def test_game_over_on_full_board():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    # Build a checkerboard board: empty where (r+c) is odd, filled where (r+c) is even.
    # This means no two empty cells are horizontally or vertically adjacent,
    # so sq2 (2x2) can never fit. No row or col is fully filled (each has 4 gaps),
    # so no line clears trigger. Place a dot at an empty cell (r+c odd) to use piece 0.
    # After placement that cell fills; remaining empty cells still isolated -> game over.
    board = np.array(
        [[(r + c) % 2 == 0 for c in range(8)] for r in range(8)], dtype=np.int8
    )
    state.board = board
    # Empty cells are where (r+c) is odd, e.g. (0,1), (1,0), etc.
    # Place dot at (0, 1): row 0 still has gaps at (0,3),(0,5),(0,7) → not complete.
    # Col 1 still has gaps at (2,1),(4,1),(6,1) → not complete. No clears.
    dot = Piece(shape=np.array([[1]], dtype=np.int8), id="dot")
    sq2 = next(p for p in PIECE_CATALOG if p.id == "sq2")
    state.current_pieces = [dot, sq2, None]
    result = handle_placement(state, 0, 0, 1, rng)
    assert result is not None
    assert result.is_game_over is True


def test_invalid_placement_returns_none():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    dot = Piece(shape=np.array([[1]], dtype=np.int8), id="dot")
    state.current_pieces[0] = dot
    state = handle_placement(state, 0, 0, 0, rng)
    state.current_pieces[1] = dot
    result = handle_placement(state, 1, 0, 0, rng)
    assert result is None


def test_seeded_rng_reproducible_game():
    rng1 = np.random.default_rng(777)
    rng2 = np.random.default_rng(777)
    state1 = init_game(rng1)
    state2 = init_game(rng2)
    ids1 = [p.id for p in state1.current_pieces]
    ids2 = [p.id for p in state2.current_pieces]
    assert ids1 == ids2


# --- Integration test ---

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

    assert moves > 10, f"Only played {moves} moves"
    assert state.score > 0
    assert state.total_lines_cleared >= 0
    assert state.highest_combo >= 0
