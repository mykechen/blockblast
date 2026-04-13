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
