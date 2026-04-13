import numpy as np
from .types import GameState, BOARD_SIZE
from .board import can_place_piece

ACTION_SPACE_SIZE = 3 * BOARD_SIZE * BOARD_SIZE  # 192


def decode_action(action: int) -> tuple[int, int, int]:
    """Decode flat action index to (piece_index, row, col)."""
    piece_index = action // (BOARD_SIZE * BOARD_SIZE)
    remainder = action % (BOARD_SIZE * BOARD_SIZE)
    row = remainder // BOARD_SIZE
    col = remainder % BOARD_SIZE
    return piece_index, row, col


def encode_action(piece_index: int, row: int, col: int) -> int:
    """Encode (piece_index, row, col) to flat action index."""
    return piece_index * BOARD_SIZE * BOARD_SIZE + row * BOARD_SIZE + col


def get_action_mask(state: GameState) -> np.ndarray:
    """Return (192,) bool array. True = action is valid."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

    for piece_idx in range(3):
        if piece_idx >= len(state.current_pieces):
            continue
        piece = state.current_pieces[piece_idx]
        if piece is None:
            continue
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if can_place_piece(state.board, piece, row, col):
                    mask[encode_action(piece_idx, row, col)] = True

    return mask
