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
