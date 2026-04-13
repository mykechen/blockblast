from dataclasses import dataclass
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
