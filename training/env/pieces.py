import numpy as np
from .types import Piece


def _p(id: str, *rows: str) -> Piece:
    shape = np.array(
        [[1 if c == "X" else 0 for c in row] for row in rows],
        dtype=np.int8,
    )
    return Piece(shape=shape, id=id)


PIECE_CATALOG: list[Piece] = [
    _p("dot", "X"),
    _p("h2", "XX"),
    _p("v2", "X", "X"),
    _p("h3", "XXX"),
    _p("v3", "X", "X", "X"),
    _p("h4", "XXXX"),
    _p("v4", "X", "X", "X", "X"),
    _p("h5", "XXXXX"),
    _p("v5", "X", "X", "X", "X", "X"),
    _p("sq2", "XX", "XX"),
    _p("sq3", "XXX", "XXX", "XXX"),
    _p("r2x3", "XXX", "XXX"),
    _p("r3x2", "XX", "XX", "XX"),
    _p("L0", "X.", "X.", "XX"),
    _p("L1", "XXX", "X.."),
    _p("L2", "XX", ".X", ".X"),
    _p("L3", "..X", "XXX"),
    _p("T0", "XXX", ".X."),
    _p("T1", ".X", "XX", ".X"),
    _p("T2", ".X.", "XXX"),
    _p("T3", "X.", "XX", "X."),
    _p("S0", ".XX", "XX."),
    _p("S1", "X.", "XX", ".X"),
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
