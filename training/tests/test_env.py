import numpy as np
from env.types import Piece, GameState, BOARD_SIZE
from env.board import create_empty_board, can_place_piece, place_piece
from env.pieces import PIECE_CATALOG, get_random_pieces
from env.game import init_game, handle_placement
from env.action_masking import get_action_mask, decode_action


def _make_piece(id: str, *rows: str) -> Piece:
    shape = np.array(
        [[1 if c == "X" else 0 for c in row] for row in rows],
        dtype=np.int8,
    )
    return Piece(shape=shape, id=id)


def test_action_mask_shape():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    mask = get_action_mask(state)
    assert mask.shape == (192,)
    assert mask.dtype == bool


def test_action_mask_has_valid_actions():
    rng = np.random.default_rng(1)
    state = init_game(rng)
    mask = get_action_mask(state)
    assert mask.any(), "Empty board should have valid actions"


def test_action_mask_validity():
    """Every True in mask must correspond to a valid can_place_piece call."""
    rng = np.random.default_rng(42)
    state = init_game(rng)
    mask = get_action_mask(state)

    for action_idx in range(192):
        piece_idx, row, col = decode_action(action_idx)
        piece = state.current_pieces[piece_idx]
        if piece is None:
            assert not mask[action_idx]
            continue
        expected = can_place_piece(state.board, piece, row, col)
        assert mask[action_idx] == expected, (
            f"Mismatch at action {action_idx}: piece={piece.id} pos=({row},{col}) "
            f"mask={mask[action_idx]} expected={expected}"
        )


def test_placed_piece_masked_out():
    rng = np.random.default_rng(10)
    state = init_game(rng)
    dot = _make_piece("dot", "X")
    state.current_pieces[0] = dot
    result = handle_placement(state, 0, 0, 0, rng)
    assert result is not None
    mask = get_action_mask(result)
    for action_idx in range(0, 64):
        assert not mask[action_idx], f"Piece 0 action {action_idx} should be masked"


def test_decode_action():
    piece_idx, row, col = decode_action(0)
    assert (piece_idx, row, col) == (0, 0, 0)

    piece_idx, row, col = decode_action(64)
    assert (piece_idx, row, col) == (1, 0, 0)

    piece_idx, row, col = decode_action(128 + 3 * 8 + 5)
    assert (piece_idx, row, col) == (2, 3, 5)

    piece_idx, row, col = decode_action(191)
    assert (piece_idx, row, col) == (2, 7, 7)
