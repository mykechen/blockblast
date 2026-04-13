import numpy as np
from env.types import Piece, GameState, BOARD_SIZE
from env.board import create_empty_board, can_place_piece, place_piece
from env.pieces import PIECE_CATALOG, get_random_pieces
from env.game import init_game, handle_placement
from env.action_masking import get_action_mask, decode_action
from env.block_blast_env import BlockBlastEnv
from env.board import find_completed_lines, clear_lines, has_valid_placement


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


def test_reset_observation_shape():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (7, 8, 8)
    assert obs.dtype == np.float32


def test_reset_info_has_action_mask():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    assert "action_mask" in info
    assert info["action_mask"].shape == (192,)
    assert info["action_mask"].dtype == bool
    assert info["action_mask"].any()


def test_observation_board_channel():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    np.testing.assert_array_equal(obs[0], np.zeros((8, 8), dtype=np.float32))


def test_observation_piece_channels():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    for ch in range(1, 4):
        assert obs[ch].sum() > 0, f"Piece channel {ch} is all zeros"
    for ch in range(4, 7):
        np.testing.assert_array_equal(obs[ch], np.ones((8, 8), dtype=np.float32))


def test_valid_step_returns_reward():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    valid_actions = np.where(info["action_mask"])[0]
    assert len(valid_actions) > 0
    action = valid_actions[0]
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == (7, 8, 8)
    assert isinstance(reward, float)
    assert not truncated
    assert "action_mask" in info2


def test_invalid_action_penalty():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    invalid_actions = np.where(~info["action_mask"])[0]
    assert len(invalid_actions) > 0
    action = invalid_actions[0]
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert reward < 0, "Invalid action should give negative reward"
    np.testing.assert_array_equal(obs2[0], obs[0])


def test_full_turn_cycle():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)

    pieces_placed = 0
    for _ in range(3):
        valid = np.where(info["action_mask"])[0]
        if len(valid) == 0:
            break
        obs, reward, terminated, truncated, info = env.step(valid[0])
        pieces_placed += 1
        if terminated:
            break

    assert pieces_placed == 3, f"Expected 3 placements, got {pieces_placed}"
    if not terminated:
        for ch in range(4, 7):
            np.testing.assert_array_equal(obs[ch], np.ones((8, 8), dtype=np.float32))


def test_line_clear_reward():
    env = BlockBlastEnv()
    env.reset(seed=1)
    env._state.board[0, :7] = 1
    dot = _make_piece("dot", "X")
    env._state.current_pieces[0] = dot
    action = 0 * 64 + 0 * 8 + 7
    mask = env.get_action_mask()
    assert mask[action], "Placing dot at (0,7) should be valid"
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward >= 80.0, f"Line clear reward too low: {reward}"


def test_game_over_terminates():
    env = BlockBlastEnv()
    env.reset(seed=1)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            env._state.board[r, c] = 1 if (r + c) % 2 == 0 else 0
    sq2 = _make_piece("sq2", "XX", "XX")
    env._state.current_pieces = [sq2, None, None]
    mask = env.get_action_mask()
    assert not mask.any(), "No valid actions on checkerboard with sq2"
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated


def test_seed_reproducibility():
    env1 = BlockBlastEnv()
    env2 = BlockBlastEnv()
    obs1, info1 = env1.reset(seed=999)
    obs2, info2 = env2.reset(seed=999)
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(info1["action_mask"], info2["action_mask"])

    valid = np.where(info1["action_mask"])[0]
    if len(valid) > 0:
        obs1b, r1, _, _, _ = env1.step(valid[0])
        obs2b, r2, _, _, _ = env2.step(valid[0])
        np.testing.assert_array_equal(obs1b, obs2b)
        assert r1 == r2
