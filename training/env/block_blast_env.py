import os
import numpy as np
import gymnasium
from gymnasium import spaces
import yaml

from .types import GameState, Piece, Board, BOARD_SIZE
from .board import can_place_piece, place_piece, find_completed_lines, clear_lines, has_valid_placement
from .pieces import get_random_pieces
from .game import init_game, handle_placement
from .action_masking import get_action_mask, decode_action, ACTION_SPACE_SIZE

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "default.yaml"
)

_DEFAULT_REWARD = {
    "cells_cleared_weight": 10.0,
    "multi_line_bonus": 15.0,
    "board_cleanliness_weight": 0.1,
    "hole_penalty": 2.0,
    "survival_bonus": 0.5,
    "game_over_penalty": 50.0,
}


def _load_config(config_path: str | None, config_override: dict | None) -> dict:
    config = {}
    path = config_path or _DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        with open(path) as f:
            config = yaml.safe_load(f) or {}

    if "reward" not in config:
        config["reward"] = dict(_DEFAULT_REWARD)
    else:
        for k, v in _DEFAULT_REWARD.items():
            config["reward"].setdefault(k, v)

    if config_override:
        for section, values in config_override.items():
            if section not in config:
                config[section] = {}
            if isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values

    return config


def _count_holes(board: Board) -> int:
    """Count empty cells with 3+ filled/OOB orthogonal neighbors."""
    holes = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != 0:
                continue
            filled_neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                    filled_neighbors += 1
                elif board[nr, nc] != 0:
                    filled_neighbors += 1
            if filled_neighbors >= 3:
                holes += 1
    return holes


def _build_observation(state: GameState) -> np.ndarray:
    """Encode game state as (7, 8, 8) float32 array."""
    obs = np.zeros((7, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    obs[0] = state.board.astype(np.float32)

    for i in range(3):
        piece = state.current_pieces[i] if i < len(state.current_pieces) else None
        if piece is not None:
            h, w = piece.shape.shape
            obs[1 + i, :h, :w] = piece.shape.astype(np.float32)

    for i in range(3):
        piece = state.current_pieces[i] if i < len(state.current_pieces) else None
        if piece is not None:
            obs[4 + i, :, :] = 1.0

    return obs


class BlockBlastEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | None = None,
        config_override: dict | None = None,
    ):
        super().__init__()
        self.config = _load_config(config_path, config_override)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self._state: GameState | None = None
        self._rng: np.random.Generator | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._state = init_game(self._rng)

        obs = _build_observation(self._state)
        info = {"action_mask": get_action_mask(self._state)}
        return obs, info

    def step(self, action: int):
        assert self._state is not None, "Call reset() before step()"
        assert self._rng is not None

        mask = get_action_mask(self._state)

        if not mask[action]:
            obs = _build_observation(self._state)
            terminated = self._state.is_game_over or not mask.any()
            info = {"action_mask": mask}
            return obs, -1.0, terminated, False, info

        piece_idx, row, col = decode_action(action)
        prev_state = self._state

        piece = self._state.current_pieces[piece_idx]
        temp_board = place_piece(self._state.board, piece, row, col)
        cleared_rows, cleared_cols = find_completed_lines(temp_board)

        new_state = handle_placement(self._state, piece_idx, row, col, self._rng)
        assert new_state is not None

        self._state = new_state
        terminated = new_state.is_game_over

        if not terminated:
            new_mask = get_action_mask(self._state)
            if not new_mask.any():
                terminated = True

        reward = self._compute_reward(prev_state, cleared_rows, cleared_cols, terminated)

        obs = _build_observation(self._state)
        info = {"action_mask": get_action_mask(self._state)}

        return obs, reward, terminated, False, info

    def get_action_mask(self) -> np.ndarray:
        assert self._state is not None, "Call reset() before get_action_mask()"
        return get_action_mask(self._state)

    def _compute_reward(
        self,
        prev_state: GameState,
        cleared_rows: list[int],
        cleared_cols: list[int],
        done: bool,
    ) -> float:
        rc = self.config["reward"]
        reward = 0.0

        cells_cleared = (
            len(cleared_rows) * BOARD_SIZE
            + len(cleared_cols) * BOARD_SIZE
            - len(cleared_rows) * len(cleared_cols)
        )
        reward += cells_cleared * rc["cells_cleared_weight"]

        lines_cleared = len(cleared_rows) + len(cleared_cols)
        if lines_cleared >= 2:
            reward += (lines_cleared - 1) * rc["multi_line_bonus"]

        occupied = int(self._state.board.sum())
        reward += (64 - occupied) * rc["board_cleanliness_weight"]

        holes = _count_holes(self._state.board)
        reward -= holes * rc["hole_penalty"]

        reward += rc["survival_bonus"]

        if done:
            reward -= rc["game_over_penalty"]

        return float(reward)
