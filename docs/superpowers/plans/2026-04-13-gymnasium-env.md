# Gymnasium Environment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gymnasium-compatible RL environment for Block Blast with configurable reward shaping and action masking.

**Architecture:** `BlockBlastEnv` wraps the Python game engine, encoding state as a 7-channel (8,8) observation. Actions are discrete (192 = 3 pieces x 8 rows x 8 cols). A YAML config controls reward weights. Action masking ensures agents only pick valid moves.

**Tech Stack:** Python 3.11+, gymnasium, numpy, pyyaml, pytest

---

## File Structure

```
training/
  pyproject.toml              — Add gymnasium, pyyaml deps
  configs/
    default.yaml              — Reward weights + training hyperparams
  env/
    action_masking.py         — get_action_mask() function
    block_blast_env.py        — BlockBlastEnv(gymnasium.Env)
  tests/
    test_env.py               — 12 env tests
```

---

### Task 1: Add Dependencies and Config

**Files:**
- Modify: `training/pyproject.toml`
- Create: `training/configs/default.yaml`

- [ ] **Step 1: Update pyproject.toml with new dependencies**

Replace the contents of `training/pyproject.toml`:

```toml
[project]
name = "blockblast-training"
version = "0.1.0"
description = "Block Blast RL training environment"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "gymnasium>=0.29",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create default config**

Create `training/configs/default.yaml`:

```yaml
reward:
  cells_cleared_weight: 10.0
  multi_line_bonus: 15.0
  board_cleanliness_weight: 0.1
  hole_penalty: 2.0
  survival_bonus: 0.5
  game_over_penalty: 50.0

training:
  total_steps: 2_000_000
  batch_size: 128
  gamma: 0.99
  learning_rate: 0.0001
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 500_000
  target_update_freq: 2000
  train_freq: 4
  replay_buffer_size: 500_000
  min_replay_size: 10_000
```

- [ ] **Step 3: Install new dependencies**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv pip install -e ".[dev]"
```

- [ ] **Step 4: Verify gymnasium imports**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run python -c "import gymnasium; import yaml; print(f'gymnasium={gymnasium.__version__}')"
```

- [ ] **Step 5: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/pyproject.toml training/configs/default.yaml
git commit -m "feat: add gymnasium, pyyaml deps and default config"
```

---

### Task 2: Action Masking

**Files:**
- Create: `training/env/action_masking.py`
- Create: `training/tests/test_env.py` (partial — mask tests)

- [ ] **Step 1: Write action mask tests**

Create `training/tests/test_env.py`:

```python
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
    # Place piece 0 somewhere valid
    dot = _make_piece("dot", "X")
    state.current_pieces[0] = dot
    result = handle_placement(state, 0, 0, 0, rng)
    assert result is not None
    mask = get_action_mask(result)
    # All actions for piece_index=0 should be False (piece was placed)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_env.py -v 2>&1 | head -15
```

Expected: FAIL (no module `env.action_masking`)

- [ ] **Step 3: Implement action masking**

Create `training/env/action_masking.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_env.py -v 2>&1
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/env/action_masking.py training/tests/test_env.py
git commit -m "feat: add action masking with encode/decode and tests"
```

---

### Task 3: BlockBlastEnv Core (reset, step, observation)

**Files:**
- Create: `training/env/block_blast_env.py`
- Modify: `training/tests/test_env.py` (append env tests)

- [ ] **Step 1: Write env core tests**

Append to `training/tests/test_env.py`:

```python
import yaml
import os
from env.block_blast_env import BlockBlastEnv


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
    """Channel 0 should match the game board state."""
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    # After reset, board is empty — channel 0 all zeros
    np.testing.assert_array_equal(obs[0], np.zeros((8, 8), dtype=np.float32))


def test_observation_piece_channels():
    """Channels 1-3 should encode piece shapes at top-left."""
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    # All 3 piece channels should have some non-zero values
    for ch in range(1, 4):
        assert obs[ch].sum() > 0, f"Piece channel {ch} is all zeros"
    # All 3 availability channels should be 1.0 (all pieces available)
    for ch in range(4, 7):
        np.testing.assert_array_equal(obs[ch], np.ones((8, 8), dtype=np.float32))


def test_valid_step_returns_reward():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=42)
    # Pick first valid action
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
    # Find an invalid action
    invalid_actions = np.where(~info["action_mask"])[0]
    assert len(invalid_actions) > 0
    action = invalid_actions[0]
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert reward < 0, "Invalid action should give negative reward"
    # State should not change — board still empty
    np.testing.assert_array_equal(obs2[0], obs[0])


def test_full_turn_cycle():
    """Place 3 pieces, verify new ones are dealt."""
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
    # After 3 placements, new pieces should be dealt
    # Piece availability channels (4-6) should be all 1s again
    if not terminated:
        for ch in range(4, 7):
            np.testing.assert_array_equal(obs[ch], np.ones((8, 8), dtype=np.float32))


def test_line_clear_reward():
    """Complete a row and verify reward includes cells_cleared bonus."""
    env = BlockBlastEnv()
    env.reset(seed=1)
    # Manually fill row 0 with 7 cells, then place a dot to complete it
    env._state.board[0, :7] = 1
    dot = _make_piece("dot", "X")
    env._state.current_pieces[0] = dot
    mask = env.get_action_mask()
    # Action: piece 0 at (0, 7)
    action = 0 * 64 + 0 * 8 + 7
    assert mask[action], "Placing dot at (0,7) should be valid"
    obs, reward, terminated, truncated, info = env.step(action)
    # Reward should include cells_cleared_weight * 8 = 80
    assert reward >= 80.0, f"Line clear reward too low: {reward}"


def test_game_over_terminates():
    """Fill the board so no moves are possible."""
    env = BlockBlastEnv()
    env.reset(seed=1)
    # Checkerboard pattern — no complete lines, no space for sq2
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            env._state.board[r, c] = 1 if (r + c) % 2 == 0 else 0
    # Give only a sq2 piece that can't fit anywhere
    sq2 = _make_piece("sq2", "XX", "XX")
    env._state.current_pieces = [sq2, None, None]
    # The env should detect game over on next mask check
    mask = env.get_action_mask()
    assert not mask.any(), "No valid actions on checkerboard with sq2"
    # Step with any action should terminate
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_env.py -v -k "reset or observation or valid_step or invalid_action or full_turn or line_clear or game_over or seed" 2>&1 | head -20
```

Expected: FAIL (no module `env.block_blast_env`)

- [ ] **Step 3: Implement BlockBlastEnv**

Create `training/env/block_blast_env.py`:

```python
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

    # Ensure reward section exists with defaults
    if "reward" not in config:
        config["reward"] = dict(_DEFAULT_REWARD)
    else:
        for k, v in _DEFAULT_REWARD.items():
            config["reward"].setdefault(k, v)

    # Apply overrides
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
                    filled_neighbors += 1  # OOB counts as filled
                elif board[nr, nc] != 0:
                    filled_neighbors += 1
            if filled_neighbors >= 3:
                holes += 1
    return holes


def _build_observation(state: GameState) -> np.ndarray:
    """Encode game state as (7, 8, 8) float32 array."""
    obs = np.zeros((7, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Channel 0: board
    obs[0] = state.board.astype(np.float32)

    # Channels 1-3: piece shapes at top-left
    for i in range(3):
        piece = state.current_pieces[i] if i < len(state.current_pieces) else None
        if piece is not None:
            h, w = piece.shape.shape
            obs[1 + i, :h, :w] = piece.shape.astype(np.float32)

    # Channels 4-6: piece availability
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

        # Invalid action handling
        if not mask[action]:
            obs = _build_observation(self._state)
            info = {"action_mask": mask}
            return obs, -1.0, self._state.is_game_over, False, info

        piece_idx, row, col = decode_action(action)
        prev_state = self._state

        # Detect lines that will clear (for reward computation)
        piece = self._state.current_pieces[piece_idx]
        temp_board = place_piece(self._state.board, piece, row, col)
        cleared_rows, cleared_cols = find_completed_lines(temp_board)

        # Execute placement
        new_state = handle_placement(self._state, piece_idx, row, col, self._rng)
        assert new_state is not None  # mask guarantees validity

        self._state = new_state
        terminated = new_state.is_game_over

        # Check if no valid actions remain (even if game doesn't think it's over)
        if not terminated:
            new_mask = get_action_mask(self._state)
            if not new_mask.any():
                terminated = True

        # Compute reward
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

        # Cells cleared
        cells_cleared = (
            len(cleared_rows) * BOARD_SIZE
            + len(cleared_cols) * BOARD_SIZE
            - len(cleared_rows) * len(cleared_cols)
        )
        reward += cells_cleared * rc["cells_cleared_weight"]

        # Multi-line bonus
        lines_cleared = len(cleared_rows) + len(cleared_cols)
        if lines_cleared >= 2:
            reward += (lines_cleared - 1) * rc["multi_line_bonus"]

        # Board cleanliness
        occupied = int(self._state.board.sum())
        reward += (64 - occupied) * rc["board_cleanliness_weight"]

        # Hole penalty
        holes = _count_holes(self._state.board)
        reward -= holes * rc["hole_penalty"]

        # Survival bonus
        reward += rc["survival_bonus"]

        # Game over penalty
        if done:
            reward -= rc["game_over_penalty"]

        return float(reward)
```

- [ ] **Step 4: Run all env tests**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_env.py -v 2>&1
```

Expected: all 17 tests PASS (5 mask + 12 env)

- [ ] **Step 5: Also verify parity tests still pass**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/ -v 2>&1 | tail -5
```

Expected: all 38 + 17 = 55 tests PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/env/block_blast_env.py training/tests/test_env.py
git commit -m "feat: add BlockBlastEnv with observation, step, reward, and tests"
```

---

### Task 4: Gymnasium Registration and Smoke Test

**Files:**
- Modify: `training/env/__init__.py` (add env registration)

- [ ] **Step 1: Write a Gymnasium API compliance smoke test**

Append to `training/tests/test_env.py`:

```python
from gymnasium.utils.env_checker import check_env


def test_gymnasium_api_compliance():
    """Verify env passes Gymnasium's built-in checker."""
    env = BlockBlastEnv()
    # check_env runs reset, step, checks spaces, dtypes, etc.
    # It will raise on any API violation.
    # We skip render check since we have no render mode.
    try:
        check_env(env.unwrapped, skip_render_check=True)
    except Exception as e:
        # check_env may complain about action masking (not standard gym)
        # Only fail on serious issues
        msg = str(e)
        if "observation" in msg.lower() or "action" in msg.lower():
            raise
```

- [ ] **Step 2: Run all tests**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_env.py -v 2>&1
```

Expected: all PASS

- [ ] **Step 3: Run a quick episode to verify the env works end-to-end**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run python -c "
from env.block_blast_env import BlockBlastEnv
import numpy as np

env = BlockBlastEnv()
obs, info = env.reset(seed=42)
total_reward = 0
steps = 0

while True:
    mask = info['action_mask']
    valid = np.where(mask)[0]
    if len(valid) == 0:
        break
    action = valid[np.random.randint(len(valid))]
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated:
        break

print(f'Episode: {steps} steps, total_reward={total_reward:.1f}, score={env._state.score}')
"
```

- [ ] **Step 4: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/env/__init__.py training/tests/test_env.py
git commit -m "feat: add Gymnasium API compliance test and smoke test"
```

---

## Summary

| Task | Component | Test Count | Depends On |
|------|-----------|------------|------------|
| 1 | Dependencies + config | 0 | — |
| 2 | Action masking | 5 | 1 |
| 3 | BlockBlastEnv | 12 | 1, 2 |
| 4 | Gym compliance + smoke | 1 | 3 |

Total: 18 new tests across 4 tasks. Tasks are sequential (each builds on the prior).
