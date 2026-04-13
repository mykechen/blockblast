# Gymnasium Environment — Design Spec

## Goal

Build a Gymnasium-compatible environment (`BlockBlastEnv`) that wraps the Python game engine for RL training. Includes configurable reward shaping, action masking, and a YAML config system.

## Dependencies

Add to `training/pyproject.toml`:
- `gymnasium>=0.29`
- `pyyaml>=6.0`

## File Structure

```
training/
  env/
    block_blast_env.py    — Gymnasium env implementation
    action_masking.py     — Valid action mask computation
  configs/
    default.yaml          — Reward weights + training hyperparams
  tests/
    test_env.py           — 12 env test cases
```

## Observation Space

Shape: `(7, 8, 8)` — `Box(0, 1, shape=(7, 8, 8), dtype=float32)`

| Channel | Content |
|---------|---------|
| 0 | Board occupancy (0=empty, 1=filled) |
| 1 | Piece 0 shape encoded at top-left of 8x8 grid (zeros if placed) |
| 2 | Piece 1 shape encoded at top-left of 8x8 grid (zeros if placed) |
| 3 | Piece 2 shape encoded at top-left of 8x8 grid (zeros if placed) |
| 4 | Piece 0 availability (uniform 8x8 plane, 1.0 if available, 0.0 if placed) |
| 5 | Piece 1 availability (uniform plane) |
| 6 | Piece 2 availability (uniform plane) |

## Action Space

`Discrete(192)` — flattened from `(piece_index, row, col)`:
```
action = piece_index * 64 + row * 8 + col
```

Decode:
```python
piece_index = action // 64
row = (action % 64) // 8
col = action % 8
```

## Action Masking

`action_masking.py` provides:

```python
def get_action_mask(game_state: GameState) -> np.ndarray:
    """Returns (192,) bool array. True = valid action."""
```

For each of the 192 possible actions:
- Check if `piece_index` refers to a non-None piece
- Check if `can_place_piece(board, piece, row, col)` returns True

The mask is returned in `info` dict from `reset()` and `step()` as `info["action_mask"]`.

## Step Logic

Each `step(action)` places one piece:

1. Decode action → `(piece_index, row, col)`
2. Validate action is in the mask (if not, return penalty reward and don't change state)
3. Call `handle_placement(state, piece_index, row, col, rng)`
4. Compute reward from reward function
5. Build new observation
6. Check termination: `state.is_game_over` or no valid actions remain
7. Return `(obs, reward, terminated, truncated, info)`

`truncated` is always `False` (no time limit).

## Reward Function

All weights loaded from `configs/default.yaml`:

```python
def compute_reward(prev_state, new_state, cleared_rows, cleared_cols, done, config):
    reward = 0.0

    # Cells cleared
    cells_cleared = (
        len(cleared_rows) * BOARD_SIZE
        + len(cleared_cols) * BOARD_SIZE
        - len(cleared_rows) * len(cleared_cols)
    )
    reward += cells_cleared * config["reward"]["cells_cleared_weight"]

    # Multi-line bonus
    lines_cleared = len(cleared_rows) + len(cleared_cols)
    if lines_cleared >= 2:
        reward += (lines_cleared - 1) * config["reward"]["multi_line_bonus"]

    # Board cleanliness
    occupied = int(new_state.board.sum())
    reward += (64 - occupied) * config["reward"]["board_cleanliness_weight"]

    # Hole penalty
    holes = count_holes(new_state.board)
    reward -= holes * config["reward"]["hole_penalty"]

    # Survival bonus
    reward += config["reward"]["survival_bonus"]

    # Game over penalty
    if done:
        reward -= config["reward"]["game_over_penalty"]

    return reward
```

A "hole" = empty cell with 3+ orthogonal neighbors (up/down/left/right) that are filled or out-of-bounds.

## Config File

`training/configs/default.yaml`:

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

The env only reads the `reward` section. The `training` section is included for A-Part 3.

## BlockBlastEnv Interface

```python
class BlockBlastEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str | None = None, config_override: dict | None = None):
        """Load config from YAML. config_override merges on top."""

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset game. Returns (obs, info). info contains 'action_mask'."""

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Place one piece. Returns (obs, reward, terminated, truncated, info)."""

    def get_action_mask(self) -> np.ndarray:
        """Current valid action mask (192,) bool."""

    def _build_observation(self) -> np.ndarray:
        """Encode current game state as (7, 8, 8) float32 array."""

    def _compute_reward(self, prev_state, cleared_rows, cleared_cols, done) -> float:
        """Compute shaped reward."""
```

## Testing

`training/tests/test_env.py` — 12 tests:

1. `test_reset_observation_shape` — obs shape is (7,8,8), dtype float32
2. `test_reset_info_has_action_mask` — info["action_mask"] shape (192,), has True values
3. `test_observation_board_channel` — channel 0 matches game board
4. `test_observation_piece_channels` — channels 1-3 encode piece shapes
5. `test_action_mask_validity` — every True in mask corresponds to a valid placement
6. `test_placed_piece_masked_out` — after placing piece 0, all piece-0 actions are False
7. `test_valid_step_returns_reward` — step with valid action returns positive reward
8. `test_invalid_action_penalty` — step with masked action returns negative reward, state unchanged
9. `test_full_turn_cycle` — place 3 pieces, verify new pieces dealt
10. `test_line_clear_reward` — complete a row, verify reward >= cells_cleared_weight * 8
11. `test_game_over_terminates` — fill board, verify terminated=True
12. `test_seed_reproducibility` — same seed produces same trajectory
