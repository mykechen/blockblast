# Agent + Training — Design Spec

## Goal

Build a Dueling Double DQN agent with prioritized experience replay that learns to play Block Blast via the Gymnasium environment. Includes a training script with TensorBoard logging and checkpoint management.

## Dependencies

Add to `training/pyproject.toml`:
- `torch>=2.0`
- `tensorboard>=2.0`

## File Structure

```
training/
  agent/
    __init__.py
    model.py              — Dueling DQN network
    replay_buffer.py      — Prioritized experience replay
    dqn.py                — Training loop + action selection
  scripts/
    train.py              — Main training entry point
  tests/
    test_agent.py         — 10 agent tests
  checkpoints/            — Saved model weights (gitignored)
  logs/                   — TensorBoard logs (gitignored)
```

## Model Architecture (`model.py`)

Dueling Double DQN:

```
Input: (batch, 7, 8, 8)

Conv2d(7, 32, 3, padding=1) → BatchNorm2d(32) → ReLU
Conv2d(32, 64, 3, padding=1) → BatchNorm2d(64) → ReLU
Conv2d(64, 128, 3, padding=1) → BatchNorm2d(128) → ReLU
Flatten → 8192

Shared: Linear(8192, 512) → ReLU → Dropout(0.1)

Value head:  Linear(512, 256) → ReLU → Linear(256, 1)
Advantage head: Linear(512, 256) → ReLU → Linear(256, 192)

Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
```

Device auto-detection: MPS → CUDA → CPU.

## Replay Buffer (`replay_buffer.py`)

Prioritized experience replay with sum-tree for O(log n) sampling.

- Capacity: configurable (default 500,000)
- Each transition: `(state, action, reward, next_state, done, next_action_mask)`
  - state/next_state: `(7, 8, 8)` float32
  - action: int
  - reward: float
  - done: bool
  - next_action_mask: `(192,)` bool
- Priority: `(|TD_error| + epsilon)^alpha`
  - epsilon = 1e-6
  - alpha = 0.6
- Importance sampling weights: `w = (N * P(i))^(-beta) / max(w)`
  - beta annealed from 0.4 → 1.0 over training

## Training Loop (`dqn.py`)

`DQNTrainer` class:

```python
class DQNTrainer:
    def __init__(self, env, config, device):
        self.policy_net = DuelingDQN().to(device)
        self.target_net = DuelingDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.buffer = PrioritizedReplayBuffer(capacity=config["replay_buffer_size"])
        self.optimizer = Adam(self.policy_net.parameters(), lr=config["learning_rate"])

    def select_action(self, state, action_mask, epsilon) -> int:
        """Epsilon-greedy with action masking."""

    def train_step(self, batch_size, beta) -> float:
        """One gradient step. Returns loss."""

    def update_target(self):
        """Hard copy policy → target."""
```

Double DQN target computation:
1. `q_next_policy = policy_net(next_states)` — mask invalid, argmax
2. `q_next_target = target_net(next_states)` — evaluate at policy's best action
3. `target = reward + gamma * q_target * (1 - done)`

Loss: Huber loss, weighted by importance sampling weights.

### Hyperparams (from `configs/default.yaml`)

All already defined:
- `total_steps: 2_000_000`
- `batch_size: 128`
- `gamma: 0.99`
- `learning_rate: 0.0001`
- `epsilon_start: 1.0`, `epsilon_end: 0.05`, `epsilon_decay_steps: 500_000`
- `target_update_freq: 2000`
- `train_freq: 4`
- `replay_buffer_size: 500_000`
- `min_replay_size: 10_000`

## Training Script (`scripts/train.py`)

```
Usage: uv run python scripts/train.py [--config configs/default.yaml] [--steps 2000000]
```

Flow:
1. Load config
2. Create env, trainer
3. Main loop:
   - Select action (epsilon-greedy with mask)
   - Step env
   - Push to buffer
   - Train every `train_freq` steps (if buffer >= `min_replay_size`)
   - Update target every `target_update_freq` steps
   - Decay epsilon linearly
   - Log to TensorBoard every episode end
   - Checkpoint every 50k steps (keep top 5 by mean score)
4. Save final model

## Logging

TensorBoard via `torch.utils.tensorboard.SummaryWriter`:

Per episode:
- `episode/reward` — total episode reward
- `episode/score` — game score
- `episode/length` — steps (pieces placed)

Per training step (every 1000 steps):
- `train/loss`
- `train/mean_q`
- `train/epsilon`
- `train/buffer_size`

## Checkpoint Management

Save every 50k steps to `training/checkpoints/`:
- `checkpoint_{step}.pt` — contains model state_dict, optimizer state_dict, step count, epsilon, best scores
- Keep top 5 by rolling mean score (delete older ones)
- `best_model.pt` — symlink/copy of the best checkpoint

## Testing

`training/tests/test_agent.py` — 10 tests:

1. `test_model_forward_pass` — input (1,7,8,8) → output (1,192)
2. `test_model_with_action_mask` — masked positions are -inf
3. `test_model_dueling_structure` — value head outputs (batch,1), advantage outputs (batch,192)
4. `test_replay_buffer_push_sample` — push 100, sample 32, shapes correct
5. `test_replay_buffer_priority_update` — high-priority transitions sampled more often
6. `test_replay_buffer_capacity` — push 200 into capacity-100 buffer, size stays 100
7. `test_select_action_greedy` — epsilon=0 picks argmax of masked Q-values
8. `test_select_action_explore` — epsilon=1 picks random valid action
9. `test_training_step` — one gradient step, loss is finite float
10. `test_checkpoint_save_load` — save, load into new model, weights match
