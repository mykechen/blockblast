# Agent + Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dueling Double DQN agent with prioritized experience replay that trains on the BlockBlastEnv Gymnasium environment.

**Architecture:** CNN backbone (3 conv layers on 7x8x8 input) with dueling value/advantage heads. Prioritized replay buffer with sum-tree. Double DQN training with action masking. TensorBoard logging, checkpoint management. MPS device for Apple Silicon.

**Tech Stack:** Python 3.11+, PyTorch (MPS backend), TensorBoard, numpy

---

## File Structure

```
training/
  pyproject.toml                — Add torch, tensorboard deps
  agent/
    __init__.py
    model.py                    — DuelingDQN network
    replay_buffer.py            — PrioritizedReplayBuffer with sum-tree
    dqn.py                      — DQNTrainer (action selection, training step, target update)
  scripts/
    train.py                    — Main training entry point
  tests/
    test_agent.py               — 10 agent tests
```

---

### Task 1: Dependencies and Package Init

**Files:**
- Modify: `training/pyproject.toml`
- Create: `training/agent/__init__.py`

- [ ] **Step 1: Update pyproject.toml**

Replace contents of `training/pyproject.toml`:

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
    "torch>=2.0",
    "tensorboard>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create agent package init**

Create `training/agent/__init__.py` (empty file).

- [ ] **Step 3: Install new dependencies**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv pip install -e ".[dev]"
```

- [ ] **Step 4: Verify PyTorch + MPS**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run python -c "import torch; print(f'torch={torch.__version__}, mps={torch.backends.mps.is_available()}')"
```

Expected: `torch=2.x.x, mps=True`

- [ ] **Step 5: Add checkpoints and logs to .gitignore**

Append to `/Users/mykechen/Desktop/APPS/blockblast/.gitignore`:

```
# Training artifacts
training/checkpoints/
training/logs/
training/.venv/
training/**/__pycache__/
training/*.egg-info/
```

- [ ] **Step 6: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/pyproject.toml training/agent/__init__.py .gitignore
git commit -m "feat: add torch, tensorboard deps and agent package"
```

---

### Task 2: Dueling DQN Model

**Files:**
- Create: `training/agent/model.py`
- Create: `training/tests/test_agent.py` (partial — model tests)

- [ ] **Step 1: Write model tests**

Create `training/tests/test_agent.py`:

```python
import torch
import numpy as np
from agent.model import DuelingDQN, get_device


def test_model_forward_pass():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(1, 7, 8, 8, device=device)
    q_values = model(x)
    assert q_values.shape == (1, 192)
    assert not torch.isnan(q_values).any()


def test_model_batch_forward():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(32, 7, 8, 8, device=device)
    q_values = model(x)
    assert q_values.shape == (32, 192)


def test_model_with_action_mask():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(1, 7, 8, 8, device=device)
    q_values = model(x)

    mask = torch.zeros(192, dtype=torch.bool, device=device)
    mask[0] = True
    mask[10] = True
    mask[100] = True

    masked_q = q_values.clone()
    masked_q[0, ~mask] = float("-inf")

    # Only 3 values should be finite
    finite = torch.isfinite(masked_q[0])
    assert finite.sum().item() == 3


def test_model_dueling_structure():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(4, 7, 8, 8, device=device)

    # Access intermediate outputs
    features = model.shared(model.flatten(model.relu3(model.bn3(model.conv3(
        model.relu2(model.bn2(model.conv2(
            model.relu1(model.bn1(model.conv1(x)))
        )))
    )))))

    value = model.value_head(features)
    advantage = model.advantage_head(features)

    assert value.shape == (4, 1)
    assert advantage.shape == (4, 192)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_agent.py -v 2>&1 | head -10
```

Expected: FAIL (no module `agent.model`)

- [ ] **Step 3: Implement the model**

Create `training/agent/model.py`:

```python
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DuelingDQN(nn.Module):
    def __init__(self, in_channels: int = 7, action_size: int = 192):
        super().__init__()
        self.action_size = action_size

        # CNN backbone
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.shared(x)

        value = self.value_head(x)           # (batch, 1)
        advantage = self.advantage_head(x)   # (batch, action_size)

        # Dueling: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_agent.py -v 2>&1
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/agent/model.py training/tests/test_agent.py
git commit -m "feat: add Dueling DQN model with tests"
```

---

### Task 3: Prioritized Replay Buffer

**Files:**
- Create: `training/agent/replay_buffer.py`
- Modify: `training/tests/test_agent.py` (append buffer tests)

- [ ] **Step 1: Write buffer tests**

Append to `training/tests/test_agent.py`:

```python
from agent.replay_buffer import PrioritizedReplayBuffer


def test_replay_buffer_push_sample():
    buf = PrioritizedReplayBuffer(capacity=1000)
    for i in range(100):
        state = np.random.randn(7, 8, 8).astype(np.float32)
        next_state = np.random.randn(7, 8, 8).astype(np.float32)
        action_mask = np.ones(192, dtype=bool)
        buf.push(state, i % 192, float(i), next_state, False, action_mask)

    assert len(buf) == 100

    batch = buf.sample(32, beta=0.4)
    states, actions, rewards, next_states, dones, next_masks, weights, indices = batch

    assert states.shape == (32, 7, 8, 8)
    assert actions.shape == (32,)
    assert rewards.shape == (32,)
    assert next_states.shape == (32, 7, 8, 8)
    assert dones.shape == (32,)
    assert next_masks.shape == (32, 192)
    assert weights.shape == (32,)
    assert len(indices) == 32


def test_replay_buffer_priority_update():
    buf = PrioritizedReplayBuffer(capacity=1000)
    for i in range(50):
        state = np.zeros((7, 8, 8), dtype=np.float32)
        next_state = np.zeros((7, 8, 8), dtype=np.float32)
        mask = np.ones(192, dtype=bool)
        buf.push(state, 0, 0.0, next_state, False, mask)

    # Sample and update priorities — give index 0 a very high priority
    batch = buf.sample(10, beta=0.4)
    *_, indices = batch
    new_priorities = np.ones(len(indices)) * 0.001
    new_priorities[0] = 100.0  # Much higher priority for first sampled
    buf.update_priorities(indices, new_priorities)

    # Sample many times, check the high-priority transition appears often
    high_idx = indices[0]
    appearances = 0
    for _ in range(100):
        batch = buf.sample(10, beta=0.4)
        *_, sample_indices = batch
        if high_idx in sample_indices:
            appearances += 1

    assert appearances > 20, f"High-priority item appeared {appearances}/100 times"


def test_replay_buffer_capacity():
    buf = PrioritizedReplayBuffer(capacity=100)
    for i in range(200):
        state = np.zeros((7, 8, 8), dtype=np.float32)
        next_state = np.zeros((7, 8, 8), dtype=np.float32)
        mask = np.ones(192, dtype=bool)
        buf.push(state, 0, 0.0, next_state, False, mask)

    assert len(buf) == 100
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_agent.py -v -k "buffer" 2>&1 | head -10
```

Expected: FAIL

- [ ] **Step 3: Implement the replay buffer**

Create `training/agent/replay_buffer.py`:

```python
import numpy as np


class SumTree:
    """Binary tree where each parent is the sum of its children.
    Leaf nodes store transition priorities. Enables O(log n) proportional sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        """Add a new priority. Returns the data index."""
        idx = self.data_pointer
        tree_idx = self.data_pointer + self.capacity - 1
        self._update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, data_idx: int, priority: float):
        tree_idx = data_idx + self.capacity - 1
        self._update(tree_idx, priority)

    def _update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, value: float) -> int:
        """Sample a data index proportional to priority."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return data_idx

    def priority(self, data_idx: int) -> float:
        return float(self.tree[data_idx + self.capacity - 1])


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 500_000, alpha: float = 0.6, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

        self.states = np.zeros((capacity, 7, 8, 8), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 7, 8, 8), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, 192), dtype=bool)

        self._max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ):
        idx = self.tree.add(self._max_priority ** self.alpha)
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.next_masks[idx] = next_action_mask

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        indices: list[int] = []
        priorities = np.zeros(batch_size, dtype=np.float64)
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx = self.tree.get(value)
            # Clamp to valid range
            idx = max(0, min(idx, len(self) - 1))
            indices.append(idx)
            priorities[i] = self.tree.priority(idx)

        # Importance sampling weights
        priorities = np.clip(priorities, self.epsilon, None)
        probs = priorities / self.tree.total()
        weights = (len(self) * probs) ** (-beta)
        weights = weights / weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.next_masks[indices],
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: list[int], td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self._max_priority = max(self._max_priority, abs(float(td_error)) + self.epsilon)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_agent.py -v 2>&1
```

Expected: 7 tests PASS (4 model + 3 buffer)

- [ ] **Step 5: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/agent/replay_buffer.py training/tests/test_agent.py
git commit -m "feat: add prioritized replay buffer with sum-tree and tests"
```

---

### Task 4: DQN Trainer

**Files:**
- Create: `training/agent/dqn.py`
- Modify: `training/tests/test_agent.py` (append trainer tests)

- [ ] **Step 1: Write trainer tests**

Append to `training/tests/test_agent.py`:

```python
from agent.dqn import DQNTrainer
from env.block_blast_env import BlockBlastEnv


def _make_trainer():
    env = BlockBlastEnv()
    config = {
        "training": {
            "batch_size": 32,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "replay_buffer_size": 1000,
            "min_replay_size": 50,
            "target_update_freq": 100,
            "train_freq": 4,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 1000,
            "total_steps": 500,
        }
    }
    return DQNTrainer(env, config, device=torch.device("cpu"))


def test_select_action_greedy():
    trainer = _make_trainer()
    obs, info = trainer.env.reset(seed=42)
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device)
    mask = info["action_mask"]

    action = trainer.select_action(state, mask, epsilon=0.0)
    assert 0 <= action < 192
    assert mask[action], "Greedy action must be valid"


def test_select_action_explore():
    trainer = _make_trainer()
    obs, info = trainer.env.reset(seed=42)
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device)
    mask = info["action_mask"]

    # With epsilon=1.0, should always pick randomly from valid actions
    actions = set()
    for _ in range(50):
        a = trainer.select_action(state, mask, epsilon=1.0)
        assert mask[a], "Random action must be valid"
        actions.add(a)

    # Should have picked multiple different actions
    assert len(actions) > 1, "epsilon=1.0 should explore different actions"


def test_training_step():
    trainer = _make_trainer()
    # Fill buffer with some transitions
    obs, info = trainer.env.reset(seed=1)
    for _ in range(60):
        mask = info["action_mask"]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            obs, info = trainer.env.reset(seed=np.random.randint(10000))
            continue
        action = valid[np.random.randint(len(valid))]
        next_obs, reward, terminated, truncated, next_info = trainer.env.step(action)
        next_mask = next_info["action_mask"]
        trainer.buffer.push(obs, action, reward, next_obs, terminated, next_mask)
        if terminated:
            obs, info = trainer.env.reset(seed=np.random.randint(10000))
        else:
            obs, info = next_obs, next_info

    loss = trainer.train_step(batch_size=32, beta=0.4)
    assert loss is not None
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_agent.py -v -k "select or training_step" 2>&1 | head -10
```

Expected: FAIL

- [ ] **Step 3: Implement the DQN trainer**

Create `training/agent/dqn.py`:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import DuelingDQN
from .replay_buffer import PrioritizedReplayBuffer


class DQNTrainer:
    def __init__(self, env, config: dict, device: torch.device):
        self.env = env
        self.config = config["training"]
        self.device = device

        self.policy_net = DuelingDQN().to(device)
        self.target_net = DuelingDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config["learning_rate"],
        )
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config["replay_buffer_size"],
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss

    def select_action(
        self,
        state: torch.Tensor,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0  # Should not happen if env is correct

        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(state).squeeze(0)  # (192,)
            self.policy_net.train()

        # Mask invalid actions
        q_np = q_values.cpu().numpy()
        q_np[~action_mask] = -np.inf
        return int(np.argmax(q_np))

    def train_step(self, batch_size: int, beta: float) -> float:
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, next_masks, weights, indices = (
            self.buffer.sample(batch_size, beta)
        )

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_masks_t = torch.tensor(next_masks, dtype=torch.bool, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        gamma = self.config["gamma"]

        # Current Q-values
        q_current = self.policy_net(states_t)
        q_current = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            # Policy net selects best next action (masked)
            q_next_policy = self.policy_net(next_states_t)
            q_next_policy[~next_masks_t] = float("-inf")
            best_next_actions = q_next_policy.argmax(dim=1)

            # Target net evaluates that action
            q_next_target = self.target_net(next_states_t)
            q_next_value = q_next_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            target = rewards_t + gamma * q_next_value * (1 - dones_t)

        # Compute loss
        td_errors = target - q_current
        loss_per_sample = self.loss_fn(q_current, target)
        loss = (loss_per_sample * weights_t).mean()

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, path: str, step: int, epsilon: float, scores: list[float]):
        torch.save(
            {
                "step": step,
                "epsilon": epsilon,
                "scores": scores,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_agent.py -v 2>&1
```

Expected: 10 tests PASS (4 model + 3 buffer + 3 trainer)

- [ ] **Step 5: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/agent/dqn.py training/tests/test_agent.py
git commit -m "feat: add DQN trainer with action masking, double DQN, and tests"
```

---

### Task 5: Training Script

**Files:**
- Create: `training/scripts/train.py`

- [ ] **Step 1: Implement the training script**

Create `training/scripts/train.py`:

```python
#!/usr/bin/env python3
"""Block Blast DQN training script.

Usage:
    cd training
    uv run python scripts/train.py
    uv run python scripts/train.py --config configs/default.yaml --steps 100000
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# Add parent dir to path so imports work when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dqn import DQNTrainer
from agent.model import get_device
from env.block_blast_env import BlockBlastEnv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Block Blast DQN agent")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total training steps",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Resume from checkpoint path",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tc = config["training"]
    if args.steps:
        tc["total_steps"] = args.steps

    device = get_device()
    print(f"Device: {device}")

    env = BlockBlastEnv(config_path=args.config)
    trainer = DQNTrainer(env, config, device)

    # Resume from checkpoint if provided
    start_step = 0
    epsilon = tc["epsilon_start"]
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = trainer.load_checkpoint(args.checkpoint)
        start_step = ckpt["step"]
        epsilon = ckpt["epsilon"]
        print(f"Resumed from step {start_step}")

    # Setup logging
    log_dir = Path("logs") / f"run_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"Logging to {log_dir}")

    # Setup checkpoints
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # Training state
    total_steps = tc["total_steps"]
    batch_size = tc["batch_size"]
    min_replay = tc["min_replay_size"]
    train_freq = tc["train_freq"]
    target_freq = tc["target_update_freq"]
    eps_start = tc["epsilon_start"]
    eps_end = tc["epsilon_end"]
    eps_decay = tc["epsilon_decay_steps"]

    # Rolling metrics
    episode_rewards = deque(maxlen=100)
    episode_scores = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_scores: list[tuple[float, str]] = []  # (score, path)

    # Episode state
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    step = start_step
    t_start = time.time()

    print(f"Training for {total_steps} steps...")

    while step < total_steps:
        # Epsilon decay
        epsilon = max(eps_end, eps_start - (step / eps_decay) * (eps_start - eps_end))

        # Select action
        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask = info["action_mask"]
        action = trainer.select_action(state_t, mask, epsilon)

        # Step environment
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_mask = next_info["action_mask"]

        # Store transition
        trainer.buffer.push(obs, action, reward, next_obs, done, next_mask)

        episode_reward += reward
        episode_length += 1

        # Train
        if step % train_freq == 0 and len(trainer.buffer) >= min_replay:
            beta = min(1.0, 0.4 + (step / total_steps) * 0.6)
            loss = trainer.train_step(batch_size, beta)

            if step % 1000 == 0:
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/epsilon", epsilon, step)
                writer.add_scalar("train/buffer_size", len(trainer.buffer), step)

        # Update target network
        if step % target_freq == 0 and step > 0:
            trainer.update_target()

        # Episode end
        if done:
            game_score = env._state.score if env._state else 0
            episode_rewards.append(episode_reward)
            episode_scores.append(game_score)
            episode_lengths.append(episode_length)
            episode_count += 1

            writer.add_scalar("episode/reward", episode_reward, step)
            writer.add_scalar("episode/score", game_score, step)
            writer.add_scalar("episode/length", episode_length, step)

            if episode_count % 10 == 0:
                elapsed = time.time() - t_start
                sps = step / max(elapsed, 1)
                mean_score = np.mean(episode_scores) if episode_scores else 0
                mean_reward = np.mean(episode_rewards) if episode_rewards else 0
                print(
                    f"Step {step:>8d}/{total_steps} | "
                    f"Ep {episode_count:>5d} | "
                    f"Score {mean_score:>6.0f} | "
                    f"Reward {mean_reward:>7.1f} | "
                    f"Eps {epsilon:.3f} | "
                    f"SPS {sps:.0f}"
                )

            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
        else:
            obs, info = next_obs, next_info

        # Checkpoint every 50k steps
        if step > 0 and step % 50_000 == 0:
            mean_score = float(np.mean(episode_scores)) if episode_scores else 0.0
            ckpt_path = str(ckpt_dir / f"checkpoint_{step}.pt")
            trainer.save_checkpoint(ckpt_path, step, epsilon, list(episode_scores))
            print(f"  Saved checkpoint: {ckpt_path} (mean_score={mean_score:.0f})")

            # Keep top 5 checkpoints by score
            best_scores.append((mean_score, ckpt_path))
            best_scores.sort(key=lambda x: x[0], reverse=True)
            while len(best_scores) > 5:
                _, old_path = best_scores.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)

        step += 1

    # Save final model
    final_path = str(ckpt_dir / "final_model.pt")
    trainer.save_checkpoint(final_path, step, epsilon, list(episode_scores))
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total episodes: {episode_count}")
    if episode_scores:
        print(f"Final mean score (last 100): {np.mean(episode_scores):.0f}")

    writer.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test the training script (short run)**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run python scripts/train.py --steps 200 2>&1 | tail -10
```

Expected: Runs without error, prints step progress, saves final model.

- [ ] **Step 3: Verify checkpoint was saved**

```bash
ls -la /Users/mykechen/Desktop/APPS/blockblast/training/checkpoints/
```

Expected: `final_model.pt` exists

- [ ] **Step 4: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/scripts/train.py
git commit -m "feat: add training script with TensorBoard logging and checkpoints"
```

---

### Task 6: Checkpoint Save/Load Test

**Files:**
- Modify: `training/tests/test_agent.py` (append checkpoint test)

- [ ] **Step 1: Write checkpoint test**

Append to `training/tests/test_agent.py`:

```python
import tempfile
import os


def test_checkpoint_save_load():
    trainer1 = _make_trainer()
    # Do a few random updates to change weights from init
    obs, info = trainer1.env.reset(seed=1)
    for _ in range(60):
        mask = info["action_mask"]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            obs, info = trainer1.env.reset(seed=np.random.randint(10000))
            continue
        action = valid[np.random.randint(len(valid))]
        next_obs, reward, terminated, _, next_info = trainer1.env.step(action)
        trainer1.buffer.push(obs, action, reward, next_obs, terminated, next_info["action_mask"])
        if terminated:
            obs, info = trainer1.env.reset(seed=np.random.randint(10000))
        else:
            obs, info = next_obs, next_info

    trainer1.train_step(32, 0.4)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")
        trainer1.save_checkpoint(path, step=100, epsilon=0.5, scores=[10.0, 20.0])

        # Load into a new trainer
        trainer2 = _make_trainer()
        ckpt = trainer2.load_checkpoint(path)

        assert ckpt["step"] == 100
        assert ckpt["epsilon"] == 0.5
        assert ckpt["scores"] == [10.0, 20.0]

        # Verify weights match
        for p1, p2 in zip(trainer1.policy_net.parameters(), trainer2.policy_net.parameters()):
            assert torch.allclose(p1, p2), "Policy net weights don't match"
```

- [ ] **Step 2: Run full agent test suite**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/test_agent.py -v 2>&1
```

Expected: 11 tests PASS

- [ ] **Step 3: Run full project test suite**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast/training
uv run pytest tests/ -v 2>&1 | tail -10
```

Expected: all tests PASS (38 parity + 16 env + 11 agent = 65)

- [ ] **Step 4: Commit**

```bash
cd /Users/mykechen/Desktop/APPS/blockblast
git add training/tests/test_agent.py
git commit -m "feat: add checkpoint save/load test, complete agent test suite"
```

---

## Summary

| Task | Component | Test Count | Depends On |
|------|-----------|------------|------------|
| 1 | Dependencies + package init | 0 | — |
| 2 | Dueling DQN model | 4 | 1 |
| 3 | Prioritized replay buffer | 3 | 1 |
| 4 | DQN trainer | 3 | 2, 3 |
| 5 | Training script | 0 (smoke test) | 4 |
| 6 | Checkpoint test | 1 | 4 |

Tasks 2 and 3 can be parallelized. Total: 11 new tests. Tasks are mostly sequential with the 2/3 parallel opportunity.
