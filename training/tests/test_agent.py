import tempfile
import os
import torch
import numpy as np
from agent.model import DuelingDQN, CategoricalDuelingDQN, get_device
from agent.replay_buffer import PrioritizedReplayBuffer
from agent.dqn import DQNTrainer, C51Trainer
from env.block_blast_env import BlockBlastEnv


def test_model_forward_pass():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(1, 9, 8, 8, device=device)
    q_values = model(x)
    assert q_values.shape == (1, 192)
    assert not torch.isnan(q_values).any()


def test_model_batch_forward():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(32, 9, 8, 8, device=device)
    q_values = model(x)
    assert q_values.shape == (32, 192)


def test_model_with_action_mask():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(1, 9, 8, 8, device=device)
    q_values = model(x)

    mask = torch.zeros(192, dtype=torch.bool, device=device)
    mask[0] = True
    mask[10] = True
    mask[100] = True

    masked_q = q_values.clone()
    masked_q[0, ~mask] = float("-inf")

    finite = torch.isfinite(masked_q[0])
    assert finite.sum().item() == 3


def test_model_dueling_structure():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(4, 9, 8, 8, device=device)

    features = model.shared(model.flatten(model.relu3(model.bn3(model.conv3(
        model.relu2(model.bn2(model.conv2(
            model.relu1(model.bn1(model.conv1(x)))
        )))
    )))))

    value = model.value_head(features)
    advantage = model.advantage_head(features)

    assert value.shape == (4, 1)
    assert advantage.shape == (4, 192)


# --- Replay buffer tests ---

def test_replay_buffer_push_sample():
    buf = PrioritizedReplayBuffer(capacity=1000)
    for i in range(100):
        state = np.random.randn(9, 8, 8).astype(np.float32)
        next_state = np.random.randn(9, 8, 8).astype(np.float32)
        action_mask = np.ones(192, dtype=bool)
        buf.push(state, i % 192, float(i), next_state, False, action_mask)

    assert len(buf) == 100

    batch = buf.sample(32, beta=0.4)
    states, actions, rewards, next_states, dones, next_masks, weights, indices = batch

    assert states.shape == (32, 9, 8, 8)
    assert actions.shape == (32,)
    assert rewards.shape == (32,)
    assert next_states.shape == (32, 9, 8, 8)
    assert dones.shape == (32,)
    assert next_masks.shape == (32, 192)
    assert weights.shape == (32,)
    assert len(indices) == 32


def test_replay_buffer_priority_update():
    buf = PrioritizedReplayBuffer(capacity=1000)
    for i in range(50):
        state = np.zeros((9, 8, 8), dtype=np.float32)
        next_state = np.zeros((9, 8, 8), dtype=np.float32)
        mask = np.ones(192, dtype=bool)
        buf.push(state, 0, 0.0, next_state, False, mask)

    batch = buf.sample(10, beta=0.4)
    *_, indices = batch
    new_priorities = np.ones(len(indices)) * 0.001
    new_priorities[0] = 100.0
    buf.update_priorities(indices, new_priorities)

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
        state = np.zeros((9, 8, 8), dtype=np.float32)
        next_state = np.zeros((9, 8, 8), dtype=np.float32)
        mask = np.ones(192, dtype=bool)
        buf.push(state, 0, 0.0, next_state, False, mask)

    assert len(buf) == 100


# --- DQN Trainer tests ---

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

    actions = set()
    for _ in range(50):
        a = trainer.select_action(state, mask, epsilon=1.0)
        assert mask[a], "Random action must be valid"
        actions.add(a)

    assert len(actions) > 1, "epsilon=1.0 should explore different actions"


def test_training_step():
    trainer = _make_trainer()
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


def test_checkpoint_save_load():
    trainer1 = _make_trainer()
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

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")
        trainer1.save_checkpoint(path, step=100, epsilon=0.5, scores=[10.0, 20.0])

        trainer2 = _make_trainer()
        ckpt = trainer2.load_checkpoint(path)

        assert ckpt["step"] == 100
        assert ckpt["epsilon"] == 0.5
        assert ckpt["scores"] == [10.0, 20.0]

        for p1, p2 in zip(trainer1.policy_net.parameters(), trainer2.policy_net.parameters()):
            assert torch.allclose(p1, p2), "Policy net weights don't match"


# --- C51 model tests ---

def test_c51_model_forward():
    device = torch.device("cpu")
    model = CategoricalDuelingDQN(n_atoms=51).to(device)
    x = torch.randn(1, 9, 8, 8, device=device)
    log_p = model(x)
    assert log_p.shape == (1, 192, 51)
    assert (log_p <= 0).all(), "log_softmax output should be <= 0"
    p = log_p.exp()
    sums = p.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_c51_model_q_values():
    device = torch.device("cpu")
    model = CategoricalDuelingDQN(n_atoms=51).to(device)
    x = torch.randn(4, 9, 8, 8, device=device)
    q = model.q_values(x)
    assert q.shape == (4, 192)
    assert not torch.isnan(q).any()


# --- C51 Trainer tests ---

def _make_c51_trainer():
    env = BlockBlastEnv()
    config = {
        "algorithm": "c51",
        "model": {
            "type": "categorical",
            "n_atoms": 51,
            "v_min": -10.0,
            "v_max": 300.0,
            "hidden_channels": 32,
            "num_blocks": 1,
            "fc_hidden": 128,
            "head_hidden": 64,
        },
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
            "n_step": 1,
        },
    }
    return C51Trainer(env, config, device=torch.device("cpu"))


def test_c51_select_action():
    trainer = _make_c51_trainer()
    obs, info = trainer.env.reset(seed=42)
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask = info["action_mask"]
    action = trainer.select_action(state, mask, epsilon=0.0)
    assert 0 <= action < 192
    assert mask[action], "C51 greedy action must be valid"


def test_c51_train_step():
    trainer = _make_c51_trainer()
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
    assert np.isfinite(loss), f"C51 loss is not finite: {loss}"


def test_c51_checkpoint_save_load():
    trainer1 = _make_c51_trainer()
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

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_c51_ckpt.pt")
        trainer1.save_checkpoint(path, step=100, epsilon=0.5, scores=[10.0, 20.0])

        trainer2 = _make_c51_trainer()
        ckpt = trainer2.load_checkpoint(path)

        assert ckpt["step"] == 100
        assert ckpt["epsilon"] == 0.5

        for p1, p2 in zip(trainer1.policy_net.parameters(), trainer2.policy_net.parameters()):
            assert torch.allclose(p1, p2), "C51 policy net weights don't match"
