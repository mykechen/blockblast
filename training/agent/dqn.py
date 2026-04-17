import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import DuelingDQN, build_model
from .replay_buffer import PrioritizedReplayBuffer


class DQNTrainer:
    def __init__(self, env, config: dict, device: torch.device):
        self.env = env
        self.config = config["training"]
        self.device = device
        self.n_step = int(self.config.get("n_step", 1))

        model_cfg = config.get("model")
        if model_cfg is None:
            self.policy_net = DuelingDQN().to(device)
            self.target_net = DuelingDQN().to(device)
        else:
            self.policy_net = build_model(model_cfg).to(device)
            self.target_net = build_model(model_cfg).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.last_mean_q: float = 0.0

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config["learning_rate"],
        )
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config["replay_buffer_size"],
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Return (B, action_size) expected Q-values. Uniform interface for eval scripts."""
        return self.policy_net(state)

    def select_action(
        self,
        state: torch.Tensor,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0

        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.get_q_values(state).squeeze(0)
            self.policy_net.train()

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

        gamma_n = self.config["gamma"] ** self.n_step

        q_current = self.policy_net(states_t)
        q_current = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_policy = self.policy_net(next_states_t)
            q_next_policy[~next_masks_t] = float("-inf")
            best_next_actions = q_next_policy.argmax(dim=1)

            q_next_target = self.target_net(next_states_t)
            q_next_value = q_next_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            target = rewards_t + gamma_n * q_next_value * (1 - dones_t)

        td_errors = target - q_current
        loss_per_sample = self.loss_fn(q_current, target)
        loss = (loss_per_sample * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.last_mean_q = float(q_current.detach().mean().item())
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


class C51Trainer:
    """Categorical DQN (C51) trainer. Same interface as DQNTrainer."""

    def __init__(self, env, config: dict, device: torch.device):
        self.env = env
        self.config = config["training"]
        self.device = device
        self.n_step = int(self.config.get("n_step", 1))

        model_cfg = config.get("model")
        self.policy_net = build_model(model_cfg).to(device)
        self.target_net = build_model(model_cfg).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.n_atoms = self.policy_net.n_atoms
        self.v_min = self.policy_net.v_min
        self.v_max = self.policy_net.v_max
        self.support = self.policy_net.support
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.last_mean_q: float = 0.0

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config["learning_rate"],
        )
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config["replay_buffer_size"],
        )

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy_net.q_values(state)

    def select_action(
        self,
        state: torch.Tensor,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0

        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.get_q_values(state).squeeze(0)
            self.policy_net.train()

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

        gamma_n = self.config["gamma"] ** self.n_step
        B = states_t.size(0)

        log_p_all = self.policy_net(states_t)
        a_idx = actions_t.view(B, 1, 1).expand(B, 1, self.n_atoms)
        log_p = log_p_all.gather(1, a_idx).squeeze(1)

        with torch.no_grad():
            q_next = self.policy_net.q_values(next_states_t)
            q_next[~next_masks_t] = float("-inf")
            best_next_actions = q_next.argmax(dim=1)

            log_p_next = self.target_net(next_states_t)
            a_next_idx = best_next_actions.view(B, 1, 1).expand(B, 1, self.n_atoms)
            p_next = log_p_next.gather(1, a_next_idx).squeeze(1).exp()

            Tz = rewards_t.unsqueeze(1) + gamma_n * (1 - dones_t.unsqueeze(1)) * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            lo = b.floor().long()
            hi = b.ceil().long()
            lo = lo.clamp(0, self.n_atoms - 1)
            hi = hi.clamp(0, self.n_atoms - 1)

            m = torch.zeros(B, self.n_atoms, device=self.device)
            m.scatter_add_(1, lo, p_next * (hi.float() - b))
            m.scatter_add_(1, hi, p_next * (b - lo.float()))

        loss_per_sample = -(m * log_p).sum(dim=1)
        loss = (loss_per_sample * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, loss_per_sample.detach().cpu().numpy())

        self.last_mean_q = float(self.get_q_values(states_t[:1]).detach().max().item())
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
