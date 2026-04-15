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
            q_values = self.policy_net(state).squeeze(0)
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

        gamma = self.config["gamma"]

        q_current = self.policy_net(states_t)
        q_current = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_policy = self.policy_net(next_states_t)
            q_next_policy[~next_masks_t] = float("-inf")
            best_next_actions = q_next_policy.argmax(dim=1)

            q_next_target = self.target_net(next_states_t)
            q_next_value = q_next_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            target = rewards_t + gamma * q_next_value * (1 - dones_t)

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
