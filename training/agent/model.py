import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DuelingDQN(nn.Module):
    def __init__(self, in_channels: int = 9, action_size: int = 192):
        super().__init__()
        self.action_size = action_size

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

        self.shared = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

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

        value = self.value_head(x)
        advantage = self.advantage_head(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResidualDuelingDQN(nn.Module):
    """Wider residual CNN with dueling heads. Drop-in replacement for DuelingDQN."""

    def __init__(
        self,
        in_channels: int = 9,
        action_size: int = 192,
        hidden_channels: int = 96,
        num_blocks: int = 4,
        fc_hidden: int = 768,
        head_hidden: int = 256,
    ):
        super().__init__()
        self.action_size = action_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResBlock(hidden_channels) for _ in range(num_blocks)]
        )

        self.flatten = nn.Flatten()
        flat_size = hidden_channels * 8 * 8

        self.shared = nn.Sequential(
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(fc_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(fc_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.flatten(x)
        x = self.shared(x)
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class CategoricalDuelingDQN(nn.Module):
    """C51 distributional variant of ResidualDuelingDQN. Outputs (B, action_size, n_atoms) log-probs."""

    def __init__(
        self,
        in_channels: int = 9,
        action_size: int = 192,
        hidden_channels: int = 96,
        num_blocks: int = 4,
        fc_hidden: int = 768,
        head_hidden: int = 256,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 300.0,
    ):
        super().__init__()
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer(
            "support", torch.linspace(v_min, v_max, n_atoms)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResBlock(hidden_channels) for _ in range(num_blocks)]
        )

        self.flatten = nn.Flatten()
        flat_size = hidden_channels * 8 * 8

        self.shared = nn.Sequential(
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(fc_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, n_atoms),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(fc_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, action_size * n_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.flatten(x)
        x = self.shared(x)

        v = self.value_head(x).view(B, 1, self.n_atoms)
        a = self.advantage_head(x).view(B, self.action_size, self.n_atoms)
        logits = v + a - a.mean(dim=1, keepdim=True)
        return F.log_softmax(logits, dim=2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        log_p = self.forward(x)
        p = log_p.exp()
        return (p * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)


MODEL_REGISTRY = {
    "dueling": DuelingDQN,
    "residual": ResidualDuelingDQN,
    "categorical": CategoricalDuelingDQN,
}


def build_model(model_cfg: dict | None) -> nn.Module:
    """Factory: build a Q-net from a config dict like {type: 'residual', hidden_channels: 128}."""
    cfg = dict(model_cfg or {})
    model_type = cfg.pop("type", "dueling")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_type](**cfg)
