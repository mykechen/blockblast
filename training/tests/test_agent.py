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

    finite = torch.isfinite(masked_q[0])
    assert finite.sum().item() == 3


def test_model_dueling_structure():
    device = torch.device("cpu")
    model = DuelingDQN().to(device)
    x = torch.randn(4, 7, 8, 8, device=device)

    features = model.shared(model.flatten(model.relu3(model.bn3(model.conv3(
        model.relu2(model.bn2(model.conv2(
            model.relu1(model.bn1(model.conv1(x)))
        )))
    )))))

    value = model.value_head(features)
    advantage = model.advantage_head(features)

    assert value.shape == (4, 1)
    assert advantage.shape == (4, 192)
