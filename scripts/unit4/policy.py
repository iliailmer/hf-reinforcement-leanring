# PyTorch
from typing import Optional
from loguru import logger
import torch
import torch.nn as nn
from torch.distributions import Categorical


device = "mps"


class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def act(self, frames, state):
        raise NotImplementedError


class Policy(BasePolicy):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x.softmax(1)

    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class PolicyV2(BasePolicy):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, 2 * h_size)
        self.fc3 = nn.Linear(2 * h_size, h_size)
        self.fc4 = nn.Linear(h_size, a_size)
        self.silu = nn.SiLU()
        self.lnorm1 = nn.LayerNorm(2 * h_size)
        self.lnorm2 = nn.LayerNorm(h_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.fc1(x))
        x = self.lnorm1(self.silu(self.fc2(x)))
        x = self.lnorm2(self.silu(self.fc3(x)))
        x = self.fc4(x)
        return x.softmax(1)

    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class PolicyCNN(BasePolicy):
    def __init__(
        self, in_channels: int, h_size: int, a_size: int, state_dim: int
    ) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=h_size,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=h_size,
                out_channels=h_size,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.agg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(h_size + state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, a_size),  # output logits
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, state_vector: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.agg(x).reshape((x.shape[0], -1))
        x = torch.cat([x, state_vector], dim=1)
        x = self.fc(x)
        return x.softmax(1)

    def act(self, frames, state):
        frames = torch.from_numpy(frames).float()
        frames = frames.permute((0, 3, 1, 2)).to(device)  # (4, 3, ..., ...)
        frames = frames.reshape((-1, *frames.shape[2:]))  # (4*3, ..., ..)
        frames = frames.unsqueeze(0)  # (1, 4*3, ..., ...)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(frames, state).cpu()
        m = Categorical(probs=probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
