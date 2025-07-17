# PyTorch
import torch
import torch.nn as nn
from torch.distributions import Categorical


device = "mps"


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
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


class PolicyV2(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(PolicyV2, self).__init__()
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
