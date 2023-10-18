"""
Definition of the Neural Network used to train the agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDQN(nn.Module):
    def __init__(self):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out
