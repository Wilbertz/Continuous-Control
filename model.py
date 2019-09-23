import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=128, leak=0.01, seed=42):
        super(Actor, self).__init__()
        self.leak = leak
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

    def forward(self, state):
        state = self.bn(state)
        x = f.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = f.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=128, fc3=128, leak=0.01, seed=42):
        super(Critic, self).__init__()

    def forward(self, *input):
        pass

