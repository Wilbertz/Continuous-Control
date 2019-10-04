import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f


def hidden_init(layer):
    """"
        Method used by both the Actor and Critic to initialize the hidden layer.

        Xavier initialisation helps to keep the signal from exploding to a high value or
        vanishing to zero. In other words, we need to initialize the weights in such a way
        that the variance remains the same for x and y.
        Args:
                layer: The hidden layer that has to be initialized.
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int, fc1_units: int = 400, fc2_units: int = 300) \
            -> None:
        """
           Initialize fields and build the model. The architecture of the network is based on the parameters
           given in  "Continuous control with deep reinforcement learning" (https://arxiv.org/abs/1509.02971).
                Args:
                    state_size (int): Dimension of each state
                    action_size (int): Dimension of each action
                    seed (int): Random seed
                    fc1_units (int): Number of nodes in first hidden layer
                    fc2_units (int): Number of nodes in second hidden layer

        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
            Forward propagation of input.
            Args:
                state (PyTorch model): The observed state used to predict actions
        """
        x = f.relu(self.bn1(self.fc1(state)))
        x = f.relu(self.fc2(x))
        return f.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int, fcs1_units: int = 400, fc2_units: int = 300) \
            -> None:
        """Initialize parameters and build model. The architecture of the network is based on the parameters
        given in  "Continuous control with deep reinforcement learning" (https://arxiv.org/abs/1509.02971).
        Params: int
        ======
            state_size (int): Dimension of state vector.
            action_size (int): Dimension of action vector.
            seed (int): Used to initialize random number generator.
            fcs1_units (int): Number of nodes in hidden layer 1
            fc2_units (int): Number of nodes in hidden layer 2
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        xs = f.relu(self.bn1(self.fcs1(state)))
        
        x = torch.cat((xs, action), dim=1)
        x = f.relu(self.fc2(x))
        return self.fc3(x)
