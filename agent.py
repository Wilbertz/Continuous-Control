import numpy as np
import copy
from collections import namedtuple, deque
import random
from model import Actor

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    pass