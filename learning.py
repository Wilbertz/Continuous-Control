from unityagents import UnityEnvironment
import numpy as np

from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from agent import Agent

env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')
