import torch
import torch.nn as nn
import torch.optim as optimize
import torch.nn.functional as F


class QNet(nn.Module):

  def __init__(self, n_states, n_actions, seed = 2534554):

    super(QNet, self).__init__()
    torch.manual_seed(seed)
    self.layer1 = nn.Linear(n_states, 400)
    self.layer2 = nn.Linear(400,200)
    self.layer3 = nn.Linear(200, n_actions)

  def forward(self, x):

    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = self.layer3(x)
    return x