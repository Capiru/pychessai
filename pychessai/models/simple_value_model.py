import chess as ch
import torch
import torch.nn as nn
import torch.nn.functional as F

from pychessai.models import ValueModel

class SimpleValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel_size = 12
        self.conv1 = nn.Conv2d(input_channel_size, 24, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SimpleValueModel(ValueModel):
    pass