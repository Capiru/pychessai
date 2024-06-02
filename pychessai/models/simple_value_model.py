from typing import Any

import chess as ch
import torch
import torch.nn as nn
import torch.nn.functional as F

from pychessai.models import ValueModel
from pychessai.utils.board_utils import get_board_as_tensor


class SimpleValueNetwork(nn.Module):
    def __init__(self, model_parameters):
        super().__init__()
        input_channel_size = model_parameters["input_channel_size"]
        self.conv1 = nn.Conv2d(input_channel_size, 24, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleValueModel(ValueModel):
    def __init__(
        self,
        nn: nn.Module = SimpleValueNetwork,
        model_parameters: dict = {"input_channel_size": 12},
    ):
        super().__init__(nn, model_parameters)

    def model_setup(self):
        self.model = self.nn(self.model_parameters)

    def predict(self, board: ch.Board) -> Any:
        model_input = self.convert_board_to_input(board)
        return self.model.forward(model_input)

    def convert_board_to_input(self, board: ch.Board):
        return get_board_as_tensor(board)
