from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch.nn as nn


class Model(ABC):
    def __init__(
        self,
        nn: nn.Module,
        model_parameters: dict,
    ):
        super().__init__()
        self.nn = nn
        self.model_parameters = model_parameters

    @abstractmethod
    def setup_model(self) -> None:
        raise Exception("You should implement a setup_model method")

    @abstractmethod
    def predict(self, board) -> Any:
        raise Exception("You should implement the predict method!")

    @abstractmethod
    def convert_board_to_input(self, board):
        raise Exception("You should implement the convert_board_to_input method!")


class ValueModel(Model):
    def __init__(
        self,
        nn: nn.Module,
        model_parameters: dict,
    ):
        super().__init__(nn, model_parameters)

    @abstractmethod
    def predict(self, board) -> float:
        raise Exception("You should implement the predict method!")


class PolicyModel(Model):
    def __init__(
        self,
        nn: nn.Module,
        model_parameters: dict,
    ):
        super().__init__(nn, model_parameters)

    @abstractmethod
    def predict(self, board) -> list[float]:
        raise Exception("You should implement the predict method!")


class TwoHeadsModel(Model):
    def __init__(
        self,
        nn: nn.Module,
        model_parameters: dict,
    ):
        super().__init__(nn, model_parameters)

    @abstractmethod
    def predict(self, board) -> Tuple[float, list[float]]:
        raise Exception("You should implement the predict method!")
