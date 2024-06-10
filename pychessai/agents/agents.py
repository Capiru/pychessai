from abc import ABC, abstractmethod
from typing import Any, Callable

import chess as ch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pychessai.models import Model


class Agent(ABC):
    def __init__(
        self,
        depth: int,
        board: ch.Board,
        is_white: bool,
        search_function: Callable,
        policy_function: Callable,
        eval_function: Callable,
    ) -> None:
        super().__init__()
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.search_function = search_function
        self.policy_function = policy_function
        self.eval_function = eval_function
        self.positions = 0
        self.positions_analysed: Any = {}

    @abstractmethod
    def choose_move(self, board: ch.Board) -> ch.Move:
        raise Exception("You should implement a choose_move method")

    def update_board(self, board: ch.Board) -> None:
        self.board = board

    @abstractmethod
    def get_board_evaluation(self, board: ch.Board) -> float:
        # TODO: This function might no longer be needed.
        raise Exception("You should implement a get_board_evaluation method")


class TrainableAgent(Agent):
    def __init__(
        self,
        depth: int,
        board: ch.Board,
        is_white: bool,
        search_function: Callable,
        policy_function: Callable,
        eval_function: Callable,
        training: bool,
        nn: nn.Module,
        model_class: Callable[[nn.Module, dict], Model],
        model_parameters: dict,
        training_parameters: dict,
    ) -> None:
        super().__init__(
            depth, board, is_white, search_function, policy_function, eval_function
        )
        self.training = training
        self.model_class = model_class(nn, model_parameters)
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters

    @abstractmethod
    def get_model_name(self) -> str:
        raise Exception("You should implement a get_model_name method")

    def create_dataloader(self, dataset: Dataset) -> DataLoader:
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_parameters["batch_size"],
            shuffle=self.training_parameters["shuffle_data"],
            num_workers=self.training_parameters["num_workers"],
        )
        return dataloader
