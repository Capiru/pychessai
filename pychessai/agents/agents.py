from abc import ABC, abstractmethod

import chess as ch


class Agent(ABC):
    def __init__(self, depth: int, board: ch.Board, is_white: bool) -> None:
        super().__init__()
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white

    @abstractmethod
    def choose_move(self, board: ch.Board) -> ch.Move:
        raise Exception("You should implement a choose_move method")

    def update_board(self, board: ch.Board) -> None:
        self.board = board

    @abstractmethod
    def get_board_evaluation(self, board: ch.Board) -> float:
        raise Exception("You should implement a get_board_evaluation method")


class TrainableAgent(Agent):
    def __init__(
        self,
        depth: int,
        board: ch.Board,
        is_white: bool,
        training: bool,
        model_parameters: dict,
    ) -> None:
        super().__init__(depth, board, is_white)
        self.training = training
        self.model_parameters = model_parameters

    @abstractmethod
    def setup_model(self) -> None:
        raise Exception("You should implement a setup_model method")

    @abstractmethod
    def get_model_name(self) -> str:
        raise Exception("You should implement a get_model_name method")
