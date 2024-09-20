from typing import Callable

import chess as ch
import torch
import torch.nn as nn
import torch.nn.functional as Fs

from pychessai.agents import TrainableAgent
from pychessai.models import Model
from pychessai.models.simple_value_model import SimpleValueModel
from pychessai.move_choice.minimax import minimax
from pychessai.move_choice.utils import legal_moves


class SimpleTrainableAgent(TrainableAgent):
    def __init__(
        self,
        depth: int,
        board: ch.Board,
        is_white: bool,
        search_function: Callable = minimax,
        policy_function: Callable = legal_moves,
        eval_function: Callable = lambda x: x,
        training: bool = True,
        model_class: Model = SimpleValueModel,
        model_parameters: dict = {"input_channel_size": 12},
    ):
        super().__init__(
            depth,
            board,
            is_white,
            search_function,
            policy_function,
            eval_function,
            training,
            model_class,
            model_parameters,
        )
        self.setup_model()

    def choose_move(self, board):
        self.board = board
        score, move, positions = self.search_function(
            self.board,
            self.depth,
            self.is_white,
            eval_function=self.eval_function,
            policy_function=self.policy_function,
        )
        self.positions += positions
        self.eval = score
        return [move]

    def get_board_evaluation(self, board):
        return self.model.predict(board)

    def setup_model(self) -> None:
        self.model_class.model_setup()
        self.model = self.model_class.model

    def create_match_data(self, board, is_white, match_status):
        if match_status == 1 and is_white:
            # winner
            out = 1
        elif match_status == 0:
            out = 0.5
        else:
            out = 0

    def create_dataset(filepath):
        return dataset

    def train_policy_model(input_tensor, move_output):
        return None

    def legal_inference(legal_moves, move_outputs):
        # first transform move_outputs to san move
        output_reshaped = torch.reshape(move_outputs, (8, 8, 12))
        dic_encoder = {
            "p": 0,
            "P": 1,
            "r": 2,
            "R": 3,
            "n": 4,
            "N": 5,
            "b": 6,
            "B": 7,
            "q": 8,
            "Q": 9,
            "k": 10,
            "K": 11,
        }
        row_encoder = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
        # i need to get the max, index
        # then compare if move is in legal moves
        for move in move_outputs:
            if list(legal_moves):
                return move
