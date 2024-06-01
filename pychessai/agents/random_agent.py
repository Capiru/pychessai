import chess as ch

from pychessai.agents import Agent
from pychessai.move_choice import legal_moves, random_choice


class RandomAgent(Agent):
    def __init__(self, board: ch.Board, is_white: bool, depth: int = 0):
        super().__init__(depth, board, is_white)

    def choose_move(self, board):
        return random_choice(legal_moves(board), None, 1)

    def get_board_evaluation(self, board: ch.Board = ch.Board()) -> float:
        return 0.0