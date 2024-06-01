import chess as ch

from pychessai.agents import Agent
from pychessai.utils.eval import get_board_evaluation
from pychessai.utils.move_choice import minimax


class MinimaxAgent(Agent):
    def __init__(self, depth=3, board=ch.Board(), is_white=True):
        super().__init__(depth, board, is_white)
        self.positions = 0

    def choose_move(self, board):
        self.board = board
        score, move, positions = minimax(self.board, self.depth, self.is_white)
        self.positions += positions
        self.eval = score
        return [move]

    def get_board_evaluation(self, board: ch.Board) -> float:
        return get_board_evaluation(board)
