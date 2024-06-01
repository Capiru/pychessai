import chess as ch

from pychessai.agents import Agent
from pychessai.move_choice import legal_moves, random_choice


class RandomAgent(Agent):
    def __init__(
        self,
        board: ch.Board,
        is_white: bool,
        depth: int = 0,
        search_function=None,
        policy_function=None,
        eval_function=None,
    ):
        super().__init__(
            depth, board, is_white, search_function, policy_function, eval_function
        )

    def choose_move(self, board):
        return random_choice(legal_moves(board), None, 1)

    def get_board_evaluation(self, board: ch.Board = ch.Board()) -> float:
        return 0.0
