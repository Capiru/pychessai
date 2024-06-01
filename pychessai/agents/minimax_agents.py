import chess as ch

from pychessai.agents import Agent
from pychessai.utils.eval import get_board_evaluation
from pychessai.utils.move_choice import minimax, minimax_with_pruning


class MinimaxAgent(Agent):
    def __init__(
        self, depth=3, board=ch.Board(), is_white=True, search_function=minimax
    ):
        super().__init__(depth, board, is_white)
        self.positions = 0
        self.search_function = search_function

    def choose_move(self, board):
        self.board = board
        score, move, positions = self.search_function(
            self.board, self.depth, self.is_white
        )
        self.positions += positions
        self.eval = score
        return [move]

    def get_board_evaluation(self, board: ch.Board) -> float:
        return get_board_evaluation(board)


class MinimaxPruningAgent(MinimaxAgent):
    def __init__(
        self,
        depth=3,
        board=ch.Board(),
        is_white=True,
        search_function=minimax_with_pruning,
    ):
        super().__init__(depth, board, is_white, search_function)
