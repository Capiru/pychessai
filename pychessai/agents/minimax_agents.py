import chess as ch

from pychessai.agents import Agent
from pychessai.move_choice.minimax import (
    minimax,
    minimax_with_pruning,
    minimax_with_pruning_positionredundancy,
)
from pychessai.move_choice.utils import legal_moves
from pychessai.utils.eval import get_simple_board_evaluation
from pychessai.utils.policy import get_sorted_move_list


class MinimaxAgent(Agent):
    def __init__(
        self,
        depth=3,
        board=ch.Board(),
        is_white=True,
        search_function=minimax,
        policy_function=legal_moves,
    ):
        super().__init__(depth, board, is_white)
        self.positions = 0
        self.search_function = search_function
        self.policy_function = policy_function

    def choose_move(self, board):
        self.board = board
        score, move, positions = self.search_function(
            self.board, self.depth, self.is_white
        )
        self.positions += positions
        self.eval = score
        return [move]

    def get_board_evaluation(self, board: ch.Board) -> float:
        return get_simple_board_evaluation(board)


class MinimaxPruningAgent(MinimaxAgent):
    def __init__(
        self,
        depth=3,
        board=ch.Board(),
        is_white=True,
        search_function=minimax_with_pruning,
        policy_function=legal_moves,
    ):
        super().__init__(depth, board, is_white, search_function, policy_function)


class MinimaxPruningPositionRedundancyAgent(MinimaxAgent):
    def __init__(
        self,
        depth=3,
        board=ch.Board(),
        is_white=True,
        search_function=minimax_with_pruning_positionredundancy,
        policy_function=get_sorted_move_list,
    ):
        super().__init__(depth, board, is_white, search_function, policy_function)
        self.positions_analysed = {}

    def choose_move(self, board):
        self.board = board
        score, move, positions, positions_analysed = self.search_function(
            self.board, self.depth, self.is_white
        )
        self.positions_analysed.update(positions_analysed)
        self.positions += positions
        self.eval = score
        return [move]
