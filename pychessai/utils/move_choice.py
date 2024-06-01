from typing import Union

import chess as ch
import numpy as np

from pychessai.utils.eval import get_board_evaluation


def random_choice(
    possible_moves: list,
    probability_map: Union[list, np.ndarray] = [],
    exploration_size: int = 1,
):
    if probability_map is None or len(probability_map) == 0:
        probability_map = [1 / len(possible_moves) for x in range(len(possible_moves))]
    return np.random.choice(possible_moves, size=exploration_size, p=probability_map)


def legal_moves(board: ch.Board):
    return list(board.legal_moves)


def minimax(board, depth, is_player, positions=0, agent=None):
    # depth 1 - 21 positions - time 0.003461
    # depth 2 - 621 positions - time 0.091520
    # depth 3 - 13781 positions - time 1.991260
    # depth 4 - 419166 positions - time 61.41497
    positions += 1
    if depth == 0 or board.is_game_over():
        # this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if agent is None:
            return get_board_evaluation(board), None, positions
        else:
            return agent.get_board_evaluation(board), None, positions
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in list(board.legal_moves):
            # print(max_eval,best_move,move,board.fen())
            board.push(move)
            eval, a, positions = minimax(board, depth - 1, False, positions, agent)
            if eval >= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
        return max_eval, best_move, positions
    else:
        min_eval = np.inf
        best_move = None
        for move in list(board.legal_moves):
            # print(min_eval,best_move,move,board.fen())
            board.push(move)
            eval, a, positions = minimax(board, depth - 1, True, positions, agent)
            if eval <= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
        return min_eval, best_move, positions
