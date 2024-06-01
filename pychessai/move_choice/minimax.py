import chess as ch
import numpy as np

from pychessai.move_choice import legal_moves
from pychessai.utils.eval import get_simple_board_evaluation
from pychessai.utils.policy import get_sorted_move_list


def minimax(
    board: ch.Board,
    depth: int,
    is_player: bool,
    positions=0,
    eval_function=get_simple_board_evaluation,
    policy_function=legal_moves,
):
    # depth 1 - 21 positions - time 0.003461
    # depth 2 - 621 positions - time 0.091520
    # depth 3 - 13781 positions - time 1.991260
    # depth 4 - 419166 positions - time 61.41497
    positions += 1
    if depth == 0 or board.is_game_over():
        # this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        return eval_function(board), None, positions
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in policy_function(board):
            # print(max_eval,best_move,move,board.fen())
            board.push(move)
            eval, a, positions = minimax(
                board, depth - 1, False, positions, eval_function
            )
            if eval >= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
        return max_eval, best_move, positions
    else:
        min_eval = np.inf
        best_move = None
        for move in policy_function(board):
            # print(min_eval,best_move,move,board.fen())
            board.push(move)
            eval, a, positions = minimax(
                board, depth - 1, True, positions, eval_function
            )
            if eval <= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
        return min_eval, best_move, positions


def minimax_with_pruning(
    board: ch.Board,
    depth: int,
    is_player: bool,
    alpha=-np.inf,
    beta=np.inf,
    eval_function=get_simple_board_evaluation,
    policy_function=legal_moves,
    positions=0,
):
    # depth 1 - 21 positions - time 0.003673
    # depth 2 - 70 positions - time 0.010080
    # depth 3 - 545 positions - time 0.0784910
    # depth 4 - 1964 positions - time 0.278105
    # depth 5 - 14877 positions - time 2.12180
    # depth 6 - 82579 positions - time 11.84326
    positions += 1
    if depth == 0 or board.is_game_over():
        # this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        return eval_function(board), None, positions
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in policy_function(board):
            board.push(move)
            eval, a, positions = minimax_with_pruning(
                board, depth - 1, False, alpha, beta, eval_function, positions=positions
            )
            if eval >= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move, positions
    else:
        min_eval = np.inf
        best_move = None
        for move in policy_function(board):
            board.push(move)
            eval, a, positions = minimax_with_pruning(
                board, depth - 1, True, alpha, beta, eval_function, positions=positions
            )
            if eval <= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move, positions


def minimax_with_pruning_positionredundancy(
    board: ch.Board,
    depth: int,
    is_player: bool,
    alpha=-np.inf,
    beta=np.inf,
    eval_function=get_simple_board_evaluation,
    policy_function=get_sorted_move_list,
    positions=0,
    positions_analysed={},
):
    # depth 1 - 21 positions - time 0.004315
    # depth 2 - 76 positions - time 0.033392
    # depth 3 - 687 positions - time 0.172937
    # depth 4 - 4007 positions - time 1.278452
    # depth 5 - 30086 positions - time 7.623218
    # depth 6 - 82579 positions - time 60.89466
    positions += 1
    if depth == 0 or board.is_game_over():
        # this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        try:
            eval = positions_analysed[board.board_fen() + str(depth)]
        except KeyError:
            eval = eval_function(board)
            positions_analysed[board.board_fen() + str(depth)] = eval
        return eval, None, positions, positions_analysed
    move_list = policy_function(board)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in move_list:
            board.push(move)
            try:
                eval = positions_analysed[board.board_fen() + str(depth)]
            except KeyError:
                (
                    eval,
                    a,
                    positions,
                    positions_analysed,
                ) = minimax_with_pruning_positionredundancy(
                    board,
                    depth - 1,
                    False,
                    alpha,
                    beta,
                    eval_function,
                    policy_function,
                    positions,
                    positions_analysed,
                )
                positions_analysed[board.board_fen() + str(depth)] = eval
            if eval >= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move, positions, positions_analysed
    else:
        min_eval = np.inf
        best_move = None
        for move in move_list:
            board.push(move)
            try:
                eval = positions_analysed[board.board_fen() + str(depth)]
            except KeyError:
                (
                    eval,
                    a,
                    positions,
                    positions_analysed,
                ) = minimax_with_pruning_positionredundancy(
                    board,
                    depth - 1,
                    True,
                    alpha,
                    beta,
                    eval_function,
                    policy_function,
                    positions,
                    positions_analysed,
                )
                positions_analysed[board.board_fen() + str(depth)] = eval
            if eval <= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move, positions, positions_analysed
