import chess as ch
import numpy as np

from pychessai.move_choice import legal_moves
from pychessai.utils.eval import get_simple_board_evaluation
from pychessai.utils.policy import get_sorted_move_list


def minimax(board: ch.Board, depth: int, is_player: bool, positions=0, agent=None):
    # depth 1 - 21 positions - time 0.003461
    # depth 2 - 621 positions - time 0.091520
    # depth 3 - 13781 positions - time 1.991260
    # depth 4 - 419166 positions - time 61.41497
    positions += 1
    if depth == 0 or board.is_game_over():
        # this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if agent is None:
            return get_simple_board_evaluation(board), None, positions
        else:
            return agent.get_simple_board_evaluation(board), None, positions
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in legal_moves(board):
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
        for move in legal_moves(board):
            # print(min_eval,best_move,move,board.fen())
            board.push(move)
            eval, a, positions = minimax(board, depth - 1, True, positions, agent)
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
    agent=None,
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
        if agent is None:
            return get_simple_board_evaluation(board), None, positions
        else:
            return agent.get_simple_board_evaluation(board), None, positions
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in legal_moves(board):
            board.push(move)
            eval, a, positions = minimax_with_pruning(
                board, depth - 1, False, alpha, beta, agent, positions=positions
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
        for move in legal_moves(board):
            board.push(move)
            eval, a, positions = minimax_with_pruning(
                board, depth - 1, True, alpha, beta, agent, positions=positions
            )
            if eval <= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move, positions


def minimax_with_pruning_and_policyeval(
    board: ch.Board,
    depth: int,
    is_player: bool,
    alpha=-np.inf,
    beta=np.inf,
    value_agent=None,
    policy_model=None,
    positions=0,
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
        if value_agent is None:
            return get_simple_board_evaluation(board), None, positions
        else:
            eval = value_agent.get_simple_board_evaluation(board)
            return eval, None, positions
    sorted_list = get_sorted_move_list(board, agent=policy_model)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            eval, a, positions = minimax_with_pruning_and_policyeval(
                board,
                depth - 1,
                False,
                alpha,
                beta,
                value_agent,
                policy_model,
                positions=positions,
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
        for move in sorted_list:
            board.push(move)
            eval, a, positions = minimax_with_pruning_and_policyeval(
                board,
                depth - 1,
                True,
                alpha,
                beta,
                value_agent,
                policy_model,
                positions=positions,
            )
            if eval <= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move, positions


def minimax_with_pruning_policyeval_positionredundancy(
    board: ch.Board,
    depth: int,
    is_player: bool,
    alpha=-np.inf,
    beta=np.inf,
    value_agent=None,
    policy_model=None,
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
            if value_agent is None:
                eval = get_simple_board_evaluation(board)
            else:
                eval = value_agent.get_simple_board_evaluation(board)
            positions_analysed[board.board_fen() + str(depth)] = eval
        return eval, None, positions, positions_analysed
    sorted_list = get_sorted_move_list(board, agent=policy_model)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            try:
                eval = positions_analysed[board.board_fen() + str(depth)]
            except KeyError:
                (
                    eval,
                    a,
                    positions,
                    positions_analysed,
                ) = minimax_with_pruning_policyeval_positionredundancy(
                    board,
                    depth - 1,
                    False,
                    alpha,
                    beta,
                    value_agent,
                    policy_model,
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
        for move in sorted_list:
            board.push(move)
            try:
                eval = positions_analysed[board.board_fen() + str(depth)]
            except KeyError:
                (
                    eval,
                    a,
                    positions,
                    positions_analysed,
                ) = minimax_with_pruning_policyeval_positionredundancy(
                    board,
                    depth - 1,
                    True,
                    alpha,
                    beta,
                    value_agent,
                    policy_model,
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