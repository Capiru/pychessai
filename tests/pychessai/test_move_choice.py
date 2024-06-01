import chess as ch
import numpy as np

from pychessai.utils.move_choice import (
    legal_moves,
    minimax,
    minimax_with_pruning,
    minimax_with_pruning_and_policyeval,
    random_choice,
)


def test_random_choice(initial_board):
    legal_moves_list = legal_moves(initial_board)
    assert isinstance(random_choice(legal_moves_list)[0], ch.Move)
    probability_map = np.zeros((len(legal_moves_list)))
    probability_map[3] = 1
    assert len(legal_moves_list) == len(probability_map)
    assert random_choice(legal_moves_list, probability_map)[0] == legal_moves_list[3]
    probability_map[3] = 0.5
    probability_map[5] = 0.5
    assert (
        random_choice(legal_moves_list, probability_map)[0] == legal_moves_list[3]
        or random_choice(legal_moves_list, probability_map)[0] == legal_moves_list[5]
    )


def test_minimax(initial_board, board_checkmate_in_1, board_checkmate_in_2):
    score, move, _ = minimax(initial_board, 1, True)
    assert score == 0
    assert isinstance(move, ch.Move)
    score, move, _ = minimax(board_checkmate_in_1, 1, True)
    assert move == ch.Move.from_uci("h5f7")
    score, move, _ = minimax(board_checkmate_in_2, 3, False)
    assert move == ch.Move.from_uci("e1h1")


def test_minimax_with_pruning(
    initial_board, board_checkmate_in_1, board_checkmate_in_2
):
    score, move, _ = minimax_with_pruning(initial_board, 1, True)
    assert score == 0
    assert isinstance(move, ch.Move)
    score, move, _ = minimax_with_pruning(board_checkmate_in_1, 1, True)
    assert move == ch.Move.from_uci("h5f7")
    score, move, _ = minimax_with_pruning(board_checkmate_in_2, 3, False)
    assert move == ch.Move.from_uci("e1h1")


def test_minimax_with_pruning_and_policyeval(
    initial_board, board_checkmate_in_1, board_checkmate_in_2
):
    score, move, _ = minimax_with_pruning_and_policyeval(initial_board, 1, True)
    assert score == 0
    assert isinstance(move, ch.Move)
    score, move, _ = minimax_with_pruning_and_policyeval(board_checkmate_in_1, 1, True)
    assert move == ch.Move.from_uci("h5f7")
    score, move, _ = minimax_with_pruning_and_policyeval(board_checkmate_in_2, 3, False)
    assert move == ch.Move.from_uci("e1h1")


if __name__ == "__main__":
    test_random_choice(ch.Board())
