import chess as ch
import numpy as np

from pychessai.utils.move_choice import legal_moves, random_choice


def test_random_choice(initial_board):
    assert isinstance(random_choice(legal_moves(initial_board))[0], ch.Move)
    probability_map = np.zeros((len(legal_moves(initial_board))))
    probability_map[3] = 1
    assert (
        random_choice(legal_moves(initial_board), probability_map)[0]
        == legal_moves(initial_board)[3]
    )
    probability_map[3] = 0.5
    probability_map[5] = 0.5
    assert (
        random_choice(legal_moves(initial_board), probability_map)[0]
        == legal_moves(initial_board)[3]
        or random_choice(legal_moves(initial_board), probability_map)[0]
        == legal_moves(initial_board)[5]
    )


if __name__ == "__main__":
    test_random_choice(ch.Board())
