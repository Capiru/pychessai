from typing import Union

import chess as ch
import numpy as np


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
