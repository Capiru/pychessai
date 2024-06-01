import chess as ch
import numpy as np


def get_board_evaluation(board: ch.Board):
    # Execution time: 0.000453
    if board.is_game_over():
        winner = board.outcome().winner
        if winner is None:
            return 0
        elif winner is True:
            return np.inf
        else:
            return -np.inf
    count_black = 0.0
    count_white = 0.0
    fen = board.shredder_fen()
    dic_ = {"p": 1.0, "r": 5.0, "n": 2.5, "b": 3.0, "q": 9.0, "k": 1000.0}
    for char in fen.split(" ")[0]:
        if str.islower(char):
            count_black += dic_[char]
        elif str.isnumeric(char) or char == "/":
            continue
        else:
            try:
                count_white += dic_[char.lower()]
            except Exception as e:
                print(fen, e)
    return count_white - count_black
