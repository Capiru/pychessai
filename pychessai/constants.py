from enum import Enum


class ChessRewards(float, Enum):
    DRAW = 0.5
    LOSE = -1.0
    WIN = 3.0


class ChessBoard(int, Enum):
    NUM_PIECES = 6
    NUM_PLAYERS = 2
    BOARD_SIZE = 8
    # PLANES EXPLANATION:
    # 4 castling
    # 1 if player is black or white
    # 1 attacking_planes
    # 1 for 50 rule move
    REAL_PLANES = 7


class ChessConstants:
    def __init__(self) -> None:
        pass

    @property
    def total_planes(self):
        return ChessBoard.NUM_PIECES * ChessBoard.NUM_PLAYERS + ChessBoard.REAL_PLANES

    @classmethod
    def pytorch_board_size(self, match_len):
        return (
            match_len,
            self.total_planes,
            ChessBoard.BOARD_SIZE,
            ChessBoard.BOARD_SIZE,
        )

    def tensorflow_board_size(self, match_len):
        return (
            match_len,
            ChessBoard.BOARD_SIZE,
            ChessBoard.BOARD_SIZE,
            self.total_planes,
        )
