import chess as ch
from pettingzoo.classic import chess_v6


class chess_env(chess_v6.raw_env):
    def __init__(
        self,
        starting_fen=None,
        render_mode: str | None = None,
        screen_height: int | None = 800,
    ):
        super().__init__(render_mode, screen_height)
        self.board = ch.Board(fen=starting_fen)


if __name__ == "__main__":
    test_env = chess_env()
    test_env.board.push_uci("e4")
    print(test_env.observe(0))
