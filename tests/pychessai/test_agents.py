import chess as ch

from pychessai.agents import MinimaxAgent, RandomAgent


def test_random_agent(initial_board):
    random_agent = RandomAgent(initial_board, True)
    assert isinstance(random_agent.choose_move(random_agent.board)[0], ch.Move)


def test_minimax_agent(board_checkmate_in_1):
    minimax_agent = MinimaxAgent(depth=3, board=board_checkmate_in_1, is_white=False)
    assert isinstance(minimax_agent.choose_move(minimax_agent.board)[0], ch.Move)


if __name__ == "__main__":
    test_random_agent(ch.Board())
