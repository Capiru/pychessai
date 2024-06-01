import chess as ch

from pychessai.agents import RandomAgent


def test_random_agent(initial_board):
    random_agent = RandomAgent(initial_board, True)
    assert isinstance(random_agent.choose_move(random_agent.board)[0], ch.Move)


if __name__ == "__main__":
    test_random_agent(ch.Board())
