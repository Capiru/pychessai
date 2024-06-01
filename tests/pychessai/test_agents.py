import chess as ch

from pychessai.agents import (
    MinimaxAgent,
    MinimaxPruningAgent,
    MinimaxPruningWithPolicyAgent,
    RandomAgent,
)


def test_random_agent(initial_board):
    random_agent = RandomAgent(initial_board, True)
    assert isinstance(random_agent.choose_move(random_agent.board)[0], ch.Move)


def test_minimax_agent(board_checkmate_in_1):
    minimax_agent = MinimaxAgent(depth=3, board=board_checkmate_in_1, is_white=True)
    move = minimax_agent.choose_move(board_checkmate_in_1)
    assert move[0] == ch.Move.from_uci("h5f7")


def test_minimax_pruning_agent(board_checkmate_in_1):
    minimax_pruning = MinimaxPruningAgent()
    move = minimax_pruning.choose_move(board_checkmate_in_1)
    assert move[0] == ch.Move.from_uci("h5f7")


def test_minimax_pruning_with_policy_agent(board_checkmate_in_1):
    minimax_pruning = MinimaxPruningWithPolicyAgent()
    move = minimax_pruning.choose_move(board_checkmate_in_1)
    assert move[0] == ch.Move.from_uci("h5f7")


if __name__ == "__main__":
    test_random_agent(ch.Board())
