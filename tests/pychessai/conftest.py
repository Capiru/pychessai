import chess as ch
import pytest


@pytest.fixture
def initial_board():
    return ch.Board()


@pytest.fixture
def board_checkmate_in_1():
    return ch.Board("rnbqkbnr/1ppp1ppp/8/p3p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4")


@pytest.fixture
def board_checkmate_in_2():
    return ch.Board("6k1/pp3pp1/2B5/3N2bp/8/1Q1P1p2/PPP2PbK/R1B1r3 b - - 0 22")
