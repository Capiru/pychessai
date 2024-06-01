import chess as ch
import pytest


@pytest.fixture
def initial_board():
    return ch.Board()
