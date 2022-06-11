import chess as ch
from move_choice import random_choice

class RandomAgent(object):
    def __init__(self):
        self.elo = 400
        self.positions = 0
        self.eval = 0
    def choose_move(self,board):
        legal_moves = list(board.legal_moves)
        return random_choice(legal_moves,None,1)

class AllMovesAgent(object):
    def __init__(self,depth=3):
        self.elo = 400
        self.positions = 0
        self.depth = depth
        self.eval = 0
    def choose_move(self,board):
        legal_moves = list(board.legal_moves)
        return random_choice(legal_moves,None,1)
