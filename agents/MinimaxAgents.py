import chess as ch
from match import match
from move_choice import minimax,minimax_with_pruning

class MinimaxPruningAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
    def choose_move(self,fen):
        self.board = ch.Board(fen)
        score,move = minimax_with_pruning(self.board,self.depth,self.is_white)
        return [move]

class MinimaxAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
    def choose_move(self,fen):
        self.board = ch.Board(fen)
        score,move = minimax(self.board,self.depth,self.is_white)
        return [move]