import chess as ch
from match import match
from move_choice import minimax,minimax_with_pruning,minimax_with_pruning_and_policyeval

class MinimaxPruningAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
    def choose_move(self,board):
        self.board = board
        positions = 0
        score,move = minimax_with_pruning(self.board,self.depth,self.is_white)
        self.positions += positions
        return [move]

class MinimaxPruningSimplePolicyAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
    def choose_move(self,board):
        self.board = board
        positions = 0
        score,move = minimax_with_pruning_and_policyeval(self.board,self.depth,self.is_white)
        self.positions += positions
        return [move]

class MinimaxAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
    def choose_move(self,board):
        self.board = board
        positions = 0
        score,move = minimax(self.board,self.depth,self.is_white)
        self.positions += positions
        return [move]