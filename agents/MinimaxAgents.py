import chess as ch
from match import match
from move_choice import *

class MinimaxPruningAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
    def choose_move(self,board):
        self.board = board
        score,move,positions = minimax_with_pruning(self.board,self.depth,self.is_white)
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
        score,move,positions = minimax_with_pruning_and_policyeval(self.board,self.depth,self.is_white)
        self.positions += positions
        return [move]

class MinimaxPruningPositionRedundancyAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
        self.positions_analysed = {}
    def choose_move(self,board):
        self.board = board
        score,move,positions,positions_analysed = minimax_with_pruning_policyeval_positionredundancy(self.board,self.depth,self.is_white,positions_analysed=self.positions_analysed)
        self.positions += positions
        self.positions_analysed.update(positions_analysed)
        return [move]

class AlphaBetaMaxDepth(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
        self.positions_analysed = {}
    def choose_move(self,board):
        self.board = board
        score,move,positions,positions_analysed = alphabeta_maxdepth(self.board,self.depth,self.is_white,positions_analysed=self.positions_analysed)
        self.positions += positions
        self.positions_analysed.update(positions_analysed)
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
        score,move,positions = minimax(self.board,self.depth,self.is_white)
        self.positions += positions
        return [move]