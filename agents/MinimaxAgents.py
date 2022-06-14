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
        self.eval = 0
    def choose_move(self,board):
        self.board = board
        score,move,positions = minimax_with_pruning(self.board,self.depth,self.is_white)
        self.positions += positions
        self.eval = score
        return [move]

class MinimaxPruningSimplePolicyAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
        self.eval = 0
    def choose_move(self,board):
        self.board = board
        score,move,positions = minimax_with_pruning_and_policyeval(self.board,self.depth,self.is_white)
        self.positions += positions
        self.eval = score
        return [move]

class MinimaxPruningPositionRedundancyAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
        self.positions_analysed = {}
        self.eval = 0
    def choose_move(self,board):
        self.board = board
        score,move,positions,positions_analysed = minimax_with_pruning_policyeval_positionredundancy(self.board,self.depth,self.is_white,positions_analysed=self.positions_analysed)
        self.positions += positions
        self.positions_analysed.update(positions_analysed)
        self.eval = score
        return [move]

class AlphaBetaMaxDepth(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True,time_based = False,time_limit = 1):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
        self.positions_analysed = {}
        self.eval = 0
        self.time_based = time_based
        self.time_limit = time_limit
        self.search_obj = AlphaBetaPruning(eval_fun = get_rival_board_evaluation,pruning = True,transposition_table = True,time_based = self.time_based,time_limit = self.time_limit,capture_eval_correction = True)
    def choose_move(self,board):
        self.board = board
        if self.time_based:
            score,move,positions,positions_analysed = self.search_obj.time_limit_search(self.board,self.is_white,positions_analysed=self.positions_analysed)
        else:
            score,move,positions,positions_analysed = self.search_obj.search(self.board,self.depth,self.is_white,positions_analysed=self.positions_analysed)
        self.positions += positions
        self.positions_analysed.update(positions_analysed)
        self.eval = score
        return [move]

class MinimaxAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
        self.eval = 0
    def choose_move(self,board):
        self.board = board
        score,move,positions = minimax(self.board,self.depth,self.is_white)
        self.positions += positions
        self.eval = score
        return [move]