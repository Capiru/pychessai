import chess as ch
from match import match
from move_choice import *
from search.search import *

class NegaMaxAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
        self.eval = 0
        self.search_fun = NegaMaxObject()
    def choose_move(self,board):
        self.board = board
        score,move = self.search_fun(self.board,self.depth,self.is_white)
        self.positions += self.search_fun.positions
        self.search_fun.positions = 0
        self.eval = score
        return [move]