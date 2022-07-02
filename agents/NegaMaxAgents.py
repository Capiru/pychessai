import chess as ch
from match import match,get_board_as_tensor
from move_choice import *
from search.search import *
from policy import map_moves_to_policy

class NegaMaxAgent(object):
    def __init__(self,depth = 3,board = ch.Board(),is_white = True,save_policy = False):
        self.elo = 400
        self.depth = depth
        self.board = board
        self.is_white = is_white
        self.positions = 0
        self.eval = 0
        self.search_obj = NegaMaxObject()
        self.save_policy = save_policy
    def choose_move(self,board):
        self.board = board
        score,move = self.search_obj.search(self.board,self.depth,self.is_white)
        if self.save_policy:
            self.save_to_memory(move)
        self.positions += self.search_obj.positions
        self.search_obj.positions = 0
        self.eval = score
        return [move]

    def save_to_memory(self,move):
        policy_label,_ = map_moves_to_policy([move],self.board,flatten = True)
        if not (CFG.count_since_last_val_match + 1) % CFG.val_every_x_games == 0:
            CFG.memory_batch[2][CFG.last_policy_index,:] = policy_label
            CFG.last_policy_index += 1
            if CFG.last_policy_index%CFG.batch_size == 0:
                CFG.last_policy_index = 0
        else:
            CFG.memory_batch[5][CFG.val_last_policy_index,:] = policy_label
            CFG.val_last_policy_index += 1
            if CFG.val_last_policy_index%CFG.batch_size == 0:
                CFG.val_last_policy_index = 0