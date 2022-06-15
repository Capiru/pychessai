import numpy as np
import chess as ch
from move_choice import *

class AlphaBetaPruning(object):
    def __init__(self,depth = 3,eval_fun = get_board_evaluation,policy_fun=get_sorted_move_list,pruning = False,transposition_table = False,time_based = False,time_limit = 1,capture_eval_correction = False):
        self.depth = depth
        self.eval_fun = eval_fun
        self.policy_fun = policy_fun
        self.pruning = pruning
        self.transposition_table = transposition_table
        self.time_based = time_based
        self.time_limit = time_limit
        self.start_time = -1
        self.capture_eval_correction = capture_eval_correction
        self.max_quiescence_depth = 5
        if self.time_based:
            self.search_fun = self.time_limit_search
        else:
            self.search_fun = self.search

    def time_limit_search(self,board,is_white = True,alpha = -np.inf,beta = np.inf,positions = 0,positions_analysed = {}):
        self.start_time = time.process_time()
        depth = 1
        is_player = is_white
        max_score,min_score = -np.inf,np.inf
        all_pos = positions
        best_move = None
        all_analysed_pos = positions_analysed
        while time.process_time()-self.start_time <= self.time_limit:
            eval,move,positions,positions_analysed = self.search(board,depth,is_player,alpha,beta,0,all_analysed_pos)
            all_pos += positions
            all_analysed_pos.update(positions_analysed)
            if is_white:
                if eval >= max_score:
                    max_score = eval
                    best_move = move
            else:
                if eval <= min_score:
                    min_score = eval
                    best_move = move
            depth += 1
        if is_white:
            return max_score,best_move,all_pos,all_analysed_pos
        else:
            return min_score,best_move,all_pos,all_analysed_pos

    def quiescence(self,board,depth,is_white = True,alpha = -np.inf,beta = np.inf,positions = 0,positions_analysed = {}):
        try:
            eval,a = positions_analysed[board.shredder_fen()+"0"]
            return eval,a,positions,positions_analysed
        except:
            positions += 0
        curr_score = self.eval_fun(board)
        capture_list = get_sorted_move_list(board,only_attacks=True)
        num_captures = len(capture_list)
        if num_captures == 0:
            positions_analysed[board.shredder_fen()+"0"] = (curr_score,board.move_stack[-1])
            return curr_score,None,positions,positions_analysed
        if depth == 1:
            alpha = curr_score
        if is_white:
            for move in capture_list:
                board.push(move)
                positions += 1
                eval,a,positions,positions_analysed = self.quiescence(board,depth+1,not is_white,alpha,beta,positions,positions_analysed)
                board.pop()
                alpha = max(alpha,eval)
                if beta <= alpha:
                    break
                if eval > alpha:
                    alpha = eval
            return alpha,None,positions,positions_analysed
        else:
            for move in capture_list:
                board.push(move)
                positions += 1
                eval,a,positions,positions_analysed = self.quiescence(board,depth+1,not is_white,alpha,beta,positions,positions_analysed)
                board.pop()
                beta = min(beta,eval)
                if beta <= alpha:
                    break
            return beta,None,positions,positions_analysed

       

    def search(self,board,depth = 3,is_white = True,alpha = -np.inf,beta = np.inf,positions = 0,positions_analysed = {}):
        try:
            eval,move = positions_analysed[board.shredder_fen()+str(depth)]
            return eval,move,positions,positions_analysed
        except:
            positions += 1
        if depth==0 or board.is_game_over():
            if self.capture_eval_correction:
                return self.quiescence(board,1,not is_white,-np.inf,np.inf,positions,positions_analysed)
            else:
                return self.eval_fun(board),None,positions,positions_analysed
            
        sorted_list = get_sorted_move_list(board)
        best_move = None
        if is_white:
            max_eval = -np.inf
            for move in sorted_list:
                board.push(move)
                if self.transposition_table:
                    try:
                        eval,a = positions_analysed[board.shredder_fen()+str(depth-1)]
                    except:
                        eval,a,positions,positions_analysed = self.search(board,depth-1,False,alpha,beta,positions,positions_analysed)
                        positions_analysed[board.shredder_fen()+str(depth-1)] = (eval,move)
                else:
                    eval,a,positions,positions_analysed = self.search(board,depth-1,False,alpha,beta,positions,positions_analysed)
                if eval >= max_eval:
                    max_eval = eval
                    best_move = move
                board.pop()
                if self.pruning:
                    alpha = max(alpha,eval)
                    if beta <= alpha:
                        break
                if self.time_based:
                    if time.process_time()-self.start_time >= self.time_limit:
                        break
            if self.transposition_table:
                positions_analysed[board.shredder_fen()+str(depth)] = (max_eval,best_move)                    
            return max_eval, best_move,positions,positions_analysed
        else:
            min_eval = np.inf
            for move in sorted_list:
                board.push(move)
                if self.transposition_table:
                    try:
                        eval,a = positions_analysed[board.shredder_fen()+str(depth-1)]
                    except:
                        eval,a,positions,positions_analysed = self.search(board,depth-1,True,alpha,beta,positions,positions_analysed)
                        positions_analysed[board.shredder_fen()+str(depth-1)] = (eval,move)
                else:
                    eval,a,positions,positions_analysed = self.search(board,depth-1,True,alpha,beta,positions,positions_analysed)
                if eval<= min_eval:
                    min_eval = eval
                    best_move = move
                board.pop()
                if self.pruning:
                    beta = min(beta,eval)
                    if beta <= alpha:
                        break
                if self.time_based:
                    if time.process_time()-self.start_time >= self.time_limit:
                        break
            if self.transposition_table:
                positions_analysed[board.shredder_fen()+str(depth)] = (min_eval,best_move) 
            return min_eval, best_move,positions,positions_analysed


class NegaMaxObject(object):
    def __init__(self,depth = 3,eval_fun = rival_eval,policy_fun=get_sorted_move_list,time_based = False,time_limit = 1):
        self.depth = depth
        self.eval_fun = eval_fun
        self.policy_fun = policy_fun
        self.time_based = time_based
        self.time_limit = time_limit
        self.start_time = -1
        self.max_quiescence_depth = 5
        self.positions = 0
        self.positions_analysed = {}
        if self.time_based:
            self.search_fun = self.time_limit_search
        else:
            self.search_fun = self.search

    def quiesce(self,board,is_white,depth,alpha,beta):
        stand_pat = self.eval_fun(board,is_white)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        capture_list = self.policy_fun(board,only_attacks = True)
        for move in capture_list:
            board.push(move)
            eval = self.quiesce(board,not is_white,depth+1,-beta,-alpha)
            eval = -eval
            board.pop()
            if eval >= beta:
                return beta
            if eval > alpha:
                alpha = eval
        return alpha

    def search(self,board,depth,is_white = True,alpha = -np.inf,beta=np.inf):
        try:
            eval,move = self.positions_analysed[board.shredder_fen()+str(depth)]
            return eval,move
        except:
            self.positions += 1
        if depth == 0 or board.is_game_over():
            return self.quiesce(board,is_white,depth = 1,alpha = alpha,beta = beta),None
        best_move = None
        move_list = self.policy_fun(board)
        for move in move_list:
            board.push(move)
            eval,a = self.search(board,depth-1,not is_white,-beta,-alpha)
            board.pop()
            eval = -eval
            if eval >= beta:
                return beta,None ##fail hard beta cutoff
            if eval > alpha:
                alpha = eval ## alpha acts as max
                best_move = move
        self.positions_analysed[board.shredder_fen()+str(depth)] = alpha,best_move
        return alpha,best_move

    def time_limit_search(self):
        self.start_time = time.process_time()
        depth = 1
        is_player = is_white
        max_score,min_score = -np.inf,np.inf
        all_pos = positions
        best_move = None
        all_analysed_pos = positions_analysed
        while time.process_time()-self.start_time <= self.time_limit:
            eval,move,positions,positions_analysed = self.search(board,depth,is_player,alpha,beta,0,all_analysed_pos)
            all_pos += positions
            all_analysed_pos.update(positions_analysed)
            if is_white:
                if eval >= max_score:
                    max_score = eval
                    best_move = move
            else:
                if eval <= min_score:
                    min_score = eval
                    best_move = move
            depth += 1
        if is_white:
            return max_score,best_move,all_pos,all_analysed_pos
        else:
            return min_score,best_move,all_pos,all_analysed_pos