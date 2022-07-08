import numpy as np
import chess as ch
from move_choice import *
import torch
import torch.nn.functional as F
from match import get_board_as_tensor
from policy import map_moves_to_policy
import copy
import math
from config import CFG

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

class MonteCarloSearchNode:
    def __init__(self,agent,prior,player_white,board,state = None,node_value = None,parent = None,parent_move = None,policy_priors=None,legal_actions = None,is_terminal_node = False):
        self.agent = agent
        self.prior = prior
        self.player_white = player_white
        self.board = board.copy()

        self.children = {}
        self.parent = parent
        self.parent_move = parent_move
        self.visit_count = 0
        self.value_sum = 0
        self.state = state
        self.prior = prior
        self.is_terminal_node = is_terminal_node
        self.current_ucb_scores = {}
        if legal_actions is None and not is_terminal_node:
            self.node_value,self.policy_priors,self.legal_actions = self.get_move_from_policy()
        else:
            self.node_value = node_value
            self.legal_actions = legal_actions
            self.policy_priors = policy_priors
        ind = 0
        self.untried_actions_dic = {}
        if self.legal_actions is not None:
            for i in range(len(self.legal_actions)):
                self.current_ucb_scores[self.legal_actions[i]] = self.policy_priors[i]
                self.untried_actions_dic[self.legal_actions[i]] = self.policy_priors[i]
        self.untried_actions = copy.deepcopy(self.legal_actions)
        self.last_child_move = None
        self.best_child_score = -np.inf
        self.best_child = None
        if not is_terminal_node:
            for i in range(len(self.legal_actions)):
                self.current_ucb_scores[self.legal_actions[i]] = self.policy_priors[i]
        self.choose_move = self.choose_move_from_ucb_weights
        

    def fill_untried_actions(self):
        self.untried_actions = copy.deepcopy(self.legal_actions)
        return self.untried_actions

    def prior_score(self):
        return math.sqrt(2) * math.sqrt((self.parent.visit_count) / (self.visit_count))

    def value_score(self):
        return self.value_sum / self.visit_count 

    def ucb_score(self):
        return -self.value_score() + self.prior_score()

    def get_board_reward(self):
        if self.board.is_game_over():
            self.is_terminal_node = True
            if self.board.is_checkmate():
                if not self.board.outcome().winner ^ self.player_white:
                    return CFG.WIN_VALUE,None
                else:
                    return CFG.LOSS_VALUE,None
            else:
                return CFG.DRAW_VALUE,None
        else:
            self.is_terminal_node = False
            if CFG.TEST:
                return rival_eval(self.board,self.player_white),map_moves_to_policy(list(self.board.legal_moves),self.board,flatten = True)[0].to(CFG.DEVICE)
            return self.agent.value_model.get_board_evaluation(self.board,self.player_white)
    
    def backpropagate(self,result):
        self.visit_count += 1
        self.value_sum += result
        if self.parent:
            self.parent.backpropagate(-self.value_score())
            self.parent.update_ucb_scores(self.parent)
        else:
            self.update_ucb_scores(node=self)

    def update_ucb_scores(self,node):
        child_ucb = node.children[node.last_child_move].ucb_score()
        node.current_ucb_scores[node.last_child_move] = child_ucb

    def find_best_child(self):
        most_visits = 0
        most_visited = None
        for k,v in self.current_ucb_scores.items():
            try:
                if self.children[k].visit_count > most_visits:
                    most_visits = self.children[k].visit_count
                    most_visited = k
            except:
                pass
            if v > self.best_child_score:
                self.best_child_score = v
                self.best_child = k
        if self.best_child_score > 10:
            return self.best_child_score,self.best_child
        return self.children[most_visited].node_value,most_visited
        

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def get_move_from_policy(self):
        legal_moves = list(self.board.legal_moves)
        reduced_actions = torch.zeros((len(legal_moves))).to(CFG.DEVICE)
        #state = get_board_as_tensor(self.board,self.player_white)
        score,policy = self.get_board_reward()
        if not self.is_terminal_node:
            legal_moves_mask,move_list = map_moves_to_policy(legal_moves,self.board,flatten = True)
            action_space = torch.mul(torch.flatten(policy),legal_moves_mask.to(CFG.DEVICE))
            for i in range(len(legal_moves)):
                reduced_actions[i] = action_space[move_list[i]]
            priors = F.softmax(reduced_actions,dim = 0)
            sorted_index = torch.argsort(priors,descending = True)
            return score,priors[sorted_index],[legal_moves[i] for i in sorted_index]
        else:
            return score,None,None

    def expand(self,agent,move,prior):
        #### needs to change to select based on probability or UCB
        next_state = self.board.push(move)
        score,priors,legal_moves = self.get_move_from_policy()
        child_node = MonteCarloSearchNode(agent = self.agent,prior = prior,board = self.board.copy(),player_white = not self.player_white,parent = self,parent_move = move,
                                        node_value = score,policy_priors = priors,legal_actions = legal_moves,is_terminal_node=self.is_terminal_node)
        self.children[move] = child_node
        self.is_terminal_node = False
        self.board.pop()
        return child_node

    def choose_move_random(self):
        index = np.random.choice([i for i in range(len(self.legal_actions))],size=1,p=self.policy_priors.cpu().detach().numpy())[0]
        return self.legal_actions[index],self.policy_priors[index]

    def choose_move_from_ucb_weights(self):
        move = np.random.choice(list(self.current_ucb_scores.keys()),size=1,p=torch.softmax(torch.FloatTensor(list(self.current_ucb_scores.values())),dim=0).numpy())[0]
        return move,self.current_ucb_scores[move] ### Maybe this needs to be prior not UCB

    def choose_move_untried_actions(self):
        move = self.untried_actions.pop()
        return move,self.untried_actions_dic[move]

    def choose_best_ucb(self):
        score,move = self.find_best_child()
        return move

    def save_to_memory(self):
        policy_label,_ = map_moves_to_policy([self.best_child],self.board,flatten = True)
        #policy_label,_ = map_moves_to_policy(self.legal_actions,self.board,flatten = True,dic = self.current_ucb_scores)
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

    def search(self,n_simulations):

        for i in range(n_simulations):
            current_node = self
            found_unexplored_node = False
            terminal_nodes_found = 0
            while not found_unexplored_node:
                if not current_node.is_fully_expanded():
                    move,prior = current_node.choose_move_untried_actions()
                else:
                    move,prior = current_node.choose_move() ## find move change this
                current_node.last_child_move = move
                try:
                    children_node = current_node.children[move]
                    if not children_node.is_terminal_node:
                        current_node = children_node
                        terminal_nodes_found = 0
                    else:
                        terminal_nodes_found += 1
                        if terminal_nodes_found == len(current_node.legal_actions):
                            break
                except:
                    children_node = current_node.expand(self.agent,move,prior)
                    found_unexplored_node = True
                    terminal_nodes_found = 0
                    break
            children_node.backpropagate(children_node.node_value)
        ### Add the best_child to the memory batch
        best_child_score, best_child = self.find_best_child()
        if self.agent.training and CFG.save_batch_to_device:
            self.save_to_memory()
        if CFG.TEST:
            print(self.current_ucb_scores)
        return best_child_score, best_child
        # except Exception as e:
        #     print(current_node.legal_actions)
        #     print(current_node.board,"\n",current_node.board.move_stack)
        #     print(children_node.board,"\n")
        #     print(e)
        #     raise BaseException("Error in search")