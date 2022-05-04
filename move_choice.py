import numpy as np
import chess as ch

def random_choice(possible_moves , probability_map , exploration_size):
    if probability_map is None:
        probability_map = [1/len(possible_moves) for x in range(len(possible_moves))]
    return np.random.choice(possible_moves,size=exploration_size,p=probability_map)

def minimax(board,depth,is_player,agent = None):
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if agent is None:
            return get_board_evaluation(board),None
        else:
            return agent.get_board_evaluation(board),None
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in list(board.legal_moves):
            #print(max_eval,best_move,move,board.fen())
            board.push(move)
            eval,a = minimax(board,depth-1,False,agent)
            if eval>= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
        return max_eval, best_move
    else:
        min_eval = np.inf
        best_move = None
        for move in list(board.legal_moves):
            #print(min_eval,best_move,move,board.fen())
            board.push(move)
            eval,a = minimax(board,depth-1,True,agent)
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
        return min_eval, best_move

def minimax_with_pruning(board,depth,is_player,alpha=-np.inf,beta=np.inf,agent = None):
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if agent is None:
            return get_board_evaluation(board),None
        else:
            return agent.get_board_evaluation(board),None
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in list(board.legal_moves):
            #print(max_eval,best_move,move,board.fen())
            board.push(move)
            eval,a = minimax_with_pruning(board,depth-1,False,alpha,beta,agent)
            if eval>= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
            alpha = max(alpha,eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = np.inf
        best_move = None
        for move in list(board.legal_moves):
            #print(min_eval,best_move,move,board.fen())
            board.push(move)
            eval,a = minimax_with_pruning(board,depth-1,True,alpha,beta,agent)
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta,eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def minimax_with_pruning_and_policyeval(board,depth,is_player,alpha=-np.inf,beta=np.inf,value_agent = None,policy_model = None):
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if agent is None:
            return get_board_evaluation(board),None
        else:
            return agent.get_board_evaluation(board),None
    sorted_list = get_sorted_move_list(board,agent = policy_model)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in sorted_list:
            #print(max_eval,best_move,move,board.fen())
            board.push(move)
            eval,a = minimax_with_pruning(board,depth-1,False,alpha,beta,value_agent)
            if eval>= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
            alpha = max(alpha,eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = np.inf
        best_move = None
        for move in sorted_list:
            #print(min_eval,best_move,move,board.fen())
            board.push(move)
            eval,a = minimax_with_pruning(board,depth-1,True,alpha,beta,value_agent)
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta,eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def get_board_evaluation(board):
    if board.is_game_over():
        winner = board.outcome().winner
        if winner is None:
            return 0
        elif winner is True:
            return np.inf
        else:
            return -np.inf
    count_black = 0
    count_white = 0
    fen = board.shredder_fen()
    dic_ = {"p":1,"r":5,"n":2.5,"b":3,"q":9,"k":1000}
    for ch in fen.split(" ")[0]:
        if str.islower(ch):
            count_black += dic_[ch]
        elif str.isnumeric(ch) or ch == "/":
            continue
        else:
            try:
                count_white += dic_[ch.lower()]
            except:
                print(fen)
    return count_white-count_black

def get_players_piece_maps(board):
  pieces = board.piece_map()
  white_map = dict()
  black_map = dict()
  for k,v in pieces.items():
    if str(v).islower():
      black_map[k] = v
    else:
      white_map[k] = v
  return white_map,black_map

def get_sorted_move_list(board,agent = None):
    if agent is None:
        checkmate_list = []
        check_list = []
        capture_list = []
        attack_list = []
        pin_list = []
        castling_list = []
        other_list = []
        move_list = list(board.legal_moves)
        for move in move_list:
            if board.is_checkmate():
                board.push(move)
                checkmate_list.append(move)
            elif board.is_check():
                board.push(move)
                check_list.append(move)
            elif board.is_capture(move):
                board.push(move)
                capture_list.append(move)
            elif board.is_castling(move):
                board.push(move)
                castling_list.append(move)
            else:
                board.push(move)
                other_list.append(move)
                attacks = board.attacks(move.to_square)
                if attacks:
                    w_map,b_map = get_players_piece_maps(board)
                    if not bool(board.turn):
                        #white to play
                        if attacks.intersection(b_map):
                            attack_list.append(move)
                            other_list.pop()
                    else:
                        #black to play
                        if attacks.intersection(w_map):
                            attack_list.append(move)
                            other_list.pop()
            board.pop()
        return_list = [*checkmate_list,*check_list,*capture_list,*attack_list,*castling_list,*other_list]
        return return_list





if __name__ == "__main:__":
    board = ch.Board()
    print(minimax_with_pruning(board,2,True))