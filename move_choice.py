import numpy as np
import chess as ch

def random_choice(possible_moves , probability_map , exploration_size):
    if probability_map is None:
        probability_map = [1/len(possible_moves) for x in range(len(possible_moves))]
    return np.random.choice(possible_moves,size=exploration_size,p=probability_map)

def minimax(board,depth,is_player,agent = None):
    ### depth 1 - 21 positions - time 0.003461
    ### depth 2 - 621 positions - time 0.091520 
    ### depth 3 - 13781 positions - time 1.991260
    ### depth 4 - 419166 positions - time 61.41497
    global positions
    positions += 1
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
    ### depth 1 - 21 positions - time 0.003673
    ### depth 2 - 70 positions - time 0.010080 
    ### depth 3 - 545 positions - time 0.0784910
    ### depth 4 - 1964 positions - time 0.278105
    ### depth 5 - 14877 positions - time 2.12180
    ### depth 6 - 82579 positions - time 11.84326
    global positions
    positions += 1
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
    ### depth 1 - 21 positions - time 0.004315
    ### depth 2 - 76 positions - time 0.033392
    ### depth 3 - 687 positions - time 0.172937
    ### depth 4 - 4007 positions - time 1.278452
    ### depth 5 - 30086 positions - time 7.623218
    ### depth 6 - 82579 positions - time 60.89466
    global positions
    positions += 1
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if value_agent is None:
            return get_board_evaluation(board),None
        else:
            return agent.get_board_evaluation(board),None
    sorted_list = get_sorted_move_list(board,agent = policy_model)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            eval,a = minimax_with_pruning_and_policyeval(board,depth-1,False,alpha,beta,value_agent,policy_model)
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
            board.push(move)
            eval,a = minimax_with_pruning_and_policyeval(board,depth-1,True,alpha,beta,value_agent,policy_model)
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta,eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def get_board_evaluation(board):
    ### Execution time: 0.000453
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
    ### Execution time: 0.000391
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
    ### Execution time: 0.001401
    if agent is None:
        checkmate_list = []
        check_list = []
        capture_list = []
        attack_list = []
        pin_list = []
        castling_list = []
        other_list = []
        move_list = list(board.legal_moves)
        w_map,b_map = get_players_piece_maps(board)
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