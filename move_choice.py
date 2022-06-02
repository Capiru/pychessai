import numpy as np
import chess as ch

def random_choice(possible_moves , probability_map , exploration_size):
    if probability_map is None:
        probability_map = [1/len(possible_moves) for x in range(len(possible_moves))]
    return np.random.choice(possible_moves,size=exploration_size,p=probability_map)

def minimax(board,depth,is_player,positions=0,agent = None):
    ### depth 1 - 21 positions - time 0.003461
    ### depth 2 - 621 positions - time 0.091520 
    ### depth 3 - 13781 positions - time 1.991260
    ### depth 4 - 419166 positions - time 61.41497
    positions += 1
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if agent is None:
            return get_board_evaluation(board),None,positions
        else:
            return agent.get_board_evaluation(board),None,positions
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in list(board.legal_moves):
            #print(max_eval,best_move,move,board.fen())
            board.push(move)
            eval,a,positions = minimax(board,depth-1,False,positions,agent)
            if eval>= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
        return max_eval, best_move,positions
    else:
        min_eval = np.inf
        best_move = None
        for move in list(board.legal_moves):
            #print(min_eval,best_move,move,board.fen())
            board.push(move)
            eval,a,positions = minimax(board,depth-1,True,positions,agent)
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
        return min_eval, best_move,positions

def minimax_with_pruning(board,depth,is_player,alpha=-np.inf,beta=np.inf,agent = None,positions = 0):
    ### depth 1 - 21 positions - time 0.003673
    ### depth 2 - 70 positions - time 0.010080 
    ### depth 3 - 545 positions - time 0.0784910
    ### depth 4 - 1964 positions - time 0.278105
    ### depth 5 - 14877 positions - time 2.12180
    ### depth 6 - 82579 positions - time 11.84326
    positions += 1
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if agent is None:
            return get_board_evaluation(board),None,positions
        else:
            return agent.get_board_evaluation(board),None,positions
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in list(board.legal_moves):
            board.push(move)
            eval,a,positions = minimax_with_pruning(board,depth-1,False,alpha,beta,agent,positions = positions)
            if eval>= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
            alpha = max(alpha,eval)
            if beta <= alpha:
                break
        return max_eval, best_move,positions
    else:
        min_eval = np.inf
        best_move = None
        for move in list(board.legal_moves):
            board.push(move)
            eval,a,positions = minimax_with_pruning(board,depth-1,True,alpha,beta,agent,positions = positions)
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta,eval)
            if beta <= alpha:
                break
        return min_eval, best_move,positions

def minimax_with_pruning_and_policyeval(board,depth,is_player,alpha=-np.inf,beta=np.inf,value_agent = None,policy_model = None,positions=0):
    ### depth 1 - 21 positions - time 0.004315
    ### depth 2 - 76 positions - time 0.033392
    ### depth 3 - 687 positions - time 0.172937
    ### depth 4 - 4007 positions - time 1.278452
    ### depth 5 - 30086 positions - time 7.623218
    ### depth 6 - 82579 positions - time 60.89466
    positions += 1
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if value_agent is None:
            return get_board_evaluation(board),None,positions
        else:
            return agent.get_board_evaluation(board),None,positions
    sorted_list = get_sorted_move_list(board,agent = policy_model)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            eval,a,positions = minimax_with_pruning_and_policyeval(board,depth-1,False,alpha,beta,value_agent,policy_model,positions=positions)
            if eval>= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
            alpha = max(alpha,eval)
            if beta <= alpha:
                break
        return max_eval, best_move,positions
    else:
        min_eval = np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            eval,a,positions = minimax_with_pruning_and_policyeval(board,depth-1,True,alpha,beta,value_agent,policy_model,positions=positions)
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta,eval)
            if beta <= alpha:
                break
        return min_eval, best_move,positions

def minimax_with_pruning_policyeval_positionredundancy(board,depth,is_player,alpha=-np.inf,beta=np.inf,value_agent = None,policy_model = None,positions=0,positions_analysed={}):
    ### depth 1 - 21 positions - time 0.004315
    ### depth 2 - 76 positions - time 0.033392
    ### depth 3 - 687 positions - time 0.172937
    ### depth 4 - 4007 positions - time 1.278452
    ### depth 5 - 30086 positions - time 7.623218
    ### depth 6 - 82579 positions - time 60.89466
    positions += 1
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        if value_agent is None:
            return get_board_evaluation(board),None,positions
        else:
            return agent.get_board_evaluation(board),None,positions
    sorted_list = get_sorted_move_list(board,agent = policy_model)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            try:
                eval = positions_analysed[board.board_fen]
            except:
                eval,a,positions,positions_analysed = minimax_with_pruning_policyeval_positionredundancy(board,depth-1,False,alpha,beta,value_agent,policy_model,positions,positions_analysed)
                positions_analysed[board.board_fen] = eval
            if eval>= max_eval:
                max_eval = eval
                best_move = move
            board.pop()
            alpha = max(alpha,eval)
            if beta <= alpha:
                break
        return max_eval, best_move,positions,positions_analysed
    else:
        min_eval = np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            try:
                eval = positions_analysed[board.board_fen]
            except:
                eval,a,positions = minimax_with_pruning_policyeval_positionredundancy(board,depth-1,True,alpha,beta,value_agent,policy_model,positions,positions_analysed)
                positions_analysed[board.board_fen] = eval
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta,eval)
            if beta <= alpha:
                break
        return min_eval, best_move,positions,positions_analysed

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

def rival_board_evaluation(board):
    ### Execution time: 0.000453
    if board.is_game_over():
        winner = board.outcome().winner
        if winner is None:
            return 0
        elif winner is True:
            return 50000
        else:
            return -50000
    eval_black = 0
    eval_white = 0
    fen = board.shredder_fen()
    dic_ = {"p":100,"r":500,"n":325,"b":340,"q":900,"k":10000}
    remainder_dic = {1:"A",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G",8:"H",}
    white_piece_map = {"p":[],"r":[],"n":[],"b":[],"q":[],"k":[]}
    black_piece_map = {"p":[],"r":[],"n":[],"b":[],"q":[],"k":[]}
    board_pos = 0
    for ch in fen.split(" ")[0]:
        if str.islower(ch):
            board_pos += 1
            eval_black += dic_[ch]
            current_pos = remainder_dic[board_pos%8]+str(board_pos//8+1)
            black_piece_map[ch].append(current_pos)
        elif str.isnumeric(ch):
            board_pos += int(ch)
            continue
        elif ch == "/":
            continue
        else:
            board_pos += 1
            eval_white += dic_[ch.lower()]
            current_pos = remainder_dic[board_pos%8]+str(board_pos//8+1)
            white_piece_map[ch.lower()].append(current_pos)
    diff_white,diff_black = rival_pos_eval(white_map,black_map)
    return eval_white-eval_black

def rival_pos_eval(white_map,black_map):
    white_diff = 0
    black_diff = 0
    ###Pawn Evaluation
    for piece_name,pieces_list in white_map:
        for piece in pieces_list:
            if piece == "p":
                white_diff += 1
            continue

    return white_diff,black_diff

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