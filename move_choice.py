import numpy as np
import chess as ch


def random_choice(possible_moves , probability_map , exploration_size):
    if probability_map is None:
        probability_map = [1/len(possible_moves) for x in range(len(possible_moves))]
    return np.random.choice(possible_moves,size=exploration_size,p=probability_map)

def legal_moves(board):
    return list(board.legal_moves)

def all_moves(board,depth,positions,nodes):
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        nodes += 1
        return positions,nodes
    for move in list(board.legal_moves):
        positions += 1
        board.push(move)
        positions,nodes = all_moves(board,depth-1,positions,nodes)
        board.pop()
    return positions,nodes

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
            return value_agent.get_board_evaluation(board),None,positions
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

def attack_search_maxdepth(board,is_player,alpha = -np.inf,beta = np.inf,positions = 0,positions_analysed = {}):
    positions += 1
    sorted_list = get_sorted_move_list(board,only_attacks=True)
    if len(sorted_list) == 0:
        eval = get_rival_board_evaluation(board)
        positions_analysed[board.board_fen()+str(0)]=eval
        return eval,None,positions,positions_analysed
    else:
        if is_player:
            max_eval = -np.inf
            best_move = None
            for move in sorted_list:
                board.push(move)
                eval,a,positions,positions_analysed = attack_search_maxdepth(board,False,alpha,beta,positions,positions_analysed)
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
                eval,a,positions,positions_analysed = attack_search_maxdepth(board,True,alpha,beta,positions,positions_analysed)
                if eval<= min_eval:
                    min_eval = eval
                    best_move = move
                board.pop()
                beta = min(beta,eval)
                if beta <= alpha:
                    break
            return min_eval, best_move,positions,positions_analysed

def alphabeta_maxdepth(board,depth,is_player,alpha=-np.inf,beta=np.inf,value_agent = None,policy_model = None,positions=0,positions_analysed={}):
    ### depth 1 - 21 positions - time 0.004315
    ### depth 2 - 76 positions - time 0.033392
    ### depth 3 - 687 positions - time 0.172937
    ### depth 4 - 4007 positions - time 1.278452
    ### depth 5 - 30086 positions - time 7.623218
    ### depth 6 - 82579 positions - time 60.89466
    positions += 1
    if depth==0 or board.is_game_over():
        #this might have problems with depth == 1, should probably return  board.pop() (MAYBE)
        try:
            eval = positions_analysed[board.board_fen()+str(depth)]
        except:
            if value_agent is None:
                eval,a,positions,positions_analysed = attack_search_maxdepth(board,is_player,-np.inf,np.inf,positions,positions_analysed)
            else:
                eval = agent.get_board_evaluation(board)
            positions_analysed[board.board_fen()+str(depth)]=eval
        return eval, None,positions,positions_analysed
    sorted_list = get_sorted_move_list(board,agent = policy_model)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            try:
                eval = positions_analysed[board.board_fen()+str(depth)]
            except:
                eval,a,positions,positions_analysed = alphabeta_maxdepth(board,depth-1,False,alpha,beta,value_agent,policy_model,positions,positions_analysed)
                positions_analysed[board.board_fen()+str(depth)] = eval
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
                eval = positions_analysed[board.board_fen()+str(depth)]
            except:
                eval,a,positions,positions_analysed = alphabeta_maxdepth(board,depth-1,True,alpha,beta,value_agent,policy_model,positions,positions_analysed)
                positions_analysed[board.board_fen()+str(depth)] = eval
            if eval<= min_eval:
                min_eval = eval
                best_move = move
            board.pop()
            beta = min(beta,eval)
            if beta <= alpha:
                break
        return min_eval, best_move,positions,positions_analysed

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
        try:
            eval = positions_analysed[board.board_fen()+str(depth)]
        except:
            if value_agent is None:
                eval = get_rival_board_evaluation(board)
            else:
                eval = agent.get_board_evaluation(board)
            positions_analysed[board.board_fen()+str(depth)]=eval
        return eval, None,positions,positions_analysed
    sorted_list = get_sorted_move_list(board,agent = policy_model)
    if is_player:
        max_eval = -np.inf
        best_move = None
        for move in sorted_list:
            board.push(move)
            try:
                eval = positions_analysed[board.board_fen()+str(depth)]
            except:
                eval,a,positions,positions_analysed = minimax_with_pruning_policyeval_positionredundancy(board,depth-1,False,alpha,beta,value_agent,policy_model,positions,positions_analysed)
                positions_analysed[board.board_fen()+str(depth)] = eval
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
                eval = positions_analysed[board.board_fen()+str(depth)]
            except:
                eval,a,positions,positions_analysed = minimax_with_pruning_policyeval_positionredundancy(board,depth-1,True,alpha,beta,value_agent,policy_model,positions,positions_analysed)
                positions_analysed[board.board_fen()+str(depth)] = eval
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

def build_diag_maps():
    diagonal_maps = {"A1":[1,22],"A2":[2,21],"A3":[3,20],"A4":[4,19],"A5":[5,18],"A6":[6,17],"A7":[7,16],"A8":[8,30],
                    "B1":[15,21],"B2":[1,20],"B3":[2,19],"B4":[3,18],"B5":[4,17],"B6":[5,16],"B7":[30,6],"B8":[7,29],
                    "C1":[14,20],"C2":[19,15],"C3":[1,18],"C4":[17,2],"C5":[3,16],"C6":[30,4],"C7":[5,29],"C8":[6,28],
                    "D1":[13,19],"D2":[14,18],"D3":[15,17],"D4":[16,1],"D5":[30,2],"D6":[3,29],"D7":[4,28],"D8":[5,27],
                    "E1":[12,18],"E2":[13,17],"E3":[14,16],"E4":[15,30],"E5":[1,29],"E6":[2,28],"E7":[3,27],"E8":[4,26],
                    "F1":[11,17],"F2":[12,16],"F3":[13,30],"F4":[14,29],"F5":[15,28],"F6":[1,27],"F7":[2,26],"F8":[3,25],
                    "G1":[10,16],"G2":[11,30],"G3":[12,29],"G4":[13,28],"G5":[14,27],"G6":[15,26],"G7":[1,25],"G8":[2,24],
                    "H1":[9,30],"H2":[10,29],"H3":[11,28],"H4":[12,27],"H5":[13,26],"H6":[14,25],"H7":[15,24],"H8":[1,23] }

    diag_num_maps = {1:"A1B2C3D4E5F6G7H8",2:"A2B3C4D5E6F7G8",3:"A3B4C5D6E7F8",4:"A4B5C6D7E8",5:"A5B6C7D8",6:"A6B7C8",
                    7:"A7B8",8:"A8",9:"H1",10:"H2G1",11:"H3G2F1",12:"H4G3F2E1",13:"H5G4F3E2D1",14:"H6G5F4E3D2C1",15:"H7G6F5E4D3C2B1",
                    16:"A7B6C5D4E3F2G1",17:"A6B5C4D3E2F1",18:"A5B4C3D2E1",19:"A4B3C2D1",20:"A3B2C1",21:"A2B1",22:"A1",23:"H8",
                    24:"H7G8",25:"H6G7F8",26:"H5G6F7E8",27:"H4G5F6E7D8",28:"H3G4F5E6D7C8",29:"H2G3F4E5D6C7B8",30:"H1G2F3E4D5C6B7A8"}
    return diagonal_maps,diag_num_maps

def rival_eval(board,is_white = True):
    if is_white:
        return get_rival_board_evaluation(board)
    else:
        return -get_rival_board_evaluation(board)

def get_rival_board_evaluation(board):
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
    remainder_dic = {1:"A",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G",8:"H"}
    white_piece_map = {"p":[],"r":[],"n":[],"b":[],"q":[],"k":[]}
    black_piece_map = {"p":[],"r":[],"n":[],"b":[],"q":[],"k":[]}
    white_piece_position_stringmap = {"p":"","r":"","n":"","b":"","q":"","k":""}
    black_piece_position_stringmap = {"p":"","r":"","n":"","b":"","q":"","k":""}
    diag_pos_map,diag_num_map = build_diag_maps()
    white_diag_map = {(i+1):"" for i in range(30)}
    black_diag_map = {(i+1):"" for i in range(30)}
    board_pos = 0
    for ch in fen.split(" ")[0]:
        if str.islower(ch):
            board_pos += 1
            eval_black += dic_[ch]
            current_pos = remainder_dic[board_pos%8+1]+str((board_pos-1)//8+1)
            black_piece_map[ch].append(current_pos)
            black_piece_position_stringmap[ch]+=current_pos
            diags = diag_pos_map[current_pos]
            black_diag_map[diags[0]]+=ch
            black_diag_map[diags[1]]+=ch
        elif str.isnumeric(ch):
            board_pos += int(ch)
            continue
        elif ch == "/":
            continue
        else:
            board_pos += 1
            eval_white += dic_[ch.lower()]
            current_pos = remainder_dic[board_pos%8+1]+str((board_pos-1)//8+1)
            white_piece_map[ch.lower()].append(current_pos)
            white_piece_position_stringmap[ch.lower()]+=current_pos
            diags = diag_pos_map[current_pos]
            white_diag_map[diags[0]]+=ch.lower()
            white_diag_map[diags[1]]+=ch.lower()
    eval_white+=pawn_rival_eval(white_piece_map["p"],white_piece_position_stringmap,black_piece_position_stringmap,is_white=True)
    eval_white+=bishop_rival_eval(white_piece_map["b"],white_diag_map,diag_pos_map)
    eval_white+=knight_rival_eval(white_piece_map["k"])
    eval_white+=rook_rival_eval(white_piece_map["r"],white_piece_position_stringmap,black_piece_position_stringmap,is_white=True)
    eval_white+=queen_rival_eval(white_piece_map["q"],black_piece_position_stringmap,white_diag_map,diag_pos_map)

    eval_black+=pawn_rival_eval(black_piece_map["p"],black_piece_position_stringmap,white_piece_position_stringmap,is_white=False)
    eval_black+=bishop_rival_eval(black_piece_map["b"],black_diag_map,diag_pos_map)
    eval_black+=knight_rival_eval(black_piece_map["k"])
    eval_black+=rook_rival_eval(black_piece_map["r"],black_piece_position_stringmap,white_piece_position_stringmap,is_white=False)
    eval_black+=queen_rival_eval(black_piece_map["q"],white_piece_position_stringmap,black_diag_map,diag_pos_map)
    return eval_white-eval_black

def pawn_rival_eval(pawn_list,player_position_map,opp_position_map,is_white):
    diff = 0
    advancement_pawn = {"1":0,"2":0,"3":1,"4":3,"5":5,"6":13,"7":34,"8":900}
    for position in pawn_list:
        if player_position_map["p"].count(position[0]) > 1:
            ##doubled pawns
            diff -= 7
        if not str(int(position[1])-1) in player_position_map["p"] or not str(int(position[1])+1) in player_position_map["p"]:
            ### isolated pawn
            diff -= 2
        if position[0] not in opp_position_map["p"]:
            ### passed pawn
            diff += 1
        if is_white:
            diff += advancement_pawn[position[1]]
        else:
            diff += advancement_pawn[str(9-int(position[1]))]
    return diff

def bishop_rival_eval(bishop_list,player_diag_map,diag_pos_map):
    diff = 0
    bishop_pair = 30
    if len(bishop_list) == 2:
        ## Bishop Pair
        diff += bishop_pair
        diags = diag_pos_map[bishop_list[0]]
        diags += diag_pos_map[bishop_list[1]]
    elif len(bishop_list) == 0:
        return 0
    else:
        diags = diag_pos_map[bishop_list[0]]
    for diag in diags:
        ## if same diagonal as a same color pawn, reduce diff
        if "p" in player_diag_map[diag]:
            diff -= 15
    return diff

def knight_rival_eval(knight_list):
    diff = 0
    return diff

def rook_rival_eval(rook_list,player_position_map,opp_position_map,is_white):
    diff = 0
    file_dic = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8}
    bonuses_king_proximity = {0:15,1:9,2:5,3:4,4:3,5:1,6:-1,7:-3}
    if len(rook_list) == 2 and (rook_list[0][0] == rook_list[1][0] or rook_list[0][1] == rook_list[1][1]):
        ## Connected rooks
        diff += 15
    for rook in rook_list:
        min_dist = min(abs(file_dic[rook[0]]-file_dic[opp_position_map["k"][0]]),abs(int(rook[1])-int(opp_position_map["k"][1])))
        diff += bonuses_king_proximity[min_dist]
        ## 3 bonus if no friendly pawns in front and enemy 10 bonus if no pawns in front
        if rook[0] not in player_position_map["p"] and rook[0] not in opp_position_map["p"]:
            diff += 10
        elif rook[0] not in player_position_map["p"] and rook[0] in opp_position_map["p"]:
            diff += 3
        ## 20 bonus if rank == 7
        if "7" in rook and is_white:
            diff += 20
        elif not is_white and "2" in rook:
            diff += 20
    return diff

def queen_rival_eval(queen_list,opp_position_map,player_diag_map,diag_pos_map):
    diff = 0
    file_dic = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8}
    bonuses_king_proximity = {0:15,1:9,2:5,3:4,4:3,5:1,6:-1,7:-3}
    ## if same diagonal as bishop + x points
    diags = []
    for queen in queen_list:
        diags += diag_pos_map[queen]
        diff += bonuses_king_proximity[abs(file_dic[queen[0]]-file_dic[opp_position_map["k"][0]])]
        diff += bonuses_king_proximity[abs(int(queen[1])-int(opp_position_map["k"][1]))]
    for diag in diags:
        if "b" in player_diag_map[diag]:
            diff += 15
    ## distance from enemy king is awarded
    
    return diff

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

def get_sorted_move_list(board,agent = None,only_attacks = False):
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
            board.push(move)
            if board.is_checkmate():
                checkmate_list.append(move)
            elif board.is_check():
                check_list.append(move)
            board.pop()
            if board.is_capture(move):
                capture_list.append(move)
            elif only_attacks:
                other_list.append(move)
            elif board.is_castling(move):
                castling_list.append(move)
            else:
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
        return_list = [*checkmate_list,*check_list,*capture_list,*attack_list,*castling_list,*other_list]
        if only_attacks:
            return [*checkmate_list,*check_list,*capture_list]
        else:
            return return_list


if __name__ == "__main:__":
    board = ch.Board()
    print(minimax_with_pruning(board,2,True))