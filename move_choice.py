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

if __name__ == "__main:__":
    board = ch.Board()
    print(minimax_with_pruning(board,2,True))