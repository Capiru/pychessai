from config import CFG
import chess as ch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time
import math
from move_choice import random_choice
from cloudops import dir_size
import os
from game import *
from policy import map_moves_to_policy

def fen_start_from_opening(openings_path = "./openings/df_openings.csv"):
    df = pd.read_csv(openings_path)
    opening_len = len(df)
    random_idx = np.random.randint(0,opening_len-1)
    return str(df.iloc[random_idx].starting_fen)

def early_resignation(board,eval_white,eval_black):
    if abs(eval_white) > 500 or abs(eval_black) > 500:
        if eval_white > 500:
            return True,True 
        else:
            return True,False
    else:
        return False,None

def match(agent_one,agent_two,is_update_elo = True,start_from_opening = False,start_from_random = False,random_start_depth=16,save_tensor = True,progress = None,is_player_one = True):
    if start_from_opening:
        game = ch.Board(fen_start_from_opening())
    else:
        game = ch.Board()
        if start_from_random and random_start_depth%2 == 0:
            for i in range(random_start_depth):
                legal_moves = list(game.legal_moves)
                if len(legal_moves) > 0:
                    move = random_choice(legal_moves,None,1)
                    game.push(move[0])
    try:
        agent_one.reset_game()
        agent_two.reset_game()
    except:
        a = 1
    agent_one.is_white = True
    agent_two.is_white = False
    agent_one.positions = 0
    agent_two.positions = 0
    early_resign = False
    early_resign_patience = 0
    while game.is_game_over() is False:
        move = agent_one.choose_move(game)
        game.push(move[0])
        if CFG.SHOW_GAME:
            CFG.fig = show_board(game, agent_one.eval, agent_two.eval,CFG.fig)
        if game.is_game_over() is True:
            break
        early_resign,outcome = early_resignation(game, agent_one.eval, agent_two.eval)
        if early_resign:
            early_resign_patience+=1
        else:
            early_resign_patience = 0
        move = agent_two.choose_move(game)
        game.push(move[0])
        if CFG.SHOW_GAME:
            CFG.fig = show_board(game, agent_one.eval, agent_two.eval,CFG.fig)
        early_resign,outcome = early_resignation(game, agent_one.eval, agent_two.eval)
        if early_resign:
            early_resign_patience+=1
        else:
            early_resign_patience = 0
        if early_resign_patience > CFG.EARLY_RESIGN_PATIENCE:
            break
        if progress is not None:
            progress.set_description()
    if not early_resign:
        try:
            winner = game.outcome().winner
            win_nxor = not (winner ^ is_player_one)
        except:
            win_nxor = None
            winner = None
    else:
        winner = outcome
        win_nxor = not (winner ^ is_player_one)
    
    if is_update_elo:
        update_elo_agents(agent_one,agent_two,winner)
    if save_tensor:
        match_tensor = get_match_as_fen_tensor(game,win_nxor,is_player_one)
        return winner,match_tensor
    else:
        return game.outcome().winner

def save_tensor(tensor):
    if CFG.cloud_operations:
        dir_path = CFG.dataset_dir_path
    else:
        dir_path = "./datasets/"
    if CFG.save_tensor_to_disk and not CFG.save_batch_to_device:
        positions,outcomes = tensor
        for i in range(positions.size(dim=0)):
            torch.save([positions[i,:,:,:],outcomes[i],CFG.memory_batch[2][CFG.last_policy_index - i,:]],os.path.join(dir_path,str(time.time())+str(i)+".pt"))
            CFG.last_policy_index = 0
        return None
    elif CFG.save_batch_to_device:
        CFG.count_since_last_val_match+=1
        size = tensor[0].size(dim=0)
        if CFG.count_since_last_val_match % CFG.val_every_x_games == 0:
            ### Save Val Batch
            if CFG.val_last_index + size >= CFG.batch_size:
                CFG.memory_batch[3][CFG.val_last_index:CFG.batch_size,:,:,:] = tensor[0][0:CFG.batch_size-CFG.val_last_index,:,:,:]
                CFG.memory_batch[4][CFG.val_last_index:CFG.batch_size] = tensor[1][0:CFG.batch_size-CFG.val_last_index]
                if CFG.save_tensor_to_disk:
                    torch.save([CFG.memory_batch[3],CFG.memory_batch[4],CFG.memory_batch[5]],os.path.join(dir_path,str(time.time())+"_valbatch.pt"))
                CFG.memory_batch[3][0:size-(CFG.batch_size-CFG.val_last_index),:,:,:] = tensor[0][CFG.batch_size-CFG.val_last_index:size,:,:,:]
                CFG.memory_batch[4][0:size-(CFG.batch_size-CFG.val_last_index)] = tensor[1][CFG.batch_size-CFG.val_last_index:size]
                CFG.val_last_index = size - (CFG.batch_size - CFG.val_last_index)
                
            else:
                CFG.memory_batch[3][CFG.val_last_index:CFG.val_last_index+size,:,:,:] = tensor[0]
                CFG.memory_batch[4][CFG.val_last_index:CFG.val_last_index+size] = tensor[1]
                CFG.val_last_index += size
        else:
            if CFG.TEST:
                try:
                    if not CFG.last_index + size >= CFG.batch_size:
                        assert CFG.last_index == CFG.last_policy_index-size
                except:
                    print(CFG.last_index,CFG.last_policy_index,size)
                    assert False
            ### Save training batch
            if CFG.last_index + size > CFG.batch_size:
                CFG.batch_full = True
                CFG.memory_batch[0][CFG.last_index:CFG.batch_size,:,:,:] = tensor[0][0:CFG.batch_size-CFG.last_index,:,:,:]
                CFG.memory_batch[1][CFG.last_index:CFG.batch_size] = tensor[1][0:CFG.batch_size-CFG.last_index]
                if CFG.save_tensor_to_disk:
                    torch.save([CFG.memory_batch[0],CFG.memory_batch[1],CFG.memory_batch[2]],os.path.join(dir_path,str(time.time())+"_batch.pt"))
                CFG.memory_batch[0][0:size-(CFG.batch_size-CFG.last_index),:,:,:] = tensor[0][CFG.batch_size-CFG.last_index:size,:,:,:]
                CFG.memory_batch[1][0:size-(CFG.batch_size-CFG.last_index)] = tensor[1][CFG.batch_size-CFG.last_index:size]
                CFG.last_index = size - (CFG.batch_size - CFG.last_index)
                
            else:
                CFG.memory_batch[0][CFG.last_index:CFG.last_index+size,:,:,:] = tensor[0]
                CFG.memory_batch[1][CFG.last_index:CFG.last_index+size] = tensor[1]
                CFG.last_index += size


def experiments(agent_one,agent_two,n=100,is_update_elo=True,start_from_opening = False,start_from_random=False,
                random_start_depth = 0,progress_bar = True,save_match_tensor = True,is_player_one = True):
    outcomes = [0, 0, 0]
    if progress_bar:
        progress = tqdm(range(n), desc="", total=n)
    else:
        progress = range(n)
    for i in progress:
        if i % 2 == 0:
            if save_match_tensor:
                outcome,tensor = match(agent_one,agent_two,start_from_opening=start_from_opening,start_from_random=start_from_random,random_start_depth =random_start_depth,is_player_one=is_player_one)
                save_tensor(tensor)
            else:
                outcome = match(agent_one,agent_two,start_from_opening=start_from_opening,start_from_random=start_from_random,random_start_depth =random_start_depth,save_tensor=False,is_player_one=is_player_one)
        else:
            if save_match_tensor:
                outcome,tensor = match(agent_two,agent_one,start_from_opening=start_from_opening,start_from_random=start_from_random,random_start_depth =random_start_depth,is_player_one=not is_player_one)
                save_tensor(tensor)
            else:
                outcome =  match(agent_two,agent_one,start_from_opening=start_from_opening,start_from_random=start_from_random,random_start_depth =random_start_depth,save_tensor=False,is_player_one=not is_player_one)
        if outcome is None:
            #draw
            outcomes[1] += 1
        elif outcome == False:
            if i % 2 == 0:
                #black win
                outcomes[2] += 1
            else:
                #white win
                outcomes[0] += 1
        else:
            if i % 2 == 0:
                #white win
                outcomes[0] += 1
            else:
                #black win
                outcomes[2] += 1
        if progress_bar:
            progress.set_description(str(outcomes)+ f" {get_elo_diff_from_outcomes(outcomes)}"+" is1_white:"+str(int(agent_one.is_white))+"  1:"+str(agent_one.eval) +"  1-pos:"+str(agent_one.positions)+"   2:"+str(agent_two.eval)+"  2-pos:"+str(agent_two.positions))
        if CFG.batch_full and (CFG.save_batch_to_device and not CFG.save_tensor_to_disk) and save_match_tensor:
            break
        if CFG.cloud_operations and dir_size(CFG.dataset_dir_path) > CFG.max_dataset_size:
            break
    return outcomes

def update_elo_agents(white,black,outcome):
    if outcome is None:
        if white.elo > black.elo:
            black.elo,white.elo = update_elo(black.elo, white.elo, is_draw = True)
        else:
            white.elo,black.elo = update_elo(white.elo, black.elo, is_draw = True)
    elif outcome == False:
        black.elo,white.elo = update_elo(black.elo, white.elo, is_draw = False)
    else:
        white.elo,black.elo = update_elo(white.elo, black.elo, is_draw = False)


def update_elo(winner_elo, loser_elo, is_draw = False):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = 64 * (1-expected_win)
    if is_draw:
        winner_elo += change_in_elo/2
        loser_elo -= change_in_elo/2
    else:
        winner_elo += change_in_elo
        loser_elo -= change_in_elo
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0/(1+10**((elo_b - elo_a)/400))
    return expect_a

def get_elo_diff_from_outcomes(outcomes):
    n_draws = outcomes[1]
    white_wins = outcomes[0]
    black_wins = outcomes[2]
    total_games = n_draws + white_wins +black_wins
    white_score = 1* white_wins + 0.5*n_draws
    black_score = 1*black_wins + 0.5*n_draws
    score = max(white_score,black_score)
    if white_score == total_games:
        return 400
    elif black_score == total_games:
        return -400
    elo_diff = round(-400*math.log(1/(score/total_games)-1)/math.log(10),0)
    if white_score >= black_score:
        return elo_diff
    else:
        return -elo_diff

def get_fen_as_tensor(fen):
    pytorch = True
    num_pieces = 6
    num_players = 2
    board_size = 8
    real_planes = 7 ## 4 castling 1 black or white 
    attacking_planes = 1
    total_num_planes = num_pieces*num_players + real_planes
    if pytorch:
        input_tensor_size = (total_num_planes,board_size,board_size)
    else:
        input_tensor_size = (board_size,board_size,total_num_planes)
    tensor = torch.zeros(input_tensor_size)
    dic_encoder = {"p":0,"P":1,"r":2,"R":3,"n":4,"N":5,"b":6,"B":7,"q":8,"Q":9,"k":10,"K":11}
    fen_position = fen.split(" ")[0]
    fen_row_positions = fen_position.split("/")
    row = 0
    file_ = 0
    for char in fen_position:
        if char == "/":
            row += 1
            file_ = 0
        elif str.isnumeric(char):
            file_ += int(char)
        else:
            if pytorch:
                tensor[dic_encoder[char],row,file_] = 1
            else:
                tensor[row,file_,dic_encoder[char]] = 1
            file_ += 1
    return tensor.to(CFG.DEVICE)

def get_board_as_tensor(board,player_white):
    num_pieces = 6
    num_players = 2
    board_size = 8
    real_planes = 7 ## 4 castling 1 black or white 
    attacking_planes = 1
    total_num_planes = num_pieces*num_players + real_planes

    pytorch = True
    if pytorch:
        input_tensor_size = (total_num_planes,board_size,board_size)
    else:
        input_tensor_size = (board_size,board_size,total_num_planes)
    tensor = torch.zeros(input_tensor_size)
    
    if board.turn == ch.WHITE ^ player_white:
        ### opponent's turn
        tensor[12,:,:] = 1
    pieces = [ch.PAWN,ch.KNIGHT,ch.BISHOP,ch.ROOK,ch.QUEEN,ch.KING]
    if player_white:
        colors = [ch.WHITE,ch.BLACK]
        starting_plane = 7
    else:
        colors = [ch.BLACK,ch.WHITE]
        starting_plane = 0
    num_before_draw = math.floor(board.halfmove_clock/2)
    if num_before_draw >= 64:
        tensor[17,7,7] = 1
    tensor[17,num_before_draw%8,((num_before_draw)//8)%8] = 1
    plane = -1
    player_color = 0
    for color in colors:
        if board.has_kingside_castling_rights(color):
            tensor[13+player_color*2,:,:] = 1
        if board.has_queenside_castling_rights(color):
            tensor[14+player_color*2,:,:] = 1
        player_color += 1
        for piece in pieces:
            plane += 1
            piece_map = board.pieces(piece,color)
            for pos in piece_map:
                tensor[plane,abs(starting_plane-pos//8),pos%8] = 1
    assert plane == int(len(pieces)*2-1)

    return tensor.to(CFG.DEVICE)

def decaying_function_cosdecay(l,x):
    return math.sin(math.pi*(x/l)**3-math.pi/2)/2+0.5

def get_match_as_fen_tensor(board,winner,player_white = True,save_policy = False):
    flip_board = np.random.choice([False,True],size=1,p = [1-CFG.random_flip_chance,CFG.random_flip_chance])[0]
    if flip_board:
        player_white = not player_white
    pytorch = True
    match_len = len(board.move_stack)-CFG.RANDOM_START
    num_pieces = 6
    num_players = 2
    board_size = 8
    real_planes = 7
    total_num_planes = num_pieces*num_players + real_planes
    if pytorch:
      input_tensor_size = (match_len,total_num_planes,board_size,board_size)
    else:
      input_tensor_size = (match_len,board_size,board_size,total_num_planes)
    target_tensor = torch.zeros((match_len,1))
    tensor = torch.zeros(input_tensor_size)
    if save_policy:
        policy_tensor = torch.zeros((match_len,73*8*8))
    for i in range(match_len):
        tensor[i,:,:,:] = get_board_as_tensor(board,player_white)
        policy_move = board.pop()
        if winner is None:
            target_tensor[i] = CFG.DRAW_VALUE
        elif not (winner ^ (player_white)):
            target_tensor[i] = decaying_function_cosdecay(match_len,match_len-i)
        else:
            target_tensor[i] = -decaying_function_cosdecay(match_len,match_len-i)
        if save_policy:
            policy_tensor[i,:] = map_moves_to_policy([policy_move],board,flatten=True)[0]
    if save_policy:
        return [tensor,target_tensor,policy_tensor]
    return [tensor.to(CFG.DEVICE),target_tensor.to(CFG.DEVICE)]




if __name__ == "__main:__":
    fen_test = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print(get_fen_as_tensor(fen_test))
    print(fen_start_from_opening())
    fen_test = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    board = ch.Board()
    print(board.pieces(ch.PAWN,ch.WHITE))
    print(board.fen())
    print(get_board_as_tensor(board,player_white = True))