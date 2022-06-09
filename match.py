import chess as ch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time
import math
from move_choice import random_choice

def fen_start_from_opening(openings_path = "./openings/df_openings.csv"):
    df = pd.read_csv(openings_path)
    opening_len = len(df)
    random_idx = np.random.randint(0,opening_len-1)
    return str(df.iloc[random_idx].starting_fen)

def match(agent_one,agent_two,is_update_elo = True,start_from_opening = False,start_from_random = False,random_start_depth=6,save_tensor = True):
    try:
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
        agent_one.is_white = True
        agent_two.is_white = False
        agent_one.positions = 0
        agent_two.positions = 0
        
        while game.is_game_over() is False:
            move = agent_one.choose_move(game)
            game.push(move[0])
            
            if game.is_game_over() is True:
                break
            move = agent_two.choose_move(game)
            game.push(move[0])

        if is_update_elo:
            update_elo_agents(agent_one,agent_two,game.outcome().winner)
        if save_tensor:
            winner = game.outcome().winner
            match_tensor = get_match_as_fen_tensor(game,winner)
            return winner,match_tensor
        else:
            return game.outcome().winner
    except Exception() as e:
        print(game.fen())
        print(e)
        raise AssertionError

def save_tensor(tensor):
    positions,outcomes = tensor
    for i in range(positions.size(dim=0)):
        torch.save([positions[i,:,:,:],outcomes[i]],str(time.time())+str(i)+".pt")
    return None


def experiments(agent_one,agent_two,n=100,is_update_elo=True,start_from_opening = False,start_from_random=False,progress_bar = True,save_match_tensor = True):
    outcomes = [0, 0, 0]
    if progress_bar:
        progress = tqdm(range(n), desc="", total=n)
    else:
        progress = range(n)
    for i in progress:
        if i % 2 == 0:
            if save_match_tensor:
                outcome,tensor = match(agent_one,agent_two,start_from_opening=start_from_opening,start_from_random=start_from_random)
                save_tensor(tensor)
            else:
                outcome = match(agent_one,agent_two,start_from_opening=start_from_opening,start_from_random=start_from_random,save_tensor=False)
        else:
            if save_match_tensor:
                outcome,tensor = match(agent_two,agent_one,start_from_opening=start_from_opening,start_from_random=start_from_random)
                save_tensor(tensor)
            else:
                outcome =  match(agent_two,agent_one,start_from_opening=start_from_opening,start_from_random=start_from_random,save_tensor=False)
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
            progress.set_description(str(outcomes)+"  1:"+str(round(agent_one.elo,2))+"  1-pos:"+str(agent_one.positions)+"   2:"+str(round(agent_two.elo,2))+"  2-pos:"+str(agent_two.positions))

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

def get_fen_as_tensor(fen):
    pytorch = True
    num_pieces = 6
    num_players = 2
    board_size = 8
    if pytorch:
        input_tensor_size = (num_pieces*num_players,board_size,board_size)
    else:
        input_tensor_size = (board_size,board_size,num_pieces*num_players)
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
    return tensor

def decaying_function_cosdecay(l,x,winner):
    if winner:
        return math.sin(math.pi*(x/l)**3-math.pi/2)/2+0.5
    else:
        return -(math.sin(math.pi*(x/l)**3-math.pi/2)/2+0.5)

def get_match_as_fen_tensor(board,winner):
    pytorch = True
    match_len = len(board.move_stack)
    num_pieces = 6
    num_players = 2
    board_size = 8
    if pytorch:
      input_tensor_size = (match_len,num_pieces*num_players,board_size,board_size)
    else:
      input_tensor_size = (match_len,board_size,board_size,num_pieces*num_players)
    target_tensor = torch.zeros((match_len,1))
    tensor = torch.zeros(input_tensor_size)
    for i in range(match_len):
        fen=board.board_fen()
        tensor[i,:,:,:] = get_fen_as_tensor(fen)
        board.pop()
        if winner is None:
            target_tensor[i] = 0.5
        elif winner:
            target_tensor[i] = decaying_function_cosdecay(match_len,match_len-i,winner=True)
        else:
            target_tensor[i] = decaying_function_cosdecay(match_len,match_len-i,winner=False)
        
    return [tensor,target_tensor]

if __name__ == "__main:__":
    fen_test = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print(get_fen_as_tensor(fen_test))
    print(fen_start_from_opening())