import chess as ch
import numpy as np
import torch

def match(agent_one,agent_two,is_update_elo = True):
    try:
        game = ch.Board()
        agent_one.is_white = True
        agent_two.is_white = False

        while game.is_game_over() is False:
            move = agent_one.choose_move(game.fen())
            game.push(move[0])
            
            if game.is_game_over() is True:
                break
            move = agent_two.choose_move(game.fen())
            game.push(move[0])

        if is_update_elo:
            update_elo_agents(agent_one,agent_two,game.outcome().winner)
        return game.outcome().winner
    except:
        print(game.fen())
        raise AssertionError

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

def get_match_as_fen_tensor(board):
    match_len = len(board.move_stack)
    num_pieces = 6
    num_players = 2
    board_size = 8
    input_tensor_size = (match_len,board_size,board_size,num_pieces*num_players)
    tensor = torch.zeros(input_tensor_size)
    for i in range(match_len):
        fen=board.board_fen()
        tensor[i,:,:,:] = get_fen_as_tensor(fen)
        board.pop()
    return tensor

if __name__ == "__main:__":
    fen_test = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print(get_fen_as_tensor(fen_test))