import math
import os

import chess as ch
import numpy as np
import torch
from torch.utils.data import Dataset

from pychessai.constants import ChessBoard
from pychessai.utils.board_utils import (
    get_board_as_tensor,
    get_reward_value,
    map_moves_to_policy,
)
from pychessai.utils.utils import get_device


class CustomMatchDataset(Dataset):
    def __init__(
        self,
        dirpath=None,
        file_list=None,
        file_extension=".pt",
        val=False,
        val_train_split=0.8,
        idxs=[],
    ):
        if dirpath is None and file_list is None:
            raise Exception("You need to specify either dirpath or file_list")
        self.dirpath = dirpath
        self.file_extension = file_extension
        if file_list is None:
            self.file_list = [
                x for x in os.listdir(self.dirpath) if x.endswith(file_extension)
            ]
            self.file_len = len(self.file_list)
            self.idxs = idxs
            if len(idxs) > 0:
                self.file_list = [self.file_list[x] for x in self.idxs]
            else:
                if not val:
                    self.file_list = self.file_list[
                        : math.floor(self.file_len * val_train_split)
                    ]
                else:
                    self.file_list = self.file_list[
                        math.floor(self.file_len * val_train_split) :
                    ]
        else:
            self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        tensor = torch.load(os.path.join(self.dirpath, self.file_list[idx]))
        match = tensor[0]
        value_target = tensor[1]
        policy_target = tensor[2]
        return match, value_target, policy_target


class BatchMemoryDataset(Dataset):
    def __init__(self, tensor, length=None):
        if length is None:
            self.tensor = tensor
        else:
            a, b, c = tensor
            tensor = [a[0:length, :, :, :], b[0:length], c[0:length, :]]
            self.tensor = tensor
        self.length = length

    def __len__(self):
        return self.tensor[0].size(dim=0)

    def __getitem__(self, idx):
        return [self.tensor[0][idx, :, :, :], self.tensor[1][idx], self.tensor[2][idx]]


def create_tensors_from_png(dir_path):
    pgn_list = [x for x in os.listdir(dir_path) if x.endswith(".pgn")]
    print(pgn_list)
    for pgn in pgn_list:
        i = 0
        print(os.path.join(dir_path, pgn))
        pgn_iostream = open(os.path.join(dir_path, pgn))
        while pgn_iostream:
            try:
                tensor = get_game_as_tensor(pgn_iostream)
                torch.save(
                    tensor, os.path.join(dir_path, pgn.replace(".pgn", f"{i}_.pt"))
                )
                i += 1
            except Exception as e:
                print(e)
                break


def get_result(result):
    if result == "1-0":
        result = True
    elif result == "0-1":
        result = False
    else:
        result = None
    return result


def get_game_as_tensor(pgn):
    game = ch.pgn.read_game(pgn)
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)

    result = get_result(game.headers["Result"])

    tensor = get_match_as_fen_tensor(board, result, True, save_policy=True)
    return tensor


def get_match_as_fen_tensor(
    board,
    winner,
    player_white=True,
    save_policy=False,
    random_start=4,
    random_flip_chance=0.2,
):
    flip_board = np.random.choice(
        [False, True], size=1, p=[1.0 - random_flip_chance, random_flip_chance]
    )[0]
    if flip_board:
        player_white = not player_white
    match_len = len(board.move_stack) - random_start
    num_pieces = ChessBoard.NUM_PIECES
    num_players = ChessBoard.NUM_PLAYERS
    board_size = ChessBoard.BOARD_SIZE
    real_planes = ChessBoard.REAL_PLANES
    total_num_planes = num_pieces * num_players + real_planes
    input_tensor_size = (match_len, total_num_planes, board_size, board_size)
    target_tensor = torch.zeros((match_len, 1))
    tensor = torch.zeros(input_tensor_size)
    if save_policy:
        policy_tensor = torch.zeros((match_len, 73 * 8 * 8))
    for i in range(match_len):
        tensor[i, :, :, :] = get_board_as_tensor(board, player_white)
        policy_move = board.pop()
        target_tensor[i] = get_reward_value(
            winner, player_white, decay=False, current_move=i, match_len=match_len
        )
        if save_policy:
            policy_tensor[i, :] = map_moves_to_policy(
                [policy_move], board, flatten=True
            )[0]
    if save_policy:
        return [
            tensor.to(get_device()),
            target_tensor.to(get_device()),
            policy_tensor.to(get_device()),
        ]
    return [tensor.to(get_device()), target_tensor.to(get_device())]
