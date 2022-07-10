import os
import math
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch
from config import CFG
import chess.pgn
import chess as ch
from match import get_match_as_fen_tensor

class CustomMatchDataset(Dataset):
    def __init__(self, dirpath = None, file_list = None ,file_extension = ".pt",val = False,val_train_split = 0.8,idxs = []):
        if dirpath is None and file_list is None:
            raise("You need to specify either dirpath or file_list")
        self.dirpath = dirpath
        self.file_extension = file_extension
        if file_list is None:
            self.file_list = [x for x in os.listdir(self.dirpath) if x.endswith(file_extension)]
            self.file_len = len(self.file_list)
            self.idxs = idxs
            if len(idxs) > 0:
                self.file_list = [self.file_list[x] for x in self.idxs]
            else:
                if not val:
                    self.file_list = self.file_list[:math.floor(self.file_len*val_train_split)]
                else:
                    self.file_list = self.file_list[math.floor(self.file_len*val_train_split):]
        else:
            self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        tensor = torch.load(os.path.join(self.dirpath,self.file_list[idx]))
        match = tensor[0]
        value_target = tensor[1]
        policy_target = tensor[2]
        return match, value_target, policy_target

class BatchMemoryDataset(Dataset):
    def __init__(self,tensor,length = None):
        if length is None:
            self.tensor = tensor
        else:
            a,b,c = tensor
            tensor = [a[0:length,:,:,:],b[0:length],c[0:length,:]]
            self.tensor = tensor
        self.length = length

    def __len__(self):
        return self.tensor[0].size(dim=0)

    def __getitem__(self,idx):
        return [self.tensor[0][idx,:,:,:],self.tensor[1][idx],self.tensor[2][idx]]

class MasterDataset(Dataset):
    def __init__(self,idxs = []):
        self.file_list = [x for x in os.listdir(CFG.master_dataset_path) if x.endswith(".pt")]
        self.idxs = idxs
        if len(idxs) > 0:
            self.file_list = [self.file_list[x] for x in self.idxs]
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self,idx):
        return torch.load(os.path.join(CFG.master_dataset_path,self.file_list[idx]))

def create_tensors_from_png():
    pgn_list = [x for x in os.listdir(CFG.pgn_path) if x.endswith(".pgn")]
    print(pgn_list)
    for pgn in pgn_list:
        i = 0
        print(os.path.join(CFG.pgn_path,pgn))
        pgn_iostream = open(os.path.join(CFG.pgn_path,pgn))
        while pgn_iostream:
            try:
                tensor = get_game_as_tensor(pgn_iostream)
                torch.save(tensor,os.path.join(CFG.master_dataset_path,pgn.replace(".pgn",f"{i}_.pt")))
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

    tensor = get_match_as_fen_tensor(board,result, True,save_policy=True)
    return tensor
    
if __name__ == "__main__":
    #create_tensors_from_png()
    pass