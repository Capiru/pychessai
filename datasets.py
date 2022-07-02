import os
import math
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch
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
        tensor = torch.load(self.file_list[idx])
        match = tensor[0]
        target = tensor[1]
        return match, target

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