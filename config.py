import torch
import torch.nn as nn

class CFG:
    board_size = 8 ## total length and height, total squares = 64
    n_players = 2
    n_pieces = 6
    real_planes = 7
    attacking_planes = 0
    n_planes = n_players * n_pieces + real_planes + attacking_planes


    lr = 0.0005
    every_x_epochs = 100
    mult_lr = 0.5

    weight_decay = 1e-7
    epochs = 10
    batch_size = 2048

    save_tensor_to_disk = False
    save_batch_to_device = True
    if save_batch_to_device:
        memory_batch = [torch.zeros((batch_size,n_planes,board_size,board_size)),torch.zeros((batch_size,1))]
        last_index = 0
        batch_full = False

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    
