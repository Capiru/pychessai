import torch
import torch.nn as nn

class CFG:
    DEBUG = False
    TEST = False

    board_size = 8 ## total length and height, total squares = 64
    n_players = 2
    n_pieces = 6
    real_planes = 7
    attacking_planes = 0
    n_planes = n_players * n_pieces + real_planes + attacking_planes


    lr = 0.0005
    every_x_epochs = 1000
    mult_lr = 0.5

    weight_decay = 1e-4
    
    epochs = 20
    patience = 10
    batch_size = 2048

    policy_output_size = 4672

    save_tensor_to_disk = False
    save_batch_to_device = True
    validate_match = True
    val_every_x_games = 5
    if save_batch_to_device:
        memory_batch = [torch.zeros((batch_size,n_planes,board_size,board_size)),torch.zeros((batch_size,1)),torch.zeros((batch_size,policy_output_size)),
                        torch.zeros((batch_size,n_planes,board_size,board_size)),torch.zeros((batch_size,1)),torch.zeros((batch_size,policy_output_size))]
        last_index = 0
        last_policy_index = 0
        val_last_policy_index = 0
        val_last_index = 0
        batch_full = False
        count_since_last_val_match = 0
        
    WIN_VALUE = 1
    LOSS_VALUE = -1
    DRAW_VALUE = -0.01
    RANDOM_START = 4
    weight_policy = 1
    weight_value = 1

    cloud_operations = False
    model_dir_path = "/content/drive/MyDrive/projects/chessai/models/"
    dataset_dir_path = "/content/drive/MyDrive/projects/chessai/datasets/"
    max_dataset_size = 25 ### in Gb

    GPU = torch.cuda.is_available()
    if GPU:
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    
