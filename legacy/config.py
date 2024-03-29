import torch
import torch.nn as nn

class CFG:
    DEBUG = False
    TEST = False
    EARLY_RESIGN = False
    EARLY_RESIGN_PATIENCE = 10
    SHOW_GAME = False
    fig = None
    ax = None

    board_size = 8 ## total length and height, total squares = 64
    n_players = 2
    n_pieces = 6
    real_planes = 7
    attacking_planes = 0
    n_planes = n_players * n_pieces + real_planes + attacking_planes


    lr = 0.001
    every_x_epochs = 1e9
    mult_lr = 1

    weight_decay = 1e-4
    
    epochs = 20
    sampling_ratio = 2
    patience = 100
    batch_size = 2048
    clip_norm = 0.5
    epsilon = 0.05

    policy_output_size = 4672

    save_tensor_to_disk = True
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
        
    WIN_VALUE = 100000
    LOSS_VALUE = -100000
    DRAW_VALUE = -0.01
    early_resignation_threshold = 800
    RANDOM_START = 4
    weight_policy = 1
    weight_value = 1
    random_flip_chance = 0.10
    start_master_train = False

    cloud_operations = True
    model_dir_path = "G:/Meine Ablage/projects/chessai/models/"
    dataset_dir_path = "G:/Meine Ablage/projects/chessai/datasets/"
    master_dataset_path = "G:/Meine Ablage/projects/chessai/datasets/master/"
    pgn_path = "./datasets"
    max_dataset_size = 25 ### in Gb
    model_dir_size_limit = 10 ### in Gb
    load_best_model = False
    
    GPU = torch.cuda.is_available()
    if GPU:
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    #print("Device set on : ",DEVICE)
    criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    
