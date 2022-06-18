import torch
import torch.nn as nn

class CFG:
    lr = 0.0005
    epochs = 10
    batch_size = 2048

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
