import torch


def get_device():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE
