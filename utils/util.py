import numpy as np
import torch
import random

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0)
    elif isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0)

def seed_everything_th(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(num)
    random.seed(num)