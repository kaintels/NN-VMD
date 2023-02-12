import numpy as np
import torch
import random
from torch.utils.data import TensorDataset, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class TrainDataset(Dataset):
    def __init__(self, x_data, *args):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data1 = None
        self.y_data2 = None
        assert len(args) < 3
        if len(args) == 1:
            self.y_data1 = torch.FloatTensor(args[0])
        if len(args) == 2:
            self.y_data1 = torch.FloatTensor(args[0])
            self.y_data2 = torch.LongTensor(args[1])

    
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        if self.y_data1 == None:
            return x.to(device)
        if self.y_data2 == None:
            y1 = torch.FloatTensor(self.y_data1[idx])
            return x.to(device), y1.to(device)
        else:
            y1 = torch.FloatTensor(self.y_data1[idx])
            y2 = torch.LongTensor(self.y_data2[idx])

        return x.to(device), y1.to(device), y2.to(device)

class TestDataset(Dataset):
    def __init__(self, x_data, *args):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data1 = None
        self.y_data2 = None
        assert len(args) < 3
        if len(args) == 1:
            self.y_data1 = torch.FloatTensor(args[0])
        if len(args) == 2:
            self.y_data1 = torch.FloatTensor(args[0])
            self.y_data2 = torch.LongTensor(args[1])

    
    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        if self.y_data1 == None:
            return x.to(device)
        if self.y_data2 == None:
            y1 = torch.FloatTensor(self.y_data1[idx])
            return x.to(device), y1.to(device)
        else:
            y1 = torch.FloatTensor(self.y_data1[idx])
            y2 = torch.LongTensor(self.y_data2[idx])

        return x.to(device), y1.to(device), y2.to(device)