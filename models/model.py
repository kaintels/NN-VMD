import torch
import torch.nn as nn
import torch.nn.functional as F

class VMDNet(nn.Module):
    def __init__(self):
        super(VMDNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, 1, padding="same")
        self.conv2 = nn.Conv1d(32, 64, 3, 1, padding="same")
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(8960, 420)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = x.reshape(-1, 3, 140)
        return x