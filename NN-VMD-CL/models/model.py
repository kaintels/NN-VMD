import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)

        self.layer1 = nn.Conv1d(1, 3, 3, 1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv1d(3, 64, 3, 1, padding=1)
        self.layer4 = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def create_encoder():
    net = nn.Sequential(
            nn.Conv1d(1, 3, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv1d(3, 64, 3, 1, padding=1),
            nn.ReLU()
    )

    return net

def gen_mode():
    net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8960, 420)
    )

    return net

def add_projection_head():
    net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8960, 140)
    )
    return net

