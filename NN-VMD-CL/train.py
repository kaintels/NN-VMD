import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff, loadmat
from sklearn.model_selection import train_test_split
from utils.util import seed_everything_th, TrainDataset, TestDataset
from models.model import create_encoder, gen_mode, add_projection_head
from torchvision.transforms import Compose

EPOCH = 100
seed = 42
seed_everything_th(seed)
device = torch.device("cpu")


input_s_t = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
input_s_t = pd.DataFrame(
    input_s_t[0], dtype=np.float32).values[:, :140].reshape(500, 1, 140)

input_l_tr = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
input_l_tr = pd.DataFrame(
    input_l_tr[0], dtype=np.float32)

input_l_tr['target'] = input_l_tr['target'].replace(3, 2)
input_l_tr['target'] = input_l_tr['target'].replace(4, 2)
input_l_tr['target'] = input_l_tr['target'].replace(5, 2)
input_l_tr = input_l_tr.values[:, 140:141].reshape(500)

input_l_tr = input_l_tr - 1

target_s_tr = loadmat(
    "./dataset/processed_train.mat")["data"].reshape(500, 3, 140)
target_s_ts = loadmat(
    "./dataset/processed_test.mat")["data"].reshape(4500, 3, 140)

import ecgmentations as E

# Declare an augmentation pipeline
transform = E.Sequential([
    E.TimeShift(p=0.5),
])

transformed = transform(ecg=input_s_t)
transformed_ecg = transformed['ecg']


train_loader = torch.utils.data.DataLoader(dataset=TrainDataset(
    transformed_ecg, target_s_tr, input_l_tr), batch_size=32, shuffle=False)

loss_fn_1 = torch.nn.MSELoss()

encoder = create_encoder()
projection_head = add_projection_head()

moder = gen_mode()

from info_nce import InfoNCE

encoder.load_state_dict(torch.load("./models/encoder_proj.pt"))

optimizer = torch.optim.Adam(list(encoder.parameters()) + 
                                  list(moder.parameters()), lr=0.001)


c_loss = InfoNCE()

print("-" * 100)

for i in range(EPOCH):
    avg_cost = 0
    for _, data in enumerate(train_loader):
        optimizer.zero_grad()

        signal = data[0].to(device) #신호 
        mode = data[1].to(device)

        encode = encoder(signal)
        decode = moder(encode)
        
        loss1 = loss_fn_1(decode.reshape(-1, 3, 140), mode)
        # loss2 = c_loss(signal.squeeze(), decode)

        total_loss = loss1 #(loss1 + loss2) / 2
        total_loss.backward()
        optimizer.step()
        avg_cost += total_loss / len(train_loader)


    if i % 10 == 0:
        a = decode.reshape(-1, 140, 3)
        # plt.plot(a[0, :, 0].cpu().detach().numpy())
        # plt.plot(data[1].reshape(-1, 140, 3)[0, :, 0].cpu().detach().numpy())
        # plt.show()
        print("epoch : {0}, loss : {1}".format(i, avg_cost.item()))


torch.save(encoder.state_dict(), f"./models/encoder_b.pt")
torch.save(moder.state_dict(), f"./models/moder_b.pt")

encoder.load_state_dict(torch.load("./models/encoder_b.pt"))
moder.load_state_dict(torch.load("./models/moder_b.pt"))
b = []
l = []
with torch.no_grad():
    for _, data in enumerate(train_loader):

        signal = data[0].to(device) #신호 
        # mode = data[1].to(device)
        label = data[2]

        encode = encoder(signal)
        decode = moder(encode)

        b.append(decode.detach().numpy())
        l.append(label)

print(np.concatenate(b).shape)
print(np.concatenate(l).shape)

np.save("deep_mode_b.npy", np.concatenate(b))
np.save("deep_mode_b_label.npy", np.concatenate(l))