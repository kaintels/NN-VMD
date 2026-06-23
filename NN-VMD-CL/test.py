import torch
import argparse
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.util import seed_everything_th, weight_init_xavier_uniform, TrainDataset, TestDataset
from models.model import create_encoder, gen_mode
from scipy.io import arff, loadmat

seed = 42
seed_everything_th(seed)


input_s_tr = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
input_s_tr = pd.DataFrame(
    input_s_tr[0], dtype=np.float32).values[:, :140].reshape(500, 1, 140)

input_l_tr = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
input_l_tr = pd.DataFrame(
    input_l_tr[0], dtype=np.float32).values[:, 140:141].reshape(500)

input_l_tr = input_l_tr - 1

input_s_ts = arff.loadarff("./dataset/ECG5000_TEST.arff")
input_s_ts = pd.DataFrame(
    input_s_ts[0], dtype=np.float32).values[:, :140].reshape(4500, 1, 140)


input_l_ts = arff.loadarff("./dataset/ECG5000_TEST.arff")
input_l_ts = pd.DataFrame(
    input_l_ts[0], dtype=np.float32)

input_l_ts['target'] = input_l_ts['target'].replace(3, 2)
input_l_ts['target'] = input_l_ts['target'].replace(4, 2)
input_l_ts['target'] = input_l_ts['target'].replace(5, 2)

input_l_ts = input_l_ts.values[:, 140:141].reshape(4500)

input_l_ts = input_l_ts - 1

target_s_tr = loadmat(
    "./dataset/processed_train.mat")["data"].reshape(500, 3, 140)
target_s_ts = loadmat(
    "./dataset/processed_test.mat")["data"].reshape(4500, 3, 140)


encoder = create_encoder()
moder = gen_mode()

encoder.load_state_dict(torch.load("./models/encoder_b.pt"))
moder.load_state_dict(torch.load("./models/moder_b.pt"))

test_loader = torch.utils.data.DataLoader(dataset=TestDataset(
    input_s_ts, target_s_ts, input_l_ts), batch_size=1, shuffle=True)

imf1_s = []
imf2_s = []
imf3_s = []

bs = []
l = []
with torch.no_grad(): 
    correct = 0
    total = 0
    for _, data in enumerate(test_loader):

        signal = data[0] #신호 
        mode = data[1]
        label = data[2]

        encode = encoder(signal)
        decode = moder(encode)

        bs.append(decode.detach().numpy())
        l.append(label)


        # plt.plot(signal.reshape(-1, 140, 1)[0, :, 0])
        # plt.plot(decode.reshape(-1, 140, 3)[0, :, 2])
        # plt.show()

        for i in range(data[0].shape[0]):
            a = data[1][i].numpy().reshape(-1, 140, 3) # 검증
            b = decode[i].numpy().reshape(-1, 140, 3)


            imf1 = np.corrcoef(a[0, :, 0], b[0, :, 0])[0, 1]
            imf2 = np.corrcoef(a[0, :, 1], b[0, :, 1])[0, 1]
            imf3 = np.corrcoef(a[0, :, 2], b[0, :, 2])[0, 1]

            imf1_s.append(imf1)
            imf2_s.append(imf2)
            imf3_s.append(imf3)

# print(np.concatenate(bs).shape)
# print(np.concatenate(l).shape)

np.save("test_deep_mode_b.npy", np.concatenate(bs))
np.save("test_deep_mode_b_label.npy", np.concatenate(l))

# plt.plot(signal.reshape(-1, 140, 1)[0, :, 0])
# plt.plot(b.sum(2).reshape(-1, 140, 1)[0, :, 0])
# plt.show()
# print(len(imf1_s))
print(np.mean(imf1_s))
print(np.mean(imf2_s))
print(np.mean(imf3_s))
print(np.std(imf1_s))
print(np.std(imf2_s))
print(np.std(imf3_s))