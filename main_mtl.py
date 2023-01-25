import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import random
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from scipy.io import arff, loadmat
from sklearn.model_selection import train_test_split
from utils.util import seed_everything_th, weight_init_xavier_uniform
from models.model import VMDNet, Classifier, TaskLayer
import subprocess

EPOCH = 100
seed = 123456
seed_everything_th(seed)

# subprocess.run("julia ./utils/preprocessing.jl")

input_s_tr = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
input_s_tr = pd.DataFrame(input_s_tr[0], dtype=np.float32).values[:,:140].reshape(500, 1, 140)

input_l_tr = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
input_l_tr = pd.DataFrame(input_l_tr[0], dtype=np.float32).values[:,140:141].reshape(500)

input_l_tr = input_l_tr - 1

input_s_ts = arff.loadarff("./dataset/ECG5000_TEST.arff")
input_s_ts = pd.DataFrame(input_s_ts[0], dtype=np.float32).values[:,:140].reshape(4500, 1, 140)


input_l_ts = arff.loadarff("./dataset/ECG5000_TEST.arff")
input_l_ts = pd.DataFrame(input_l_ts[0], dtype=np.float32).values[:,140:141].reshape(4500)

input_l_ts = input_l_ts - 1

target_s_tr = loadmat("./dataset/processed_train.mat")["data"].reshape(500, 3, 140)
target_s_ts = loadmat("./dataset/processed_test.mat")["data"].reshape(4500, 3, 140)

train_s_loader = torch.utils.data.DataLoader(dataset = TensorDataset(torch.FloatTensor(input_s_tr), torch.FloatTensor(target_s_tr)), batch_size = 32, shuffle = True)
test_s_loader = torch.utils.data.DataLoader(dataset = TensorDataset(torch.FloatTensor(input_s_ts), torch.FloatTensor(target_s_ts)), batch_size = 32, shuffle = True)

train_l_loader = torch.utils.data.DataLoader(dataset = TensorDataset(torch.FloatTensor(input_s_tr), torch.LongTensor(input_l_tr)), batch_size = 32, shuffle = True)
test_l_loader = torch.utils.data.DataLoader(dataset = TensorDataset(torch.FloatTensor(input_s_ts), torch.LongTensor(input_l_ts)), batch_size = 32, shuffle = True)


model = TaskLayer()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

loss_1 = torch.nn.MSELoss()
loss_2 = torch.nn.CrossEntropyLoss()

for i in range(EPOCH):

    for _, data in enumerate(zip(train_s_loader, train_l_loader)):
        optimizer.zero_grad()

        output1, output2 = model(data[0][0])
        loss1 = loss_1(output1, data[0][1])
        loss2 = loss_2(output2, data[1][1])

        # weight = F.softmax(torch.randn(2), dim=-1)
        # total_loss = (loss1 + loss2) * weight

        total_loss = (loss1 + loss2) / 2

        total_loss.backward()

        optimizer.step()
    if i % 10 == 0:
        print("epoch : {0}, loss : {1}".format(i, total_loss.item()))

# model1 = VMDNet()
# model2 = Classifier()

# models_pm = list(model1.parameters())+list(model2.parameters())

# optimizer = torch.optim.Adam(models_pm, lr = 0.01, eps=1e-8, betas=(0.9, 0.999))

# loss_1 = torch.nn.MSELoss()
# loss_2 = torch.nn.CrossEntropyLoss()

# for i in range(EPOCH):
#     avg_cost = 0
#     for _, data in enumerate(zip(train_s_loader, train_l_loader)):
#         optimizer.zero_grad()

#         output1 = model1(data[0][0])
#         loss1 = loss_1(output1, data[0][1])

#         output2 = model2(data[1][0])
#         loss2 = loss_2(output2, data[1][1])

#         total_loss = torch.norm(loss1 + loss2)

#         total_loss.backward(retain_graph=True)

#         optimizer.step()

#     if i % 10 == 0:
#         print("epoch : {0}, loss : {1}".format(i, total_loss.item()))

# # imf1_s = []
# # imf2_s = []
# # imf3_s = []

# # imf_all = []

# # model.eval()
# # with torch.no_grad(): 
# #     correct = 0
# #     total = 0

# #     for i, (datax, targetx) in enumerate(test_loader):
# #         out = model(datax)

# #         imf_all.append(out.cpu().numpy())
# #         a = targetx.numpy()[0].reshape(140, 3)
# #         b = out.numpy()[0].reshape(140, 3)

# #         imf1 = np.corrcoef(a[:, 0], b[:, 0])[0, 1]
# #         imf2 = np.corrcoef(a[:, 1], b[:, 1])[0, 1]
# #         imf3 = np.corrcoef(a[:, 2], b[:, 2])[0, 1]

# #         imf1_s.append(imf1)
# #         imf2_s.append(imf2)
# #         imf3_s.append(imf3)

# # print(np.mean(imf1_s))
# # print(np.mean(imf2_s))
# # print(np.mean(imf3_s))
# # print(np.std(imf1_s))
# # print(np.std(imf2_s))
# # print(np.std(imf3_s))

# # plt.figure()
# # plt.subplot(2,1,1)
# # plt.plot(a)
# # plt.title('Original signal')
# # plt.xlabel('time (s)')
# # plt.subplot(2,1,2)
# # plt.plot(b)
# # plt.title('Decomposed modes')
# # plt.xlabel('time (s)')
# # plt.tight_layout()
# # plt.show()