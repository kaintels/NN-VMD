import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from scipy.io import arff
import torch
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset 
from utils.util import seed_everything_th, vmd_execute, weight_init_xavier_uniform
from models.model import VMDNet_TH

EPOCH = 10
seed = 123456
seed_everything_th(seed)

tr = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
tr = pd.DataFrame(tr[0], dtype=np.float32)

ts = arff.loadarff("./dataset/ECG5000_TEST.arff")
ts = pd.DataFrame(ts[0], dtype=np.float32)

input_tr, target_tr = vmd_execute(tr, is_torch=True)
input_ts, target_ts = vmd_execute(ts, is_torch=True)

train_loader = torch.utils.data.DataLoader(dataset = TensorDataset(torch.FloatTensor(input_tr), torch.FloatTensor(target_tr)), batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = TensorDataset(torch.FloatTensor(input_ts), torch.FloatTensor(target_ts)), batch_size = 32, shuffle = True)

model = VMDNet_TH()
model.apply(weight_init_xavier_uniform)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, eps=1e-8, betas=(0.9, 0.999))
criterion = torch.nn.MSELoss()

for i in range(EPOCH):
    avg_cost = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)

        cost = criterion(output, target)
        cost.backward()
        optimizer.step()
        avg_cost += cost / len(train_loader)
        
    if i % 10 == 0:
        print("epoch ", avg_cost)

imf1_s = []
imf2_s = []
imf3_s = []

imf_all = []

model.eval()
with torch.no_grad(): 
    correct = 0
    total = 0

    for i, (datax, targetx) in enumerate(test_loader):
        out = model(datax)

        imf_all.append(out.cpu().numpy())
        a = np.swapaxes(targetx.numpy(), 1, 2)[0]
        b = np.swapaxes(out.numpy(), 1, 2)[0]

        imf1 = np.corrcoef(a[:, 0], b[:, 0])[0, 1]
        imf2 = np.corrcoef(a[:, 1], b[:, 1])[0, 1]
        imf3 = np.corrcoef(a[:, 2], b[:, 2])[0, 1]

        imf1_s.append(imf1)
        imf2_s.append(imf2)
        imf3_s.append(imf3)

print(np.mean(imf1_s))
print(np.mean(imf2_s))
print(np.mean(imf3_s))
print(np.std(imf1_s))
print(np.std(imf2_s))
print(np.std(imf3_s))

plt.figure()
plt.subplot(2,1,1)
plt.plot(a)
plt.title('Original signal')
plt.xlabel('time (s)')
plt.subplot(2,1,2)
plt.plot(b)
plt.title('Decomposed modes')
plt.xlabel('time (s)')
plt.tight_layout()
plt.show()