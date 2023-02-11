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
from models.model import VMD_VAE_DNN
import subprocess

EPOCH = 300
seed = 123456
seed_everything_th(seed)

subprocess.run("julia ./utils/preprocessing.jl", shell=True)

input_tr = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
input_tr = pd.DataFrame(input_tr[0], dtype=np.float32).values[:,:140].reshape(500, 140)

input_ts = arff.loadarff("./dataset/ECG5000_TEST.arff")
input_ts = pd.DataFrame(input_ts[0], dtype=np.float32).values[:,:140].reshape(4500, 140)

target_tr = loadmat("./dataset/processed_train.mat")["data"].reshape(500, 3, 140)
target_ts = loadmat("./dataset/processed_test.mat")["data"].reshape(4500, 3, 140)

train_loader = torch.utils.data.DataLoader(dataset = TensorDataset(torch.FloatTensor(input_tr), torch.FloatTensor(target_tr)), batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = TensorDataset(torch.FloatTensor(input_ts), torch.FloatTensor(target_ts)), batch_size = 32, shuffle = True)

model = VMD_VAE_DNN()
model.apply(weight_init_xavier_uniform)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, eps=1e-8, betas=(0.9, 0.999))

def custom_fn(outputs, inputs, mu, log_var):
    BCE = F.mse_loss(outputs, inputs, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD


for i in range(EPOCH):
    avg_cost = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        out, mu, log_var = model(data)

        cost = custom_fn(out, data, mu, log_var)
        cost.backward()
        optimizer.step()
        avg_cost += cost / len(train_loader)
        
    if i % 10 == 0:
        print("epoch : {0}, loss : {1}".format(i, avg_cost.item()))

imf1_s = []
imf2_s = []
imf3_s = []

imf_all = []

model.eval()
with torch.no_grad(): 
    correct = 0
    total = 0

    for i, (datax, targetx) in enumerate(test_loader):
        out, mu, log_var = model(datax)
        break

plt.figure()
plt.subplot(2,1,1)
plt.plot(datax[0])
plt.title('Original signal')
plt.xlabel('time (s)')
plt.subplot(2,1,2)
plt.plot(out[0])
plt.title('Reconstructed signal')
plt.xlabel('time (s)')
plt.tight_layout()
plt.show()