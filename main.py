import torch
import torch.nn.functional as F
import random
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
import argparse
from scipy.io import arff, loadmat
from sklearn.model_selection import train_test_split
from utils.util import seed_everything_th, weight_init_xavier_uniform, TrainDataset, TestDataset
from models.model import VMDNet, Classifier, TaskLayer, VMD_VAE_DNN

parser = argparse.ArgumentParser(description="Arguments classifiers.")
parser.add_argument("--model_name", type=str, required=True,
                        help="model_name.",
                        dest="model_name")
parser.add_argument("--epoch", type=str, required=False,
                        default="200",
                        help="epoch numer.",
                        dest="epoch")
parser.add_argument("--seed", type=str, required=True,
                        default="123456",
                        help="seed numer.",
                        dest="seed")

args = parser.parse_args()

EPOCH = int(args.epoch)
seed = int(args.seed)
seed_everything_th(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(f"current model name: {args.model_name.upper()}")
print(f"current device : {device}")
print(f"EPOCH : {EPOCH}")
print(f"seed : {seed}")

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


def data_loader(args):
    assert len(args) > 0
    if args == "cnn":
        train_loader = torch.utils.data.DataLoader(dataset = TrainDataset(input_s_tr, target_s_tr), batch_size = 32, shuffle = True)
        test_loader = torch.utils.data.DataLoader(dataset = TestDataset(input_s_ts, target_s_ts), batch_size = 32, shuffle = True)
        model = VMDNet().to(device)

    if args == "vae":
        train_loader = torch.utils.data.DataLoader(dataset = TrainDataset(input_s_tr), batch_size = 32, shuffle = True)
        test_loader = torch.utils.data.DataLoader(dataset = TestDataset(input_s_ts), batch_size = 32, shuffle = True)
        model = VMD_VAE_DNN().to(device)

    if args == "mtl":
        train_loader = torch.utils.data.DataLoader(dataset = TrainDataset(input_s_tr, target_s_tr, input_l_tr), batch_size = 32, shuffle = True)
        test_loader = torch.utils.data.DataLoader(dataset = TestDataset(input_s_ts, target_s_ts, input_l_ts), batch_size = 32, shuffle = True)
        model = TaskLayer().to(device)
    
    return train_loader, test_loader, model


train_loader, test_loader, model = data_loader(args.model_name)

loss_1 = torch.nn.MSELoss()
loss_2 = torch.nn.CrossEntropyLoss()
def custom_fn(outputs, inputs, mu, log_var):
    BCE = F.mse_loss(outputs, inputs, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    print("-" * 100)
    alpha = 0.9
    for i in range(EPOCH):
        avg_cost = 0
        if args == "cnn":
            for _, data in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data[0].to(device))

                cost = loss_1(output, data[1])
                cost.backward()
                optimizer.step()
                avg_cost += cost / len(train_loader)

            if i % 10 == 0:
                print("epoch : {0}, loss : {1}".format(i, avg_cost.item()))

        if args == "vae":
            for _, data in enumerate(train_loader):
                optimizer.zero_grad()
                out, mu, log_var = model(data[0])

                cost = custom_fn(out, data[0], mu, log_var)
                cost.backward()
                optimizer.step()
                avg_cost += cost / len(train_loader)
            
            if i % 10 == 0:
                print("epoch : {0}, loss : {1}".format(i, avg_cost.item()))


        if args == "mtl":
            for _, data in enumerate(train_loader):
                optimizer.zero_grad()

                output1, output2 = model(data[0])
                loss1 = loss_1(output1, data[1]) * (1-alpha)
                loss2 = loss_2(output2, data[2]) * alpha

                total_loss = loss1 + loss2

                total_loss.backward()

                optimizer.step()
                avg_cost += total_loss / len(train_loader)

            if i % 10 == 0:
                print("epoch : {0}, loss : {1}".format(i, avg_cost.item()))

    print("-" * 100)

train(model, args.model_name)