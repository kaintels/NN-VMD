import torch
import argparse
import pandas as pd
import numpy as np
from utils.util import seed_everything_th, weight_init_xavier_uniform, TrainDataset, TestDataset
from models.model import VMDNet, Classifier, TaskLayer, VMD_VAE_DNN
from scipy.io import arff, loadmat

parser = argparse.ArgumentParser(description="Arguments classifiers.")
parser.add_argument("--model_name", type=str, required=True,
                    help="model_name.",
                    dest="model_name")
parser.add_argument("--seed", type=str, required=True,
                    default="123456",
                    help="seed numer.",
                    dest="seed")

args = parser.parse_args()

seed = int(args.seed)
seed_everything_th(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"current model name: {args.model_name.upper()}")
print(f"current device : {device}")
print(f"seed : {seed}")

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
    input_l_ts[0], dtype=np.float32).values[:, 140:141].reshape(4500)

input_l_ts = input_l_ts - 1

target_s_tr = loadmat(
    "./dataset/processed_train.mat")["data"].reshape(500, 3, 140)
target_s_ts = loadmat(
    "./dataset/processed_test.mat")["data"].reshape(4500, 3, 140)


def data_loader(args):
    assert len(args) > 0
    if args == "cnn":
        train_loader = torch.utils.data.DataLoader(dataset=TrainDataset(
            input_s_tr, target_s_tr), batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=TestDataset(
            input_s_ts, target_s_ts), batch_size=32, shuffle=True)
        model = VMDNet().to(device)

    if args == "vae":
        train_loader = torch.utils.data.DataLoader(
            dataset=TrainDataset(input_s_tr), batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=TestDataset(input_s_ts), batch_size=32, shuffle=True)
        model = VMD_VAE_DNN().to(device)

    if args == "mtl":
        train_loader = torch.utils.data.DataLoader(dataset=TrainDataset(
            input_s_tr, target_s_tr, input_l_tr), batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=TestDataset(
            input_s_ts, target_s_ts, input_l_ts), batch_size=32, shuffle=True)
        model = TaskLayer().to(device)

    return train_loader, test_loader, model

train_loader, test_loader, model = data_loader(args.model_name)
imf1_s = []
imf2_s = []
imf3_s = []
if args.model_name == "cnn":
    model.load_state_dict(torch.load("./models/cnn_eph100.pt"))
if args.model_name == "mtl":
    model.load_state_dict(torch.load("./models/mtl_eph10.pt"))


with torch.no_grad(): 
    correct = 0
    total = 0

    if args.model_name == "cnn":
        for _, data in enumerate(test_loader):
            output = model(data[0])

            for i in range(data[0].shape[0]):
                a = data[1][i].to(device).numpy().reshape(-1, 140, 3)
                b = output[i].to(device).numpy().reshape(-1, 140, 3)
                imf1 = np.corrcoef(a[0, :, 0], b[0, :, 0])[0, 1]
                imf2 = np.corrcoef(a[0, :, 1], b[0, :, 1])[0, 1]
                imf3 = np.corrcoef(a[0, :, 2], b[0, :, 2])[0, 1]

                imf1_s.append(imf1)
                imf2_s.append(imf2)
                imf3_s.append(imf3)

    if args.model_name == "mtl":
        for _, data in enumerate(test_loader):
            output1, output2 = model(data[0])

            for i in range(data[1].shape[0]):
                a = data[1][i].to(device).numpy().reshape(-1, 140, 3)
                b = output1[i].to(device).numpy().reshape(-1, 140, 3)
                imf1 = np.corrcoef(a[0, :, 0], b[0, :, 0])[0, 1]
                imf2 = np.corrcoef(a[0, :, 1], b[0, :, 1])[0, 1]
                imf3 = np.corrcoef(a[0, :, 2], b[0, :, 2])[0, 1]

                imf1_s.append(imf1)
                imf2_s.append(imf2)
                imf3_s.append(imf3)

print(len(imf1_s))
print(np.mean(imf1_s))
print(np.mean(imf2_s))
print(np.mean(imf3_s))
print(np.std(imf1_s))
print(np.std(imf2_s))
print(np.std(imf3_s))