from vmdpy import VMD  
import numpy as np
import torch
import tensorflow as tf
import random
import os

def vmd_execute(data, is_torch=False):
    signal = data.values[:,:140]

    alpha = 2000       # moderate bandwidth constraint  
    tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
    K = 3              # 3 modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7  

    output = []
    for i in range(len(data)):
        u, u_hat, omega = VMD(signal[i].reshape(140, -1), alpha, tau, K, DC, init, tol)
        output.append(u.T)
    signal = np.expand_dims(signal, 2)
    output = np.array(output)
    if is_torch:
        signal = np.swapaxes(signal, 2, 1)
        output = np.swapaxes(output, 2, 1)

    return signal, output

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

def seed_everything_tf(num):
    os.environ['PYTHONHASHSEED'] = str(num)
    random.seed(num)
    tf.random.set_seed(num)
    np.random.seed(num)

