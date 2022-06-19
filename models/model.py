import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from tensorflow.keras import layers, Model, initializers
import tensorflow as tf

class VMDNet_TH(nn.Module):
    def __init__(self):
        super(VMDNet_TH, self).__init__()

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

class VMDNet_TF(Model):
    def __init__(self):
        super(VMDNet_TF, self).__init__()
        kernel_init = initializers.GlorotNormal()
        bias_init = tf.constant_initializer(0)
        self.conv1 = layers.Conv1D(32, 3, padding="same",kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.conv2 = layers.Conv1D(64, 3, padding="same",kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(420,kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.reshape = layers.Reshape((-1, 140, 3))
    
    def call(self, x):
        x = self.conv1(x)
        x = tf.nn.sigmoid(x)
        x = self.conv2(x)
        x = tf.nn.sigmoid(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.reshape(x)
        return x