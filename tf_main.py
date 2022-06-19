import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD
import pandas as pd
import numpy as np
from scipy.io import arff
from utils.util import seed_everything_tf, vmd_execute, weight_init_xavier_uniform
from models.model import VMDNet_TF

EPOCH = 10
seed = 123456
seed_everything_tf(seed)

tr = arff.loadarff("./dataset/ECG5000_TRAIN.arff")
tr = pd.DataFrame(tr[0], dtype=np.float32)

ts = arff.loadarff("./dataset/ECG5000_TEST.arff")
ts = pd.DataFrame(ts[0], dtype=np.float32)

input_tr, target_tr = vmd_execute(tr, is_torch=False)
input_ts, target_ts = vmd_execute(ts, is_torch=False)

model = VMDNet_TF()

train_ds = tf.data.Dataset.from_tensor_slices((input_tr, target_tr)).shuffle(500).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((input_ts, target_ts)).shuffle(4500).batch(32)

optimizer = optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
criterion = losses.MeanSquaredError()
train_loss = metrics.Mean()

@tf.function
def train(data, target):
    with tf.GradientTape() as tape:
        output = model(data)
        loss = criterion(output, target)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    tr_loss = train_loss(loss)

    return tr_loss

for i in range(EPOCH):
    for data, target in train_ds:
        tr_loss = train(data, target)

    if i % 10 == 0:
        print(tr_loss.numpy().item())


imf1_s = []
imf2_s = []
imf3_s = []

imf_all = []
for i, (data, target) in enumerate(test_ds):
    out = model(data)
    imf_all.append(out.numpy())
    a = target.numpy()[0]
    b = out.numpy().squeeze()[0]
    imf1 = np.corrcoef(a[:, 0], b[:, 0])[0, 1]
    imf2 = np.corrcoef(a[:, 1], b[:, 1])[0, 1]
    imf3 = np.corrcoef(a[:, 2], b[:, 2])[0, 1]

    imf1_s.append(imf1)
    imf2_s.append(imf2)
    imf3_s.append(imf3)

print("--")
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