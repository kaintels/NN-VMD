import numpy as np

x_train = np.load("./deep_mode_b.npy").reshape(-1, 140, 3)
y_train = np.load("./deep_mode_b_label.npy")

x_test = np.load("./test_deep_mode_b.npy").reshape(-1, 140, 3)
y_test = np.load("./test_deep_mode_b_label.npy")

import matplotlib.pyplot as plt

x_train = np.swapaxes(x_train, 1, 2)
x_test = np.swapaxes(x_test, 1, 2)

print(x_train.shape)
# plt.plot(x_train[0, 0, :])
# # plt.plot(target_s_tg[0, :, 0], '--')
# plt.show()

target_s_tr = x_train.astype(np.float64)
print(target_s_tr.shape)
target_s_ts = x_test.astype(np.float64)
from mne.decoding import CSP
from sklearn.metrics import classification_report, confusion_matrix

csp = CSP(n_components=3)

csp_x_train = csp.fit_transform(target_s_tr, y_train)
csp_x_test = csp.transform(target_s_ts)

from sklearn.svm import SVC

svm = SVC()

svm.fit(csp_x_train, y_train)

pred = svm.predict(csp_x_test)

print(confusion_matrix(pred, y_test))
print(classification_report(pred, y_test, digits=4))