import numpy as np
import pandas as pd
import time
# import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import get_custom_objects
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
import tsa
from data import load_data_3



def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# get_custom_objects().clear()
get_custom_objects()["rmse"] = rmse

x1_h1, x1_h2, x1_l1, x1_l2, y1_h1, y1_h2, y1_l1, y1_l2 = load_data_3(leak_size='*', inx='all', m=19, d=1)
# print(x1_h1.shape, y1_h1.shape, x1_h2.shape, y1_h2.shape, x1_l1.shape, y1_l1.shape, x1_l2.shape, y1_l2.shape)

t2 = time.time()

m2 = load_model('cnn_all_leaks_1.h5')
t3 = time.time()
print(1000*(t3-t2), 'ms')
y1h_h1 = m2.predict(x1_h1)
y1h_h2 = m2.predict(x1_h2)
y1h_l1 = m2.predict(x1_l1)
y1h_l2 = m2.predict(x1_l2)
t4 = time.time()
print(1000*(t4-t3), 'ms')
print('yh:', y1h_h1.shape)

x2_h1 = np.concatenate((y1h_h1, y1_h1), axis=1, out=None)
x2_h2 = np.concatenate((y1h_h2, y1_h2), axis=1, out=None)
x2_l1 = np.concatenate((y1h_l1, y1_l1), axis=1, out=None)
x2_l2 = np.concatenate((y1h_l2, y1_l2), axis=1, out=None)

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

acc_1svm = []
acc_h_1svm = []
acc_l_1svm = []
TPR_1svm = []
FPR_1svm = []
for run in range(100):
    print(run)
    clf = OneClassSVM(nu=0.001, kernel='rbf', gamma=5.3)
    # clf = IsolationForest(n_estimators=300, max_samples=100000, contamination=0.001, max_features=1.0, n_jobs=10, behaviour='new')

    y_pred_train = clf.fit_predict(x2_h1)

    y_pred = clf.predict(np.concatenate((x2_h2, x2_l1, x2_l2), axis=0))
    y2_tst = np.concatenate((np.ones(x2_h2.shape[0]), -1*np.ones(x2_l1.shape[0]), -1*np.ones(x2_l2.shape[0])))
    confm = confusion_matrix(-1*y2_tst, -1*y_pred)
    print('Confusion Matrix : \n', confm)
    tn, fp, fn, tp = confm.ravel()
    total1 = np.sum(np.sum(confm))
    acc_1svm = np.append(acc_1svm, (confm[0, 0] + confm[1, 1]) / total1)
    acc_h_1svm = np.append(acc_h_1svm, confm[0, 0] / (confm[0, 0] + confm[0, 1]))
    acc_l_1svm = np.append(acc_l_1svm, confm[1, 1] / (confm[1, 1] + confm[1, 0]))
    TPR_1svm = np.append(TPR_1svm, tp / (tp + fn))
    FPR_1svm = np.append(FPR_1svm, fp / (fp + tn))

acc_ifo = []
acc_h_ifo = []
acc_l_ifo = []
TPR_ifo = []
FPR_ifo = []
for run in range(100):
    print(run)
    # clf = OneClassSVM(nu=0.001, kernel='rbf', gamma=5.3)
    clf = IsolationForest(n_estimators=300, max_samples=100000, contamination=0.001, max_features=1.0, n_jobs=10) #  , behaviour='new')

    y_pred_train = clf.fit_predict(x2_h1)

    y_pred = clf.predict(np.concatenate((x2_h2, x2_l1, x2_l2), axis=0))
    y2_tst = np.concatenate((np.ones(x2_h2.shape[0]), -1*np.ones(x2_l1.shape[0]), -1*np.ones(x2_l2.shape[0])))
    confm = confusion_matrix(-1*y2_tst, -1*y_pred)
    print('Confusion Matrix : \n', confm)
    tn, fp, fn, tp = confm.ravel()
    total1 = np.sum(np.sum(confm))
    acc_ifo = np.append(acc_ifo, (confm[0, 0] + confm[1, 1]) / total1)
    acc_h_ifo = np.append(acc_h_ifo, confm[0, 0] / (confm[0, 0] + confm[0, 1]))
    acc_l_ifo = np.append(acc_l_ifo, confm[1, 1] / (confm[1, 1] + confm[1, 0]))
    TPR_ifo = np.append(TPR_ifo, tp / (tp + fn))
    FPR_ifo = np.append(FPR_ifo, fp / (fp + tn))

# print('Accuracy : ', np.mean(acc_1svm))
# print('Acc healthy : ', np.mean(acc_h_1svm))
# print('Acc leak : ', np.mean(acc_l_1svm))
# print('TPR_1svm : ', np.mean(TPR_1svm))
# print('FPR_1svm : ', np.mean(FPR_1svm))

from scipy.stats import wilcoxon, ttest_ind, ttest_rel

print(wilcoxon(acc_1svm, acc_ifo))

print(wilcoxon(acc_1svm, acc_ifo, alternative='less'))

print(ttest_ind(acc_1svm, acc_ifo))

print(ttest_rel(acc_1svm, acc_ifo))
