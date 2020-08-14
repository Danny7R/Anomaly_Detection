import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsa


def load_data(inx=slice(4)):
    all_files = ['csv/08_0.49R_FD_5_LK_EPQ10.csv', 'csv/09_0.49R_FD_5_LK_PT10.csv', 'csv/24_2.20R_FI_5_LK_APQ10.csv', 'csv/25_2.20R_FI_5_LK_PT10.csv',
                 'csv/13_2.20R_FD_30_LK.csv', 'csv/26_2.20R_FI_30_LK.csv', 'csv/39_2.20R_ST_30_LK.csv']
    try:
        files = all_files[inx]
        print('datasets:', inx, files)
    except IndexError:
        files = all_files
        print('all datasets')

    m = 19
    d = 1
    x1_h1_t = np.array([]).reshape((0, m, 4))
    x1_l1_t = np.array([]).reshape((0, m, 4))
    x1_l2_t = np.array([]).reshape((0, m, 4))
    y1_h1_t = np.array([]).reshape((0, 4))
    y1_l1_t = np.array([]).reshape((0, 4))
    y1_l2_t = np.array([]).reshape((0, 4))
    for filename in files:
        df = pd.read_csv(filename, sep=',', header=None)  # delim_whitespace=False)
        ds = df.iloc[:, 2]
        ds = ds.values
        ds = ds.reshape((-1, 4))
        ds = ds[3:]
        leak_start = np.where(ds[:, 0] == '04:00:00')
        leak_start = int(np.squeeze(leak_start))
        # print(ds[leak_start])
        leak_split = np.where(ds[:, 0] == '06:00:00')
        leak_split = int(np.squeeze(leak_split))
        # print(ds[leak_split])

        ds = df.iloc[:, 4]
        ds = ds.values
        ds = ds.astype('float32')
        ds = ds.reshape((-1, 4))
        ds = ds[3:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        ds = scaler.fit_transform(ds)
        healthy1 = ds[:leak_start]
        leak1 = ds[leak_start:leak_split]
        leak2 = ds[leak_split:]
        # print(healthy1.shape, leak1.shape, leak2.shape)

        x1_h1, y1_h1 = tsa.multi_delay_embed(healthy1, m, d)
        x1_l1, y1_l1 = tsa.multi_delay_embed(leak1, m, d)
        x1_l2, y1_l2 = tsa.multi_delay_embed(leak2, m, d)

        x1_h1_t = np.concatenate((x1_h1_t, x1_h1), axis=0, out=None)
        x1_l1_t = np.concatenate((x1_l1_t, x1_l1), axis=0, out=None)
        x1_l2_t = np.concatenate((x1_l2_t, x1_l2), axis=0, out=None)
        y1_h1_t = np.concatenate((y1_h1_t, y1_h1), axis=0, out=None)
        y1_l1_t = np.concatenate((y1_l1_t, y1_l1), axis=0, out=None)
        y1_l2_t = np.concatenate((y1_l2_t, y1_l2), axis=0, out=None)
        # del(x1_h1, x1_l1, x1_l2, y1_h1, y1_l1, y1_l2)

        # print(x1_h1_t.shape, y1_h1_t.shape, x1_l1_t.shape, y1_l1_t.shape, x1_l2_t.shape, y1_l2_t.shape)

        # y2_tra = np.concatenate((np.zeros(x2_tra1.shape[0]), np.ones(x2_tra2.shape[0])), axis=0, out=None)
        # y2_tst = np.ones(x2_tst.shape[0])
    return x1_h1_t, x1_l1_t, x1_l2_t, y1_h1_t, y1_l1_t, y1_l2_t


def load_data_2(inx=slice(7), m=19, d=1):
    all_files = ['csv/08_0.49R_FD_5_LK_EPQ10.csv', 'csv/09_0.49R_FD_5_LK_PT10.csv', 'csv/24_2.20R_FI_5_LK_APQ10.csv', 'csv/25_2.20R_FI_5_LK_PT10.csv',
                 'csv/13_2.20R_FD_30_LK.csv', 'csv/26_2.20R_FI_30_LK.csv', 'csv/39_2.20R_ST_30_LK.csv']
    try:
        files = all_files[inx]
        print('datasets:', inx, files)
    except IndexError:
        files = all_files
        print('all datasets')

    x1_h1_t = np.array([]).reshape((0, m, 4))
    x1_h2_t = np.array([]).reshape((0, m, 4))
    x1_l1_t = np.array([]).reshape((0, m, 4))
    x1_l2_t = np.array([]).reshape((0, m, 4))
    y1_h1_t = np.array([]).reshape((0, 4))
    y1_h2_t = np.array([]).reshape((0, 4))
    y1_l1_t = np.array([]).reshape((0, 4))
    y1_l2_t = np.array([]).reshape((0, 4))
    for filename in files:
        df = pd.read_csv(filename, sep=',', header=None)  # delim_whitespace=False)
        print(df.describe)
        ds = df.iloc[:, 2]
        ds = ds.values
        ds = ds.reshape((-1, 4))
        ds = ds[3:]
        healthy_split = np.where(ds[:, 0] == '03:00:03')
        healthy_split = int(np.squeeze(healthy_split))

        leak_start = np.where(ds[:, 0] == '04:00:00')
        leak_start = int(np.squeeze(leak_start))
        # print(ds[leak_start])

        leak_split = np.where(ds[:, 0] == '06:00:00')
        leak_split = int(np.squeeze(leak_split))
        # print(ds[leak_split])

        ds = df.iloc[:, 4]
        ds = ds.values
        ds = ds.astype('float32')
        ds = ds.reshape((-1, 4))
        ds = ds[3:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        ds = scaler.fit_transform(ds)
        healthy1 = ds[:healthy_split]
        healthy2 = ds[healthy_split:leak_start]
        leak1 = ds[leak_start:leak_split]
        leak2 = ds[leak_split:]
        # print(healthy1.shape, leak1.shape, leak2.shape)

        x1_h1, y1_h1 = tsa.multi_delay_embed(healthy1, m, d)
        x1_h2, y1_h2 = tsa.multi_delay_embed(healthy2, m, d)
        x1_l1, y1_l1 = tsa.multi_delay_embed(leak1, m, d)
        x1_l2, y1_l2 = tsa.multi_delay_embed(leak2, m, d)

        x1_h1_t = np.concatenate((x1_h1_t, x1_h1), axis=0, out=None)
        x1_h2_t = np.concatenate((x1_h2_t, x1_h2), axis=0, out=None)
        x1_l1_t = np.concatenate((x1_l1_t, x1_l1), axis=0, out=None)
        x1_l2_t = np.concatenate((x1_l2_t, x1_l2), axis=0, out=None)

        y1_h1_t = np.concatenate((y1_h1_t, y1_h1), axis=0, out=None)
        y1_h2_t = np.concatenate((y1_h2_t, y1_h2), axis=0, out=None)
        y1_l1_t = np.concatenate((y1_l1_t, y1_l1), axis=0, out=None)
        y1_l2_t = np.concatenate((y1_l2_t, y1_l2), axis=0, out=None)
        # del(x1_h1, x1_l1, x1_l2, y1_h1, y1_l1, y1_l2)

        # print(x1_h1_t.shape, y1_h1_t.shape, x1_l1_t.shape, y1_l1_t.shape, x1_l2_t.shape, y1_l2_t.shape)

        # y2_tra = np.concatenate((np.zeros(x2_tra1.shape[0]), np.ones(x2_tra2.shape[0])), axis=0, out=None)
        # y2_tst = np.ones(x2_tst.shape[0])
    return x1_h1_t, x1_h2_t, x1_l1_t, x1_l2_t, y1_h1_t, y1_h2_t, y1_l1_t, y1_l2_t


def load_data_3(leak_size=2, inx='all', m=19, d=1):
    if leak_size not in [2, 5, 30, '*']:
        raise Exception('leak_size should be 2, 5, 30, or "*" for all leaks. Passed value was:{}'.format(leak_size))

    files = sorted(glob.glob(os.path.join('data/comp_gen/', ('*_' + str(leak_size) + '_LK*.txt'))))

    if inx != 'all':
        files = files[inx]

    x1_h1_t = np.array([]).reshape((0, m, 4))
    x1_h2_t = np.array([]).reshape((0, m, 4))
    x1_l1_t = np.array([]).reshape((0, m, 4))
    x1_l2_t = np.array([]).reshape((0, m, 4))
    y1_h1_t = np.array([]).reshape((0, 4))
    y1_h2_t = np.array([]).reshape((0, 4))
    y1_l1_t = np.array([]).reshape((0, 4))
    y1_l2_t = np.array([]).reshape((0, 4))
    for filename in files:
        print(filename)
        df = pd.read_csv(filename, header=None, delim_whitespace=True)  # , sep=','
        # print(df.describe)
        ds = df.iloc[:, 2]
        ds = ds.values
        ds = ds.reshape((-1, 4))
        ds = ds[3:].astype(str)
        healthy_split = np.flatnonzero(np.core.defchararray.find(ds[:, 0], '03:00:0') != -1)
        healthy_split = healthy_split[0]
        # print(ds[healthy_split])

        leak_start = np.flatnonzero(np.core.defchararray.find(ds[:, 0], '04:00:0') != -1)
        leak_start = leak_start[0]
        # print(ds[leak_start])

        leak_split = np.flatnonzero(np.core.defchararray.find(ds[:, 0], '06:00:0') != -1)
        leak_split = leak_split[0]
        # print(ds[leak_split])

        ds = df.iloc[:, 4]
        ds = ds.values
        ds = ds.astype('float32')
        ds = ds.reshape((-1, 4))
        ds = ds[3:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        ds = scaler.fit_transform(ds)
        healthy1 = ds[:healthy_split]
        healthy2 = ds[healthy_split:leak_start]
        leak1 = ds[leak_start:leak_split]
        leak2 = ds[leak_split:]
        # print(healthy1.shape, leak1.shape, leak2.shape)

        x1_h1, y1_h1 = tsa.multi_delay_embed(healthy1, m, d)
        x1_h2, y1_h2 = tsa.multi_delay_embed(healthy2, m, d)
        x1_l1, y1_l1 = tsa.multi_delay_embed(leak1, m, d)
        x1_l2, y1_l2 = tsa.multi_delay_embed(leak2, m, d)

        x1_h1_t = np.concatenate((x1_h1_t, x1_h1), axis=0, out=None)
        x1_h2_t = np.concatenate((x1_h2_t, x1_h2), axis=0, out=None)
        x1_l1_t = np.concatenate((x1_l1_t, x1_l1), axis=0, out=None)
        x1_l2_t = np.concatenate((x1_l2_t, x1_l2), axis=0, out=None)

        y1_h1_t = np.concatenate((y1_h1_t, y1_h1), axis=0, out=None)
        y1_h2_t = np.concatenate((y1_h2_t, y1_h2), axis=0, out=None)
        y1_l1_t = np.concatenate((y1_l1_t, y1_l1), axis=0, out=None)
        y1_l2_t = np.concatenate((y1_l2_t, y1_l2), axis=0, out=None)
        # del(x1_h1, x1_l1, x1_l2, y1_h1, y1_l1, y1_l2)

        # print(x1_h1_t.shape, y1_h1_t.shape, x1_l1_t.shape, y1_l1_t.shape, x1_l2_t.shape, y1_l2_t.shape)

        # y2_tra = np.concatenate((np.zeros(x2_tra1.shape[0]), np.ones(x2_tra2.shape[0])), axis=0, out=None)
        # y2_tst = np.ones(x2_tst.shape[0])
    return x1_h1_t, x1_h2_t, x1_l1_t, x1_l2_t, y1_h1_t, y1_h2_t, y1_l1_t, y1_l2_t


if __name__ == '__main__':
    # x1_h1_t, x1_h2_t, x1_l1_t, x1_l2_t, y1_h1_t, y1_h2_t, y1_l1_t, y1_l2_t = load_data_2()
    # print(x1_h1_t.shape, y1_h1_t.shape, x1_h2_t.shape, y1_h2_t.shape, x1_l1_t.shape, y1_l1_t.shape, x1_l2_t.shape, y1_l2_t.shape)

    x1_h1_t, x1_h2_t, x1_l1_t, x1_l2_t, y1_h1_t, y1_h2_t, y1_l1_t, y1_l2_t = load_data_3(leak_size='*', inx='all', m=19, d=1)
    # load_data_2(inx=slice(1), m=19, d=1)
