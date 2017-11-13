# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import numpy as np
import pandas as pd


def x_sin(x):
    return x * np.sin(x)


def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def generate_ir_data(filepath):
    data = pd.read_csv(filepath, index_col = 0, parse_dates = True)
    data['y'] = data['bond'].shift(-3)
    #data['y'] = np.sign(data['y'].diff())
    #data['y'] = data['pl'].shift(-3)
    data = data.fillna(method = 'pad')
    data = data.fillna(0.0)
    data_v = data.values
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(90):
        train_x.append(data_v[i:i+12, :-1])
        train_y.append(data_v[i+11, -1])

    for i in range(90, 131):
        test_x.append(data_v[i:i+12, :-1])
        test_y.append(data_v[i+11, -1])

    train_x = np.array(train_x, dtype = np.float32)
    test_x = np.array(test_x, dtype = np.float32)

    train_y = np.asarray(np.expand_dims(train_y, 2), dtype = np.float32)
    test_y = np.asarray(np.expand_dims(test_y, 2), dtype = np.float32)

    return dict(train=train_x, test=test_x), dict(train=train_y, test=test_y)

def generate_ir_data_2(filepath, time_steps, start = 30):
    data = pd.read_csv(filepath, index_col = 0, parse_dates = True, encoding = 'utf8')
    print(data.head())
    data  = data.loc[:, [u'中债国债到期收益率:10年', u'公开市场操作:货币净投放']]
    total_num = len(data)
    data['y'] = data[u'中债国债到期收益率:10年'].shift(-30)
    data = data.replace(0.0, np.nan)
    data = data.fillna(method = 'pad')
    data = data.fillna(0.0)
    data_v = data.values
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(total_num//2 + start):
        train_x.append(data_v[i:i+time_steps, :-1])
        train_y.append(data_v[i+time_steps-1, -1])

    for i in range(total_num//2 + start, total_num-time_steps+1):
        test_x.append(data_v[i:i+time_steps, :-1])
        test_y.append(data_v[i+time_steps-1, -1])

    train_x = np.array(train_x, dtype = np.float32)
    test_x = np.array(test_x, dtype = np.float32)

    train_y = np.asarray(np.expand_dims(train_y, 2), dtype = np.float32)
    test_y = np.asarray(np.expand_dims(test_y, 2), dtype = np.float32)

    train_index = data.index[time_steps-1: total_num//2 + time_steps-1 + start]
    test_index = data.index[total_num//2 + time_steps-1 + start: total_num]
    return dict(train=train_x, test=test_x), dict(train=train_y, test=test_y), dict(train=train_index, test=test_index)

if __name__ == '__main__':
    X, y = generate_ir_data()
    #X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), 3, seperate=False)
    print y['train'].shape
