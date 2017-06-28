#coding=utf8
from __future__ import division

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('../assets/sh300.csv', index_col = 0, parse_dates = True)
df['dummy'] = np.sign(df['pct_chg'])
df['label'] = df['dummy'].shift(-1)
df.dropna(inplace = True)

x_train, x_test = df.loc[:, 'close':'dummy'][:-100], df.loc[:, 'close':'dummy'][-100:]
y_train, y_test = df.loc[:, 'label'][:-100], df.loc[:, 'label'][-100:]

model = Sequential()
input_dim = len(x_train.columns)

pre_states = []
test_num = 100
window = 300
for i in range(test_num):
    model.add(Dense(10, activation='relu', input_dim=input_dim))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    x_train, x_test = df.loc[:, 'close':'dummy'][-test_num-window+i:-test_num+i], df.loc[:, 'close':'dummy'][-test_num+i:]
    y_train = df.loc[:, 'label'][-test_num-window+i: -test_num+i]
    model.fit(x_train.values, y_train, epochs=1, batch_size = 32)
    pre_state = model.predict(x_test.values).reshape(1, -1)[0][0]
    pre_states.append(pre_state)

print (np.array(pre_states) == y_test).sum()/100
