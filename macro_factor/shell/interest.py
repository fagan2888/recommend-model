#coding=utf8


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D

import pandas as pd
import numpy as np


if __name__ == '__main__':


    df = pd.read_csv('./data/interest.csv', index_col = ['date'], parse_dates = ['date'])
    dates = pd.date_range(df.index[0], df.index[-1])
    df = df.reindex(dates)
    df = df.fillna(method = 'pad')
    dates = df['shibor_3m'].dropna().index
    df = df.loc[dates]
    df = df.fillna(0.0)
    df = df.resample('M').last()
    #print df
    #df['national_debt_10y_ma6'] = df['national_debt_10y'].rolling(window = 6, min_periods = 1).mean()
    #df.to_csv('interest.csv')

    sequence_length = 1000
    x = []
    y = []
    for i in range(sequence_length, len(dates) - 30):
        x.append(df.iloc[i - sequence_length : i,].values)
        y.append(df['shibor_3m'].iloc[i + 30])

    x = np.array(x)
    y = np.array(y)

    x_train = x[0 : 2500]
    y_train = y[0 : 2500]
    x_test  = x[2500 : ]
    y_test  = y[2500 : ]

    model = Sequential()

    model.add(LSTM(input_shape = (sequence_length, 12), units = 50))
    model.add(Dropout(0.2))

    #model.add(LSTM(input_shape = (50, 50), units = 50))
    #model.add(Dropout(0.2))

    model.add(Dense( units = 1))
    model.add(Activation('linear'))

    model.compile(loss = 'mse', optimizer = 'adam')

    model.fit(x_train, y_train, batch_size = 128, epochs = 5000, validation_data = (x_test, y_test))

    #predicted = model.predict(x_test)
    #print predicted
    #print y_test
