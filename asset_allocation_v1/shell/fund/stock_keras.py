#coding=utf8



import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, LSTM, RepeatVector, Embedding
from keras import optimizers
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr
from numpy import *
from keras.preprocessing import sequence
from keras.layers import  Embedding



if __name__ == '__main__':


    df = pd.read_csv('./data/stock_price_adjust.csv', parse_dates = ['date'], index_col = ['date'])


    df = df.iloc[-2000:]

    df = df.fillna(method = 'pad').dropna(axis = 1)

    dfr = df.pct_change().fillna(0.0)

    mean_dfr = dfr.rolling(window = 252, min_periods = 1).mean()

    x = []
    y = []
    for d in mean_dfr.index:
        x.append(list(dfr.loc[d].values))
        y.append(list(mean_dfr.loc[d].values))


    #model = Sequential()
    #model.add(Dense(len(dfr.columns) / 2, input_shape(len(dfr.columns)))

    (x_train, y_train), (x_test, y_test) = (x[0:1600], y[0:1600]),(x[1600:], y[1600:])

    print('Build model...')
    model = Sequential()
    model.add(Embedding(1567, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=252,
              epochs=15,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=252)
    print('Test score:', score)
    print('Test accuracy:', acc)
