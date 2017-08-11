#coding=utf8



import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input


if __name__ == '__main__':


    df = pd.read_csv('./tmp/fund_nav.csv', parse_dates = ['date'], index_col = ['date'])


    df = df.iloc[-1000 : ]
    df = df.dropna(axis = 1)
    dfr = df.pct_change().fillna(0.0)

    #print dfr.columns
    x_train = dfr.values[0 : 800]
    x_test  = dfr.values[800 : 200]

    input_img = Input(shape=(760, ))

    encoding_dim = 10
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    decoded = Dense(10, activation='relu')(encoder_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(760, activation='softmax')(decoded)

    autoencoder = Model(inputs = input_img, outputs = decoded)

    autoencoder.compile(optimizer = 'sgd', loss = 'mse')

    autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size = 10, shuffle = True, validation_data = (x_test, x_test))

    print autoencoder
