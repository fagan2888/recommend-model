#coding=utf-8

import pandas as pd
import numpy as np
import datetime
#'''
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
import tensorflow
#'''
from dateindex import dateindex
import sys
import random

data = pd.read_csv('capital.csv')
data = dateindex(data)

ratetmp = data.iloc[:,0:2]
rate = ratetmp.copy()
rate2 = ratetmp.copy()
for i in range(1,90):
    rate2 = rate2 * ratetmp.shift(i)
    rate = pd.concat([rate,rate2],axis=1)
rate = rate.dropna()

index = list(rate.index)
#random.shuffle(index) #随机排序可能造成last model或过拟合问题
rate = rate.ix[index,:]
indata = np.matrix(rate)
ansdata = np.matrix(data.ix[rate.index,['result1','result2']])
length = rate.shape[1]

in_train = indata[0:int(len(indata)*3/5)]
in_test = indata[int(len(indata)*3/5):int(len(indata)*4/5)]
in_verify = indata[int(len(indata)*4/5):len(indata)]
ans_train = ansdata[0:int(len(ansdata)*3/5)]
ans_test = ansdata[int(len(ansdata)*3/5):int(len(ansdata)*4/5)]
ans_verify = ansdata[int(len(ansdata)*4/5):len(ansdata)]
#'''
try:
    epofit = int(sys.argv[1])
    lr = np.power(0.1,int(sys.argv[2]))
    decay = np.power(0.1,int(sys.argv[3]))
    momentum = float(sys.argv[4])
    #drop = float(sys.argv[5])
except:
    epofit = 10000
    lr = 1e-4
    decay = 1e-5
    momentum = 0.999
    #drop = 0

model = Sequential()
model.add(Dense(units=length*20,activation='relu',input_dim=length))
model.add(Dense(units=length*20,activation='linear'))
#model.add(Dropout(drop))
#model.add(Dense(units=length*int(20*drop),activation='relu'))
model.add(Dense(units=length*20,activation='relu'))
model.add(Dense(units=2,activation='linear'))
model.add(Dense(units=2,activation='softmax'))
model.compile(loss='mse', optimizer=SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True), metrics=['accuracy'])
model.fit(in_train, ans_train, epochs=epofit, batch_size=1280)

predict = model.predict(in_train, batch_size = 1280)
print '##########################################################'
print ans_train.flat[:30]
print predict.flat[:30]

vloss = model.evaluate(in_test, ans_test, batch_size=1280)
pre = model.predict(in_test, batch_size=1280)
print '###################################################################'
print vloss
print ans_test.flat[:30]
print pre.flat[:30]
#'''
