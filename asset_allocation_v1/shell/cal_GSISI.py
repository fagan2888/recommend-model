import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import os
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
#from utils import day_2_week2 as d2w
#from db import asset_trade_dates as load_td

Rf = 0.03/52
filedir = '~/recommend_model/asset_allocation_v1/sw_gsisi_100w.csv'

def cal_gsisi():
#    data = pd.read_csv('~/recommend_model/asset_allocation_v1/sw_component.csv', \
#            index_col = 0, parse_dates = True)
#    trade_dates = load_td.load_trade_dates()
#    data = d2w(data, trade_dates)
    close = pd.read_csv('~/recommend_model/asset_allocation_v1/sw_component.csv', \
            index_col = 0, parse_dates = True)

    index = close.index
    #columns = [str(column) for column in close.columns]
    ret = close.pct_change()
    date_list = []
    gsisi_list = []

    cycle = 50
    for i in range(cycle, len(index) - 1):
        date = index[i].date()
        tmp_ret = ret.loc[index[i]].tolist()[:-1]
        tmp_beta = []
        for j in range(1, 29):
            ass_ret = ret.iloc[i - cycle + 1:i + 1, j]
            mkt_ret = ret.iloc[i - cycle + 1:i + 1, 0]
            beta = cal_beta(ass_ret, mkt_ret)
            tmp_beta.append(beta)

        gsisi = spearmanr(tmp_beta, tmp_ret).correlation
        gsisi_list.append(gsisi)
        date_list.append(date)

    df = pd.DataFrame({'gsisi':gsisi_list}, index = date_list)
    df.to_csv(filedir)

def cal_beta(mkt, ass):
    mkt = np.array(mkt.tolist())
    ass = np.array(ass.tolist())
    mkt = mkt - Rf
    ass = ass - Rf
    finite_loc = np.isfinite(mkt)*np.isfinite(ass)
    mkt = mkt[finite_loc]
    ass = ass[finite_loc]

    lr = LinearRegression().fit(mkt.reshape(-1,1), ass)
    beta = lr.coef_[0]

    return beta

def cal_signal(threshold):
    #df = pd.read_csv('~/recommend_model/asset_allocation_v1/gsisi_sh300.csv', \
    #        index_col = 0, parse_dates = True)

    df = merge_data()
    df.dropna(inplace = True)
    positions = []
    turning_points= []
    signal = 0
    position = 0
    tp = 0
    for idx, indic in enumerate(df['gsisi']):
        if (indic > threshold) and (df['pct_chg'][idx] > 0):
            if position == 0:
                if signal == -1:
                    signal = 0
                elif signal == 1:
                    (signal, position) = (0, 1)
                    tp = 1
                else:
                    signal = 1

            if position == 1:
                if signal == -1:
                    signal = 0

        if (indic < -threshold) and (df['pct_chg'][idx] < 0):
            if position == 1:
                if signal == 1:
                    signal = 0
                elif signal == -1:
                    (signal, position) = (0, 0)
                    tp = -1
                else:
                    signal = -1

            if position == 0:
                if signal == 1:
                    signal = 0

        positions.append(position)
        turning_points.append(tp)
        tp = 0

    df['signal'] = turning_points
    df['position'] = positions

    return df

def cal_signal_fisi(threshold):
    df = merge_data()
    signal = []
    for i in range(len(df)):
        if (df['gsisi'][i] > threshold) and (df['pct_chg'][i] > 0):
            signal.append(1)
        elif (df['gsisi'][i] > threshold) and (df['pct_chg'][i] < 0):
            signal.append(2)
        elif (df['gsisi'][i] < threshold) and (df['pct_chg'][i] > 0):
            signal.append(5)
        elif (df['gsisi'][i] < threshold) and (df['pct_chg'][i] < 0):
            signal.append(6)
        elif df['pct_chg'][i] > 0:
            signal.append(3)
        elif df['pct_chg'][i] <= 0:
            signal.append(4)
    df['signal'] = signal
    df['fisi'] = df['signal'].rolling(10).apply(cal_fisi)
    df = df.dropna()
    return df

def cal_fisi(x):
    x = list(x)
    fisi = (x.count(1) + x.count(5) - x.count(2) - x.count(6))/\
            float(x.count(1) + x.count(5) + x.count(2) + x.count(6))
    return fisi

def merge_data():
    gsisi = pd.read_csv(filedir, index_col = 0, parse_dates = True)
    sh300 = pd.read_csv('~/recommend_model/asset_allocation_v1/120000001_ori_day_data.csv', \
            index_col = 0, parse_dates = True)
    merged_data = pd.merge(gsisi, sh300, left_index = True, right_index = True, \
            how = 'left')
    merged_data['pct_chg'] = merged_data.close.pct_change()
    merged_data.dropna(inplace = True)
    #print merged_data
    return merged_data

def training(x):
    model = sm.tsa.VARMAX(x, order = (5,0))
    result = model.fit(disp = False)
    #print result.summary()
    return result.forecast(1)[0, 1]

def predict():
    window = 100
    df = cal_signal(threshold = 0.317)
    x = df.loc[:, ['gsisi', 'pct_chg']].values
    result = [training(x[i: i+window]) for i in range(len(x)-window+1)]
    df = df[window-1:]
    df['pre_pct_chg'] = result
    df.to_csv('gsisi_ts_pre.csv')
    return df

def plot(method = 'gsisi'):
    if method == 'gsisi':
        df = cal_signal(threshold = 0.317)
        fig = plt.figure(figsize = (30, 20))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.plot(df.index, df.close)
        #tp_up = df[df.signal == 1]
        tp_up = df[(df.gsisi > 0.317) & (df.pct_chg > 0)]
        ax1.plot(tp_up.index, tp_up.close, '.', markersize = 24, color = 'r')
        #tp_down = df[df.signal == -1]
        #tp_down = df[(df.signal < 0.317) & (df.pct_chg < 0)]
        #ax1.plot(tp_down.index, tp_down.close, '.', markersize = 24, color = 'g')
        ax2.bar(df.index, df.gsisi, width = 2.0, color = 'y')
        fig.savefig('gsisi_alltp_100w.png')

    if method == 'fisi':
        df = cal_signal_fisi(threshold = 0.317)
        fig = plt.figure(figsize = (30, 20))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.plot(df.index, df.close)
        ax2.bar(df.index, df.fisi, width = 2.0, color = 'y')
        fig.savefig('sh300_fisi.png')


if __name__ == '__main__':
    #cal_gsisi()
    #plot('fisi')
    #training()
    predict()
