#!/usr/bin/python
# coding=utf-8

from ipdb import set_trace
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_features():
    '''
    IP:工业生产指数, 滞后18天
    MP:制造业产出指数，滞后18天
    BM:基础货币，滞后5天
    UN:失业人数（少于5周），滞后6天
    IUI:当周初次申领失业金人数，滞后7天
    IPD:工业生产指数，耐用品，滞后18天
    MI:制造业存货量，滞后37天
    DNO:Durity new orders, lag23
    MUO:Mfg. uncompleted orders, lag37
    CEI:consumers expectation index, lag(-10)
    ECRI: ECRI leading indicators, lag6
    OECD: OECD leading indicators, lag36
    '''
    data = pd.read_csv('sp500_features.csv', index_col = 0, parse_dates = True)
    return data

def lag():
    data = smooth()
    ip = data.IP.copy().diff().apply(np.sign)
    ip.index = ip.index+timedelta(18)
    mp = data.MP.copy().diff().apply(np.sign)
    mp.index = mp.index+timedelta(18)
    bm = data.BM.copy().diff().apply(np.sign)
    bm.index = bm.index+timedelta(5)
   # bm[bm.index <= '2009'] = 0
    un = data.UN.copy().diff().apply(np.sign)
    un.index = un.index+timedelta(6)
    iui = data.IUI.copy().diff().apply(np.sign)
    iui.index = iui.index+timedelta(7)
    ipd = data.IPD.copy().diff().apply(np.sign)
    ipd.index= ipd.index+timedelta(18)
    mi = data.MI.copy()
    mi.index = mi.index+timedelta(37)
    mi = mi.dropna().diff().apply(np.sign)
    dno = data.DNO.copy()
    dno.index = dno.index+timedelta(23)
    dno = dno.dropna().diff().apply(np.sign)
    muo = data.MUO.copy()
    muo.index = muo.index+timedelta(37)
    muo = muo.dropna().diff().apply(np.sign)
    cei = data.CEI.copy()
    cei.index = cei.index-timedelta(10)
    cei = cei.dropna().diff().apply(np.sign)
   # ym = data['10Y3M'].copy()
   # ym = ym.dropna().diff().apply(np.sign)
   # rs = data.RS.copy()
   # rs.index = rs.index-timedelta(35)
   # rs = rs.dropna().diff().apply(np.sign)
   # ecri = data.ECRI.copy()
   # ecri.index = ecri.index-timedelta(6)
   # ecri = ecri.dropna().diff().apply(np.sign)
   # oecd = data.OECD.copy()
   # oecd.index = oecd.index-timedelta(36)
   # oecd = oecd.dropna().diff().apply(np.sign)*5

    sp500 = pd.read_csv('sp500.csv', index_col=0, parse_dates=True)
    sp500 = sp500[sp500.index >= '1992-01-01']
    data_lag = pd.concat([ip,mp,bm,un,iui,ipd,mi,dno,muo,cei,sp500],1)
    data_lag = data_lag.fillna(method = 'pad')
    data_lag = data_lag.dropna()

    return data_lag

def smooth():
    data = load_features()
    sp500 = data.SP500.copy()
    data = data.rolling(4).apply(filter)
    data['SP500'] = sp500
    #data.to_csv('smoothed_feature.csv', index_label = 'date')
    #set_trace()

    return data

def filter(sequence):
    '''
    This is a band-filter, which keep cycle from 4 months to 12 months.
    '''
    return sequence[0]+2*sequence[1]+2*sequence[2]+sequence[3]
    #return np.mean(sequence)

def cal_view():
    data = lag()
    data['view'] = data.loc[:, ['IP', 'MP', 'BM', 'UN', 'IUI', 'IPD', 'MI', 'DNO', 'MUO', 'CEI']].sum(axis = 1)
    data['sp500_R'] = data.loc[:, ['sp500']].rolling(2).apply(lambda x: x[1]/x[0]).fillna(1.0)
    data['timing_R'] = data.loc[:, ['view', 'sp500_R']].apply(lambda x:x[1] if x[0] > 0 else 1, axis = 1)
    data['sp500_nav'] = data['sp500_R'].cumprod()
    data['timing_nav'] = data['timing_R'].cumprod()
    #data.to_csv('sp500_view.csv', index_label = 'date')

    return data

def plot():
    data = cal_view()
    print data
    data.loc[:, ['sp500_nav', 'timing_nav']].plot()
    plt.show()

if __name__ == '__main__':
    #data = load_features()
    #lag()
    #cal_view()
    plot()
