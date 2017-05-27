#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import Financial as fin
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./data/pingan_timing.csv', index_col = ['date'], parse_dates = ['date'])


    data = df['mid_small_t']
    base = df['mid_small']

    data_r = data.pct_change().fillna(0.0)
    base_r = base.pct_change().fillna(0.0)

    rf = 0.03 / 365
    sharp = (np.mean(data_r) * 365 - rf * 365) / (np.std(data_r) * (365 ** 0.5))
    print 'sharp', sharp
    std = np.std(data_r) * (360 ** 0.5)
    print 'std', std
    print 'jensen', fin.jensen(data_r, base_r, rf) * 365
    print 'semivariance', fin.semivariance(data_r)
    print 'treynor', fin.treynor(data_r, base_r, rf)
    print 'ir', fin.ir(data_r, base_r)
    print 'sortino', fin.sortino(data_r, rf)
