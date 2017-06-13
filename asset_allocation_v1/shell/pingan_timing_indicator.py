#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import Financial as fin
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./export_nav.csv', index_col = ['date'], parse_dates = ['date'])

    data = df['92101:12:9']
    base = df['zz500']

    data_r = data.pct_change().fillna(0.0)
    base_r = base.pct_change().fillna(0.0)

    rf = 0.03 / 252
    print 'annual' , (np.mean(data_r)) * 252
    sharp = (np.mean(data_r) * 252 - rf * 252) / (np.std(data_r) * (252 ** 0.5))
    print 'sharp', sharp
    std = np.std(data_r) * (252 ** 0.5)
    print 'std', std
    print 'jensen', fin.jensen(data_r, base_r, rf) * 252
    print 'semivariance', fin.semivariance(data_r)
    print 'treynor', fin.treynor(data_r, base_r, rf)
    print 'ir', fin.ir(data_r, base_r)
    print 'sortino', fin.sortino(data_r, rf)
