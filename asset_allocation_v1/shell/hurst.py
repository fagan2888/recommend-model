from __future__ import division
import pandas as pd
import numpy as np
#import os

#from nolds import hurst_rs, logarithmic_n
#from nolds.measures import rs
from nolds import hurst_rs
from sklearn.linear_model import LinearRegression

class Hurst(object):
    def __init__(self, data, window = 400, short_mean = 10, long_mean = 30):
        self.data = data
        self.window = window
        self.m1 = short_mean
        self.m2 = long_mean
        self.data['pct_chg'] = self.data['tc_close'].pct_change()
        self.data = self.data.dropna()
        if window > 200:
            self.n = np.arange(50, window/2)
        else:
            self.n = np.arange(2, window/2)

    def cal_hurst(self):
        self.data['hurst'] = self.data['pct_chg'].rolling(self.window).apply(hurst_rs, \
                kwargs = {'nvals':self.n, 'fit':'poly'})
        self.data['short'] = self.data['hurst'].rolling(self.m1).mean()
        self.data['long'] = self.data['hurst'].rolling(self.m2).mean()
        self.data = self.data.dropna()

    def cal_signal(self):
        turning_point = [0]
        pre = self.data['short'][0] - self.data['long'][0]
        for i in range(1, len(self.data)):
            now = self.data.ix[i, 'short'] - self.data.ix[i, 'long']
            if pre > 0 and now < 0:
                turning_point.append(1)
            else:
                turning_point.append(0)
            pre = now

        self.data['tp'] = turning_point

    @staticmethod
    def cal_expt_rs(n):
        const = ((n-0.5)/n)*((n*np.pi/2)**(-0.5))
        T = 0
        for r in range(1, n):
            T += ((n-r)/r)**(0.5)
        expt_rs = const*T
        return expt_rs

    def cal_expt_hurst(self):
        X = np.log(self.n).reshape(-1,1)
        y = np.log([self.cal_expt_rs(int(i)) for i in self.n])
        LR = LinearRegression(fit_intercept = True)
        LR.fit(X, y)
        expt_hurst = LR.coef_[0]
        self.data['expt_hurst'] = expt_hurst

    def cal_tp_dates(self):
        self.cal_hurst()
        self.cal_signal()
        #self.cal_expt_hurst()
        tp_dates = self.data.index[self.data.tp == 1]
        return tp_dates

if __name__ == '__main__':
    data = pd.read_csv('/home/yaojiahui/recommend_model/asset_allocation_v1/120000001_ori_day_data.csv', \
            index_col = 0, parse_dates = True)
    data = data.rename(columns = {'close':'tc_close', 'high':'tc_high', \
            'low':'tc_low'})
    hurst = Hurst(data[:500])
    tp_dates = hurst.cal_tp_dates()
    print tp_dates
