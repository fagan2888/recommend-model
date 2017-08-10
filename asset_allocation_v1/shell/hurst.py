from __future__ import division
import pandas as pd
import numpy as np
#import os

#from nolds.measures import rs
from nolds import hurst_rs, logarithmic_n
from sklearn.linear_model import LinearRegression

class Hurst(object):
    def __init__(self, data, window = 200, short_mean = 10, long_mean = 30, \
            baseline = -0.03):
        self.data = data
        self.window = window
        self.m1 = short_mean
        self.m2 = long_mean
        self.data['pct_chg'] = self.data['tc_close'].pct_change()
        self.data = self.data.dropna()
        self.n = logarithmic_n(50, 100, 1.05)
        self.baseline = baseline

    def cal_hurst(self):
        self.data['hurst'] = self.data['pct_chg'].rolling(self.window).apply(hurst_rs, \
                kwargs = {'nvals':self.n, 'fit':'poly'})
        #self.data['hurst'] = self.data['pct_chg'].rolling(self.window).apply(hurst_rs, \
        #        kwargs = {'nvals':self.n})
        self.data['short'] = self.data['hurst'].rolling(self.m1).mean()
        self.data['long'] = self.data['hurst'].rolling(self.m2).mean()
        self.data = self.data.dropna()
        #self.data.to_csv('hurst_zz500.csv')

    def cal_signal(self, method = 'mean'):
        if method == 'mean':
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

        if method == 'diff':
            turning_point = [0]
            dhurst = self.data['hurst'].diff(5)
            self.data['dhurst'] = dhurst
            dhurstm = self.data['dhurst'].rolling(20).mean()
            self.data['dhurstm'] = dhurstm
            tp_func = lambda x: 1 if (x[0] > self.baseline and x[1] < self.baseline) else 0
            self.data['tp'] = self.data['dhurstm'].rolling(2).apply(tp_func)

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

    def cal_tp_dates(self, method = 'mean'):
        self.cal_hurst()
        self.cal_signal(method = 'diff')
        #self.cal_expt_hurst()
        tp_dates = self.data.index[self.data.tp == 1]
        tp_dates = np.array(tp_dates.tolist())
        np.save('../tp_dates.npy', tp_dates)
        return tp_dates

if __name__ == '__main__':
    data = pd.read_csv('/home/yaojiahui/recommend_model/asset_allocation_v1/120000002_ori_day_data.csv', \
            index_col = 0, parse_dates = True)
    data = data.rename(columns = {'close':'tc_close', 'high':'tc_high', \
            'low':'tc_low'})
    hurst = Hurst(data)
    tp_dates = hurst.cal_tp_dates()
    print tp_dates
