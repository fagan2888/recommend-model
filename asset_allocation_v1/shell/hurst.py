from __future__ import division
import pandas as pd
import numpy as np
#import os

from nolds import hurst_rs, logarithmic_n
from nolds.measures import rs
from sklearn.linear_model import LinearRegression

def load_data():
    df = pd.read_csv('./data/120000002_ori_day_data.csv', index_col=0, \
            parse_dates = True)
    df['pct_chg'] = df.close.pct_change()
    df = df.dropna()
    return df

def cal_hurst(ori_data, window, frac_len):
    ori_data['hurst'] = ori_data['pct_chg'].rolling(window).apply(hurst_rs, \
            kwargs = {'nvals':frac_len})
    ori_data['hurst_mean20'] = ori_data['hurst'].rolling(20).mean()
    ori_data = ori_data.dropna()
    return ori_data

def cal_signal(ori_data, expt_hurst):
    get_signal = lambda x: 1 if (x[0] > expt_hurst and x[-1] < expt_hurst) else 0
    signal = ori_data['hurst'].rolling(2).apply(get_signal)
    ori_data['signal'] = signal
    ori_data = ori_data.dropna()
    return ori_data

def cal_signal2(ori_data, expt_hurst, window):
    s_num = 0
    in_num = window
    e_num = len(ori_data)
    signal = []
    hurst_expo = np.array(ori_data['hurst'])
    pct_chg = np.array(ori_data['pct_chg'])
    while True:
        tmp_pctchg = pct_chg[s_num:in_num]
        tmp_ret = np.product(tmp_pctchg[-10:] + 1)
        tmp_hurst = hurst_expo[in_num - 1]
        '''
        if tmp_hurst > expt_hurst + 0.01:
            if tmp_ret > 1.01:
                signal.append(1)
            elif tmp_ret < 0.99:
                signal.append(-1)
            else:
                signal.append(0)

        elif tmp_hurst < expt_hurst - 0.01:
            if tmp_ret > 1.01:
                signal.append(-1)
            elif tmp_ret < 0.99:
                signal.append(1)
            else:
                signal.append(0)

        else:
            signal.append(0)
            '''

        if tmp_hurst > expt_hurst:
            if tmp_ret > 1:
                signal.append(1)
            else:
                signal.append(-1)

        else:
            if tmp_ret > 1:
                signal.append(-1)
            else:
                signal.append(1)

        s_num += 1
        in_num += 1
        if in_num == e_num:
            break

    ori_data = ori_data[window:]
    print len(ori_data)
    print signal
    print len(signal)
    ori_data['signal'] = signal
    return ori_data

def get_cycle(ori_data, n = np.arange(2, 1301)):
    ret = np.array(ori_data['pct_chg'])
    result = {}
    logn = np.log10(n)
    RS = np.array([rs(ret, i) for i in n])
    logRS = np.log10(RS)
    Vn = RS/(n**0.5)
    result['logn'] = logn
    result['logRS'] = logRS
    result['Vn'] = Vn
    result_df = pd.DataFrame(result, index = n)
    result_df.to_csv('./output/hs300_cycle.csv')
    print result_df
    return result_df

def cal_expt_rs(n):
    const = ((n-0.5)/n)*((n*np.pi/2)**(-0.5))
    T = 0
    for r in range(1, n):
        T += ((n-r)/r)**(0.5)
    expt_rs = const*T
    return expt_rs

def cal_expt_hurst(frac_len):
    X = np.log(frac_len).reshape(-1,1)
    y = np.log([cal_expt_rs(i) for i in frac_len])
    LR = LinearRegression(fit_intercept = True)
    LR.fit(X, y)
    expt_hurst = LR.coef_[0]
    return expt_hurst

def handle():
    window = 400
    frac_len = logarithmic_n(50, 200, 1.05)
    expt_hurst = cal_expt_hurst(frac_len)
    print expt_hurst
    ori_data = load_data()
    #cycle = get_cycle(ori_data, np.arange(2, 401))
    #os._exit(0)
    ori_data = cal_hurst(ori_data, window, frac_len)
    #print ori_data
    #print ori_data['hurst_mean20'].min(), ori_data['hurst_mean20'].mean(), \
    #        ori_data['hurst_mean20'].max()
    #ori_data = cal_signal2(ori_data, expt_hurst, window)
    #print np.sum(ori_data['signal'])
    ori_data['expt_hurst'] = expt_hurst
    ori_data.to_csv('./output/hurst_day_zz500.csv')


if __name__ == '__main__':
    #handle()
    data = load_data()
    get_cycle(data)
