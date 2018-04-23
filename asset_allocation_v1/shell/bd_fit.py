#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import scipy.stats as ss
import numpy as np
import pandas as pd
from ipdb import set_trace
from scipy.stats import norm, binom_test


def likelihood_f(P, x, neg=1):
    n=np.round(P[0]) #by definition, it should be an integer 
    p=P[1]
    loc=np.round(P[2])
    return neg*(np.log(ss.binom.pmf(x, n, p, loc))).sum()


def cal_bd_ci(n, p, alpha):
    q = 1 - p
    c = norm.ppf((1+alpha)/2)
    interval = c*(p*q/n)**0.5

    return (p - interval, p + interval)


def cal_bf_ci(df):
    bf_ids = df.columns
    alpha = 0.95
    for bf_id in bf_ids:
        tmp_df = df[bf_id].dropna()
        n = len(tmp_df)
        p = len(tmp_df[tmp_df > 0])
        test = binom_test(p, n)
        # p = p/float(n)
        # ci = cal_bd_ci(n, p, alpha)
        # if ci[0] < 0.5 < ci[1]:
        #     status = 0
        # else:
        #     status = 1
        if test < 0.05:
            status = 1
        else:
            status = 0

        print bf_id, test, status


if __name__ == '__main__':
    '''
    x=ss.binom.rvs(n=1, p=0.5, loc=0, size=1000)
    result=so.fmin(likelihood_f, [1, 0.8, 0], args=(x,-1), full_output=True, disp=False)
    set_trace()
    print result
    '''
    # print cal_bd_ci(1000, 0.55, 0.95)
    df = pd.read_csv('data/layer_ic_df.csv', index_col = 0, parse_dates = True)
    cal_bf_ci(df)
