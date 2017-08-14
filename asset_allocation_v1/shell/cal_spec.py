from __future__ import division
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#import os

def cal_t(data, q = 2, dtype = 1):
    Sq = []
    measure = [1, 1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/16, 1/24, 1/48]
    if dtype == 2:
        measure.extend([1/60, 1/80, 1/120, 1/240])

    for i in measure:
        n = int(len(data)*i)
        m = int(1/i)
        Pi = data.reshape(m,n).sum(1)/data.reshape(m,n).sum()
        Pi = Pi**q
        Sq.append(Pi.sum())

    log_measure = np.log(measure)
    log_Sq = np.log(Sq)

    keep = np.isfinite(log_measure)*np.isfinite(log_Sq)
    log_measure = log_measure[keep]
    log_Sq = log_Sq[keep]

    LR = LinearRegression().fit(log_measure.reshape(-1, 1), log_Sq)
    return LR.coef_[0]

def cal_indic(data, qs = np.arange(-10, 10, 0.25), dtype = 1):
    tq = []
    for q in qs:
        t = cal_t(data, q = q, dtype = 1)
        tq.append(t)

    alpha = np.diff(tq)/(qs[1] - qs[0])
    f_alpha = alpha*qs[:-1] - tq[:-1]
    da = np.max(alpha) - np.min(alpha)
    df = f_alpha[np.argmin(alpha)] - f_alpha[np.argmax(alpha)]
    Rf = da * sign(df) * np.exp(np.abs(df))
    #df = np.min(f_alpha[-10:]) - np.min(f_alpha[:10])
    #print alpha
    #print f_alpha
    #print df

    return da, df, Rf

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

if __name__ == '__main__':
    r1 = np.random.randn(48)
    r2 = r1 + 3000
#    print cal_indic(r1)
#    print cal_indic(r2)
    print cal_indic(r2)
