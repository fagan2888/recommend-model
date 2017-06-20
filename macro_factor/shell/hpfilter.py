#coding=utf8


import scipy
from scipy import linalg as la
from scipy import sparse
import pandas as pd
import numpy as np


def hp_filter(y,w):
    # make sure the inputs are the right shape
    m,n  = y.shape
    if m < n:
        y = y.T
        m = n

    a    = scipy.array([w, -4*w, ((6*w+1)/2.)])
    d    = scipy.tile(a, (m,1))

    d[0,1]   = -2.*w
    d[m-2,1] = -2.*w
    d[0,2]   = (1+w)/2.
    d[m-1,2] = (1+w)/2.
    d[1,2]   = (5*w+1)/2.
    d[m-2,2] = (5*w+1)/2.

    B = sparse.spdiags(d.T, [-2,-1,0], m, m)
    B = B+B.T

    # report the filtered series, s
    s = scipy.dot(la.inv(B.todense()),y)
    return s


if __name__ == '__main__':


    df = pd.read_csv('./data/index.csv', parse_dates = ['date'], index_col = ['date'])
    df = df['000300.SH']
    df = df.iloc[0:-483]
    y = scipy.atleast_2d(df.values)
    s = hp_filter(y, 1600)
    s = s.reshape(1, -1)
    #print s
    df = pd.DataFrame(np.matrix([df.values, s[0]]).T, index = df.index, columns = ['000300.SH', 'hpfilter'])
    df.index.name = 'date'
    #print df
    df.to_csv('hpfilter.csv')
    print df
