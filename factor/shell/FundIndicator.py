#coding=utf8


import numpy  as np
import pandas as pd
import math


rf = 0.025 / 52


#计算下方差
def semivariance(prs):
    mean       =        np.mean(prs)
    var        =        0.0
    for r in prs:
        if r <= mean:
            var = var + (r - mean) ** 2
    var        =    var / (len(prs) - 1)
    var        =    math.sqrt(var)

    return     var


#计算jensen测度
def jensen(prs, market, rf):

    p = []
    m = []
    for i in range(0, len(prs)):
        #print portfolio[i], market[i]
        p.append(prs[i] - rf)
        m.append(market[i] - rf)

    p = np.array(p)
    m = np.array(m)

    clf       = linear_model.LinearRegression()
    clf.fit(m.reshape(len(m),1), p.reshape(len(p), 1))
    #clf.fit(m, p)
    alpha = clf.intercept_[0]
    beta  = clf.coef_[0]

    #if np.isnan(alpha) or np.isinf(alpha):
    #    alpha = 0.0

    return alpha


