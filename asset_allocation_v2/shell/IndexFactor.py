#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from __future__ import division
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
sys.path.append('shell')
from ipdb import set_trace


def risk_budget_objective(x,pars):
    # calculate portfolio risk
    sfe = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk

    pfe = np.dot(x, sfe)

    # target = -np.dot(x_t, pfe**2)
    vf = pfe[abs(x_t) == 1]
    # ivf = pfe[abs(x_t) == 0]
    # target = np.sum(x_t)*(np.sign(vf)*(vf**2)).sum() - np.abs(ivf).sum()
    # target = np.sum(x_t)*(np.sign(vf)*(vf**2)).mean() - (ivf**2).mean()
    target = np.sum(x_t)*(np.sign(vf)*(vf**2)).sum()
    # target = np.sum(x_t)*vf - (ivf**2).sum()

    return -target


def total_weight_constraint(x):
    return np.sum(x)-1.0


# def cal_weight(sfe, x_t, lower_limit = None, upper_limit = None, cons2 = None, w0 = None):

#     stock_num = sfe.shape[0]

#     if lower_limit is None:
#         lower_limit = 0.0

#     if upper_limit is None:
#         upper_limit = 0.02

#     if w0 is None:
#         w0 = np.array([1.0/stock_num]*stock_num)

#     cons = (
#         {'type': 'eq', 'fun': total_weight_constraint},
#     )

#     bnds = tuple([(lower_limit, upper_limit) for i in range(stock_num)])

#     res = minimize(risk_budget_objective, w0, args=[sfe, x_t], method='SLSQP', constraints=cons, options={'disp': False}, bounds = bnds)

#     return res.x


def cal_weight(sfe, x_t, target_num = 100, lower_limit = None, upper_limit = None, cons2 = None, w0 = None):

    target_exposure = np.dot(sfe, x_t)
    df = pd.Series(data = target_exposure, index = sfe.index)
    df = df.nlargest(target_num)

    tmp_df = df[df == max(df)]
    if len(tmp_df) >= target_num:
        pos = tmp_df.index
    else:
        pos = df.nlargest(target_num).index

    weight = pd.Series(0.0, index = sfe.index)
    weight.loc[pos] = 1 / len(pos)

    return weight.values


if __name__ == '__main__':

    sfe = np.random.randn(100, 5)
    x_t = np.array([1,0,0,0,0])
    weight = cal_weight(sfe, x_t)
    set_trace()




