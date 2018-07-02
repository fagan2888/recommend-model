#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from __future__ import division
import numpy as np
import pandas as pd
from scipy.optimize import minimize
# from ipdb import set_trace
import sys
sys.path.append('shell')
from ipdb import set_trace

 # risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]


def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC


def risk_budget_objective(x,pars):
    # calculate portfolio risk
    sfe = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    x_v = pars[2]

    pfe = np.dot(x, sfe)
    pfe = pfe[x_v == 1]
    x_t = x_t[x_v == 1]

    target = np.sum(np.power(pfe - x_t, 2))
    # target = np.sum(np.abs(pfe - x_t))

    return target


def total_weight_constraint(x):
    return np.sum(x)-1.0


def long_only_constraint(x):
    return x


def upper_constraint(x):
    return 0.1 -x


def cal_weight(sfe, x_t, x_v = None, cons2 = None, w0 = None):

    stock_num = sfe.shape[0]
    if w0 is None:
        w0 = np.array([1.0/stock_num]*stock_num)
    if x_v is None:
        x_v = np.ones_like(x_t)

    cons = (
        {'type': 'eq', 'fun': total_weight_constraint},
    )

    bnds = tuple([(0.0,0.1) for i in range(stock_num)])

    # res = minimize(risk_budget_objective, w0, args=[sfe, x_t], method='L-BFGS-B', constraints=cons, options={'disp': False}, bounds = bnds)
    res = minimize(risk_budget_objective, w0, args=[sfe, x_t, x_v], method='SLSQP', constraints=cons, options={'disp': False}, bounds = bnds)
    print('Loss value:', res.fun)

    return res.x


if __name__ == '__main__':

    sfe = np.random.randn(100, 5)
    x_t = np.array([1,0,0,0,0])
    weight = cal_weight(sfe, x_t)
    set_trace()



