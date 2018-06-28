#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from __future__ import division
import numpy as np
import pandas as pd
from scipy.optimize import minimize
# from ipdb import set_trace
import sys
sys.path.append('shell')
from db import asset_mz_markowitz_nav, base_trade_dates
from ipdb import set_trace
import DBData

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

    pfe = np.dot(x, sfe)
    target = np.sum(np.power(pfe - x_t, 2))

    return target

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

def upper_constraint(x):
    return 0.1 -x

def cal_weight(sfe, x_t, cons2 = None, w0 = None):

    stock_num = sfe.shape[0]
    if w0 is None:
        w0 = np.array([1.0/stock_num]*stock_num)

    cons1 = (
        {'type': 'eq', 'fun': total_weight_constraint},
        # {'type': 'ineq', 'fun': long_only_constraint},
        # {'type': 'ineq', 'fun': upper_constraint},
    )
    if cons2 is not None:
        cons = cons1 + cons2
    else:
        cons = cons1

    bnds = tuple([(0,0.05) for i in range(stock_num)])

    res = minimize(risk_budget_objective, w0, args=[sfe, x_t], method='L-BFGS-B', constraints=cons, options={'disp': False}, bounds = bnds)
    # w_rb = np.asmatrix(res.x)

    return res.x


if __name__ == '__main__':

    sfe = np.random.randn(100, 5)
    x_t = np.array([1,0,0,0,0])
    weight = cal_weight(sfe, x_t)
    set_trace()





