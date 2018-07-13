#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8


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

    pv = np.dot(np.dot(w, V), w.T)

    return pv


def risk_budget_objective(x,pars):

    V = pars[0]
    sig_p = calculate_portfolio_var(x,V)

    return sig_p


def total_weight_constraint(x):
    return np.sum(x)-1.0


def long_only_constraint(x):
    return x


def cal_weight(V, w0 = None):

    V = V * 10000
    asset_num = V.shape[0]

    if w0 is None:
        w0 = [1 / asset_num]*asset_num

    b_ = [(0, 1.0) for i in range(asset_num)]

    cons = (
        {'type': 'eq', 'fun': total_weight_constraint},
    )
    res = minimize(risk_budget_objective, w0, args=[V], method='SLSQP', constraints = cons, bounds = b_, options={'disp': False})

    return res.x


if __name__ == '__main__':
    pass



