#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ipdb import set_trace
import sys
sys.path.append('shell')
from db import asset_mz_markowitz_nav, base_trade_dates
import DBData

 # risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    pvar = np.dot(np.dot(w,V),w.T)[0,0]
    # return (w*V*w.T)[0,0]
    return pvar


def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = np.dot(V,w.T)
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC


def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p = np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    # J = sum(np.power(asset_RC-risk_target.T, 4))[0,0] # sum of squared error
    return J


def total_weight_constraint(x):
    return np.sum(x)-1.0


def long_only_constraint(x):
    return x


def cal_weight(V, x_t, cons2 = None, w0 = None):
    if w0 is None:
        w0 = [1/len(x_t)]*len(x_t)
        # w0 = np.zeros(len(x_t))
        # w0[0] = 1.0
    cons1 = (
        {'type': 'eq', 'fun': total_weight_constraint},
        {'type': 'ineq', 'fun': long_only_constraint}
    )
    if cons2 is not None:
        cons = cons1 + cons2
    else:
        cons = cons1
        res = minimize(risk_budget_objective, w0, args=[V*3000,x_t], method='SLSQP', constraints=cons, options={'disp': False})
        # res = minimize(risk_budget_objective, w0, args=[V,x_t], method='SLSQP', constraints=cons, options={'disp': True, 'ftol':1e-100})
    # print()
    # print(calculate_risk_contribution(res.x, V).T)
    if not res.success:
        print('ERROR!!!')
    # w_rb = np.asmatrix(res.x)

    return res.x


def cal_each_risk_contribution():
    trade_dates = DBData.trade_date_index(start_date = '2010-03-01')
    high_nav = asset_mz_markowitz_nav.load_series('MZ.000010')
    high_nav = high_nav.reindex(trade_dates).dropna()
    low_nav = asset_mz_markowitz_nav.load_series('MZ.000011')
    low_nav = low_nav.reindex(trade_dates).dropna()
    high_ret = high_nav.pct_change().fillna(0.0)
    low_ret = low_nav.pct_change().fillna(0.0)
    df_ret = pd.concat([high_ret, low_ret], 1).dropna()
    V = np.cov(df_ret.T)
    for risk in range(1, 11):
        ratio_h = (risk-1)/9.0
        ratio_l = (10 - risk)/9.0
        rc = calculate_risk_contribution([ratio_h, ratio_l], V).flat[:]
        rc = np.array(rc)[0]
        rc = rc/np.sum(rc)
        print(risk ,rc)


if __name__ == '__main__':
    '''
    x_t = [0.8, 0.2] # initial guess
    V = [
        [1e-4,5e-5],
        [5e-5,1e-6],
    ] # expected variance
    V = np.array(V)
    w = cal_weight(V*10000, x_t)
    print w
    '''
    cal_each_risk_contribution()
