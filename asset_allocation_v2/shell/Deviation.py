#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ipdb import set_trace
import sys
sys.path.append('shell')
from db import asset_mz_markowitz_nav, base_trade_dates, base_ra_index_nav
from asset import Asset
import DBData

def load_asset_ret(asset_ids):

    df_nav = {}
    for asset_id in asset_ids:
        df_nav[asset_id] = Asset.load_nav_series(asset_id)
    df_nav = pd.DataFrame(df_nav)
    df_ret = df_nav.dropna().pct_change().dropna()

    return df_ret

# risk budgeting optimization
def calculate_portfolio_var(w, V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    pvar = np.dot(np.dot(w,V),w.T)[0,0]
    # return (w*V*w.T)[0,0]
    return pvar

def calculate_risk_contribution(w, V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w, V))
    # Marginal Risk Contribution
    MRC = np.dot(V, w.T)
    # Risk Contribution
    RC = np.multiply(MRC, w.T) / sigma / sigma
    RC = np.ravel(RC)

    return RC, sigma

def cal_deviation(user_weight, best_weight, V):

    RC_user, sigma_user = calculate_risk_contribution(user_weight, V)
    RC_best, sigma_best = calculate_risk_contribution(best_weight, V)

    deviation = np.abs(RC_user - RC_best).sum()
    deviation = deviation + np.abs(sigma_user / sigma_best - 1)

    return deviation


if __name__ == '__main__':

    asset_ids_1 = ['120000010', '120000011', '120000014', '120000039', '120000053', '120000056', '120000058', '120000073', '120000081', '120000082']
    asset_ids_2 = ['ERI000001', 'ERI000002', 'MZ.FA0010', 'MZ.FA0050', 'MZ.FA0070', 'MZ.FA1010']
    asset_ids = asset_ids_1 + asset_ids_2
    df_ret = load_asset_ret(asset_ids)
    V = df_ret.cov()
    user_weight = np.array([1.0 / len(asset_ids)] * len(asset_ids))  # 0.0625 for all factor
    print(user_weight)

    best_weight = user_weight.copy()
    best_weight[0] = 0.2
    best_weight = best_weight / best_weight.sum()
    print(best_weight)
    deviation = cal_deviation(user_weight, best_weight, V)
    print(deviation)

    best_weight = user_weight.copy()
    best_weight[0] += 0.05
    best_weight[1] -= 0.05
    deviation = cal_deviation(user_weight, best_weight, V)
    print(deviation)

    best_weight = user_weight.copy()
    best_weight[-1] = 0.2
    best_weight = best_weight / best_weight.sum()
    deviation = cal_deviation(user_weight, best_weight, V)
    print(deviation)


