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

    deviation1 = np.abs(RC_user - RC_best).sum()
    deviation2 = np.abs((user_weight - best_weight)).sum()
    # deviation = deviation + np.abs(sigma_user / sigma_best - 1)
    print(deviation1, deviation2)
    deviation = (deviation1 + deviation2) * 0.1

    return deviation


if __name__ == '__main__':

    # asset_ids_1 = ['120000010', '120000011', '120000014', '120000039', '120000053', '120000056', '120000058', '120000073', '120000081', '120000082']
    # asset_ids_2 = ['ERI000001', 'ERI000002', 'MZ.FA0010', 'MZ.FA0050', 'MZ.FA0070', 'MZ.FA1010']
    # asset_ids = asset_ids_1 + asset_ids_2
    # df_ret = load_asset_ret(asset_ids)
    # V = df_ret.cov()
    # user_weight = np.array([1.0 / len(asset_ids)] * len(asset_ids))  # 0.0625 for all factor
    # print(user_weight)

    # best_weight = user_weight.copy()
    # best_weight[0] = 0.2
    # best_weight = best_weight / best_weight.sum()
    # print(best_weight)
    # deviation = cal_deviation(user_weight, best_weight, V)
    # print(deviation)

    # best_weight = user_weight.copy()
    # best_weight[0] += 0.05
    # best_weight[1] -= 0.05
    # deviation = cal_deviation(user_weight, best_weight, V)
    # print(deviation)

    # best_weight = user_weight.copy()
    # best_weight[-1] = 0.2
    # best_weight = best_weight / best_weight.sum()
    # deviation = cal_deviation(user_weight, best_weight, V)
    # print(deviation)

    # user_weight = [0,0,0,0,0,0,0.0020745263283731,0,0.2346006071113,0.06713850608467,0.31198286644651,0.23212020382551,0.0,0.098030725190447,0.0]
    # best_weight = [0.0069, 0.0067,0.0066,0.0062,0.0067,0.0109,0.0082,0,0.3737,0.0728,0.1333,0.1113,0.141,0.0477,0.068]

    # user_weight = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4229740041536,0.10147138931187,0.14071549729879,0.0713686823611,0.0,0.15284559148429,0.0]
    # user_weight = [0.0,0.0202,0.0098,0.0091,0.0099,0.0159,0.012,0.0,0.375,0.0786,0.1094,0.0889,0.1567,0.0446,0.0699]
    # best_weight = [0.0103,0.0099,0.0098,0.0091,0.0099,0.0159,0.012,0.0,0.375,0.0786,0.1094,0.0889,0.1567,0.0446,0.0699]

    user_weight = [0.0085, 0,0,0,0,0,0,0.0168,0.0413,0.0195,0.4748,0.3949,0,0.0422,0]
    best_weight = [0.0050, 0.0048, 0.0047, 0.0044, 0.0048, 0.0077, 0.0058, 0.0194, 0, 0.036, 0.2485, 0.2284, 0.3137, 0.0889, 0.0279]

    user_weight = np.array(user_weight)
    best_weight = np.array(best_weight)
    df_cov = pd.read_csv('ra_pool_cov.csv')
    df_cov = df_cov.set_index(['ra_poola_id', 'ra_poolb_id'])
    df_cov = df_cov[['ra_cov']]
    df_cov = df_cov.unstack()
    df_cov.columns = df_cov.columns.levels[1]

    score = cal_deviation(user_weight, best_weight, df_cov)
    score = 100*max(0, 1 - 0.5*score)
    print(score)
    set_trace()

