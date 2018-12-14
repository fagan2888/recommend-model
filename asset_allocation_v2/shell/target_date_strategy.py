# -*- coding: utf-8 -*-

import sys
sys.path.append('shell')
import pandas as pd
import numpy as np
from ipdb import set_trace
from dateutil import rrule
from scipy.stats import norm
from scipy.optimize import minimize

from asset import Asset
from db import asset_on_online_nav

class TargetDateStrategy():

    def __init__(self, invest_horison, target_asset, p, ret, sigma, initial_amount=0, invest_periods=24):
        self.invest_horison = invest_horison
        self.target_asset = target_asset
        self.p = p
        self.ret = ret
        self.sigma = sigma
        self.initial_amount = initial_amount
        self.invest_periods = invest_periods

    def cal_glide_path(self):
        m = []
        s = []
        ret_t = []
        asset_num = len(self.ret)
        w0 = np.array([1/asset_num]*asset_num)
        df_ws = pd.DataFrame(columns=['stock', 'bond', 'money'])
        for i in range(self.invest_horison, 0, -1):
            cons = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x}
            )
            res = minimize(self.quantile_func, w0, args=[m, s], method='SLSQP', constraints=cons, options={'disp': False, 'eps': 1e-10})
            w = res.x
            ret_t.insert(0, res.fun)
            m1 = np.dot(self.ret, w)
            s1 = np.sqrt(np.dot(np.dot(w, self.sigma), w))
            m.insert(0, m1)
            s.insert(0, s1)
            df_ws.loc[i] = w
            # print(w)
            # print(df_ws)
        self.m = m
        self.s = s
        df_ws = df_ws.sort_index()
        self.df_ws = df_ws
        self.df_ret_t = 1 / pd.Series(data=ret_t)
        df_ws.to_csv('data/tds/df_weight.csv', index_label='decision_date')
        return df_ws

    def cal_invest_amount(self):

        total_multiple = sum(self.df_ret_t[:self.invest_periods])
        target_amount = self.target_asset - self.initial_amount*self.df_ret_t.loc[0]
        invest_amount = target_amount / total_multiple
        self.invest_amount = invest_amount

        return invest_amount

    def quantile_func(self, w, pars):
        m = pars[0]
        s = pars[1]

        m1 = np.dot(self.ret, w)
        s1 = np.sqrt(np.dot(np.dot(w, self.sigma), w))
        m = np.append(m1, m)
        s = np.append(s1, s)

        C1 = (1+m).prod()
        C2 = (1+(s/(1+m))**2).prod()
        Q = np.sqrt(C2) / C1 * np.exp(norm.ppf(self.p) * np.sqrt(np.log(C2)))
        return Q

    def return_dist(self, p, m, s):
        C1 = (1+m).prod()
        C2 = (1+(s/(1+m))**2).prod()
        Q = np.sqrt(C2) / C1 * np.exp(norm.ppf(p) * np.sqrt(np.log(C2)))
        retQ = 1 / Q
        return retQ

    def cal_return_dist(self):
        df_return_dist = pd.DataFrame(columns=['return'])
        m = np.array(self.m)
        s = np.array(self.s)
        for p in np.arange(0.01, 1.00, 0.01):
            df_mul = pd.Series()
            for i in range(self.invest_periods):
                retQ = self.return_dist(p, m[i:], s[i:])
                # df_return_dist.loc[p] = retQ**(1/8)-1
                df_mul.loc[i] = retQ
            final_amount = df_mul.sum()*self.invest_amount
            df_return_dist.loc[p] = final_amount
            # print(df_return_dist)
        df_return_dist.to_csv('data/tds/df_return_dist.csv', index_label='prob')
        return df_return_dist


def load_ret_sigma(end_date):

    drate = [0.6, 0.8, 0.8]
    # drate = [0.6, 1, 1]
    df_stock = asset_on_online_nav.load_series('800000', 8)
    df_stock_all = df_stock.copy()
    df_stock = df_stock[df_stock.index <= end_date]
    stock_months = (df_stock.index[-1] - df_stock.index[0]).days / 30
    stock_ret = (df_stock.iloc[-1]-1)*drate[0]+1
    stock_ret = stock_ret**(1 / stock_months) - 1
    if stock_ret > 0.005:
        stock_ret = 0.005

    df_bond = Asset.load_nav_series('120000009')
    df_bond_all = df_bond.copy()
    df_bond = df_bond[df_bond.index <= end_date]
    df_bond = df_bond / df_bond.iloc[0]
    bond_months = (df_bond.index[-1] - df_bond.index[0]).days / 30
    bond_ret = (df_bond.iloc[-1]-1)*drate[1]+1
    bond_ret = bond_ret**(1 / bond_months) - 1

    df_money = Asset.load_nav_series('120000039')
    df_money_all = df_money.copy()
    df_money = df_money[df_money.index <= end_date]
    df_money = df_money / df_money.iloc[0]
    money_months = (df_money.index[-1] - df_money.index[0]).days / 30
    money_ret = (df_money.iloc[-1]-1)*drate[2]+1
    money_ret = money_ret**(1 / money_months) - 1

    df_nav = pd.concat([df_stock_all, df_bond_all, df_money_all], 1).dropna()
    df_ret = df_nav.pct_change()
    df_ret = df_ret.resample('m').sum()
    df_ret.columns = ['stock', 'bond', 'money']
    sigma = df_ret.cov()
    ret = np.array([stock_ret, bond_ret, money_ret])

    # sigma = sigma.iloc[:2, :2]
    # ret = ret[:2]

    return ret, sigma


def load_monthly_ret():

    df_stock = asset_on_online_nav.load_series('800000', 8)
    df_stock = df_stock.pct_change()
    df_stock = df_stock.resample('m').sum()
    df_stock = df_stock.loc['2012-09':'2018-08']

    df_bond = Asset.load_nav_series('120000009')
    df_bond = df_bond.pct_change()
    df_bond = df_bond.resample('m').sum()
    df_bond = df_bond.loc['2012-09':'2018-08']

    df_money = Asset.load_nav_series('120000039')
    df_money = df_money.pct_change()
    df_money = df_money.resample('m').sum()
    df_money = df_money.loc['2012-09':'2018-08']

    return df_stock, df_bond, df_money


def main():

    ret, sigma = load_ret_sigma(end_date='2018-10-31')
    invest_horison = 96
    invest_periods = 96
    target_amount = 225
    initial_amount = 0
    p = 0.98
    tds = TargetDateStrategy(invest_horison, target_amount, p, ret, sigma, initial_amount=initial_amount, invest_periods=invest_periods)
    tds.cal_glide_path()
    invest_amount = tds.cal_invest_amount()
    print({'invest_amount': invest_amount})
    tds.cal_return_dist()


def test():

    df_stock, df_bond, df_money = load_monthly_ret()
    # ret, sigma = load_ret_sigma()
    invest_horison = 72
    # invest_periods = 72
    target_amount = 225
    initial_amount = 0
    p = 0.80
    df_invest_amount = pd.Series()
    df_total_amount = pd.Series()
    df_ws = pd.DataFrame(columns=['stock', 'bond', 'money'])
    for i in range(invest_horison, 0, -1):
        date = df_stock.index[invest_horison - i]
        ret, sigma = load_ret_sigma(end_date=date)
        invest_periods = invest_horison
        tds = TargetDateStrategy(i, target_amount, p, ret, sigma, initial_amount=initial_amount, invest_periods=invest_periods)
        tds.cal_glide_path()
        invest_amount = tds.cal_invest_amount()
        ws = tds.df_ws.iloc[0]
        asset_ret = [df_stock.loc[date], df_bond.loc[date], df_money.loc[date]]
        earning = np.dot(ws.values, asset_ret)
        initial_amount = (invest_amount + initial_amount) * (1 + earning)
        print(date, initial_amount, invest_amount, ws.values)
        df_invest_amount.loc[date] = invest_amount
        df_total_amount.loc[date] = initial_amount
        df_ws.loc[date] = ws
    cf = df_invest_amount.values
    cf = np.append(-cf, target_amount)
    irr = np.irr(cf)
    print({'irr': irr})
    df_total_amount.to_csv('df_total_amount.csv')
    df_invest_amount.to_csv('df_invest_amount.csv')
    df_ws.to_csv('df_ws.csv')
    # tds.cal_return_dist()


if __name__ == '__main__':
    # main()
    test()


