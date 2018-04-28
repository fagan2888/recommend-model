#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata, spearmanr
from mk_test import mk_test
from ipdb import set_trace

import DBData
from CommandMarkowitz import load_nav_series


class ValidFactor(object):

    def __init__(self, factor_ids, start_date, end_date):

        self.factor_ids = factor_ids
        # self.base_ids = ['120000001', '120000002', '120000018']
        self.base_ids = ['120000001', '120000002']

        self.asset_navs, self.asset_incs = self.load_asset_nav(factor_ids, start_date, end_date)
        self.base_navs, self.base_incs = self.load_asset_nav(self.base_ids, start_date, end_date)


    def load_asset_nav(self, factor_ids, start_date, end_date):

        trade_dates = DBData.trade_dates(start_date, end_date)
        asset_navs = {}
        for factor_id in factor_ids:
            asset_navs[factor_id] = load_nav_series(factor_id, reindex = trade_dates)

        df_asset_navs = pd.DataFrame(asset_navs)
        df_asset_incs = df_asset_navs.pct_change().dropna()

        return df_asset_navs, df_asset_incs


    def cal_base_beta(self):

        for asset in self.factor_ids:
            betas = []
            for base in self.base_ids:
                df_asset = self.asset_incs[asset].values
                df_base = self.base_incs[[base]].values
                lr = LinearRegression()
                lr.fit(df_base, df_asset)
                betas.append(lr.coef_[0])

            betas = np.array(betas)
            invalid_betas = betas[(betas >= 0.9) & (betas < 1.1)]
            if len(invalid_betas) > 0:
                print asset, np.round(betas, 5), 0, betas.argmax()
            else:
                print asset, np.round(betas, 5), 1, betas.argmax()
            # print asset, np.round(betas, 2)


    def cal_rank_coef(self):

        # spearmanr between last week and next week
        # asset_rank = self.asset_incs.apply(rankdata, 1)
        # last_row = None
        # for idx, row in self.asset_incs.iterrows():
        #     if last_row is None:
        #         last_row = row
        #         continue
        #     print idx, spearmanr(last_row, row)

        # spearmanr between last 3m and next 3m
        # dates = self.asset_navs.index
        # coefs = []
        # ps = []
        # for i in range(13, len(self.asset_navs)-13):
        #     past_ret = self.asset_navs.iloc[i]/self.asset_navs.iloc[i-13] - 1
        #     future_ret = self.asset_navs.iloc[i+13]/self.asset_navs.iloc[i] - 1
        #     coef = spearmanr(past_ret, future_ret)
        #     coefs.append(coef[0])
        #     ps.append(coef[1])
        #     print dates[i], round(coef[0], 2), round(coef[1], 2)

        dates = self.asset_navs.index
        best_ranks = []
        last_weeks = 52
        next_weeks = 13
        for i in range(last_weeks, len(self.asset_navs)-next_weeks):
            past_ret = self.asset_navs.iloc[i]/self.asset_navs.iloc[i-last_weeks] - 1
            future_ret = self.asset_navs.iloc[i+next_weeks]/self.asset_navs.iloc[i] - 1
            # best_factor = past_ret[past_ret >= sorted(past_ret)[-1]].index
            best_factor = past_ret[past_ret <= sorted(past_ret)[0]].index
            best_rank = pd.Series(data = rankdata(future_ret), index = future_ret.index)
            best_rank = best_rank.loc[best_factor]
            best_ranks.append(np.mean(best_rank))
            print dates[i], np.mean(best_rank), best_rank.values
        print np.mean(best_ranks)


    def trend_test(self):

        dates = self.asset_navs.index
        window = 52
        for asset in factor_ids:
            asset_nav = self.asset_navs[asset]
            asset_nav = asset_nav.to_frame(name = 'nav')
            signals = []
            for i in range(window, len(dates)):
                tmp_asset_nav = asset_nav.iloc[i-window:i]
                date = dates[i-1]
                mk_test_res = mk_test(tmp_asset_nav.values)
                signals.append(mk_test_res[1])
                print date, mk_test(tmp_asset_nav.values)
            asset_nav_signal = asset_nav.iloc[window:]
            asset_nav_signal['signal'] = signals
            asset_nav_increasing = asset_nav_signal[asset_nav_signal.signal == 1]
            asset_nav_notrend = asset_nav_signal[asset_nav_signal.signal == 0]
            asset_nav_decreasing = asset_nav_signal[asset_nav_signal.signal == -1]
            fig = plt.figure(figsize = (30,20))
            ax = fig.add_subplot(111)
            ax.plot(asset_nav_increasing.index, asset_nav_increasing.nav, '.', color = 'red', markersize = 10)
            ax.plot(asset_nav_notrend.index, asset_nav_notrend.nav, '.', color = 'blue', markersize = 10)
            ax.plot(asset_nav_decreasing.index, asset_nav_decreasing.nav, '.', color = 'green', markersize = 10)
            fig.savefig('png/%s_trend.png'%asset)

            # asset_nav = self.asset_navs[asset].values
            # print asset, mk_test(asset_nav)


    def handle(self):
        # self.cal_base_beta()
        # self.cal_rank_coef()
        self.trend_test()


if __name__ == '__main__':

    factor_ids = ['1200000%d'%i for i in range(52, 80)]
    # start_date = '2004-01-01'
    # start_date = '2010-06-01'
    start_date = '2005-06-01'
    end_date = '2018-01-22'
    vf = ValidFactor(factor_ids, start_date, end_date)
    vf.handle()
