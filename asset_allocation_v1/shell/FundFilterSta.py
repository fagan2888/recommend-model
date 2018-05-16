#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import sys
sys.path.append('shell')
import pandas as pd
import numpy as np
from ipdb import set_trace
from scipy.stats import rankdata, ttest_ind

from db import *
from trade_date import ATradeDate

class FundFilterSta(object):


    def __init__(self, begin_date, end_date, code = None):
        if code is None:
            self.code = base_ra_fund.find_type_fund(1).ra_code.values
        self.nav = self.get_nav(begin_date, end_date, self.code)
        self.dates = ATradeDate.trade_date('2000-01-01', end_date)
        self.trade_dates = self.nav.index[10:]
        self.lookback = 126


    def get_nav(self, begin_date, end_date, code):

        # df_nav_fund = base_ra_fund_nav.load_daily(begin_date, end_date, codes=code)
        # df_nav_fund.to_csv('data/fund.csv', index_label = 'date')
        df_nav_fund = pd.read_csv('data/fund.csv', index_col = ['date'], parse_dates = ['date'])

        return df_nav_fund


    def cal_tshare(self):
        # df_secode = caihui_fund.get_secode(self.code)
        # racode_2_secode = df_secode.secode.to_dict()
        # secode = df_secode.secode.ravel()
        # secode_2_racode = {v:k for (k,v) in racode_2_secode.iteritems()}
        # fund_tshare = caihui_fund.get_tshare(secode, self.trade_dates)
        # fund_tshare = fund_tshare.rename(columns = secode_2_racode)
        # fund_tshare.to_csv('data/fund_tshare.csv', index_label = 'date')
        fund_tshare = pd.read_csv('data/fund_tshare.csv', index_col = ['date'], parse_dates = ['date'])

        df_ret = pd.DataFrame(columns = ['layer%d'%i for i in range(5)])
        df_valid = pd.DataFrame(columns = ['layer%d'%i for i in range(5)])
        for sdate, edate in zip(self.trade_dates[:-self.lookback], self.trade_dates[self.lookback:]):
            tmp_tshare = fund_tshare.loc[sdate]
            tmp_tshare = tmp_tshare.replace(0.0, np.nan)
            tmp_tshare = tmp_tshare.dropna()

            tmp_nav = self.nav.loc[sdate:edate]
            tmp_nav = tmp_nav.dropna(1)
            tmp_ret = tmp_nav.iloc[-1] / tmp_nav.iloc[0] - 1

            tmp_tshare.name = 'tshare'
            tmp_ret.name = 'ret'
            df_tshare_ret = pd.concat([tmp_tshare, tmp_ret], 1)
            df_tshare_ret = df_tshare_ret.dropna()
            df_tshare_ret = df_tshare_ret[df_tshare_ret.tshare > 2e8]
            df_tshare_ret = df_tshare_ret.sort_values('tshare')
            # fund_ret_mean = df_tshare_ret.ret.mean()

            fund_num = len(df_tshare_ret)
            layer_num = fund_num / 5
            layer_ret = []
            layer_valid = []
            for layer in range(5):
                tmp_tshare_ret = df_tshare_ret.iloc[layer*layer_num:(layer+1)*layer_num]
                tmp_tshare_ret_mean = tmp_tshare_ret.ret.mean()
                t_stat = ttest_ind(tmp_tshare_ret.ret.values, df_tshare_ret.ret.values)
                if t_stat[0] > 0 and t_stat[1] < 0.05:
                    layer_valid.append(1)
                else:
                    layer_valid.append(0)
                layer_ret.append(tmp_tshare_ret_mean)
            df_ret.loc[sdate] = layer_ret
            df_valid.loc[sdate] = layer_valid
            print df_ret.tail(1)
            print df_valid.tail(1)
        set_trace()
        df_ret.to_csv('data/tshare_ret_sta.csv', index_label = 'date')
        df_valid.to_csv('data/tshare_valid_sta.csv', index_label = 'date')


    def cal_tshare_layer_nav(self):
        # df_secode = caihui_fund.get_secode(self.code)
        # racode_2_secode = df_secode.secode.to_dict()
        # secode = df_secode.secode.ravel()
        # secode_2_racode = {v:k for (k,v) in racode_2_secode.iteritems()}
        # fund_tshare = caihui_fund.get_tshare(secode, self.trade_dates)
        # fund_tshare = fund_tshare.rename(columns = secode_2_racode)
        # fund_tshare.to_csv('data/fund_tshare.csv', index_label = 'date')
        fund_tshare = pd.read_csv('data/fund_tshare.csv', index_col = ['date'], parse_dates = ['date'])

        df_ret = pd.DataFrame(columns = ['benchmark'] + ['layer%d'%i for i in range(5)])
        # df_valid = pd.DataFrame(columns = ['benchmark'] +['layer%d'%i for i in range(5)])
        for sdate, edate in zip(self.trade_dates[:-self.lookback], self.trade_dates[self.lookback:]):
            tmp_tshare = fund_tshare.loc[sdate]
            tmp_tshare = tmp_tshare.replace(0.0, np.nan)
            tmp_tshare = tmp_tshare.dropna()

            tmp_nav = self.nav.loc[sdate:edate]
            tmp_nav = tmp_nav.dropna(1)
            tmp_ret = tmp_nav.iloc[1] / tmp_nav.iloc[0] - 1

            tmp_tshare.name = 'tshare'
            tmp_ret.name = 'ret'
            df_tshare_ret = pd.concat([tmp_tshare, tmp_ret], 1)
            df_tshare_ret = df_tshare_ret.dropna()
            df_tshare_ret = df_tshare_ret[df_tshare_ret.tshare > 2e8]
            df_tshare_ret = df_tshare_ret.sort_values('tshare')
            # fund_ret_mean = df_tshare_ret.ret.mean()

            fund_num = len(df_tshare_ret)
            layer_num = fund_num / 5
            tmp_layer_ret = [tmp_ret.mean()]
            for layer in range(5):
                tmp_tshare_ret = df_tshare_ret.iloc[layer*layer_num:(layer+1)*layer_num]
                tmp_tshare_ret_mean = tmp_tshare_ret.ret.mean()
                tmp_layer_ret.append(tmp_tshare_ret_mean)

            df_ret.loc[edate] = tmp_layer_ret

            print df_ret.tail(1)
        set_trace()
        df_nav = (1 + df_ret).cumprod()
        df_nav.to_csv('data/tshare_layer_nav.csv', index_label = 'date')


    def cal_iratio(self):
        # df_secode = caihui_fund.get_secode(self.code)
        # racode_2_secode = df_secode.secode.to_dict()
        # secode = df_secode.secode.ravel()
        # secode_2_racode = {v:k for (k,v) in racode_2_secode.iteritems()}
        # fund_tshare = caihui_fund.get_iratio(secode, self.trade_dates)
        # fund_tshare = fund_tshare.rename(columns = secode_2_racode)
        # fund_tshare.to_csv('data/fund_iratio.csv', index_label = 'date')
        fund_tshare = pd.read_csv('data/fund_iratio.csv', index_col = ['date'], parse_dates = ['date'])

        df_ret = pd.DataFrame(columns = ['layer%d'%i for i in range(5)])
        df_valid = pd.DataFrame(columns = ['layer%d'%i for i in range(5)])
        for sdate, edate in zip(self.trade_dates[:-self.lookback], self.trade_dates[self.lookback:]):
            tmp_tshare = fund_tshare.loc[sdate]
            tmp_tshare = tmp_tshare.replace(0.0, np.nan)
            tmp_tshare = tmp_tshare.dropna()

            tmp_nav = self.nav.loc[sdate:edate]
            tmp_nav = tmp_nav.dropna(1)
            tmp_ret = tmp_nav.iloc[-1] / tmp_nav.iloc[0] - 1

            tmp_tshare.name = 'iratio'
            tmp_ret.name = 'ret'
            df_tshare_ret = pd.concat([tmp_tshare, tmp_ret], 1)
            df_tshare_ret = df_tshare_ret.dropna()
            df_tshare_ret = df_tshare_ret.sort_values('iratio')
            # fund_ret_mean = df_tshare_ret.ret.mean()

            fund_num = len(df_tshare_ret)
            layer_num = fund_num / 5
            layer_ret = []
            layer_valid = []
            for layer in range(5):
                tmp_tshare_ret = df_tshare_ret.iloc[layer*layer_num:(layer+1)*layer_num]
                tmp_tshare_ret_mean = tmp_tshare_ret.ret.mean()
                t_stat = ttest_ind(tmp_tshare_ret.ret.values, df_tshare_ret.ret.values)
                if t_stat[0] > 0 and t_stat[1] < 0.05:
                    layer_valid.append(1)
                else:
                    layer_valid.append(0)
                layer_ret.append(tmp_tshare_ret_mean)
            df_ret.loc[sdate] = layer_ret
            df_valid.loc[sdate] = layer_valid
            print df_ret.tail(1)
            print df_valid.tail(1)
        set_trace()
        df_ret.to_csv('data/iratio_ret_sta.csv', index_label = 'date')
        df_valid.to_csv('data/iratio_valid_sta.csv', index_label = 'date')


    def cal_iratio_layer_nav(self):
        # df_secode = caihui_fund.get_secode(self.code)
        # racode_2_secode = df_secode.secode.to_dict()
        # secode = df_secode.secode.ravel()
        # secode_2_racode = {v:k for (k,v) in racode_2_secode.iteritems()}
        # fund_tshare = caihui_fund.get_tshare(secode, self.trade_dates)
        # fund_tshare = fund_tshare.rename(columns = secode_2_racode)
        # fund_tshare.to_csv('data/fund_tshare.csv', index_label = 'date')
        fund_tshare = pd.read_csv('data/fund_iratio.csv', index_col = ['date'], parse_dates = ['date'])

        df_ret = pd.DataFrame(columns = ['benchmark'] + ['layer%d'%i for i in range(5)])
        # df_valid = pd.DataFrame(columns = ['benchmark'] +['layer%d'%i for i in range(5)])
        for sdate, edate in zip(self.trade_dates[:-self.lookback], self.trade_dates[self.lookback:]):
            tmp_tshare = fund_tshare.loc[sdate]
            tmp_tshare = tmp_tshare.replace(0.0, np.nan)
            tmp_tshare = tmp_tshare.dropna()

            tmp_nav = self.nav.loc[sdate:edate]
            tmp_nav = tmp_nav.dropna(1)
            tmp_ret = tmp_nav.iloc[1] / tmp_nav.iloc[0] - 1

            tmp_tshare.name = 'iratio'
            tmp_ret.name = 'ret'
            df_tshare_ret = pd.concat([tmp_tshare, tmp_ret], 1)
            df_tshare_ret = df_tshare_ret.dropna()
            df_tshare_ret = df_tshare_ret.sort_values('iratio')
            # fund_ret_mean = df_tshare_ret.ret.mean()

            fund_num = len(df_tshare_ret)
            layer_num = fund_num / 5
            tmp_layer_ret = [tmp_ret.mean()]
            for layer in range(5):
                tmp_tshare_ret = df_tshare_ret.iloc[layer*layer_num:(layer+1)*layer_num]
                tmp_tshare_ret_mean = tmp_tshare_ret.ret.mean()
                tmp_layer_ret.append(tmp_tshare_ret_mean)

            df_ret.loc[edate] = tmp_layer_ret

            print df_ret.tail(1)
        set_trace()
        df_ret = df_ret.dropna()
        df_nav = (1 + df_ret).cumprod()
        df_nav.to_csv('data/iratio_layer_nav.csv', index_label = 'date')


    def cal_totyears(self):
        df_secode = caihui_fund.get_secode(self.code)
        racode_2_secode = df_secode.secode.to_dict()
        secode = df_secode.secode.ravel()
        secode_2_racode = {v:k for (k,v) in racode_2_secode.iteritems()}
        # fund_tshare = caihui_fund.get_totyears(secode, self.trade_dates)
        fund_tshare = pd.read_csv('data/m_totyears.csv', index_col = ['date'], parse_dates = ['date'])
        fund_tshare = fund_tshare.rename(columns = secode_2_racode)
        # fund_tshare.to_csv('data/fund_totyears.csv', index_label = 'date')
        # fund_tshare = pd.read_csv('data/fund_totyears.csv', index_col = ['date'], parse_dates = ['date'])

        df_ret = pd.DataFrame(columns = ['layer%d'%i for i in range(5)])
        df_valid = pd.DataFrame(columns = ['layer%d'%i for i in range(5)])
        for sdate, edate in zip(self.trade_dates[:-self.lookback], self.trade_dates[self.lookback:]):
            tmp_tshare = fund_tshare.loc[sdate]
            tmp_tshare = tmp_tshare.replace(0.0, np.nan)
            tmp_tshare = tmp_tshare.dropna()

            tmp_nav = self.nav.loc[sdate:edate]
            tmp_nav = tmp_nav.dropna(1)
            tmp_ret = tmp_nav.iloc[-1] / tmp_nav.iloc[0] - 1

            tmp_tshare.name = 'iratio'
            tmp_ret.name = 'ret'
            df_tshare_ret = pd.concat([tmp_tshare, tmp_ret], 1)
            df_tshare_ret = df_tshare_ret.dropna()
            df_tshare_ret = df_tshare_ret.sort_values('iratio')
            # fund_ret_mean = df_tshare_ret.ret.mean()

            fund_num = len(df_tshare_ret)
            layer_num = fund_num / 5
            layer_ret = []
            layer_valid = []
            for layer in range(5):
                tmp_tshare_ret = df_tshare_ret.iloc[layer*layer_num:(layer+1)*layer_num]
                tmp_tshare_ret_mean = tmp_tshare_ret.ret.mean()
                t_stat = ttest_ind(tmp_tshare_ret.ret.values, df_tshare_ret.ret.values)
                if t_stat[0] > 0 and t_stat[1] < 0.05:
                    layer_valid.append(1)
                else:
                    layer_valid.append(0)
                layer_ret.append(tmp_tshare_ret_mean)
            df_ret.loc[sdate] = layer_ret
            df_valid.loc[sdate] = layer_valid
            print df_ret.tail(1)
            print df_valid.tail(1)
        set_trace()
        df_ret.to_csv('data/myears_ret_sta.csv', index_label = 'date')
        df_valid.to_csv('data/myears_valid_sta.csv', index_label = 'date')


    def cal_totyears_layer_nav(self):
        df_secode = caihui_fund.get_secode(self.code)
        racode_2_secode = df_secode.secode.to_dict()
        # secode = df_secode.secode.ravel()
        secode_2_racode = {v:k for (k,v) in racode_2_secode.iteritems()}
        # fund_tshare = caihui_fund.get_tshare(secode, self.trade_dates)
        # fund_tshare = fund_tshare.rename(columns = secode_2_racode)
        # fund_tshare.to_csv('data/fund_tshare.csv', index_label = 'date')
        fund_tshare = pd.read_csv('data/m_totyears.csv', index_col = ['date'], parse_dates = ['date'])
        fund_tshare = fund_tshare.rename(columns = secode_2_racode)

        df_ret = pd.DataFrame(columns = ['benchmark'] + ['layer%d'%i for i in range(5)])
        # df_valid = pd.DataFrame(columns = ['benchmark'] +['layer%d'%i for i in range(5)])
        for sdate, edate in zip(self.trade_dates[:-self.lookback], self.trade_dates[self.lookback:]):
            tmp_tshare = fund_tshare.loc[sdate]
            tmp_tshare = tmp_tshare.replace(0.0, np.nan)
            tmp_tshare = tmp_tshare.dropna()

            tmp_nav = self.nav.loc[sdate:edate]
            tmp_nav = tmp_nav.dropna(1)
            tmp_ret = tmp_nav.iloc[1] / tmp_nav.iloc[0] - 1

            tmp_tshare.name = 'totyears'
            tmp_ret.name = 'ret'
            df_tshare_ret = pd.concat([tmp_tshare, tmp_ret], 1)
            df_tshare_ret = df_tshare_ret.dropna()
            df_tshare_ret = df_tshare_ret.sort_values('totyears')
            # fund_ret_mean = df_tshare_ret.ret.mean()

            fund_num = len(df_tshare_ret)
            layer_num = fund_num / 5
            tmp_layer_ret = [tmp_ret.mean()]
            for layer in range(5):
                tmp_tshare_ret = df_tshare_ret.iloc[layer*layer_num:(layer+1)*layer_num]
                tmp_tshare_ret_mean = tmp_tshare_ret.ret.mean()
                tmp_layer_ret.append(tmp_tshare_ret_mean)

            df_ret.loc[edate] = tmp_layer_ret

            print df_ret.tail(1)
        set_trace()
        df_ret = df_ret.dropna()
        df_nav = (1 + df_ret).cumprod()
        df_nav.to_csv('data/mtotyears_layer_nav.csv', index_label = 'date')


    def factor_sta(self):
        df_sta = pd.read_csv('data/tshare_ret_sta.csv', index_col = ['date'], parse_dates = ['date'])
        df_sta = df_sta.dropna()
        df_rank = df_sta.apply(rankdata, 1)
        set_trace()


    def handle(self):
        # self.cal_tshare()
        # self.cal_iratio()
        # self.cal_totyears()

        # self.cal_tshare_layer_nav()
        # self.cal_iratio_layer_nav()
        self.cal_totyears_layer_nav()

        # self.factor_sta()



if __name__ == '__main__':

    begin_date = '2009-01-01'
    end_date = '2018-05-15'
    ffs = FundFilterSta(begin_date, end_date)
    ffs.handle()
