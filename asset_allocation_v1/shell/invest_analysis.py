#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import sys
sys.path.append('shell')
from db import *
from ipdb import set_trace


def stock_bond_balance():

    dfs = base_ra_index_nav.load_series('120000016', begin_date = '2015-05-31').pct_change().fillna(0.0)
    dfb = base_ra_index_nav.load_series('120000010', begin_date = '2015-05-31').pct_change().fillna(0.0)
    dfp = 0.5*dfs + 0.5*dfb
    dfp_nav = (1 + dfp).cumprod()
    dfp_nav.to_csv('data/requirement/stock_bond_balance.csv', index_label = 'date')


def stock_fi():

    dfs = base_ra_index_nav.load_series('120000016', begin_date = '2015-05-31').pct_change().fillna(0.0)
    dfm = base_ra_index_nav.load_series('120000039', begin_date = '2015-05-31').pct_change().fillna(0.0)
    dfm = dfm.reindex(dfs.index).fillna(method = 'pad')
    dfw = pd.Series(1.0/36, index = pd.date_range('2015-5-31', '2018-5-31', freq = 'm'))
    dfw = dfw.cumsum()
    dfw.iloc[-1] = 1
    dfw = dfw.resample('d').fillna(method = 'pad')
    dfw = dfw.iloc[:-1]
    dfp = dfs*dfw + dfm*(1-dfw)
    dfp = dfp.fillna(0.0)
    dfp = (1+dfp).cumprod()
    dfp.to_csv('data/requirement/stock_fi_3.csv', index_label = 'date')


def stock_fi_1():

    dfs = base_ra_index_nav.load_series('120000016', begin_date = '2015-05-31').pct_change().fillna(0.0)
    dfm = base_ra_index_nav.load_series('120000039', begin_date = '2015-05-31').pct_change().fillna(0.0)
    dfm = dfm.reindex(dfs.index).fillna(method = 'pad')
    dfw = pd.Series(1.0/12, index = pd.date_range('2015-5-31', '2016-5-31', freq = 'm'))
    dfw = dfw.cumsum()
    dfw.iloc[-1] = 1
    dfw = dfw.reindex(dfs.index).fillna(method = 'pad')
    dfp = dfs*dfw + dfm*(1-dfw)
    dfp = dfp.fillna(0.0)
    dfp = (1+dfp).cumprod()
    dfp.to_csv('data/requirement/stock_fi_1.csv', index_label = 'date')


if __name__ == '__main__':

    stock_bond_balance()
    stock_fi()
    stock_fi_1()
