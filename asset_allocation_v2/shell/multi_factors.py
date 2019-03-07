#coding=utf-8
'''
Created on: Mar. 6, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import click
import warnings
import numpy as np
import pandas as pd
sys.path.append('shell')
from db import caihui_tq_ix_comp, caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic
from trade_date import ATradeDate
from util_timestamp import *
from ipdb import set_trace


logger = logging.getLogger(__name__)


def load_CSI800():

    # df_CSI800_pool = caihui_tq_ix_comp.load_index_constituents('2070000191',date='20171231')
    df_CSI800_constituents = pd.read_csv('data/CSI800_constituents.csv')
    df_CSI800_pool = caihui_tq_sk_basicinfo.load_stock_code_info(stock_codes=df_CSI800_constituents['Constituents Code'], current=False)

    df_CSI800_market_data = caihui_tq_sk_dquoteindic.load_stock_market_data(begin_date='20000101', stock_ids=df_CSI800_pool.index)
    df_CSI800_financial_data = caihui_tq_sk_finindic.load_stock_financial_data(begin_date='20000101', stock_ids=df_CSI800_pool.index)
    df_CSI800_industry = caihui_tq_sk_basicinfo.load_stock_industry(stock_ids=df_CSI800_pool.index)
    set_trace()
    result = pd.merge(df_CSI800_dquote_indic, df_CSI800_financial_data, how='outer', on=['stock_id', 'date'])

    set_trace()


if __name__ == '__main__':

    load_CSI800()

