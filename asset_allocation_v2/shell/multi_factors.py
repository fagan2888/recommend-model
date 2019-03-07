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
from db import caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic
from trade_date import ATradeDate
from util_timestamp import *
from ipdb import set_trace


logger = logging.getLogger(__name__)


def load_CSI800():

    df_CSI800_constinuents = pd.read_csv('data/CSI800_constinuents.csv')
    ser_CSI800_pool = caihui_tq_sk_basicinfo.load_stock_basic_info(stock_codes=df_CSI800_constinuents['Consituents Code'], current=True)
    ser = caihui_tq_sk_basicinfo.load_stock_basic_info(stock_codes=df_CSI800_constinuents['Consituents Code'], current=False)
    set_trace()
    df_CSI800_dquote_indic = caihui_tq_sk_dquoteindic.load_stock_dquote_indic(begin_date='20180101', stock_ids=ser_CSI800_pool.index)
    df_CSI800_fin_indic = caihui_tq_sk_finindic.load_stock_fin_indic(begin_date='20180101', stock_ids=ser_CSI_800_pool.index)
    set_trace()


if __name__ == '__main__':

    load_CSI800()

