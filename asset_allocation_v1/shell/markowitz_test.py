#coding=utf8


import os
import string
import click
import logging
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF
import DBData
import DFUtil
import AllocationData
import random
import tensorflow_markowitz as tm

from Const import datapath

logger = logging.getLogger(__name__)

if __name__ == '__main__':


    index = DBData.db_index_value_daily('2010-01-01', '2017-01-20', None)
    index = index.resample('W-FRI').last()
    index = index.fillna(method = 'pad').dropna()
    index = index / index.iloc[0]
    #print index.columns
    #print index.columns
    cols = ['000300.SH', '000905.SH', 'GLNC', 'HSCI.HI', 'SP500.SPI']
    #cols = ['000300.SH', '000905.SH', 'GLNC', 'HSCI.HI', 'SP500.SPI', 'H11001.CSI']
    index = index[cols]
    index.to_csv('index.csv')
    df_inc = index.pct_change().fillna(0.0)

    ds = []
    rs = []
    dates = df_inc.index
    look_back = 52
    interval = 13
    weight = None

    for i in range(look_back, len(dates)):
        d = dates[i]
        if i % interval == 0:
            df_inc = df_inc.iloc[i - look_back : i]
            tm.tensorflow_markowitz(df_inc)
