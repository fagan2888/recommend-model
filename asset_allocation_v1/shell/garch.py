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
import scipy
import scipy.optimize
import arch
from Const import datapath

logger = logging.getLogger(__name__)

rf = 0.025 / 52

if __name__ == '__main__':

    index = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates = ['date'])
    index = index.fillna(method = 'pad')[['000300.SH']]
    indexr = index.pct_change().fillna(0.0)
    #index['diff'] = index['000300.SH'].diff()
    #index['diff2'] = index['diff'].diff()
    #print index
    #index.to_csv('index_diff.csv')
    am = arch.arch_model(indexr['000300.SH'], p =1 ,q = 1)
    res = am.fit(update_freq = 10)
    print res.summary()
    #print res.params
