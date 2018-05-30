#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('shell/')
from ipdb import set_trace
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA

from db import *
from trade_date import ATradeDate
from util_optimize import ConstrRegression




def tshare_filter(fund_nav):

    secode = caihui_fund.get_secode(fund_nav.columns)
    tshare = caihui_fund.get_tshare(secode.secode.ravel(), end_date = fund_nav.index[-1])
    tshare = tshare.tail(1).T
    tshare.columns = ['tshare']
    secode = secode.reset_index()
    tshare = tshare.reset_index()
    df = pd.merge(tshare, secode)
    df = df[df.tshare > 2e8]
    fund_nav = fund_nav.loc[:, df.ra_code]

    return fund_nav
