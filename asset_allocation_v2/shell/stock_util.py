#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import pandas as pd
import numpy as np
from asset import Asset

from ipdb import set_trace


def cal_stock_portfolio_nav(stock_nav_df, stock_pos, reindex):

    tmp_nav = stock_nav_df.loc[reindex, stock_pos.index]
    tmp_nav = tmp_nav / tmp_nav.iloc[0]
    pnav = np.dot(tmp_nav, stock_pos)
    pnav = pd.Series(data = pnav, index = reindex)

    return pnav

