#coding=utf8

import sys
import logging
import pandas as pd
import numpy as np
import datetime
import calendar
import heapq
import TradeNav
import readline
import pdb



from datetime import datetime, timedelta
from sqlalchemy import *
from util.xdebug import dd

from db import *

logger = logging.getLogger(__name__)
readline.parse_and_bind('tab: complete')


class Investor(object):
    def __init__(self, df_ts_order):

        self.df_ts_order = df_ts_order.set_index(['ts_placed_date'], drop=False)
        
        # # 所有用到的基金ID
        # fund_ids = df_pos.index.get_level_values(1).unique()
        # # dd(fund_ids)

        # #
        # # 赎回费率，延迟加载，因为跟持仓基金有关
        # self.df_redeem_fee = base_fund_fee.load_redeem(fund_ids)

        # # 购买费率，延迟加载，因为跟持仓基金有关
        # self.df_buy_fee = base_fund_fee.load_buy(fund_ids)
        # # dd(self.df_buy_fee.head(), self.df_redeem_fee.head())

    def get_sdate(self):
        
        self.sdate =  pd.to_datetime(self.df_ts_order.index.get_level_values(0).min())
        return self.sdate

    def get_edate(self):
        return None

    def get_buys(self, day):
        orders = [];

        if day in self.df_ts_order.index:
            df_tmp = self.df_ts_order.loc[[day], :]
            for k, v in df_tmp.iterrows():
                if v['ts_trade_type'] == 3:
                    dt = pd.to_datetime(datetime.combine(day, v['ts_placed_time']))

                    order = {'dt': dt, 'ts_txn_id': v['ts_txn_id'], 'argv': v}
                    orders.append(order)
        return orders

    def get_redeems(self, day):
        redeems = [];
        return redeems;

    
    def get_adjusts(self, day):
        orders = []
        if day in self.df_ts_order.index:
            df_tmp = self.df_ts_order.loc[[day], :]
            for k, v in df_tmp.iterrows():
                if v['ts_trade_type'] == 6:
                    dt = pd.to_datetime(datetime.combine(day, v['ts_placed_time']))

                    order = {'dt': dt, 'ts_txn_id': v['ts_txn_id'], 'argv': v}
                    orders.append(order)
        return orders

    
