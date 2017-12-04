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
    def __init__(self, df_ts_order, df_ts_dividend_fund):

        self.df_ts_order = df_ts_order.set_index(['ts_placed_date'], drop=False)
        self.monetary = ['000509', '002847', '202301']

        df_ts_dividend_fund.set_index(['ts_fund_code'], drop=False, inplace=True)

        self.sdate =  pd.to_datetime(self.df_ts_order.index.get_level_values(0).min())

        carried = {}
        dividend = {}
        codes = df_ts_dividend_fund.index.unique().tolist()
        for code in codes:
            # pdb.set_trace()
            df = df_ts_dividend_fund.loc[[code]]
            if code in self.monetary:
                # 任何自然日所属交易日及其对应的净值序列
                df = df.reset_index(drop=True).set_index(['ts_dividend_date'], drop=False)

                nav = Nav.Nav().load_nav(code, self.sdate)
                nav['tm1'] = nav['ra_date'].shift(1)
                df['ts_carried_date'] = nav['tm1']
                df.reset_index(drop=True, inplace=True)

                carried[code] = df
            else:
                dividend[code] = df

        pdb.set_trace()
        if carried:
            self.df_carried = pd.concat(carried).reset_index(level=1, drop=True)
        else:
            self.df_carried = pd.DataFrame([], index=[], columns=[
                'ts_uid', 'ts_portfolio_id', 'ts_fund_code', 'ts_pay_method', 'ts_record_date', 'ts_dividend_date', 'ts_dividend_amount', 'ts_dividend_share', 'ts_carried_date'])

        if dividend:
            self.df_dividend = pd.concat(dividend)
        else:
            self.df_dividend = pd.DataFrame([], index=[], columns=[
                'ts_uid', 'portfolio_id', 'fund_code', 'pay_method', 'record_date', 'dividend_date', 'dividend_amount', 'dividend_share'])
            
    def get_sdate(self):
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
    

    def get_uncarried(self, day, fund_code, sr_uncarried):
        sr_uncarried = ((sr_uncarried * 100).apply(np.floor)) / 100.0
        return sr_uncarried

    def get_carried(self, day, fund_code, df_uncarried):
        if fund_code not in self.df_carried.index:
            return None
        
        df = self.df_carried.loc[fund_code]
        df = df.loc[df['ts_carried_date'] == day]

        if not df.empty:
            sr_uncarried = -df_uncarried['ts_stat_uncarried'].groupby(level=[0]).sum()
            df['ts_stat_uncarried'] = sr_uncarried
            return df
        else:
            return None
