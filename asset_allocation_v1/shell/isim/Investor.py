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
                df = df.reset_index(drop=True).set_index(['ts_dividend_date'], drop=False)
                nav = Nav.Nav().load_nav(code, self.sdate)
                nav['tm1'] = nav['ra_date'].shift(1)
                df['ts_exec_date'] = nav['tm1']
                df.reset_index(drop=True, inplace=True)
                
                dividend[code] = df

        if carried:
            self.df_carried = pd.concat(carried).reset_index(level=1, drop=True)
        else:
            self.df_carried = pd.DataFrame([], index=[], columns=[
                'ts_uid', 'ts_portfolio_id', 'ts_fund_code', 'ts_pay_method', 'ts_record_date', 'ts_dividend_date', 'ts_dividend_amount', 'ts_dividend_share', 'ts_carried_date'])

        if dividend:
            self.df_dividend = pd.concat(dividend).reset_index(level=1, drop=True)
        else:
            self.df_dividend = pd.DataFrame([], index=[], columns=[
                'ts_uid', 'portfolio_id', 'fund_code', 'pay_method', 'record_date', 'dividend_date', 'dividend_amount', 'dividend_share'])
        pdb.set_trace()
            
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
        df = df.reset_index().set_index(['ts_portfolio_id', 'ts_fund_code', 'ts_pay_method'], drop=False)

        if not df.empty:
            sr_uncarried = -df_uncarried['ts_stat_uncarried'].groupby(level=df_uncarried.index.names).sum()
            df['ts_stat_uncarried'] = sr_uncarried
            return df
        else:
            return None

    def perform_dividend(self, code, day, df):
        i_amount = df.columns.get_loc('ts_bonus_amount')
        i_share =  df.columns.get_loc('ts_bonus_share')
        
        for i in xrange(0, len(df.index)):
            sr = df.iloc[i]
            mask = (self.df_dividend['ts_portfolio_id'] == sr['ts_portfolio_id']) & \
                   (self.df_dividend['ts_fund_code'] == sr['ts_fund_code']) & \
                   (self.df_dividend['ts_pay_method'] == sr['ts_pay_method']) & \
                   (self.df_dividend['ts_exec_date'] == sr['ts_dividend_date'])

            df_tmp = self.df_dividend.loc[mask]
            if df_tmp.empty:
                df.iloc[i, i_amount] = sr['ts_estimate_amount']
                df.iloc[i, i_share] = (sr['ts_estimate_amount'] / sr['ts_bonus_nav']).round(2)
            else:
                x = df_tmp.iloc[0]
                if x['ts_dividend_amount'] > 0.0099:
                    # 发生现金分红的情况
                    df.iloc[i, i_amount] = x['ts_dividend_amount']
                    df.iloc[i, i_share] = x['ts_dividend_share']
                    df.iloc[i, 'ts_div_mode'] = 0
                else:
                    df.iloc[i, i_amount] = sr['ts_estimate_amount']
                    df.iloc[i, i_share] = x['ts_dividend_share']
                     
                    
        return df
 
