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

def calc_buy_fee(amount, df):
    sr = df.loc[df['ff_max_value'] > amount].iloc[0]
    if sr['ff_fee_type'] == 2: # 固定费用模式，一般是申购额足够大，费用封顶
        fee = sr['ff_fee']
    else:
        fee = amount - amount / (1 + sr['ff_fee'] * 0.2) # 标准费率计算方式
    return round(fee, 2)

def calc_redeem_fee(row, df):
    sr = df.loc[df['ff_max_value'] >= row['ndays']].iloc[0]
    if sr['ff_fee_type'] == 2: # 固定费用模式，一般是0元，持有期足够长，赎回免费
        fee = sr['ff_fee']
    else:
        fee = row['ts_share'] * row['ts_trade_nav'] * sr['ff_fee']  # 标准费率计算方式
    return fee
    

class FundRule(object):    

    def __init__(self, sdate, edate, optfee=True):

        self.sdate = sdate
        self.edate = edate
        self.optfee = optfee
        
        self.df_ack = {}
        self.dt_nav = {}
        self.dt_redeem_fee = {}
        self.dt_buy_fee = {}
        
        # 赎回到账期限 记录了每个基金的到账日到底是T+n；
        # 购买确认日期 记录了每个基金的到购买从份额确认到可以赎回需要T+n；

        # # 未来交易日 记录了每个交易日的t+n是哪个交易日
        # max_n = int(max(self.df_ack['buy'].max(), self.df_ack['redeem'].max()))
        # dates = base_trade_dates.load_index(sdate, edate)
        # self.df_t_plus_n = pd.DataFrame(dates, index=dates, columns=["td_date"])
        # for i in xrange(0, max_n + 1):
        #     self.df_t_plus_n["T+%d" % i] = self.df_t_plus_n['td_date'].shift(-i)
        # self.df_t_plus_n = self.df_t_plus_n.drop('td_date', axis=1)
        # self.df_t_plus_n.fillna(pd.to_datetime('2029-01-01'), inplace=True)
        # # dd(self.df_t_plus_n)
        
        # #
        # # 加载分红信息
        # #
        # self.df_bonus = base_ra_fund_bonus.load(fund_ids, sdate=sdate, edate=edate)
        # #
        # # 我们发现基础数据里面有部分基金缺少派息日信息(000930)，这里统
        # # 一用除息日填充。
        # #
        # if not self.df_bonus.loc[self.df_bonus['ra_payment_date'].isnull(), 'ra_payment_date'].empty:
        #     self.df_bonus.loc[self.df_bonus['ra_payment_date'].isnull(), 'ra_payment_date'] = self.df_bonus['ra_dividend_date']
        # # dd(self.df_bonus, self.df_bonus.loc[('2016-01-20', [523, 524]), :])
        
        # # 
        # # 加载分拆信息
        # # 
        # #
        # self.df_split = base_fund_split.load(fund_ids, sdate, edate)
        # self.df_split.sort_index(inplace=True)
        # # dd(self.df_split)

        # # 
        # # 加载基金净值
        # #
        # self.df_nav = Nav.Nav().load_nav_and_date(fund_ids, sdate, edate)
        # self.df_nav = self.df_nav.swaplevel(0, 1, axis=0)
        # self.df_nav.sort_index(inplace=True)
        # # dd(self.df_nav.head(20))

        # #
        # # 加载调仓信息
        # #
        # for day, v0 in df_pos.groupby(level=0):
        #     argv = {'pos': v0.loc[day]}
        #     ev = (day + timedelta(hours=17), 8, 0, 0, argv)
        #     heapq.heappush(self.events, ev)
        
        pass

    
    
    def get_buy_ack_days(self, fund_code):
        if fund_code not in self.df_ack:
            df_tmp = base_fund_infos.load_ack2(codes=[fund_code])
            if not df_tmp.empty:
                self.df_ack[fund_code] = df_tmp.iloc[0]
            else:
                self.df_ack[fund_code] = pd.Series({'code':fund_code, 'buy': 0, 'redeem': 0})
        # dd(self.df_ack.loc[(self.df_ack['buy'] > 10) | (self.df_ack['redeem'] > 10) ])
        # dd(self.df_ack)
        return self.df_ack[fund_code].loc['buy']

    def get_redeem_ack_days(self, fund_code):
        if fund_code not in self.df_ack:
            df_tmp = base_fund_infos.load_ack2(codes=[fund_code])
            if not df_tmp.empty:
                self.df_ack[fund_code] = df_tmp.iloc[0]
            else:
                self.df_ack[fund_code] = pd.Series({'code':fund_code, 'buy': 0, 'redeem': 0})
        # dd(self.df_ack.loc[(self.df_ack['buy'] > 10) | (self.df_ack['redeem'] > 10) ])
        # dd(self.df_ack)
        return self.df_ack[fund_code].loc['redeem']
    
    def get_nav(self, fund_code, day):
        if fund_code not in self.dt_nav:
            # 任何自然日所属交易日及其对应的净值序列
            df = Nav.Nav().load_nav(fund_code, day, self.edate)
            self.dt_nav[fund_code] = df
        else:
            df = self.dt_nav[fund_code]

        if day in df.index:
            return df.loc[day]
        else:
            return None

    def get_buy_fee(self, fund_code, sr_amount):
        '''
        获取申购费用和净申购金额

        计算公式如下（来自中国证监会官网）：
        
            净申购金额=申购金额/（1+申购费率）
            申购费用=申购金额-净申购金额

        '''
        if self.optfee == False:
            return sr_amount * 0

        if fund_code not in self.dt_buy_fee:
            df = base_fund_fee.load_buy(codes=[fund_code])
            if df.empty:
                df = pd.DataFrame([(fund_code, np.inf, 0, 2)], columns=['ff_code', 'ff_max_value', 'ff_fee', 'ff_fee_type'])
            self.dt_buy_fee[fund_code] = df
        else:
            df = self.dt_buy_fee[fund_code]

        sr_fee = sr_amount.apply(calc_buy_fee, args=(df,))
        return sr_fee
        
    def get_redeem_fee(self, fund_code, df_redeeming, today):
        '''
        获取赎回费用

        如果在赎回费用表中不存在， 则默认无费率. 此外，需要注意的是基
        金赎回的持有日期是按自然日算的。

        赎回费计算公式（来自中国证监会网站）：
            赎回总金额=赎回份额×T日基金份额净值
            赎回费用=赎回总金额×赎回费率

        '''
        if self.optfee == False:
            return 0
        
        if fund_code not in self.dt_redeem_fee:
            df = base_fund_fee.load_redeem(codes=[fund_code])
            if df.empty:
                df = pd.DataFrame([(fund_code, np.inf, 0, 2)], columns=['ff_code', 'ff_max_value', 'ff_fee', 'ff_fee_type'])
            self.dt_redeem_fee[fund_code] = df
        else:
            df = self.dt_redeem_fee[fund_code]

        # 持有日期
        df_redeeming['ndays'] = (today - df_redeeming['ts_buy_date']).dt.days
            
        sr_fee = df_redeeming.apply(calc_redeem_fee, axis=1, args=(df,))

        return sr_fee
        
