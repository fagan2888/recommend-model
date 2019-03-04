#coding=utf8


import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import time
import logging
import re
import Const
import DFUtil
import DBData
import util_numpy as npu
import Portfolio as PF
import Financial as fin
from TimingWavelet import TimingWt
import multiprocessing
from multiprocessing import Manager
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, asset_mz_markowitz, asset_mz_markowitz_alloc, asset_mz_markowitz_argv,  asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe, asset_wt_filter_nav
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl, asset_stock
from db.asset_stock_factor import *
from util import xdict
from util.xdebug import dd
from asset import Asset, WaveletAsset
from allocate import Allocate, AssetBound
from trade_date import ATradeDate
from view import View
import RiskParity
import util_optimize
from multiprocessing import Pool

import PureFactor
import IndexFactor
import traceback, code
from monetary_fund_filter import MonetaryFundFilter
from dateutil.relativedelta import relativedelta


logger = logging.getLogger(__name__)



class Trade(object):

    def __init__(self, globalid, asset_pos, reindex, asset_trade_delay = None, ops = None, init_amount = 10000):

        self.asset_pos = asset_pos.reindex(reindex | asset_pos.index).fillna(method='pad').dropna()
        self.asset_pos = self.asset_pos.div(self.asset_pos.sum(axis = 1), axis = 0)
        self.index = reindex
        first_day = self.index[0]

        #用户净值
        self.user_nav = pd.DataFrame(columns = ['nav'], index = self.index)
        self.user_nav.loc[first_day] = 1.0

        #用户持有每个资产的价值
        self.user_hold_asset = pd.DataFrame(0, columns = self.asset_pos.columns, index = self.index)
        self.user_hold_asset.loc[first_day] = self.asset_pos.loc[first_day] * init_amount

        #用户持有每个资产的权重
        self.user_hold_weight = pd.DataFrame(columns = self.asset_pos.columns, index = self.index)
        self.user_hold_weight.loc[first_day] = self.asset_pos.loc[first_day]

        #用户正在调仓赎回中每类资产
        #self.user_redeem_and_buy = pd.DataFrame(0, columns = self.asset_pos.columns, index = self.index)

        #用户正在购买中每类资产
        #self.user_buying = pd.DataFrame(0, columns = self.asset_pos.columns, index = self.index)

        #每个资产净值和收益率
        self.asset_navs = {}
        for asset_id in asset_pos.columns:
            self.asset_navs[asset_id] = Asset(asset_id).nav()
        self.asset_navs = pd.DataFrame(self.asset_navs).fillna(method='pad')
        self.asset_inc = self.asset_navs.pct_change()

        #self.asset_navs = self.asset_navs.reindex(self.index).fillna(method='pad')
        self.asset_inc = self.asset_inc.reindex(self.index).fillna(0.0)


    def rebuy(self, today, amount):

        index = self.index.tolist().index(today)
        tomorrow = self.index[index+1]
        allocate_weight = self.asset_pos.loc[today]
        user_weight = self.user_hold_weight.loc[today]
        for asset_id in allocate_weight.index:
            self.user_hold_asset.loc[tomorrow, asset_id] = self.user_hold_asset.loc[tomorrow, asset_id] + amount * allocate_weight.loc[asset_id]


    def cal_score(self, today):

        user_weight = self.user_hold_weight.loc[today]
        allocate_weight = self.asset_pos.loc[today]
        deviation = abs(user_weight - allocate_weight).sum() / 2
        return (1.0 - deviation) * 100


    def adjust(self, today):

        index = self.index.tolist().index(today)
        tomorrow = self.index[index+1]
        t = index + 1
        n = 1
        t_plus_n = self.index[t + n]

        user_weight = self.user_hold_weight.loc[today]
        allocate_weight = self.asset_pos.loc[today]
        tomorrow_total_amount = ((self.user_hold_asset.loc[today]) * (1 + self.asset_inc.loc[tomorrow])).sum()
        for asset_id in allocate_weight.index:
            u_amount = user_weight.loc[asset_id] * tomorrow_total_amount
            a_amount = allocate_weight.loc[asset_id] * tomorrow_total_amount
            if u_amount >= a_amount:
                redeem = u_amount - a_amount
                self.user_hold_asset.loc[self.index[t+1], asset_id] = -1.0 * redeem
            else:
                buy = a_amount - u_amount
                self.user_hold_asset.loc[self.index[t+2], asset_id] = buy


    def redeem(today, ratio = 0.0):
        index = self.index.tolist().index(today)
        tomorrow = self.index[index+1]
        every_asset = self.user_hold_asset[self.user_hold_asset.index >= today].sum(axis = 0)
        for asset_id in every_asset.index:
           self.user_hold_asset.loc[tomorrow, asset_id] = -1.0 * every_asset.loc[asset_id] * ratio


    def trade_strategy(self):

        self.asset_inc = self.asset_inc[self.asset_pos.columns]
        pos = pd.DataFrame(columns = self.asset_pos.columns)
        every_asset = self.user_hold_asset.sum(axis = 0)
        pos.loc[self.index[0]] = every_asset / every_asset.sum()
        num_adjust = 0

        for i in range(1, len(self.index) - 3):
            today, yesterday, tomorrow = self.index[i], self.index[i-1], self.index[i+1]
            #计算当天可获得收益的资产总额
            self.user_hold_asset.loc[today] = self.user_hold_asset.loc[yesterday] + self.user_hold_asset.loc[today]
            #计算当日收益率
            #today_inc = (self.user_hold_asset.loc[today] * (1 + self.asset_inc.loc[today])) / (self.user_hold_asset.loc[today]).sum()
            #self.user_nav.loc[today] = self.user_nav.loc[yesterday] * (1 + today_inc)
            #当日获得收益后的资产额度
            self.user_hold_asset.loc[today] = self.user_hold_asset.loc[today] * (1 + self.asset_inc.loc[today])

            #self.redeem(today, ratio)
            #self.rebuy(today, amount)

            #计算当天配置比例
            every_asset = self.user_hold_asset[self.user_hold_asset.index >= today].sum(axis = 0)
            self.user_hold_weight.loc[today] = every_asset / every_asset.sum()

            score = self.cal_score(today)
            if score < 90:
                #print(today, score, self.asset_pos.loc[today], self.user_hold_weight.loc[today])
                #print('----------')
                num_adjust += 1
                self.adjust(today)
                every_asset = self.user_hold_asset[self.user_hold_asset.index >= today].sum(axis = 0)
                pos.loc[tomorrow] = every_asset / every_asset.sum()

            #every_asset = self.user_hold_asset.loc[today] - self.user_redeeming.loc[today] + self.user_buying.loc[today]
            #sys.exit(0)
            #self.user_hold_weight =
            #print(date)

        print(num_adjust)

        return pos
        #set_trace()
        #return ws


class TradeNew(Trade):

    def __init__(self, globalid, asset_pos, reindex, asset_trade_delay = None, ops = None, init_amount = 10000):

        super(TradeNew, self).__init__(globalid, asset_pos, reindex, asset_trade_delay, ops, init_amount)
        self.last_adjust_day = reindex[0]
        self.count = 0

    def cal_score(self, today):

        user_weight = self.user_hold_weight
        allocate_weight = self.asset_pos
        delta_weight = (user_weight - allocate_weight).loc[self.last_adjust_day:today].abs()
        # if delta_weight.shape[0] > 15:
        #     delta_weight = delta_weight.iloc[-15:]
        assets = user_weight.columns

        cov = np.cov(np.array(delta_weight.T).tolist())
        res = np.dot(np.mat([allocate_weight.loc[today]]), np.dot(cov, np.mat([allocate_weight.loc[today]]).T))[0, 0]
        res = res ** 0.5
        deviation = abs(user_weight.loc[today] - allocate_weight.loc[today]).sum() / 2
        # print(res)
        if res > 0.012  and today > self.last_adjust_day + relativedelta(days=+7):
            # print(today)
            return 70
        else:
            return 100

    def trade_strategy(self):

        self.asset_inc = self.asset_inc[self.asset_pos.columns]
        pos = pd.DataFrame(columns = self.asset_pos.columns)
        every_asset = self.user_hold_asset.sum(axis = 0)
        pos.loc[self.index[0]] = every_asset / every_asset.sum()

        for i in range(1, len(self.index) - 2):
            today, yesterday, tomorrow = self.index[i], self.index[i-1], self.index[i+1]
            #计算当天可获得收益的资产总额
            self.user_hold_asset.loc[today] = self.user_hold_asset.loc[yesterday] + self.user_hold_asset.loc[today]
            #计算当日收益率
            #today_inc = (self.user_hold_asset.loc[today] * (1 + self.asset_inc.loc[today])) / (self.user_hold_asset.loc[today]).sum()
            #self.user_nav.loc[today] = self.user_nav.loc[yesterday] * (1 + today_inc)
            #当日获得收益后的资产额度
            self.user_hold_asset.loc[today] = self.user_hold_asset.loc[today] * (1 + self.asset_inc.loc[today])

            #self.redeem(today, ratio)
            #self.rebuy(today, amount)

            #计算当天配置比例
            every_asset = self.user_hold_asset[self.user_hold_asset.index >= today].sum(axis = 0)
            self.user_hold_weight.loc[today] = every_asset / every_asset.sum()

            score = self.cal_score(today)
            if score < 80:
                #print(today, score, self.asset_pos.loc[today], self.user_hold_weight.loc[today])
                #print('----------')
                self.adjust(today)
                every_asset = self.user_hold_asset[self.user_hold_asset.index >= today].sum(axis = 0)
                pos.loc[tomorrow] = every_asset / every_asset.sum()
                self.last_adjust_day = tomorrow

            #every_asset = self.user_hold_asset.loc[today] - self.user_redeeming.loc[today] + self.user_buying.loc[today]
            #sys.exit(0)
            #self.user_hold_weight =
            #print(date)

        return pos

class TradeVolatility(Trade):

    def __init__(self, globalid, asset_pos, reindex, asset_trade_delay = None, ops = None, init_amount = 10000, trade_risk = None):

        super(TradeVolatility, self).__init__(globalid, asset_pos, reindex, asset_trade_delay, ops, init_amount)
        # self.asset_volability = self.asset_inc.cov()
        self.count_adjust = 0
        self.trade_risk = trade_risk

    def cal_score(self, today):

        user_weight = self.user_hold_weight.loc[today]
        allocate_weight = self.asset_pos.loc[today]
        delta_weight = (user_weight - allocate_weight).abs()
        asset_inc = self.asset_inc.loc[:today]
        asset_inc = asset_inc[-120:]
        asset_volatility = np.diag(asset_inc.cov())

        score = (delta_weight * (asset_volatility ** 0.5)).sum()
        score = max(0.6, 100 - score / self.trade_risk * 10)
        # print(score)
        # set_trace()
        #if score < 0.00124:
        user_weight = self.user_hold_weight.loc[today]
        allocate_weight = self.asset_pos.loc[today]
        deviation = abs(user_weight - allocate_weight).sum() / 2
        deviation = max(0.6, 100 - deviation / 0.2 * 10)
        return (score + deviation) / 2
        #else:
        #    self.count_adjust += 1
        #    print(self.count_adjust)
        #delta_weight = (user_weight - allocate_weight).abs() / 2
        #asset_inc = self.asset_inc.loc[:today]
        #asset_inc = asset_inc[-120:]

        #week_trade_date = ATradeDate.week_trade_date()
        #week_trade_date = week_trade_date[week_trade_date <= today]
        #week_trade_date = week_trade_date[-26:]

        #asset_week_navs = self.asset_navs.loc[week_trade_date]
        #asset_week_inc = asset_week_navs.pct_change().fillna(0.0)

        #asset_volability = np.diag(asset_inc.cov()) ** 0.5
        #user_inc = (asset_inc * user_weight).sum(axis = 1)
        #allocate_inc = (asset_inc * allocate_weight).sum(axis = 1)
        #delta_weight_inc = (asset_week_inc * delta_weight).sum(axis = 1)
        #user_shape = user_inc.mean() / user_inc.std()
        #allocate_shape = allocate_inc.mean() / allocate_inc.std()

        #user_std = user_inc.std()
        #allocate_std = allocate_inc.std()
        #deviation = abs(user_std - allocate_std) / allocate_std
        #print(today, delta_weight_inc.mean(), delta_weight_inc.std())
        #delta_std = delta_weight_inc.std()
        #if delta_std < 0.015 * 0.15:
        #    return 100
        #else:
        #    print(today, delta_std)
        #    return 70
        #user_volability()

        #volability_score = (delta_weight * asset_volability).sum()
        #print(volability_score)

        # print(score)
        # set_trace()
        #0.005
        #if volability_score < 0.005:
        #if volability_score - 0.015 < 0.015 * 0.1:
        #if user_shape < 0 and allocate_shape < 0:
        #    if user_shape > allocate_shape * 1.1:
        #        return 100
        #    else:
        #        return 70
        #return 100

def cal_tracking_error(asset1, asset2, reindex=None):

    ser_asset1_nav = Asset(asset1).nav()
    ser_asset2_nav = Asset(asset2).nav()
    tracking_error = (ser_asset1_nav - ser_asset2_nav).std()

    return tracking_error

if __name__ == '__main__':

    print(cal_tracking_error('MZ.T00011', 'MZ.000071'))
    print(cal_tracking_error('MZ.T00012', 'MZ.000072'))
    print(cal_tracking_error('MZ.T00013', 'MZ.000073'))
    print(cal_tracking_error('MZ.T00014', 'MZ.000074'))
    print(cal_tracking_error('MZ.T00015', 'MZ.000075'))
    print(cal_tracking_error('MZ.T00016', 'MZ.000076'))
    print(cal_tracking_error('MZ.T00017', 'MZ.000077'))
    print(cal_tracking_error('MZ.T00018', 'MZ.000078'))
    print(cal_tracking_error('MZ.T00019', 'MZ.000079'))
    print(cal_tracking_error('MZ.T00010', 'MZ.000070'))
