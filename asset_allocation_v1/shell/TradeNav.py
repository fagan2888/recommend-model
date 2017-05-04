#coding=utf8

import logging
import pandas as pd
import numpy as np
import datetime
import calendar
import heapq
from sqlalchemy import *

from db import database

logger = logging.getLogger(__name__)

class TradeNav(object):
    
    def __init__(self, debug=False):
        #
        # 用户当前持仓: DataFrame
        #      columns        share  share_buying nav, nav_date, share_redeeming nav_redeeming, share_bonusing amount_bonusing div_mode
        #      index
        # (fund_id, order_id) 
        # 
        #
        self.df_share = pd.DataFrame()

        #
        # 历史持仓
        #
        self.holdings = {}

        #
        # 净值/市值序列
        #
        self.nav = {}

        #
        # 业绩归因序列
        #
        self.contrib = {}
        
        # 
        # 订单列表，用于保存整个过程中产生的订单, 每个订单是一个字典，包含如下项目
        #     （gid, portfolio_id, fund_id, fund_code, op, place_date, place_time, amount, share, fee, nav, nav_date, ack_date, ack_amount, ack_share, div_mode）
        # 
        self.orders = [];

        self.debug = debug

    def calc(self, df_pos):

        sdate = df_pos.index.min()
        edate = datetime.now() - timedelta(days=1)

        #
        # 事件类型:0:净值更新;1:申购;2:赎回;3:分红;8:调仓;11:申购确认;12:赎回到账;15:分红登记;16:分红除息;17:分红派息;18:基金分拆;19:记录当前持仓;
        #
        # 事件队列，基于heapq实现，元组来实现
        #
        #     (dt, op, fund_id, amount, share, nav, nav_date, {others})
        #
        # 其中，
        #
        #     dt： 日内事件处理时间，主要用于事件在队列中的排序，不一
        #          定等同于下单时间。比如：3:00pm前下的购买订单的处理
        #          时间可能是16:00，因为15:30净值才出来，而购买赎回都
        #          需要使用当天的净值。
        #
        #          各个事件的处理时间如下：
        #
        #          01:00:00 分拆
        #          15:00:00 基金净值
        #          16:00:00 购买
        #          16:30:00 赎回
        #          17:00:00 调仓处理
        #          22:00:00 分红权益登记
        #          22:30:00 分红除息
        #          22:45:00 分红派息
        #          23:59:59 记录当日持仓
        # 
        #
        events = [];
        
        #
        # 加载分红信息
        #
        df = base_ra_fund_bounus.load(fund_ids, sdate, edate)
        for _, row in df.iterrows():
            argv = {
                'bonus_ratio': row['ra_bonus'],
                'bonus_nav': row['ra_bonus_nav'],
                'dividend_date': row['ra_dividend_date'] + timedelta(hours=22,minites=30),
                'payment_date': row['ra_payment_date'] + timedelta(hours=22,minites=45),
            }
            ev = (row['ra_record_date'] + timedelta(hours=22), 15, row['ra_fund_id'], argv)
            heapq.heappush(events, ev)

        # 
        # 加载分拆信息
        # 
        #
        df = base_fund_split.load(fund_ids, sdate, edate)
        for _, row in df.iterrows():
            argv = {'fs_split_proportion': row['fs_split_proportion']}
            ev = (row['fs_split_date'] + timedelta(hours=1), 18, row['fs_fund_id'], argv)
            heapq.heappush(events, ev)
        
        # 
        # 加载基金净值
        #
        df = base_ra_fund_nav.load_nav_date(fund_ids, sdate, edate)
        for key, row in df.iterrows():
            day, fund_id = key
            if row['ra_nav']:
                argv = {'nav': row['ra_nav'], 'nav_date': row['ra_nav_date']}
                ev = (day + timedelta(hours=15), 0, fund_id, argv)
                heapq.heappush(events, ev)
            else:
                logger.error('zero ra_nav detected(ra_fund_id: %d, ra_date:%s)', fund_id, day.strftime("%Y-%m-%d"))
        #
        # 加载调仓信息
        #
        for day, v0 in df_pos.groupby(level=0):
            argv = {'pos': v0}
            ev = (day + timedelta(hours=17), 0, fund_id, argv)
            heapq.heappush(events, ev)
        
        #
        # 记录持仓事件
        #
        dates = pd.date_range(sdate, edate)
        for day in dates:
            argv = {}
            ev = (day + timedelta(hours=23, minutes=59, seconds=59), 19, 0, argv)
            heapq.heappush(events, ev)

        #
        # 依次处理每个事件
        #
        while True:
            ev = heapq.heappop(events)
            if ev is None:
                break

            evs = self.process(ev)

            for ev in evs:
                heapq.heappush(events, ev)
        # #
        # # 保存持仓数据
        # #
        # AssetInvestorHoldingFund::where('ih_uid', $this->uid)
        #     ->where('ih_fund_id', $this->productId)
        #     ->where('ih_origin', $this->origin)
        #     ->delete();
        # AssetInvestorHoldingFund::insert($this->holdings);

        # #
        # # 保存交易数据
        # #
        # AssetInvestorTradeFlow::where('it_uid', $this->uid)
        #     ->where('it_type', 3)
        #     ->where('it_product_id', $this->productId)
        #     ->where('it_origin', $this->origin)
        #     ->delete();
        # AssetInvestorTradeFlow::insert($this->bonusOrders);

    def process(self， ev):
        result = []
        # 事件类型:0:净值更新;1:申购;2:赎回;3:分红;11:申购确认;12:赎回到账;15:分红登记;16:分红除息;17:分红派息;18:基金分拆;19:记录当前持仓;
        dt, op, fund_id, argv = ev
        if op == 0:
            #
            # 净值更新
            #
            df = self.df_share.loc[fund_id]
            if not df.empty:
                df['yield'] = (df['share'] + df['share_buying']) * (df['nav'] - argv['nav'])
                df['nav'] = argv['nav']
                df['nav_date'] = argv['nav_date']

        elif op == 1:
            #
            # 申购
            #
            self.cash -= argv['share'] * argv['nav'] + argv['fee']
            self.df_share.loc[(fund_id, argv['order_id'])] = {
                'share': 0, 'share_buying': argv['share'], 'share_redeeming': 0
            }

        elif op == 11:
            #
            # 申购确认
            #
            self.df_share.loc[(fund_id, argv['order_id']), 'share'] = self.df_share.loc[(fund_id, argv['order_id']), 'share_buying']
            self.df_share.loc[(fund_id, argv['order_id']), 'share_buying'] = 0

        elif op == 2:
            #
            # 赎回
            #
            self.df_share.loc[(fund_id, argv['share_id']), 'share'] -= argv['share']
            self.df_share.loc[(fund_id, argv['share_id']), 'share_redeeming'] += argv['share']

        elif op == 12:
            #
            # 赎回确认
            #
            self.df_share.loc[(fund_id, argv['share_id']), 'share_redeeming'] -= argv['share']
            self.cash += argv['share'] * argv['nav'] - argv['fee']

        elif op == 8:
            #
            # 调仓处理
            # 
            # df_share: 是当前状态；argv['pos']: 是目标状态
            #
            # 调仓逻辑里面，首先取消掉所有未提交的订单，然后在根据当前状态
            # 重新生成订单
            #
            self.remove_flying_op(events)
            result = self.adjust(self.df_share, argv['pos'])

        elif op == 15:
            #
            # 基金分红：权益登记
            #
            df = self.df_share.loc[fund_id]
            df['share_bonusing'] = (df['share'] + df['share_buying'])
            share = df['share_bonusing'].sum()
            argv2 = {
                'share': share,
                'bonus_ratio': argv['bonus_ratio'],
                'bonus_nav': argv['bound_nav'],
                'payment_date': argv['payment_date'],
            }
            ev2 = (argv['dividend_date'], 16, fund_id, argv2)
            result.append(ev2)
            #
            # 记录分红操作
            #
            share_dividend = df.loc[df['div_mode'] == 1, 'share_bonusing'].sum()
            share_cash = df['share_bonusing'].sum() - share_dividend
            if share_dividend > 0:
                self.orders.append(
                    self.make_bonus_order(dt, fund_id, share_dividend, argv, 1))

            if share_cash > 0:
                self.orders.append(
                    self.make_bonus_order(dt, fund_id, share_cash, argv, 0))

        elif op == 16:
            #
            # 基金分红：除息
            #
            df = self.df_share.loc[fund_id]
            df['amount_bonusing'] = df['share_bonusing'] * argv['bonus_ratio']
            argv2 = {
                'bound_nav': argv['bonus_nav'],
                'payment_date': argv['payment_date'],
            }
            ev2 = (argv['payment_date'], 17, fund_id, argv2)
            result.append(ev2)
            #
            # 记录在途分红资金
            #
            # self.cash_bonusing += df['amount_bonusing'].sum()

        elif op == 17:
            #
            # 基金分红：派息
            #
            df = self.df_share.loc[fund_id]
            #
            # 现金分红
            #
            mask = (df['div_mode'] == 0)
            self.cash_bounused = df.loc[mask, 'amount_bonusing'].sum()
            #
            # 红利再投
            #
            mask = ~mask
            df.loc[mask, 'share'] += df.loc[mask, 'amount_bonusing'] / argv['bonus_nav']
            #
            # 清除分红上下文
            #
            df.loc['share_bonusing']  = 0
            df.loc['amount_bonusing'] = 0

        elif op == 18:
            #
            # 基金分拆
            #
            df = self.df_share.loc[fund_id]
            df['share'] *= argv['fs_split_proportion']
            df['share_buying'] *= argv['fs_split_proportion']
            df['share_redeeming'] *= argv['fs_split_proportion']
            #
            # 理论上此时的净值为昨天的净值, 因为分拆是在调整今天净值之前处理的.
            #
            # 要保证后面日收益计算的正确, 这个地方需要对净值进行折算.
            #
            df['nav'] /= argv['fs_split_proportion']

        elif op == 19:
            #
            # 记录持仓
            #
            if self.debug and self.df_share.sum().sum() > 0.000099:
                self.holdings[dt] = self.df_share.copy()
            #
            # 记录净值
            #
            amount = (self.df_share['share'] + self.df_share['share_buying']) * self.df_share['nav'] \
                     + self.df_share['share_redeeming'] * self.df_share['nav_redeeming'] \
                     + self.df_share['amount_bonusing']
            self.nav[dt] = amount
            #
            # 记录业绩归因
            #
            self.contrib[dt] = self.df_share['yield'].groupby(level=1).sum()

        return result

    def make_bonus_order(dt, fund_id, order_share, bonus, div_mode):
        amount = order_share * bonus['bonus_ratio']
        if div_mode == 1:
            #
            # 红利再投
            #
            nav, nav_date, ack_amount = bonus['nav'], bonus['nav_date'], 0
            ack_share = amount / nav
        else:
            nav, nav_date, ack_share, ack_amount = 0, '0000-00-00', 0, amount
                
        return {
            'gid': self.new_order_id(dt),
            'portfolio_id': self.portfolio_id,
            'fund_id': fund_id,
            # 'fund_code': fund_code,
            'op': 3,
            'place_date': bonus['dividend_date'],
            'place_time': '15:00:00',
            'share': order_share,
            'amount': amount,
            'fee': 0,
            'nav': nav,
            'nav_date': nav_date,
            'ack_date': bonus['payment_date'],
            'ack_amount': ack_amount,
            'ack_share': ack_share,
            'div_mode': div_mode,
        }

    def adjust(df_share, df_dst):
        '''
        生成调仓订单
        '''
        
        df_src = ((df_share['share'] + df_share['share_buying']) * df_share['nav'] + df_share['amount_bonusing']).groupby(level=0).sum()
        total = df_src.sum() + (df_share['share_redeeming'] * df_share['nav_redeeming']).sum()
