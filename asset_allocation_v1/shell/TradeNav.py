#coding=utf8

import sys
import logging
import pandas as pd
import numpy as np
import datetime
import calendar
import heapq
import TradeNav

from datetime import datetime, timedelta
from sqlalchemy import *
from util.xdebug import dd

from db import *

logger = logging.getLogger(__name__)

class TradeNav(object):
    
    def __init__(self, debug=False):

        # 调试模式 会记录整个计算的上下文，默认为关闭
        self.debug = debug

        #
        # 状态变量：用户当前持仓: DataFrame
        # 
        #      columns        share, yield, share_buying, ack_date,  nav, nav_date, share_bonusing amount_bonusing div_mode
        #      index
        # (fund_id, share_id)
        #      
        #  其中，
        #
        #    share_id: 持仓ID，直接采用购买操作的订单号order_id，因
        #              为每次购买会生成唯一的持仓
        #
        #    ack_date: 购买确认日期（实际上是份额可赎回开始日期，不同
        #              于交易中的订单确认日期，因为订单确认日期在费率
        #              计算中没有任何实际意义。
        #
        self.df_share = pd.DataFrame({
            'share': pd.Series(dtype=float),
            'yield': pd.Series(dtype=float),
            'share_buying': pd.Series(dtype=float),
            'nav': pd.Series(dtype=float),
            'nav_date':  pd.Series(dtype='datetime64[ns]'),
            'ack_date':  pd.Series(dtype=float),
            'share_bonusing':  pd.Series(dtype=float),
            'amount_bonusing':  pd.Series(dtype=float),
            'div_mode': pd.Series(dtype=int)},
            index=pd.MultiIndex(names=['fund_id','share_id'], levels=[[], []], labels=[[],[]]),
        )
        # dd(self.df_share, self.df_share['nav'], self.df_share['nav_date'])

        #
        # 状态变量：赎回上下文：DataFrame
        #      columns        share_id, share,  nav, nav_date, ack_date
        #      index
        # (fund_id, redeem_id)
        #
        #  其中，
        #
        #    redeem_id: 赎回ID，直接采用赎回操作的订单号order_id，因为
        #               每个赎回会生成唯一的赎回上下文
        #
        #    share_id : 被赎回的持仓的ID，为被赎回的持仓的ID，多个赎回
        #               上下文可能会具有相同的share_id，也就意味着被提
        #               交了多个赎回的请求（这个是留着为将来支持用户赎
        #               回和调仓同时进行的伏笔）
        #
        #    share & nav : 赎回的份额 和 净值（用于赎回确认的净值）
        #
        #    ack_date: 到账日期，也就是赎回款最终到账，可用于进一步购买日期
        #
        self.df_redeem = pd.DataFrame(
            index=pd.MultiIndex(names=['fund_id','redeem_id'], levels=[[], []], labels=[[],[]]),
            columns=['share_id', 'share', 'nav', 'nav_date', 'ack_date']
        )

        #
        # 状态变量：当日订单计数
        #
        self.dsn = 0 # day sequenece no
        
        #
        # 输出变量：历史持仓
        #
        self.dt_holding = {}

        # 
        # 输出变量：订单列表，用于保存整个过程中产生的订单, 每个订单是一个字典，包含如下项目
        #     （order_id, fund_id, fund_code, op, place_date, place_time, amount, share, fee, nav, nav_date, ack_date, ack_amount, ack_share, div_mode）
        # 
        self.orders = []

        #
        # 输出变量：业绩归因序列
        #
        self.contrib = {}

        #
        # 输出变量：计算的净值/市值序列
        #
        self.nav = {}

        #
        # 输入变量：基金自然日使用的交易日及其净值
        #
        # 延迟加载，因为跟具体配置有关
        #
        self.dt_nav = None

        #
        # 输入变量：赎回费率，延迟加载，因为跟持仓基金有关
        #
        self.df_redeem_fee = None
        #
        # 购买费率，延迟加载，因为跟持仓基金有关
        #
        self.df_buy_fee = None

        #
        # 输入变量：
        # redeem 赎回到账期限 记录了每个基金的到账日到底是T+n；
        # buy    购买确认日期 记录了每个基金的到购买从份额确认到可以赎回需要T+n；
        # 
        # 延迟加载，因为跟具体持仓基金有关
        #
        self.df_ack = None

        #
        # 输入变量：未来交易日 记录了每个交易日的t+n是哪个交易日
        #
        # 延迟加载，具体跟调仓序列有关
        #
        self.df_t_plus_n = None

        #
        # 输入变量：分红信息
        #
        # 延迟加载，具体跟调仓序列有关
        #
        self.df_bonus = None
        
        #
        # 输入变量：拆分信息
        #
        # 延迟加载，具体跟调仓序列有关
        #
        self.df_split = None

        #
        # 输入变量：基金净值
        #
        # 延迟加载，因为跟具体配置有关
        #
        self.df_nav = None
        

    def calc(self, df_pos, principal):

        self.cash = principal
        
        sdate = df_pos.index.get_level_values(0).min()
        edate = datetime.now() - timedelta(days=1)
        # dd(sdate, edate)
        #

        # 初始化用到的计算数据
        #

        # 所有用到的基金ID
        fund_ids = df_pos.index.get_level_values(1).unique()
        # dd(fund_ids)

        #
        # 赎回费率，延迟加载，因为跟持仓基金有关
        self.df_redeem_fee = base_fund_fee.load_redeem(fund_ids)

        # 购买费率，延迟加载，因为跟持仓基金有关
        self.df_buy_fee = base_fund_fee.load_buy(fund_ids)
        # dd(self.df_buy_fee.head(), self.df_redeem_fee.head())



        # 赎回到账期限 记录了每个基金的到账日到底是T+n；
        # 购买确认日期 记录了每个基金的到购买从份额确认到可以赎回需要T+n；
        self.df_ack = base_fund_infos.load_ack(fund_ids)
        # dd(self.df_ack.loc[(self.df_ack['buy'] > 10) | (self.df_ack['redeem'] > 10) ])

        # 未来交易日 记录了每个交易日的t+n是哪个交易日
        max_n = int(max(self.df_ack['buy'].max(), self.df_ack['redeem'].max()))
        dates = base_trade_dates.load_index(sdate, edate)
        self.df_t_plus_n = pd.DataFrame(dates, index=dates, columns=["td_date"])
        for i in xrange(1, max_n + 1):
            self.df_t_plus_n["T+%d" % i] = self.df_t_plus_n['td_date'].shift(-i)
        self.df_t_plus_n = self.df_t_plus_n.drop('td_date', axis=1)
        # dd(self.df_t_plus_n, max_n, self.df_t_plus_n.index)


        # 任何自然日所属交易日及其对应的净值序列
        self.dt_nav = Nav.Nav().load_tdate_and_nav(fund_ids, sdate, edate)
        # dd(self.dt_nav)
        
        #
        # 加载分红信息
        #
        self.df_bonus = base_ra_fund_bonus.load(fund_ids, sdate=sdate, edate=edate)
        # dd(self.df_bonus, self.df_bonus.loc[('2016-01-20', [523, 524]), :])
        
        # 
        # 加载分拆信息
        # 
        #
        self.df_split = base_fund_split.load(fund_ids, sdate, edate)

        # 
        # 加载基金净值
        #
        self.df_nav = Nav.Nav().load_nav_and_date(fund_ids, sdate, edate)
        self.df_nav = self.df_nav.swaplevel(0, 1, axis=0)
        self.df_nav.sort_index(inplace=True)
        # dd(self.df_nav.head(20))
        
        #
        # 事件类型:0:净值更新;1:申购;2:赎回;3:分红;8:调仓;11:申购确认;12:赎回到账;15:分红登记;16:分红除息;17:分红派息;18:基金分拆;
        #         100:例行事件生成;101:记录当前持仓;
        #
        # 事件队列，基于heapq实现，元组来实现
        #
        #     (dt, op, order_id, fund_id, argv)
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
        #          00:00:01 生成持仓相关的当日例行事件（更新净值，分拆，分红，记录当日持仓）
        #          01:00:00 分拆
        #          02:00:00 购买确认
        #          03:00:00 赎回确认
        #          15:00:00 更新净值
        #          16:00:00 购买
        #          16:30:00 赎回
        #          17:00:00 调仓处理
        #          22:00:00 分红权益登记
        #          22:30:00 分红除息
        #          22:45:00 分红派息
        #          23:59:59 记录当日持仓
        # 
        #
        self.events = [];
        
        #
        # 加载调仓信息
        #
        for day, v0 in df_pos.groupby(level=0):
            argv = {'pos': v0.loc[day]}
            ev = (day + timedelta(hours=17), 8, 0, 0, argv)
            heapq.heappush(self.events, ev)
        
        #
        # 生成每日例行事件的事件
        #
        dates = pd.date_range(sdate, edate)
        for day in dates:
            argv = {}
            ev = (day + timedelta(seconds=1), 100, 0, 0, argv)
            heapq.heappush(self.events, ev)

        #
        # 依次处理每个事件
        #
        while self.events:
            ev = heapq.heappop(self.events)
            if ev is None:
                break

            self.dump_event('-', ev)
            evs = self.process(ev)

            for ev in evs:
                self.dump_event('+', ev)
                heapq.heappush(self.events, ev)


        #
        # 保存计算结果
        #
        # 本模块不负责保存具体的计算结果
        #
        df_result_nav = pd.Series(self.nav).to_frame('nav') 
        dd(df_result_nav, "completed")

    def process(self, ev):
        result = []
        # 事件类型:0:净值更新;1:申购;2:赎回;3:分红;8:调仓;11:申购确认;12:赎回到账;15:分红登记;16:分红除息;17:分红派息;18:基金分拆;
        #         100:例行事件生成;101:记录当前持仓;
        dt, op, order_id, fund_id, argv = ev
        if op == 0:
            #
            # 净值更新
            #
            if fund_id in self.df_share.index.levels[0]:
                df = self.df_share.loc[fund_id]
                df['yield'] = (df['share'] + df['share_buying']) * (df['nav'] - argv['nav'])
                df['nav'] = argv['nav']
                df['nav_date'] = argv['nav_date']
            else:
                dd("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                

        elif op == 1:
            #
            # 申购
            #
            self.cash -= argv['share'] * argv['nav'] + argv['fee']
            sr = pd.Series({
                'share': 0,
                'yield': 0,
                'share_buying': argv['share'],
                'nav': argv['nav'],
                'nav_date': argv['nav_date'],
                'ack_date': argv['ack_date'],
                'share_bonusing': 0,
                'amount_bonusing': 0,
                'div_mode': argv['div_mode']
            })
            self.df_share.loc[(fund_id, argv['order_id']), :] = sr
            #
            # 生成申购确认事件
            #
            ev2 = (argv['ack_date'] + timedelta(hours=2), 11, order_id, fund_id, argv)
            result.append(ev2)

        elif op == 11:
            #
            # 申购确认
            #
            self.df_share.loc[(fund_id, order_id), 'share'] = self.df_share.loc[(fund_id, order_id), 'share_buying']
            self.df_share.loc[(fund_id, order_id), 'share_buying'] = 0

        elif op == 2:
            #
            # 赎回
            #
            self.df_share.loc[(fund_id, argv['share_id']), 'share'] -= argv['share']
            # 生成赎回上下文
            self.df_redeem.loc[(fund_id, argv['order_id'])] = {
                'share_id': argv['share_id'],
                'share': argv['share'],
                'nav': argv['nav'],
                'nav_date': argv['nav_date'],
                'ack_date': argv['ack_date']
            }

        elif op == 12:
            #
            # 赎回确认
            #
            self.cash += argv['share'] * argv['nav'] - argv['fee']
            self.df_redeem.drop((fund_id, argv['redeem_id']))

        elif op == 8:
            #
            # 调仓处理
            # 
            # df_share: 是当前状态；argv['pos']: 是目标状态
            #
            # 调仓逻辑里面，首先取消掉所有未提交的订单，然后在根据当前状态
            # 重新生成订单
            #
            self.remove_flying_op()
            orders = self.adjust(dt, argv['pos'])

            #
            # 将订单插入事件队列
            #
            for order in orders:
                argv = order
                if order['op'] == 1:
                    ev2 = (order['nav_date'] + timedelta(hours=16), 1, order['order_id'], order['fund_id'], argv)
                else:
                    ev2 = (order['nav_date'] + timedelta(hours=16, minutes=30), 2, order['order_id'], order['fund_id'], argv)
                result.append(ev2)

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
            ev2 = (argv['dividend_date'], 16, 0, fund_id, argv2)
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
            ev2 = (argv['payment_date'], 17, 0, fund_id, argv2)
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
            #
            # 理论上此时的净值为昨天的净值, 因为分拆是在调整今天净值之前处理的.
            #
            # 要保证后面日收益计算的正确, 这个地方需要对净值进行折算.
            #
            df['nav'] /= argv['fs_split_proportion']

        elif op == 100:
            #
            # 当日持仓相关的例行事件
            #
            result.extend(self.share_routine(dt, argv))
            
        elif op == 101:
            #
            # 记录持仓
            #
            if self.debug and self.df_share.sum().sum() > 0.000099:
                self.dt_holding[dt] = self.df_share.copy()
            #
            # 记录净值
            #
            amount = ((self.df_share['share'] + self.df_share['share_buying']) * self.df_share['nav'] + self.df_share['amount_bonusing']).sum()
            
            amount += (self.df_redeem['share'] * self.df_redeem['nav']).sum()
            self.nav[dt] = amount
            #
            # 记录业绩归因
            #
            self.contrib[dt] = self.df_share['yield'].groupby(level=1).sum()

        return result

    def make_bonus_order(self, dt, fund_id, order_share, bonus, div_mode):
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
            'order_id': self.new_order_id(dt),
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

    def adjust(self, dt, df_dst):
        '''
        生成调仓订单
        '''
        result = []
        #
        # 计算当前持仓的持仓比例
        #
        # 计算当前基金的持仓金额
        sr_src = ((self.df_share['share'] + self.df_share['share_buying']) * self.df_share['nav'] + self.df_share['amount_bonusing']).groupby(level=0).sum()
        # 计算用户总持仓金额
        total = sr_src.sum() + (self.df_redeem['share'] * self.df_redeem['nav']).sum() + self.cash
        # 计算当前基金的持仓比例（金额比例）
        df_src = (sr_src / total).to_frame('src_ratio')
        df_src.index.name = 'ra_fund_id'

        #
        # 计算各个基金的调仓比例和调仓金额
        #
        df = df_src.merge(df_dst, how='outer', left_index=True, right_index=True).fillna(0)
        df['diff_ratio'] = df['ra_fund_ratio'] - df['src_ratio']
        df['diff_amount'] = total * df['diff_ratio']

        #
        # 生成调仓订单。 具体地，
        #
        #     1. 首先生成赎回订单，针对每只基金，按照赎回费率少的优先
        #        的方式赎回。 因为基金的最小赎回份额和最小保留份额都是
        #        按整个基金算的，所以简单的贪心算法就可以生成最优的赎
        #        回方案。
        #
        #     2. 在赎回订单到账的基础上，按照调入金额的买入比例买入。
        #        理论上来说，这里有两种买入策略可以选择：（1）按照调入
        #        金额的比例买入；（2）优先买入单支基金；这里选择按比例
        #        买入，主要是考虑QDII的赎回到账时间可能会比较长（比如
        #        T+7），那么先到账的基金优先买入A股还是货币显然会引入
        #        比较大的差别。（由于我们不考虑最小申购金额的限制，所
        #        以按比例买入能够更好的提现调仓的本质）。如果考虑最小
        #        申购金额限制，可以采用能平均买入就平均，否则就买单支
        #        的方式）
        #
        #     3. 为了防止购买订单碎片化，我们同一天到账的金额进行了汇
        #        总，利用汇总的金额生成购买订单。
        #
        #     4. 需要考虑在途资金（也就是以前的赎回尚未到账的资金，因
        #        为跟该赎回相关的购买订单已经被需要，所以需要重新生成），
        #        生成方法也是根据在途资金的到账日期来下购买订单。
        #

        dt_flying = {}          # 在途资金

        #
        # 贪心算法生成赎回订单
        #
        for fund_id, row in df[df['diff_amount'] < 0].iterrows():
            amount = abs(row['diff_amount'])
            df_share_fund = self.df_share.loc[fund_id]
            count = len(df_share_fund.index)

            for k, v in df_share_fund.iterrows():
                fund_id, share_id = k
                redeem_amount = min(amount, (v['share'] + v['share_buying']) * v['nav'])
                redeem_share = redeem_amount / v['nav']
                if v['share'] > 0:
                    place_date = dt
                    # sr['share'] -= redeem_share
                    nav = v['nav']
                else:
                    place_date = v['ack_date']
                    # sr['share_buying'] -= redeem_share
                    nav = None
                #
                # 生成赎回订单
                #
                order = self.make_redeem_order(
                    place_date, fund_id, redeem_share, v['buy_date'], nav)
                result.append(order)
                #
                # 记录赎回到账金额，为购买记录做准备
                #
                if order['ack_date'] in dt_flying:
                    dt_flying[order['ack_date']] += order['ack_amount']
                else:
                    dt_flying[order['ack_date']] = order['ack_amount']
                #
                # 调整需要赎回的金额，看是否需要进一步赎回
                #
                amount -= redeem_amount
                if amount < 0.0001:
                    break

            #
            # 退出赎回循环，正确性检查
            #
            if amount >= 0.0001:
                # logger.error("SNH: bad redeem amount, something left: fund_id: %d, total: %f, left: %f", fund_id, abs(row['diff_amount']), amount)
                print "SNH: bad redeem amount, something left: fund_id: %d, total: %f, left: %f" % (fund_id, abs(row['diff_amount']), amount)
                sys.exit(0)
                
        #
        # 统计所有赎回到账日期和到账金额
        #
        df_new_redeem = pd.DataFrame({'new_redeem': dt_flying})
        df_new_redeem.index.name = 'date'
        df_old_redeem  = (self.df_redeem['share'] * self.df_redeem['nav']).groupby(by=self.df_redeem['ack_date']).sum().to_frame('old_redeem')
        
        df_flying = df_new_redeem.merge(df_old_redeem, left_index=True, right_index=True)
        df_flying['amount'] = df_flying['old_redeem'] + df_flying['new_redeem']

        #
        # 根据赎回到账日期，生成购买订单
        #
        # 就算不同基金的购买比例
        sr_ratio =  df.loc[df['diff_amount'] > 0, 'diff_amount']
        buy_total =  sr_ratio.sum()
        if buy_total > 0:
            sr_ratio = sr_ratio / buy_total
            # 现金持仓生成购买订单
            sr_buy = (sr_ratio * self.cash).round(2)
            for fund_id, amount in sr_buy.iteritems():
                order = self.make_buy_order(dt, fund_id, amount)
                result.append(order)
            
            # 在途资金生成购买订单
            for day, v in df_flying.iterrows():
                sr_buy = (sr_ratio * v['amount']).round(2)

                for fund_id, amount in sr_buy.iterrows():
                    order = self.make_buy_order(day, fund_id, amount)
                    result.append(order)

        #
        # 返回所有订单，即调仓计划
        #
        self.dump_orders(result, False)
        return result
                
            
    def make_redeem_order(self, dt, fund_id, share, buy_date, nav):
        if nav is None:
            (nav, nav_date) = self.get_tdate_and_nav(fund_id, dt)
        else:
            nav_date = dt

        amount = share * nav
        fee = self.get_redeem_fee(dt, fund_id, buy_date, amount)
        
        return {
            'order_id': self.new_order_id(dt),
            'fund_id': fund_id,
            # 'fund_code': fund_code,
            'op': 2,
            'place_date': dt,
            'place_time': '14:30:00',
            'share': share,
            'amount': amount,
            'fee': fee,
            'nav': nav,
            'nav_date': nav_date,
            'ack_date': self.get_redeem_ack_date(fund_id, nav_date), 
            'ack_amount': amount - fee,
            'ack_share': share,
            'div_mode': 0,
        }

    def get_tdate_and_nav(self, fund_id, day):
        '''
        获取day对应的交易日和该交易日基金的净值。

        对于国内基金，情形比较直接，无需废话。但对于QDII基金：

        （1）美股基金（比如大成标普096001），购买份额确认和赎回的到账
             的T+N采用的A股交易日，如果遇到A股交易日和美股非交易日，一
             般这类基金会暂停申购赎回。

        （2）港股基金（比如华夏恒生000071），同美股一样。也即，购买份
             额确认和赎回的到账的T+N采用的A股交易日，如果遇到A股交易日
             和美股非交易日，一般这类基金会暂停申购赎回。

        （3）黄金基金（比如华安黄金000217），同A股一样，无需特殊处理。

        '''
        if fund_id not in self.dt_nav:
            print "SNH: missing nav: fund_id: %d, day: %s" % (fund_id, day.strftime("%Y-%m-%d"))
            sys.exit(0)

        df_nav = self.dt_nav[fund_id]

        return df_nav.loc[day.date(), ['ra_nav', 'ra_nav_date']]

    def get_redeem_fee(self, dt, fund_id, buy_date, amount):
        '''
        获取赎回费用

        如果在赎回费用表中不存在， 则默认无费率. 此外，需要注意的是基
        金赎回的持有日期是按自然日算的。

        赎回费计算公式（来自中国证监会网站）：
            赎回总金额=赎回份额×T日基金份额净值
            赎回费用=赎回总金额×赎回费率

        '''
        df = self.df_redeem_fee.loc[fund_id]

        if df.empty: return 0

        # 持有日期
        ndays = (dt - buy_date).days

        sr = df.loc[df['ff_max_value'] >= ndays].iloc[0]
        if sr['ff_fee_type'] == 2: # 固定费用模式，一般是0元，持有期足够长，赎回免费
            fee = sr['ff_fee']
        else:
            fee = amount * sr['ff_fee'] # 标准费率计算方式

        return fee

    def get_redeem_ack_date(self, fund_id, tdate):
        '''
        获取赎回到账的A股交易日

        所有的赎回到账以A股交易日为准。目前，我们涉及的QDII基金的赎回到账是按A股交易日计。
        '''

        #
        # 涉及到两个全局数据:
        #   df_ack 记录了每个基金的到账日是T+n；
        #   df_t_plus_n 记录了每个交易日的t+n是哪一天
        #
        n = 3 # 默认t+3到账
        if fund_id in self.df_ack.index:
            n = self.df_ack.at[fund_id, 'redeem']
        else:
            print "WARN: missing yingmi_to_account_time, use default(t+3): fund_id: %d" % fund_id

            
        ack_date = self.df_t_plus_n.at[tdate, "T+%d" % n]

        return ack_date

    def make_buy_order(self, dt, fund_id, amount):
        (nav, nav_date) = self.get_tdate_and_nav(fund_id, dt)

        fee = self.get_buy_fee(dt, fund_id, amount)
        ack_amount = amount - fee
        share = ack_amount / nav
        
        return {
            'order_id': self.new_order_id(dt),
            'fund_id': fund_id,
            # 'fund_code': fund_code,
            'op': 1,
            'place_date': nav_date,
            'place_time': '14:30:00',
            'share': share,
            'amount': amount,
            'fee': fee,
            'nav': nav,
            'nav_date': nav_date,
            'ack_date': self.get_buy_ack_date(fund_id, nav_date), 
            'ack_amount': amount - fee,
            'ack_share': share,
            'div_mode': 1,
        }

    def get_buy_fee(self, dt, fund_id, amount):
        '''
        获取申购费用和净申购金额

        计算公式如下（来自中国证监会官网）：
        
            净申购金额=申购金额/（1+申购费率）
            申购费用=申购金额-净申购金额

        '''
        if fund_id not in self.df_buy_fee.index:
            return 0
        
        df = self.df_buy_fee.loc[fund_id]

        sr = df.loc[df['ff_max_value'] > amount].iloc[0]
        if sr['ff_fee_type'] == 2: # 固定费用模式，一般是申购额足够大，费用封顶
            fee = sr['ff_fee']
        else:
            fee = amount - amount / (1 + sr['ff_fee']) # 标准费率计算方式

        return fee

    def get_buy_ack_date(self, fund_id, buy_date):
        '''
        获取购买可赎回日期

        所有的可赎回日期以A股交易日为准。目前，我们涉及的QDII基金的可赎回日期是按A股交易日计。
        '''
        #
        # 涉及到两个全局数据:
        #   df_ack 记录了每个基金的可赎回是T+n；
        #   df_t_plus_n 记录了每个交易日的t+n是哪一天
        #
        n = 2 # 默认t+2到账
        if fund_id in self.df_ack.index:
            n = self.df_ack.at[fund_id, 'buy']
        else:
            print "WARN: missing yingmi_to_confirm_time, use default(t+2): fund_id: %d" % fund_id
            
        ack_date = self.df_t_plus_n.at[buy_date, "T+%d" % n]

        return ack_date

    def share_routine(self, dt, argv):
        '''
        生成与当日持仓相关的例行事件
        '''
        evs = []
        fund_ids = self.df_share.index.get_level_values(0).unique().tolist()

        #
        # 清空当日订单计数
        #
        self.dsn = 0
        
        day = pd.to_datetime(dt.date())
        if len(fund_ids) != 0:
            #
            # 如果当日有分红事件
            #
            if day in self.df_bonus.index.levels[0]:
                df = self.df_bonus.loc[(day, fund_ids), :]
                for key, row in df.iterrows():
                    record_date, fund_id = key
                    argv = {
                        'bonus_ratio': row['ra_bonus'],
                        'bonus_nav': row['ra_bonus_nav'],
                        'bonus_nav_date': row['ra_bonus_nav_date'],
                        'dividend_date': row['ra_dividend_date'] + timedelta(hours=22,minutes=30),
                        'payment_date': row['ra_payment_date'] + timedelta(hours=22,minutes=45),
                    }
                    ev = (record_date + timedelta(hours=22), 15, 0, fund_id, argv)
                    # dd(row['ra_record_date'] + timedelta(hours=22), argv)
                    evs.append(ev)

            # 
            # 加载分拆信息
            # 
            #
            if day in self.df_bonus.index.levels[0]:
                df = self.df_bonus.loc[(day, fund_ids), :]
                for key, row in df.iterrows():
                    split_date, fund_id = key
                    argv = {'ra_split_proportion': row['ra_split_proportion']}
                    ev = (split_date + timedelta(hours=1), 18, 0, fund_id, argv)
                    evs.append(ev)

            # 
            # 加载基金净值
            #
            if day in self.df_nav.index.levels[0]:
                df = self.df_nav.loc[(day, fund_ids), :]
                # dd(df, day, type(fund_ids))
                for key, row in df.iterrows():
                    day, fund_id = key
                    if row['ra_nav']:
                        argv = {'nav': row['ra_nav'], 'nav_date': day}
                        ev = (day + timedelta(hours=15), 0, 0, fund_id, argv)
                        evs.append(ev)
                    else:
                        logger.error('zero ra_nav detected(ra_fund_id: %d, ra_date:%s)', fund_id, day.strftime("%Y-%m-%d"))

        #
        # 记录持仓事件
        #
        argv = {}
        ev = (day + timedelta(hours=23, minutes=59, seconds=59), 101, 0, 0, argv)
        evs.append(ev)

        return evs

    def remove_flying_op(self):
        pass

    def new_order_id(self, dt):
        '''
        获取下一个订单号
        '''
        self.dsn  += 1

        return "%s%02d" % (dt.strftime("%Y%m%d"), self.dsn)

    def dump_orders(self, orders, die=True):
        if len(orders) > 0:
            print "order|%10s|%2s|%8s|%10s|%10s|%8s|%10s|%6s|%7s|%10s|%10s|%10s|%10s" % (
                "order_id", 'op', 'fund_id', 'place_date', 'place_time', 'amount', 'share', 'fee', 'nav', 'nav_date', 'ack_date', 'ack_amount', 'ack_share')
        for o in orders:
            #     （order_id, fund_id, fund_code, op, place_date, place_time, amount, share, fee, nav, nav_date, ack_date, ack_amount, ack_share, div_mode）
            print "order|%s|%2d|%d|%s|%10s|%8.2f|%10.4f|%6.2f|%7.4f|%s|%s|%10.2f|%10.4f" % (
                o['order_id'], o['op'], o['fund_id'], o['place_date'].strftime("%Y-%m-%d"), o['place_time'], o['amount'], o['share'], o['fee'], o['nav'],o['nav_date'].strftime("%Y-%m-%d"),o['ack_date'].strftime("%Y-%m-%d"), o['ack_amount'], o['ack_share'])

        if die:
            dd('----end orders----')

    def dump_event(self, x, e, die=False):
        dt, op, order_id, fund_id, argv = e
        if op == 0:
            s = "nav:%.4f, nav_date:%s" % (argv['nav'], argv['nav_date'].strftime("%Y-%m-%d"))
        elif op == 1:
            s = "at:%s %s, share:%.4f, nav:%.4f, fee:%.2f" % (argv['place_date'].strftime("%Y-%m-%d"), argv['place_time'], argv['share'], argv['nav'], argv['fee'])
        elif op == 11:
            s = "ack_date:%s, share:%.4f " % (argv['ack_date'].strftime("%Y-%m-%d"), argv['share'])
        elif op == 2:
            s = "at:%s %s, share:%.4f, nav:%.4f, fee:%.2f" % (argv['place_date'].strftime("%Y-%m-%d"), argv['place_time'], argv['share'], argv['nav'], argv['fee'])
        elif op == 12:
            s = "ack_date:%s, share:%.4f " % (argv['ack_date'].strftime("%Y-%m-%d"), argv['share'])
        elif op == 8:
            s = ''
        elif op == 15:
            s = ''
        elif op == 16:
            s = ''
        elif op == 17:
            s = ''
        elif op == 18:
            s = ''
        elif op == 100:
            s = ''
        elif op == 101:
            s = ''
        
        print "%c xev|%20s|%3d|%10s|%10s|%s" % (
                x, dt.strftime("%Y-%m-%d %H:%M:%S"), op, order_id, fund_id, s)

        if die:
            dd('----end orders----')
        

