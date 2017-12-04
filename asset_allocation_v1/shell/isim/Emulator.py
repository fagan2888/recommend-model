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

# 1000270128
# 1000134808

ST_SUB = 1
ST_REDEEM = 2
ST_SUB_FEE = 3
ST_REDEEM_FEE = 4
ST_YIELD = 5
ST_UNCARRY = 6
ST_BONUS_OUT = 7
ST_BONUS_IN = 8
ST_CARRY = 9
ST_CARRY_FORCE = 10
ST_ADD_FORCE = 11
ST_DEL_FORCE = 12

def xfun_last(x, y, ph=False):
    return y.combine_first(x)

def dump_orders(orders, s, die=True):
    if len(orders) > 0:
        # print "\n", s
        logger.info("order|%10s|%2s|%8s|%10s|%10s|%8s|%10s|%8s|%7s|%10s|%10s|%10s|%10s|%10s" % (
            "order_id", 'op', 'fund_code', 'place_date', 'place_time', 'amount', 'share', 'fee', 'nav', 'nav_date', 'ack_date', 'ack_amount', 'ack_share', 'share_id'))
    for o in orders:
        #     （order_id, fund_code, fund_code, op, place_date, place_time, amount, share, fee, nav, nav_date, ack_date, ack_amount, ack_share, div_mode）
        logger.info("order|%s|%2d|%s|%s|%10s|%8.6f|%10.6f|%6.6f|%7.4f|%s|%s|%10.6f|%10.6f|%10s" % (
            o['order_id'], o['op'], o['fund_code'], o['place_date'].strftime("%Y-%m-%d"), o['place_time'], o['amount'], o['share'], o['fee'], o['nav'],o['nav_date'].strftime("%Y-%m-%d"),o['ack_date'].strftime("%Y-%m-%d"), o['ack_amount'], o['ack_share'], o['share_id']))

    if die:
        dd('----end orders----')

def dump_event(x, e, die=False):
    dt, op, order_id, fund_code, argv = e
    if op == 0:
        s = "nav:%.4f, nav_date:%s" % (argv['nav'], argv['nav_date'].strftime("%Y-%m-%d"))
    elif op == 3:
        s = "amount:%.4f" % (argv['ts_placed_amount'])
    elif op in [30, 31, 63]:
        s = "at:%s, amount:%.4f, fee:%.2f" % (argv['ts_scheduled_at'].strftime("%Y-%m-%d %H:%M:%S"), argv['ts_placed_amount'], argv['ts_placed_fee'])
    elif op == 11:
        s = "ack_date:%s, share:%.4f " % (argv['ack_date'].strftime("%Y-%m-%d"), argv['share'])
    elif op == 2:
        s = "at:%s %s, share:%.4f, nav:%.4f, fee:%.2f" % (argv['place_date'].strftime("%Y-%m-%d"), argv['place_time'], argv['share'], argv['nav'], argv['fee'])
    elif op == 12:
        s = "ack_date:%s, share:%.4f " % (argv['ack_date'].strftime("%Y-%m-%d"), argv['share'])
    else:
        s = ''

    logger.info("%c xev|%20s|%3d|%10s|%10s|%s" % (
            x, dt.strftime("%Y-%m-%d %H:%M:%S"), op, order_id, fund_code, s))

    if die:
        dd('----end events----')
    
class Emulator(object):
    
    def __init__(self, investor, policy, rule, debug=True, optfee=True, optt0=False):

        # 调试模式 会记录整个计算的上下文，默认为关闭
        self.investor = investor
        self.policy = policy
        self.rule = rule
        self.debug = debug
        self.optfee = optfee
        self.optt0 = optt0

        #
        # 状态变量：用户当前持仓: DataFrame
        # 
        #      columns        share, yield, share_buying, buy_date, ack_date,  nav, nav_date, share_bonusing amount_bonusing div_mode
        #      index
        # (fund_code, share_id)
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
        # self.df_share =  pd.DataFrame({
        #     'ts_order_id': pd.Series(dtype=str),
        #     'ts_fund_code': pd.Series(dtype=str),
        #     'ts_date':  pd.Series(dtype='datetime64[ns]'),
        #     'ts_nav': pd.Series(dtype=float),
        #     'ts_share': pd.Series(dtype=float),
        #     'ts_amount': pd.Series(dtype=float),
        #     'ts_trade_date': pd.Series(dtype='datetime64[ns]'),
        #     'ts_acked_date': pd.Series(dtype='datetime64[ns]'),
        #     'ts_redeemable_date': pd.Series(dtype='datetime64[ns]'),
        # },
        #     index=pd.MultiIndex(names=['fund_code','share_id'], levels=[[], []], labels=[[],[]]),
        #     columns=['ts_order_id', 'ts_fund_code', 'ts_date', 'ts_nav', 'ts_share', 'ts_amount', 'ts_trade_date', 'ts_acked_date', 'ts_redeemable_date'],
        # )
        # dd(self.df_share, self.df_share['nav'], self.df_share['nav_date'])
        self.df_share = None

        # self.df_share_buying = pd.DataFrame({
        #     'ts_order_id': pd.Series(dtype=str),
        #     'ts_fund_code': pd.Series(dtype=str),
        #     'ts_trade_date': pd.Series(dtype='datetime64[ns]'),
        #     'ts_acked_date': pd.Series(dtype='datetime64[ns]'),
        #     'ts_redeemable_date': pd.Series(dtype='datetime64[ns]'),
        #     'ts_nav': pd.Series(dtype=float),
        #     'ts_share': pd.Series(dtype=float),
        #     'ts_amount': pd.Series(dtype=float),
        # },
        #     index=pd.MultiIndex(names=['fund_code','order_id'], levels=[[], []], labels=[[],[]]),
        #     columns=['ts_order_id', 'ts_fund_code', 'ts_trade_date', 'ts_acked_date', 'ts_redeemable_date', 'ts_nav', 'ts_share', 'ts_amount'],
        # )
        self.df_share_buying = None
        self.df_share_redeeming = None
        # self.df_uncarried = None
        self.sr_uncarried = None

        #
        # 状态变量：赎回上下文：DataFrame
        #      columns        share_id, share,  nav, nav_date, ack_date
        #      index
        # (fund_code, redeem_id)
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
            {'share_id': pd.Series(dtype=str), 'share': pd.Series(dtype=float), 'nav': pd.Series(dtype=float), 'nav_date':pd.Series(dtype='datetime64[ns]'), 'ack_date': pd.Series(dtype='datetime64[ns]'), 'fee': pd.Series(dtype=float)},
            index=pd.MultiIndex(names=['fund_code','redeem_id'], levels=[[], []], labels=[[],[]]),
            columns=['share_id', 'share', 'nav', 'nav_date', 'ack_date', 'fee']
        )

        #
        # 状态变量：当日订单计数
        #
        self.dsn = 0 # day sequenece no
        self.day = 0 # 当天日期
        self.ts = pd.to_datetime('1990-01-01')  # 当前模拟时钟
        self.df_stat = None
        self.sr_holding_last = None
        
        #
        # 输出变量：历史持仓
        #
        self.dt_holding = {}
        self.dt_stat = {}

        # 
        # 输出变量：订单列表，用于保存整个过程中产生的订单, 每个订单是一个字典，包含如下项目
        #     （order_id, fund_code, fund_code, op, place_date, place_time, amount, share, fee, nav, nav_date, ack_date, ack_amount, ack_share, div_mode, share_id）
        # 
        self.orders = []

        self.df_ts_order = None
        self.df_ts_order_fund = None

        #
        # 输出变量：业绩归因序列
        #
        self.contrib = {}
        self.dt_today_fee_buy = {}   # 业绩归因的当日申赎上下文
        self.dt_today_fee_redeem = {}   # 业绩归因的当日申赎上下文
        self.dt_today_bonus = {} # 业绩归因的当日分红上下文

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
        

    def run(self):

        self.cash = 0
        sdate = self.investor.get_sdate()
        edate = self.investor.get_edate()
        if edate is None:
            # edate = pd.to_datetime(datetime.now().date()) - timedelta(days=1)
            edate = pd.to_datetime(datetime.now().date())

        self.sdate = sdate
        self.edate = edate
        
        #
        # 初始化用到的计算数据
        #

        #
        # 事件类型:0:净值更新;2:赎回;7:分红;8:调仓;11:申购确认;12:赎回到账;15:分红登记;16:分红除息;17:分红派息;18:基金分拆;
        #         3: 组合购买
        #         4: 组合赎回
        #         6: 组合调仓
        #         30: 银行卡购买
        #         31: 钱包购买
        #         100:例行事件生成;101:记录当前持仓;102:每天3点的例行事件;103:每天2:50的例行事件
        #
        # 事件队列，基于heapq实现，元组来实现
        #
        #     (dt, op, order_id, fund_code, argv)
        #
        # 其中，
        #
        #     dt： 日内事件处理时间，主要用于事件在队列中的排序，不一
        #          定等同于下单时间。比如：3:00pm前下的购买订单的处理
        #          时间可能是16:00，因为15:00净值才出来，而购买赎回都
        #          需要使用当天的净值。
        #
        #          各个事件的处理时间如下：
        #
        #          00:00:01 生成持仓相关的当日例行事件（更新净值，分拆，分红，记录当日持仓）
        #          01:00:00 分拆
        #          02:00:00 购买确认
        #          03:00:00 赎回确认
        #          15:00:00 更新净值
        #          16:00:00 分红权益登记
        #          16:30:00 分红除息
        #          16:45:00 分红派息
        #          17:00:00 调仓处理
        #          18:00:00 购买
        #          18:30:00 赎回
        #          23:59:59 记录当日持仓
        #
        #          时间的处理顺序之所以这样排序，原因如下：
        #
        #          由于分拆日的净值是日末净值，也就是分拆之后的，所以
        #          分拆需要在更新净值之前进行。
        #
        #          购买的确认和赎回确认也需要在更新净值在前进行，因为T
        #          日确认的购买份额在T日是可以赎回的，所以应该在开盘之
        #          前完成结算。T日确认的赎回份额，在T日不再计算收益，
        #          所以需要在在更新净值之前完成
        #
        #          理论上，分拆和购买/赎回确认的顺序可以互换，甚至先处
        #          理确认，再处理分拆可能更合理，但代码写出先分拆了，
        #          因为不影响最后结果，也就不改了。
        #
        #          分红，申购，赎回依赖当日净值，所以需要在净值更新之
        #          后进行。具体的，红利在投资的份额是依据除息日的净值
        #          的。申购和赎回的是按T日的净值计算金额的。
        #
        #          分红需要在购买和赎回之前进行，因为权益登记日购买的
        #          份额是不享受分红的，权益登记赎回的份额是享受分红的。
        #
        #          分红本身的权益登记、除息、派息需要按顺序进行，因为
        #          三个日子可能是一天。
        # 
        #
        self.events = [];

        # 未来交易日 记录了每个交易日的t+n是哪个交易日
        dates = base_trade_dates.load_index(sdate, edate)
        self.df_t_plus_n = pd.DataFrame(dates, index=dates, columns=["td_date"])
        for i in xrange(0, 10 + 1):
            self.df_t_plus_n["T+%d" % i] = self.df_t_plus_n['td_date'].shift(-i)
        self.df_t_plus_n = self.df_t_plus_n.drop('td_date', axis=1)
        self.df_t_plus_n.fillna(pd.to_datetime('2029-01-01'), inplace=True)

        # 调整edate, 我们只算到最后一个交易日
        self.edate = edate = self.df_t_plus_n.index.max()
        days = pd.date_range(sdate, edate)
        self.df_t_plus_n = self.df_t_plus_n.reindex(days, method='bfill')
        
        #
        # 生成每日例行事件的事件
        #
        dates = pd.date_range(sdate, edate)
        for day in dates:
            argv = {}
            ev = (day + timedelta(seconds=1), 100, 0, 0, argv)
            heapq.heappush(self.events, ev)

        self.trade_date_last = None
        self.trade_date_current = self.df_t_plus_n.loc[sdate, "T+0"]
            
        #
        # 依次处理每个事件
        #
        max_ts = edate + timedelta(hours=24)
        while self.events:
            ev = heapq.heappop(self.events)
            if ev is None:
                break

            if self.debug:
                dump_event('-', ev)
            evs = self.process(ev)

            for ev in evs:
                if ev[0] >= max_ts:
                    continue
                if self.debug:
                    dump_event('+', ev)
                heapq.heappush(self.events, ev)


        #
        # 保存计算结果
        #
        # 本模块不负责保存具体的计算结果
        #
        # df_result_nav = pd.Series(self.nav).to_frame('nav') 
        # dd(df_result_nav, "completed")
        # print self.df_share
        return self.df_share

    def process(self, ev):
        result = []
        # 事件类型:0:净值更新;2:赎回;7:分红;8:调仓;11:申购确认;12:赎回到账;15:分红登记;16:分红除息;17:分红派息;18:基金分拆;
        #         3:组合购买
        #         30: 银行卡购买
        #         31: 钱包购买
        #         100:例行事件生成;101:记录当前持仓;
        dt, op, order_id, fund_code, argv = ev

        #
        # 更新模拟时钟
        #
        if dt >= self.ts:
            self.ts = dt
        else:
            #dd("SNH: ev out of date", self.ts, ev)
            pass

        if op == 0:
            #
            # 净值更新
            #
            # if fund_code in self.df_share.index.levels[0]:
            #     # if dt.strftime("%Y-%m-%d") == '2014-11-11' and fund_code == 30000237:
            #     #     pdb.set_trace()
            #     # df = self.df_share.query('fund_code in [%d]' % fund_code)
            #     df = self.df_share.loc[self.df_share.index.get_level_values(0) == fund_code]
            #     self.df_share.loc[fund_code, 'yield'] = (df['share'] + df['share_buying']) * (argv['nav'] - df['nav'])
            #     self.df_share.loc[fund_code, 'nav'] = argv['nav']
            #     self.df_share.loc[fund_code, 'nav_date'] = argv['nav_date']
            # else:
            #     dd("SNH: fund_code not in df_share.index", fund_code, self.df_share)
            pass

        if op == 3:
            #
            # 处理组合购买
            #
            ts_order, ts_order_fund = self.policy.place_buy_order(dt, argv)
            
            if self.df_ts_order is None:
                self.df_ts_order = pd.DataFrame([ts_order], index=[ts_order['ts_txn_id']])
            else:
                self.df_ts_order.loc[ts_order['ts_txn_id']] = ts_order

            if self.df_ts_order_fund is None:
                self.df_ts_order_fund = ts_order_fund
            else:
                dd(self.df_ts_order_fund, ts_order_fund, ev)
                self.df_ts_order_fund = ts_order_fund.combine_first(self.df_ts_order_fund)
                self.df_ts_order_fund['ts_trade_type'] = self.df_ts_order_fund['ts_trade_type'].astype(int)
                self.df_ts_order_fund['ts_trade_status'] = self.df_ts_order_fund['ts_trade_status'].astype(int)
            
            result.extend(self.issue_order_place_event())
            
        elif op in [30, 31, 63]:
            result.extend(self.place_buy_order(dt, order_id, fund_code, argv))

        elif op in [40, 41, 64]:
            result.extend(self.place_redeem_order(dt, order_id, fund_code, argv))
            
        elif op == 6:
            #
            # 处理组合调仓
            #
            # pdb.set_trace()
            ts_order, ts_order_fund = self.policy.place_adjust_order(dt, argv)
            
            if self.df_ts_order is None:
                self.df_ts_order = pd.DataFrame([ts_order], index=[ts_order['ts_txn_id']])
            else:
                self.df_ts_order.loc[ts_order['ts_txn_id']] = ts_order

            if self.df_ts_order_fund is None:
                self.df_ts_order_fund = ts_order_fund
            else:
                self.df_ts_order_fund = ts_order_fund.combine_first(self.df_ts_order_fund)
                self.df_ts_order_fund['ts_trade_type'] = self.df_ts_order_fund['ts_trade_type'].astype(int)
                self.df_ts_order_fund['ts_trade_status'] = self.df_ts_order_fund['ts_trade_status'].astype(int)
            
            result.extend(self.issue_order_place_event())
               

        # elif op == 2:
        #     #
        #     # 赎回
        #     #
        #     left = self.df_share.loc[(fund_code, argv['share_id']), 'share'] - argv['share']
        #     self.df_share.loc[(fund_code, argv['share_id']), 'share'] = left if left > 0.00000099 else 0
        #     # 生成赎回上下文
        #     row = (fund_code, argv['order_id'], argv['share_id'], argv['share'], argv['nav'], argv['nav_date'], argv['ack_date'], argv['fee'])
        #     tmp = pd.DataFrame([row], columns=['fund_code', 'redeem_id', 'share_id', 'share', 'nav', 'nav_date', 'ack_date', 'fee'])
        #     self.df_redeem = self.df_redeem.append(tmp.set_index(['fund_code', 'redeem_id']))
        #     # 记录赎回费率
        #     self.dt_today_fee_redeem[fund_code] = self.dt_today_fee_redeem.setdefault(fund_code, 0) + (-argv['fee'])

        #     #
        #     # 生成赎回确认事件
        #     #
        #     if argv['ack_date'].date() == dt.date():
        #         hours = 19
        #     else:
        #         hours = 2
                
        #     ev2 = (argv['ack_date'] + timedelta(hours=hours), 12, order_id, fund_code, argv)
        #     result.append(ev2)

        elif op == 15:
            #
            # 基金分红：权益登记
            #
            # [XXX] 这里啰嗦几句基金分红的处理：基金的分红方式有两种，
            # 现金分红和红利再投。
            #
            # 对于红利再投的方式，再投份额的折算发生在除息日，按照除息
            # 日的净值，但该部分份额不可赎回，直到派息日方可赎回。
            #
            # 对于现金分红方式，除息日扣除相应的净值，派息日打款
            #  
            #
            df = self.df_share.loc[[fund_code]]
            df['share_bonusing'] = df['share'] + df['share_buying']
            # if dt.strftime("%Y-%m-%d") == '2013-05-16':
            #     pdb.set_trace()
        
            #
            # 生成分红订单，记录分红操作
            #
            share_dividend = df.loc[df['div_mode'] == 1, 'share_bonusing'].sum()
            share_cash = df['share_bonusing'].sum() - share_dividend
            orders, dividend_id, cash_id = [], "0", "0"
            if share_dividend > 0.000000001:
                order = self.make_bonus_order(dt, fund_code, share_dividend, argv, 1)
                orders.append(order)
                dividend_id = order['order_id']

            if share_cash > 0.000000001:
                order = self.make_bonus_order(dt, fund_code, share_cash, argv, 0)
                orders.append(order)
                cash_id = order['order_id']
            if orders:
                dump_orders(orders, "bonus %s" % dt.strftime("%Y-%m-%d"), False)
                self.orders.extend(orders)

            self.df_share.loc[[fund_code]] = df
            # dd(orders, argv, self.df_share.loc[[fund_code]], dt, share_dividend, share_cash, df)
            
            #
            # 调度除息事件
            #
            share = df['share_bonusing'].sum()
            if share > 0.000000001:
                argv2 = {
                    'share': share,
                    'bonus_ratio': argv['bonus_ratio'],
                    'bonus_nav': argv['bonus_nav'],
                    'bonus_nav_date': argv['bonus_nav_date'],
                    'payment_date': argv['payment_date'],
                    'order_dividend': dividend_id,
                    'order_cash': cash_id,
                }
                ev2 = (argv['dividend_date']+ timedelta(hours=16,minutes=30), 16, 0, fund_code, argv2)
                result.append(ev2)

        elif op == 16:
            #
            # 基金分红：除息
            #
            # pdb.set_trace()
            df = self.df_share.loc[[fund_code]]
            df['amount_bonusing'] = df['share_bonusing'] * argv['bonus_ratio']
            ev2 = (argv['payment_date'] + timedelta(hours=16,minutes=45), 17, 0, fund_code, argv)
            result.append(ev2)
            # 记录分红业绩
            self.dt_today_bonus[fund_code] = self.dt_today_bonus.setdefault(fund_code, 0) + df['amount_bonusing'].sum()
            
            #
            # 处理红利再投的份额转换
            #
            mask = (df['div_mode'] == 1)
            share = df.loc[mask, 'amount_bonusing'].sum() /  argv['bonus_nav']
            #
            # 清除红利再投上下文
            #
            df.loc[mask, 'share_bonusing']  = 0
            df.loc[mask, 'amount_bonusing'] = 0

            self.df_share.loc[[fund_code]] = df
            
            #
            # 生成再投份额
            #
            df_tmp = self.make_share(
                fund_code, argv['order_dividend'], share, pd.to_datetime(dt.date()), argv['bonus_nav'], argv['bonus_nav_date'], argv['payment_date'], 1)
            self.df_share = self.df_share.append(df_tmp)

            # tmp = {
            #     'share': 0.0,
            #     'yield': 0.0,
            #     'share_buying': share,
            #     'nav': argv['bonus_nav'],
            #     'nav_date': argv['bonus_nav_date'],
            #     'buy_date': pd.to_datetime(dt.date()),
            #     'ack_date': argv['payment_date'],
            #     'share_bonusing': 0.0,
            #     'amount_bonusing': 0.0,
            #     'div_mode': 1,
            # }
            # self.df_share.loc[(fund_code, argv['order_dividend']), :] = tmp
            
            

        elif op == 17:
            # pdb.set_trace()
            #
            # 基金分红：派息
            #
            if fund_code in self.df_share.index.get_level_values(0):
                df = self.df_share.loc[[fund_code]]
                #
                # 现金分红
                #
                mask = (df['div_mode'] == 0)
                if mask.any():
                    self.cash_bounused = df.loc[mask, 'amount_bonusing'].sum()
                    #
                    # 清除现金分红上下文
                    #
                    df.loc[mask, 'share_bonusing']  = 0
                    df.loc[mask, 'amount_bonusing'] = 0

                #
                # 红利再投, 确认红利再投份额
                #
                if argv['order_dividend'] in df.index.get_level_values(1):
                    df.loc[(fund_code, argv['order_dividend']), 'share'] = df.loc[(fund_code, argv['order_dividend']), 'share_buying']
                    df.loc[(fund_code, argv['order_dividend']), 'share_buying'] = 0

                self.df_share.loc[[fund_code]] = df

        elif op == 18:
            #
            # 基金分拆
            #
            
            df = self.df_share.loc[[fund_code]]
            df['share'] *= argv['ra_split_proportion']
            df['share_buying'] *= argv['ra_split_proportion']
            #
            # 理论上此时的净值为昨天的净值, 因为分拆是在调整今天净值之前处理的.
            #
            # 要保证后面日收益计算的正确, 这个地方需要对净值进行折算.
            #
            df['nav'] /= argv['ra_split_proportion']
            self.df_share.loc[[fund_code]] = df
            

        elif op == 100:
            #
            # 当日持仓相关的例行事件
            #
            result.extend(self.share_routine(dt, argv))
            
        elif op == 101:
            # if dt.strftime("%Y-%m-%d") in ['2014-02-10']:
            #     pdb.set_trace()
            day = pd.to_datetime(dt.date())
            # if dt.strftime("%Y-%m-%d") == '2016-10-10':
            #     pdb.set_trace()
            result.extend(self.record_holding(dt, argv))

            # self.idebug(dt)
        elif op == 102:
            #
            # 每日15:00例行事件
            #
            result.extend(self.routine_15pm(dt, argv))

        elif op == 103:
            #
            # 每日02:50调仓跑批事件
            #
            result.extend(self.routine_0259pm(dt, argv))

        else:
            dd("unknown event", ev)
            
        return result


    def routine_15pm(self, dt, argv):
        '''
        生成与当日15例行事件
        '''
        evs=[]
        codes = set([])
        if self.df_share is not None:
            codes = codes.union(set(self.df_share['ts_fund_code']))
        if self.df_share_buying is not None:
            codes = codes.union(set(self.df_share_buying['ts_fund_code']))

        for code in codes:
            nav = self.rule.get_nav(code, self.day)
            if nav is not None:
                evs.extend(self.handle_nav_update(code, nav))

        # if dt == pd.to_datetime('2017-06-26 15:00:00'):
        #     pdb.set_trace()

        #
        # 切换交易日
        #
        tomorrow = dt + timedelta(hours=9)

        if tomorrow <= self.edate:
            # dd(self.df_t_plus_n)
            self.trade_date_last = self.trade_date_current
            self.trade_date_current = self.df_t_plus_n.loc[tomorrow, "T+0"]
        
        return evs

    def routine_0259pm(self, dt, argv):
        '''
        生成与当日15例行事件
        '''
        evs=[]

        # if dt == pd.to_datetime('2017-06-28')
        
        evs.extend(self.issue_order_place_event())

        return evs

    def share_routine(self, dt, argv):
        '''
        生成与当日持仓相关的例行事件
        '''
        #
        # 清空当日订单计数
        #
        self.dsn = 0
        self.day = pd.to_datetime(dt.date())
        # print "day", dt.strftime("%Y-%m-%d")


        #
        # 清空当日收益
        #
        print dt, self.df_stat
        self.df_stat = None
        self.dt_today_fee_buy = {}
        self.dt_today_fee_redeem = {}
        self.dt_today_bonus = {}

        # if dt.strftime("%Y-%m-%d") == '2013-05-16':
        #     pdb.set_trace()
        

        evs = []
        day = pd.to_datetime(dt.date())

        #
        # 生成3点例行事件
        #
        evs.append((day + timedelta(hours=14, minutes=59, seconds=59), 103, 0, 0, {}))
        evs.append((day + timedelta(hours=15), 102, 0, 0, {}))

        #
        # 如果当日有购买
        #
        buys = self.investor.get_buys(day);
        evs.extend([(x['dt'], 3, x['ts_txn_id'], 0, x['argv']) for x in buys]);
        
        #
        # 如果当日有赎回
        #
        redeems = self.investor.get_redeems(day)
        evs.extend([(x['dt'], 4, 0, 0, x['argv']) for x in redeems]);
        
        #
        # 如果当日有调仓
        #
        adjusts = self.investor.get_adjusts(day)
        evs.extend([(x['dt'], 6, 0, 0, x['argv']) for x in adjusts]);

        #
        # 记录持仓事件
        #
        argv = {}
        ev = (day + timedelta(hours=23, minutes=59, seconds=59), 101, 0, 0, argv)
        evs.append(ev)

        #
        # 处理当天需要确认的订单
        #
        evs.extend(self.handle_order_confirm(dt, argv))

        #
        # 处理当天需要下单的订单
        #
        evs.extend(self.handle_order_continue(dt, argv))

        return evs


    def idebug(self, dt):
        while True:
            line = raw_input('enter debug [%s]. Command?  (s: share, r: redeem. Enter to continue, q to quit) ' % dt.strftime("%Y-%m-%d"))
            if line == '':
                break
            elif line == 'q':
                sys.exit(0)
            elif line == 's':
                print ""
                print self.df_share
                print ""
            elif line == 'r':
                print ""
                print self.df_redeem
                print ""
            else:
                print 'unknown command "%s"' % line

    def issue_order_place_event(self):
        #
        # 处理下单
        #
        evs = []

        mask = (self.df_ts_order_fund['ts_trade_status'] == 0)
        df = self.df_ts_order_fund.loc[mask]
        # if not df.empty:
        #     pdb.set_trace()
        for k, v in df.iterrows():
            # ev = (self.ts, v['ts_trade_type'], v['ts_txn_id'], v['ts_fund_code'], v)
            ev = (v['ts_scheduled_at'], v['ts_trade_type'], v['ts_txn_id'], v['ts_fund_code'], v)
            evs.append(ev)
        self.df_ts_order_fund.loc[mask, 'ts_trade_status'] = 2
        # if not df.empty:
        #     pdb.set_trace()

        return evs

    
    def place_buy_order(self, dt, order_id, fund_code, argv):
        evs = []

        if order_id not in self.df_ts_order_fund.index:
            dd("SNH: missng ts_order_fund in context %s " % order_id)
            return evs

        if self.df_share_buying is not None and order_id in self.df_share_buying:
            dd("buying order exists", [order_id, fund_code, argv])
            # Log::error($this->logtag.'SNH: buying order exists', [$this->buying, $e]);
            # $alert = sprintf($this->logtag."基金份额计算检测到重复购买订单:[%s]", $e['it_order_id']);
            # SmsService::smsAlert($alert, 'kun');
        #
        # 获取订单确认日
        #
        nr_acked_days = self.rule.get_buy_ack_days(fund_code)
        acked_date = self.df_t_plus_n.loc[self.trade_date_current, "T+%d" % nr_acked_days]
        
        tmp = {
            'ts_trade_status': 1,
            'ts_placed_date': dt.date(),
            'ts_placed_time': dt.time(),
            'ts_trade_date': self.trade_date_current,
            'ts_acked_date': acked_date,
        }
        self.df_ts_order_fund.loc[order_id, tmp.keys() ] = tmp.values()

        #
        # 购买下单时我们并不生成购买份额, 因为我们并不知道当天的净值
        #
        # pdb.set_trace()
        buying = pd.Series({
            'ts_order_id': order_id,
            'ts_portfolio_id':argv['ts_portfolio_id'],
            'ts_fund_code': fund_code,
            'ts_pay_method': argv['ts_pay_method'],
            'ts_trade_date': self.trade_date_current,
            'ts_acked_date': acked_date,
            'ts_redeemable_date': acked_date,
            'ts_nav': 0.0000,
            'ts_share': 0.0000,
            'ts_amount': argv['ts_placed_amount'],
            'ts_date': pd.to_datetime('2000-01-01'),
        })
        if self.df_share_buying is None:
            self.df_share_buying = pd.DataFrame([buying], index=[order_id], columns=[
                'ts_order_id', 'ts_portfolio_id', 'ts_fund_code', 'ts_pay_method', 'ts_date', 'ts_nav', 'ts_share',
                'ts_amount', 'ts_trade_date', 'ts_acked_date', 'ts_redeemable_date'
            ])
        else:
            self.df_share_buying.loc[order_id] = buying 

        return evs
        #
        # 记录购买对账单: 购买时记账手续费要计入购买金额(it_amount是包含手续费)
        #
        # self.adjust_stat(self::ST_SUB, fund_code, argv['ts_placed_amount']);
        # self.adjust_stat(self::ST_SUB_FEE, -$e['it_fee'], 0);

    def place_redeem_order(self, dt, order_id, fund_code, argv):
        evs = []

        if order_id not in self.df_ts_order_fund.index:
            dd("SNH: missng ts_order_fund in context %s " % order_id)
            return evs

        if self.df_share_redeeming is not None and order_id in self.df_share_redeeming:
            dd("redeem order exists", [order_id, fund_code, argv])
            # Log::error($this->logtag.'SNH: buying order exists', [$this->buying, $e]);
            # $alert = sprintf($this->logtag."基金份额计算检测到重复购买订单:[%s]", $e['it_order_id']);
            # SmsService::smsAlert($alert, 'kun');

        #
        # 获取订单确认日
        #
        nr_acked_days = self.rule.get_redeem_ack_days(fund_code)
        acked_date = self.df_t_plus_n.loc[self.trade_date_current, "T+%d" % nr_acked_days]
        
        tmp = {
            'ts_trade_status': 1,
            'ts_placed_date': dt.date(),
            'ts_placed_time': dt.time(),
            'ts_trade_date': self.trade_date_current,
            'ts_acked_date': acked_date,
        }
        self.df_ts_order_fund.loc[order_id, tmp.keys() ] = tmp.values()

        #
        # 从份额上扣减
        #
        redeeming = []
        left = argv['ts_placed_share']
        mask = (self.df_share['ts_fund_code'] == fund_code) & (self.df_share['ts_portfolio_id'] == argv['ts_portfolio_id']) & (self.df_share['ts_pay_method'] == argv['ts_pay_method'])
        keys = self.df_share.loc[mask].index.tolist()
        for key in keys:
            share = self.df_share.loc[key]
            if share['ts_share'] - left > 0.00001:
                redeemed = left
                self.df_share.loc[key, 'ts_share'] -= left
                left = 0
            else:
                redeemed = share['ts_share']
                left -= share['ts_share']
                self.df_share.loc[key, 'ts_share'] = 0
            redeeming.append((
                order_id,                # 赎回订单号
                argv['ts_portfolio_id'], # 组合ID
                fund_code,               # 基金代码
                argv['ts_pay_method'],   # 支付方式
                share['ts_order_id'],    # 份额订单号
                redeemed,                # 赎回份额
                self.trade_date_current, # 交易日期
                0,                       # 交易净值
                acked_date,              # 确认日期
                0,                       # 最新净值
                share['ts_trade_date'],  # 份额购买日期
            ))
            if left < 0.000099:
                break

        columns = ['ts_order_id', 'ts_portfolio_id', 'ts_fund_code', 'ts_pay_method', 'ts_share_id', 'ts_share', 'ts_trade_date', 'ts_trade_nav', 'ts_acked_date', 'ts_latest_nav', 'ts_buy_date']
        df_redeeming = pd.DataFrame(redeeming, columns=columns)
        
                             
        if self.df_share_redeeming is None:
            self.df_share_redeeming = df_redeeming
        else:
            self.df_share_redeeming = pd.concat([self.df_share_redeeming, df_redeeming])
                             
        if left > 0.00001:
            dd('SNH: insuffient share for redeem', [self.df_share, left, order_id, fund_code, argv])
         
        # #
        # # 记录赎回对账单：赎回时手续费单独记录，赎回金额实际是到账金额
        # #
        # $amount = $e['it_amount'];
        # if ($e['it_amount'] < 0.00001 && $e['it_share'] > 0.00001) {
        #     $amount = round($e['it_share'] * $this->nav, 2);
        # }
        # // $this->stat[self::ST_REDEEM] -= $e['it_amount'];
        # $this->adjustStat(self::ST_REDEEM, -$amount, -$e['it_share']);
        # // $this->stat[self::ST_REDEEM_FEE] -= $e['it_fee'];
        # if (abs($e['it_fee']) > 0.00001) {
        #     $this->adjustStat(self::ST_REDEEM_FEE, -$e['it_fee'], 0);
        # }
        
        return evs
        

    def handle_nav_update(self, fund_code, nav):
        '''
        更新基金净值是需要一些例行的步骤
        '''
        evs = []
        df = self.df_ts_order_fund
        # if fund_code == '000509' and nav['ra_date'] == pd.to_datetime('2017-06-26'):
        #     pdb.set_trace()

        #
        # 对购买订单进行预确认
        #
        mask = (df['ts_fund_code'] == fund_code) \
               & (df['ts_trade_status'] == 1) \
               & (df['ts_trade_date'] == nav['ra_date']) \
               & (df['ts_trade_type'].isin([30, 31, 50, 51, 63]))
        df_order_fund = df.loc[mask]
        if not df_order_fund.empty:
            #
            # 计算基金的确认金额、份额和手续费
            #
            #dd(df, df_order_fund, fund_code, nav)
            # pdb.set_trace()
            sr_fee = self.rule.get_buy_fee(fund_code, df_order_fund['ts_placed_amount'])
            df.loc[mask, 'ts_trade_nav'] = nav['ra_nav']
            df.loc[mask, 'ts_acked_fee'] = sr_fee
            df.loc[mask, 'ts_acked_amount'] = df_order_fund['ts_placed_amount']
            df.loc[mask, 'ts_acked_share'] = ((df_order_fund['ts_placed_amount'] - sr_fee) / nav['ra_nav']).round(2)
            #
            # 对份额进行预确认
            #
            sr_share = df.loc[mask, 'ts_acked_share']
            self.df_share_buying.loc[sr_share.index, 'ts_share'] = sr_share
            self.df_share_buying.loc[sr_share.index, 'ts_nav'] = nav['ra_nav']
            self.df_share_buying.loc[sr_share.index, 'ts_date'] = nav['ra_date']
            #
            # 记录购买对账单
            #
            gbcols = [df_order_fund['ts_portfolio_id'], df_order_fund['ts_fund_code'], df_order_fund['ts_pay_method']]
            sr_amount =  df.loc[mask, 'ts_acked_amount'].groupby(gbcols).sum()
            sr_share = df.loc[mask, 'ts_acked_share'].groupby(gbcols).sum()
            # if amount > 0.0099 or share > 0.000099:
            df_tmp_stat = pd.DataFrame({'ts_stat_amount': sr_amount, 'ts_stat_share': sr_share})
            self.adjust_stat_df(ST_SUB, df_tmp_stat)

            df_tmp_stat = (- df.loc[mask, 'ts_acked_fee'].groupby(gbcols).sum()).to_frame('ts_stat_amount') # [XXX] 注意有个负号，因为费用以负数记账
            self.adjust_stat_df(ST_SUB_FEE, df_tmp_stat)

        #
        # 对赎回订单进行确认
        #
        mask = (df['ts_fund_code'] == fund_code) \
               & (df['ts_trade_status'] == 1) \
               & (df['ts_trade_date'] == nav['ra_date']) \
               & (df['ts_trade_type'].isin([40, 41, 64]))
        df_order_fund = df.loc[mask]
        if not df_order_fund.empty:
            df.loc[mask, 'ts_trade_nav'] = nav['ra_nav']
            df.loc[mask, 'ts_acked_share'] = df_order_fund['ts_placed_share']

            #
            # 计算确认金额和手续费
            #
            mask2 = self.df_share_redeeming['ts_order_id'].isin(df_order_fund['ts_txn_id'])
            self.df_share_redeeming.loc[mask2, 'ts_trade_nav'] = nav['ra_nav']

            df_redeeming = self.df_share_redeeming.loc[mask2, ['ts_order_id', 'ts_share', 'ts_trade_nav', 'ts_buy_date']]
            df_redeeming['ts_acked_fee'] = self.rule.get_redeem_fee(fund_code, df_redeeming, nav['ra_date'])

            sr_fee = df_redeeming['ts_acked_fee'].groupby(by=df_redeeming['ts_order_id']).sum().round(2)
            
            sr_amount = (df_order_fund['ts_placed_share'] * nav['ra_nav']).round(2)
            df.loc[mask, 'ts_acked_fee'] = sr_fee
            df.loc[mask, 'ts_acked_amount'] = sr_amount - sr_fee
            #
            # 记录赎回对账单
            #
            df_order_fund = df.loc[mask]
            gbcols = [df_order_fund['ts_portfolio_id'], df_order_fund['ts_fund_code'], df_order_fund['ts_pay_method']]
            sr_amount =  - (df.loc[mask, 'ts_acked_amount'].groupby(gbcols).sum())
            sr_share = - (df.loc[mask, 'ts_acked_share'].groupby(gbcols).sum())

            df_tmp_stat = pd.DataFrame({'ts_stat_amount': sr_amount, 'ts_stat_share': sr_share})
            self.adjust_stat_df(ST_REDEEM, df_tmp_stat)

            df_tmp_stat = (- df.loc[mask, 'ts_acked_fee'].groupby(gbcols).sum()).to_frame('ts_stat_amount') # [XXX] 注意有个负号，因为费用以负数记账
            self.adjust_stat_df(ST_REDEEM_FEE, df_tmp_stat)

        #
        # 更新基金净值
        #
        if self.df_share is not None:
            if nav['ra_type'] != 3:
                #
                # 非货币基金，直接更新净值即可
                #
                mask = (self.df_share['ts_fund_code'] == fund_code) & (self.df_share['ts_share'] > 0.000099)
                self.df_share.loc[mask, 'ts_nav'] = nav['ra_nav']
                self.df_share.loc[mask, 'ts_date'] = self.day
            else:
                #
                # 货币基金，需要调整基金的份额
                #
                mask = (self.df_share['ts_fund_code'] == fund_code) & (self.df_share['ts_share'] > 0.000099)
                df_tmp = self.df_share.loc[mask]
                sr_total_share = df_tmp['ts_share'].groupby([df_tmp['ts_portfolio_id'], df_tmp['ts_fund_code'], df_tmp['ts_pay_method']]).sum()
                sr_uncarried =  sr_total_share * nav['ra_return_daily'] / 10000
                #
                # 让策略对未结转收益进行调整
                #
                sr_uncarried = self.investor.get_uncarried(nav['ra_date'], fund_code, sr_uncarried)
                df_uncarried = sr_uncarried.to_frame('ts_stat_uncarried')
                df_uncarried['ts_date'] = nav['ra_date']
                
                #
                # 记录未结转收益
                #
                if self.sr_uncarried is None:
                    self.df_uncarried = df_uncarried
                else:
                    self.df_uncarried = pd.concat(self.df_uncarried, df_uncarried)

                #
                # 记录对账单
                #
                if not df_uncarried.empty:
                    self.adjust_stat_df(ST_UNCARRY, df_uncarried[['ts_stat_uncarried']])

                #
                # 处理货币基金的收益结转
                # 
                # pdb.set_trace()
                df_carried = self.investor.get_carried(nav['ra_date'], fund_code, self.df_uncarried)
                if df_carried is not None:
                    self.df_uncarried = None

                    if not df_carried.empty:
                        # 记录结转对账单
                        df_tmp_stat = pd.DataFrame({
                            'ts_stat_amount': df_carried['ts_dividend_share'],
                            'ts_stat_share' : df_carried['ts_dividend_share'],
                            'ts_stat_uncarried': df_carried['ts_stat_uncarried']})
                        self.adjust_stat_df(ST_CARRY, df_tmp_stat)

                        # 结转的数据单独记录份额，份额的交易日期和可赎回日期均按照0000-00-00记录
                        for k, x in df_carried.iterrows():
                            order_id = "%s|%s|%s" % (x['ts_portfolio_id'], x['ts_fund_code'], x['ts_pay_method'])
                            if order_id in self.df_share.index:
                                self.df_share.loc[order_id, 'ts_share'] += x['ts_dividend_share']
                                self.df_share.loc[order_id, 'ts_amount'] += x['ts_dividend_share']
                            else:
                                day = pd.to_datetime('2000-01-01')
                                sr = pd.Series({
                                    'ts_order_id': order_id,
                                    'ts_portfolio_id': x['ts_portfolio_id'],
                                    'ts_fund_code': fund_code,
                                    'ts_pay_method': x['ts_pay_method'],
                                    'ts_date': nav['ra_date'],
                                    'ts_nav': 1,
                                    'ts_share': x['ts_dividend_share'],
                                    'ts_amount': x['ts_dividend_share'],
                                    'ts_trade_date': day,
                                    'ts_acked_date': day,
                                    'ts_redeemable_date': day,
                                })
                                self.df_share.loc[order_id] = sr

        return evs
        # #
        # # 如果当日有分红事件
        # #
        # if day in self.df_bonus.index.levels[0]:
        #     df = self.df_bonus.loc[(day, fund_codes), :]
        #     for key, row in df.iterrows():
        #         record_date, fund_code = key
        #         argv = {
        #             'bonus_ratio': row['ra_bonus'],
        #             'bonus_nav': row['ra_bonus_nav'],
        #             'bonus_nav_date': row['ra_bonus_nav_date'],
        #             'dividend_date': row['ra_dividend_date'],
        #             'payment_date': row['ra_payment_date']
        #         }
        #         ev = (record_date + timedelta(hours=16), 15, 0, fund_code, argv)
        #         # dd(record_date + timedelta(hours=16), ev)
        #         evs.append(ev)

        # # 
        # # 如果当日有分拆，加载分拆信息
        # # 
        # #
        # if day in self.df_split.index.levels[0]:
        #     # dd(self.df_split, day)
        #     df = self.df_split.loc[(day, fund_codes), :]
        #     for key, row in df.iterrows():
        #         split_date, fund_code = key
        #         argv = {'ra_split_proportion': row['ra_split_proportion']}
        #         ev = (split_date + timedelta(hours=1), 18, 0, fund_code, argv)
        #         evs.append(ev)

        

    def handle_order_continue(self, dt, argv):
        '''
        对需要确认的订单进行确认
        '''
        evs = []

        #
        # 对未完成的订单进行continue
        #
        # if dt == pd.to_datetime('2017-06-27 00:00:01'):
        #     pdb.set_trace()

        if self.df_ts_order is not None:
            ptxns = self.df_ts_order.loc[self.df_ts_order['ts_trade_status'].isin([0,1])].index.tolist()            

            for ptxn in ptxns:
                df_order_fund = self.df_ts_order_fund.loc[self.df_ts_order_fund['ts_portfolio_txn_id'] == ptxn]
                ts_order_fund = self.policy.place_plan_order(dt, ptxn, df_order_fund)

                if ts_order_fund is not None and not ts_order_fund.empty:
                    if self.df_ts_order_fund is None:
                        self.df_ts_order_fund = ts_order_fund
                    else:
                        self.df_ts_order_fund = ts_order_fund.combine_first(self.df_ts_order_fund)
                        self.df_ts_order_fund['ts_trade_type'] = self.df_ts_order_fund['ts_trade_type'].astype(int)
                        self.df_ts_order_fund['ts_trade_status'] = self.df_ts_order_fund['ts_trade_status'].astype(int)

            evs.extend(self.issue_order_place_event())
        
        return evs

    def handle_order_confirm(self, dt, argv):
        '''
        对需要确认的订单进行确认
        '''
        evs = []

        # pdb.set_trace()
        if self.df_ts_order_fund is not None:
            mask = (self.df_ts_order_fund['ts_trade_status'] == 1) \
                   & (self.df_ts_order_fund['ts_acked_date'] == self.day)
            df = self.df_ts_order_fund.loc[mask]
            if not df.empty:
                # 对订单进行确认
                self.df_ts_order_fund.loc[mask, 'ts_trade_status'] = 6
                # dd(df, self.df_ts_order_fund)

        #
        # 更新组合订单状态
        #
        if self.df_ts_order is not None:
            ptxns = self.df_ts_order.loc[self.df_ts_order['ts_trade_status'].isin([0,1])].index
            # 需要处理更新主订单状态
            self.update_ts_order_status(ptxns)
    
        #
        # 对购买持仓进行确认
        #
        if self.df_share_buying is not None:
            mask = (self.df_share_buying['ts_acked_date'] == self.day)
            df = self.df_share_buying.loc[mask]
            if not df.empty:
                if self.df_share is None:
                    self.df_share = df.copy()
                else:
                    self.df_share = df.combine_first(self.df_share)
                self.df_share_buying.drop(df.index, inplace=True)

        #
        # 对赎回持仓进行确认
        #
        if self.df_share_redeeming is not None:
            mask = (self.df_share_redeeming['ts_acked_date'] == self.day)
            df = self.df_share_redeeming.loc[mask]
            if not df.empty:
                self.df_share_redeeming.drop(df.index, inplace=True)
        
        return evs
    
    def update_ts_order_status(self, ptxns):
        # if self.ts.strftime("%Y-%m-%d") == '2017-04-27':
        #     pdb.set_trace()
        for ptxn in ptxns:
            self.policy.continue_order(self.ts, ptxn)

            ret = self.policy.check_order(self.ts, ptxn)
            if ret == 20000:
                is_plan_finished = 1
                status = 6
            elif ret == 40001:
                is_plan_finished = -1
                status = -1
            else:
                is_plan_finished = 0
                status = 0

            self.df_ts_order.loc[ptxn, 'ts_trade_status'] = status
        
        
    
    def record_holding(self, dt, argv):
        '''
        记录持仓事件
        '''
        evs = []
        #
        # 记录当日持仓
        #
        if self.df_share is not None or self.df_share_buying is not None:
            df_share = pd.concat([self.df_share, self.df_share_buying])
            mask = df_share['ts_share'] > 0.000099;
            df_holding = df_share.loc[mask].groupby(['ts_portfolio_id', 'ts_fund_code', 'ts_pay_method']).agg({
               'ts_nav':'first', 'ts_share':'sum'
            })
            df_holding['ts_amount'] = (df_holding['ts_nav'] * df_holding['ts_share']).round(2)

            self.dt_holding[self.day] = df_holding.copy()

            #
            # 记录当日对账
            #
            #
            # 计算日收益：
            #
            #    日末持仓 + (资金流出 - 资金流入) - 昨日日末持仓
            #
            # 也即：
            #
            #    日末持仓 - 资金进出结余 - 昨日日末持仓
            #
            if self.df_stat is None:
                sr_balance = pd.Series(0, index=df_holding.index)
            else:
                sr_balance = self.df_stat['ts_stat_amount'].groupby(level=[0, 1, 2]).sum()  # 当日资金进出结余
            df_holding['ts_balance'] = sr_balance

            if self.sr_holding_last is None:
                df_holding['ts_holding_last'] = 0
            else:
                df_holding['ts_holding_last'] = self.sr_holding_last

            df_holding.fillna(0, inplace=True)
            
            sr_yield = df_holding['ts_amount'] - df_holding['ts_balance'] - df_holding['ts_holding_last']
            
            #
            # 记录日收益对账单，而是要到记录日末持仓时再记录对账单
            #
            # [XXX] 从产品的角度，只要当日有持仓，且基金有净值，就应该有
            # 日收益，哪怕是为0也要记录
            #
            df_yield = sr_yield.to_frame('ts_stat_amount')

            self.adjust_stat_df(ST_YIELD, df_yield)
            
            self.sr_holding_last = df_holding['ts_amount']
            # dd(self.df_stat, sr_balance, df_holding, sr_yield, dt)

        if self.df_stat is not None:
            mask = (self.df_stat['ts_stat_amount'].abs() > 0.0099) | (self.df_stat['ts_stat_share'].abs() > 0.000099)
            self.dt_stat[self.day] = self.df_stat.loc[mask]
        
        return evs


    def adjust_stat(self, code , xtype, amount, share, uncarrid = 0):
        sr = pd.Series({'ts_stat_amount': amount, 'ts_stat_share': share, 'ts_stat_uncarried': uncarrid})
        if self.df_stat is None:
            self.df_stat = pd.DataFrame([sr], index=[[code], [xtype]])
            self.df_stat.index.names=['ts_fund_code', 'ts_stat_type']
        else:
            if (code, xtype) in self.df_stat.index:
                self.df_stat.loc[(code, xtype), :] += sr
            else:
                self.df_stat.loc[(code, xtype), :] = sr

    def adjust_stat_df(self, xtype, df):
        if 'ts_stat_amount' not in df.columns:
            df['ts_stat_amount'] = 0
        if 'ts_stat_share' not in df.columns:
            df['ts_stat_share'] = 0
        if 'ts_stat_uncarried' not in df.columns:
            df['ts_stat_uncarried'] = 0

        df = df.loc[(df['ts_stat_amount'].abs() > 0.0099) | (df['ts_stat_share'].abs() > 0.000099) | (df['ts_stat_uncarried'].abs() > 0.0099)]

        if not df.empty:
            df.index.names = ['ts_portfolio_id', 'ts_fund_code', 'ts_pay_method']
            df['ts_stat_type'] = xtype

            df.set_index(['ts_stat_type'], append=True, inplace=True)

            if  self.df_stat is None:
                self.df_stat = df
            else:
                self.df_stat = pd.concat([self.df_stat, df]).sort_index()
                        
        
