#coding=utf8

import logging
import pandas as pd
import numpy as np
import datetime
import calendar
from sqlalchemy import *
import hmm_incremental

from db import database

logger = logging.getLogger(__name__)

class TimingHmm(object):
    
    def __init__(self):

        self.ass_id = globalid
        self.start_date = start_date
        self.end_date = end_date
        self.assets = {
            '120000001':'2070000060', #沪深300
            '120000002':'2070000187', #中证500
            '120000013':'2070006545', #标普500指数
            '120000014':'2070000626', #黄金指数
            '120000015':'2070000076', #恒生指数
            '120000028':'2070006521', #标普高盛原油商品指数收益率
            '120000029':'2070006789', #南华商品指数

        }
        self.feature_selected = {
            '120000001':list(['bias', 'pct_chg', 'priceosc', 'roc']),
            '120000002':list(['sobv', 'pct_chg', 'bias', 'pvt']),
            '120000013':list(['sobv', 'pct_chg', 'vstd', 'macd']),
            '120000014':list(['vstd', 'pct_chg', 'roc', 'wvad']),
            '120000015':list(['priceosc', 'pct_chg', 'bias', 'roc']),
            '120000028':list(['macd', 'pct_chg', 'atr']),
            '120000029':list(['priceosc', 'pct_chg', 'bias', 'roc']),
        }
        # 隐形状态数目
        self.state_num = 5
        self.features = ['macd', 'atr', 'cci', 'rsi', 'sobv', 'mtm', 'roc', \
                        'slowkd', 'pct_chg', 'pvt', 'wvad', 'priceosc', \
                        'bias', 'vma', 'vstd', 'dpo']
        # 模型训练用到的样本数, 149加1即为这个样本数
        self.train_num = 249
        # 训练开始时间
        self.t_start = datetime.datetime(2005, 8, 1)
        # 训练结束时间
        self.t_end = datetime.datetime(2010, 8, 1)

        # 验证开始时间
        self.v_start = datetime.datetime(2003, 1, 1)
        # 验证结束时间
        self.v_end = datetime.datetime(2003, 12, 31)

        # 测试开始时间
        self.test_start = datetime.datetime(2009, 1, 1)
        # 测试结束时间
        self.test_end = datetime.datetime(2016, 12, 31)
        # 单边交易费用
        self.fee_ratio = 0.0
        # 多头收益率阈值，用于选取多头状态
        self.filter_ratio = 0.03
        # 市场状态划分阈值
        self.sharpe_ratio = 0.2
        # 特征评价指标，0: 平均排名（收益）, 1: 最大回撤排名, 2: 收益回撤比排名, 3: 平均胜率排名
        self.eva_indic = [0, 3]
        # 选取指标中排名前几的特征，2代表选取某一指标中排名前2的特征作为最终使用的特征
        self.rank_num = 2


    def timing(self, df_nav):
        #
        # step 1: 计算 ud
        #
        sr_ud = df_nav['tc_close']\
            .rolling(window=self.n1+1, min_periods=self.n1+1)\
            .apply(lambda x: cmp(x[-1], x[0])).fillna(0)
        df_nav['tc_ud'] = sr_ud
        #
        # step 2: 对ud[i]进行累加,且当其值与上一个不等时停止累加
        #
        sr_flip = sr_ud\
            .rolling(window=2, min_periods=1)\
            .apply(lambda x: 0 if x[0] == x[-1] else 1)
        # df_nav['tc_ud_flip'] = sr_flip
        
        sr_flip_acc = sr_flip.cumsum()
        #df_nav['flip_acc'] = sr_flip_acc

        ud_acc = sr_ud\
            .groupby(sr_flip_acc, group_keys=False)\
            .apply(lambda x: np.add.accumulate(x))
        df_nav['tc_ud_acc'] = ud_acc

        #
        # step 3: 计算买入和卖出启动信号buy_start,sell_start
        #
        sr_buy_start = ud_acc[::-1]\
            .rolling(2, 1)\
            .apply(lambda x: 1 if x[-1] <= -self.n2 and (x[0] >= 0 or len(x) == 1) else 0)
        sr_buy_start = sr_buy_start[::-1]
        
        sr_sell_start = ud_acc[::-1]\
            .rolling(2, 1)\
            .apply(lambda x: 1 if x[-1] >= self.n2 and (x[0] <= 0 or len(x) == 1) else 0)
        sr_sell_start = sr_sell_start[::-1]

        df_nav['tc_buy_start'] = sr_buy_start

        #
        # 买入计数, 累加, 发出买入信号
        #
        (row_1, row_2, row_last, count) = (None, None, None, None)
        (sr_kstick_buy, sr_count_buy, sr_signal_buy) = ({}, {}, {})
        for key, row in df_nav.iterrows():
            if row['tc_buy_start'] == 1:
                (kstick, count, signal, row_last) = (0, 0, 0, None)
            else:
                if count is None:
                    (kstick, signal) = (0, 0)
                else:
                    cond1 = row['tc_close'] >= row_2['tc_high']
                    cond2 = row['tc_high'] > row_1['tc_high']
                    cond3 = row['tc_close'] > row_last['tc_close'] if row_last is not None else True

                    kstick = 1 if cond1 and cond2 and cond3 else 0
                    count += kstick
                    signal = 1 if count == self.n3 and kstick == 1 else 0

            sr_kstick_buy[key] = kstick
            sr_count_buy[key] = count if count is not None and kstick == 1 else 0
            sr_signal_buy[key] = signal
            
            row_2 = row_1
            row_1 = row
            if kstick:
                row_last = row
                
        # df_nav['tc_buy_kstick'] = pd.Series(sr_kstick_buy)
        df_nav['tc_buy_count'] = pd.Series(sr_count_buy)
        df_nav['tc_buy_signal'] = pd.Series(sr_signal_buy)

        #
        # 卖出计数, 累加, 发出卖出信号
        #
        df_nav['tc_sell_start'] = sr_sell_start
            
        (row_1, row_2, row_last, count) = (None, None, None, None)
        (sr_kstick_sell, sr_count_sell, sr_signal_sell) = ({}, {}, {})
        for key, row in df_nav.iterrows():
            if row['tc_sell_start'] == 1:
                (kstick, count, signal, row_last) = (0, 0, 0, None)
            else:
                if count is None:
                    (kstick, signal) = (0, 0)
                else:
                    cond1 = row['tc_close'] <= row_2['tc_low']
                    cond2 = row['tc_low'] < row_1['tc_low']
                    cond3 = row['tc_close'] < row_last['tc_close'] if row_last is not None else True

                    kstick = 1 if cond1 and cond2 and cond3 else 0
                    count += kstick
                    signal = 1 if count == self.n4 and kstick == 1 else 0

            sr_kstick_sell[key] = kstick
            sr_count_sell[key] = count if count is not None and kstick == 1 else 0
            sr_signal_sell[key] = signal
            
            row_2 = row_1
            row_1 = row
            if kstick:
                row_last = row
                
        # df_nav['tc_sell_kstick'] = pd.Series(sr_kstick_sell)
        df_nav['tc_sell_count'] = pd.Series(sr_count_sell)
        df_nav['tc_sell_signal'] = pd.Series(sr_signal_sell)

        #
        # 生成交易信号和止损信号
        #
        status = -1 # 持仓状态: -1:空仓; 1:持有
        action = 0  # -3:买入止损; -2:卖出启动; -1:卖出信号; 0:不动; 1:买入信号; 2:买入启动; 3:沽空止损
        (high, low, high_recording, low_recording) = (float('inf'), 0, None, None) # 起作用止损线 & 正在记录止损线

        dict_status = {}
        dict_action = {}
        dict_stop_high = {}
        dict_stop_low = {}
        dict_recording_high = {}
        dict_recording_low = {}
        for key, row in df_nav.iterrows():
            if  row['tc_buy_start'] + row['tc_buy_signal'] +\
                row['tc_sell_start'] + row['tc_sell_signal'] >= 2:
                logger.warn("multiple event occur: %s {'buy_start':%d, 'buy_signal':%d, 'sell_start':%d, 'sell_signal':%d}"\
                    % (key.strftime("%Y-%m-%d"), row['tc_buy_start'], row['tc_buy_signal'], row['tc_sell_start'], row['tc_sell_signal']))
            #
            # 处理事件
            #
            action = 0   
            if row['tc_buy_start'] == 1:
                # 买入启动
                (action, low_recording) = (2, row['tc_low'])
            elif row['tc_buy_signal'] == 1:
                # 买入信号
                low = min(row['tc_low'], low_recording)
                (status, action, low_recording) = (1, 1, None)
                # if low is None:
                #     print low, low_recording, key, row['tc_low']
                
            if row['tc_sell_start'] == 1:
                # 卖出启动
                (action, high_recording) = (-2, row['tc_high'])
            elif row['tc_sell_signal'] == 1:
                # 卖出信号
                high = max(row['tc_high'], high_recording)
                (status, action, high_recording) = (-1, -1, None)
                
            #
            # 处理止损
            #
            if status == -1:
                # 空仓状态
                if row['tc_close'] >= high:
                    # 沽空止损
                    logger.info("empty stop: day:%s, OHLC-H:%f, HSTOP:%f", key, row['tc_high'], high)
                    low = min(row['tc_low'], low_recording)
                    (status, action) = (1, 3)
            else:
                # 持仓状态, status == 1
                if row['tc_close'] <= low:
                    # 买入止损
                    logger.info("buy stop: day:%s, OHLC-L: %f, LSTOP: %f", key, row['tc_low'], low)
                    high = max(row['tc_high'], high_recording)
                    (status, action) = (-1, -3)
            #
            # 记录空仓和止损线
            #
            if high_recording is not None:
                high_recording = max(row['tc_high'], high_recording)
            if low_recording is not None:
                low_recording = min(row['tc_low'], low_recording)

            #stop =  (high if status == -1 else low)
            dict_status[key] = status
            dict_action[key] = action
            dict_stop_high[key] = high if high is not None else 0
            dict_stop_low[key] = low if low is not None else 0
            # if key == pd.to_datetime('2015-01-22'):
            #     print key, status, high, low
            #     print stop
            #     print dict_stop[key]
            dict_recording_high[key] = high_recording if high_recording else 0
            dict_recording_low[key] = low_recording if low_recording else 0

        tmp = {
            'tc_signal': dict_status,
            'tc_action': dict_action,
            'tc_stop_high': dict_stop_high,
            'tc_stop_low': dict_stop_low,
            'tc_recording_high': dict_recording_high,
            'tc_recording_low': dict_recording_low,
        }    
        df_tmp = pd.DataFrame(tmp, index=df_nav.index)
        df_tmp.loc[df_tmp['tc_stop_high'] == float('inf'), 'tc_stop_high'] = 0
        df_nav = pd.concat([df_nav, df_tmp], axis=1)
             
        # print df_nav.head(5000)
        # print df_nav.tail(60)
        return df_nav
