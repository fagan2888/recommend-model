#coding=utf8


import pandas as pd
import numpy as np
import datetime
import calendar
from sqlalchemy import *

import database

class TimingGFTD(object):
    
    def __init__(self, n1=4, n2=4, n3=4):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

    def timing(self, df_nav):
        n1, n2, n3 = (self.n1, self.n2, self.n3)
        #
        # step 1: 计算 ud
        #
        sr_ud = df_nav['tc_close']\
            .rolling(window=n1, min_periods=n1)\
            .apply(lambda x: cmp(x[-1], x[0])).fillna(0)
        df_nav['tc_ud'] = sr_ud
        #
        # step 2: 对ud[i]进行累加,且当其值与上一个不等时停止累加
        #
        sr_flip = sr_ud\
            .rolling(window=2, min_periods=1)\
            .apply(lambda x: 0 if x[0] == x[-1] else 1)
        df_nav['tc_ud_flip'] = sr_flip
        
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

        #
        # step 4.1: 买入和卖出计数
        #
        sr_kstick_buy = ((df_nav['tc_close'] >= df_nav['tc_high'].shift(2))\
            & (df_nav['tc_high'] >= df_nav['tc_high'].shift(1)) \
            & (df_nav['tc_close'] >= df_nav['tc_close'].shift(1))).astype(int)
        sr_kstick_sell = ((df_nav['tc_close'] <= df_nav['tc_low'].shift(2))\
            & (df_nav['tc_low'] <= df_nav['tc_low'].shift(1)) \
            & (df_nav['tc_close'] <= df_nav['tc_close'].shift(1))).astype(int)

        #
        # step 4.2: 对买入和卖出信号进行累加
        #
        sr_count_buy = sr_kstick_buy\
            .groupby(sr_buy_start.cumsum(), group_keys=False)\
            .apply(lambda x: np.add.accumulate(x) - x[0])
        sr_count_buy[:sr_buy_start.index[sr_buy_start == 1].min()] = 0
        sr_count_sell=sr_kstick_sell\
            .groupby(sr_sell_start.cumsum(), group_keys=False)\
            .apply(lambda x: np.add.accumulate(x) - x[0])
        sr_count_sell[:sr_sell_start.index[sr_sell_start == 1].min()] = 0

        #
        # step 5: 发出买入和卖出信号
        #
        sr_signal_buy = sr_count_buy.rolling(2, 1)\
            .apply(lambda x: 1 if x[0] != self.n3 and x[-1] == self.n3 else 0)
        sr_signal_sell = sr_count_sell.rolling(2, 1)\
            .apply(lambda x: 1 if x[0] != self.n3 and x[-1] == self.n3 else 0)

        df_nav['tc_buy_start'] = sr_buy_start
        df_nav['tc_buy_kstick'] = sr_kstick_buy
        df_nav['tc_buy_count'] = sr_count_buy
        df_nav['tc_buy_signal'] = sr_signal_buy
        df_nav['tc_sell_start'] = sr_sell_start
        df_nav['tc_sell_kstick'] = sr_kstick_sell
        df_nav['tc_sell_count'] = sr_count_sell
        df_nav['tc_sell_signal'] = sr_signal_sell

        #
        # 生成交易信号和止损信号
        #
        status = -1 # 持仓状态: -1:空仓; 1:持有
        action = 0  # -3:买入止损; -2:卖出启动; -1:卖出信号; 0:不动; 1:买入信号; 2:买入启动; 3:沽空止损
        (high, low, high_recording, low_recording) = (float('inf'), 0, None, None) # 起作用止损线 & 正在记录止损线

        dict_status = {}
        dict_action = {}
        dict_stop = {}
        dict_recording_high = {}
        dict_recording_low = {}
        for key, row in df_nav.iterrows():
            if  row['tc_buy_start'] + row['tc_buy_signal'] +\
                row['tc_sell_start'] + row['tc_sell_signal'] >= 2:
                print "warning: multiple event occur: %s {'buy_start':%d, 'buy_signal':%d, 'sell_start':%d, 'sell_signal':%d}"\
                    % (key.strftime("%Y-%m-%d"), row['tc_buy_start'], row['tc_buy_signal'], row['tc_sell_start'], row['tc_sell_signal'])
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
                    print "empty stop:", key, row['tc_high'], high
                    low = min(row['tc_low'], low_recording)
                    (status, action) = (1, 3)
            else:
                # 持仓状态, status == 1
                if row['tc_close'] <= low:
                    # 买入止损
                    print "buy stop:", key, row['tc_low'], low
                    high = max(row['tc_high'], high_recording)
                    (status, action) = (-1, -3)
            #
            # 记录空仓和止损线
            #
            if high_recording is not None:
                high_recording = max(row['tc_high'], high_recording)
            if low_recording is not None:
                low_recording = min(row['tc_low'], low_recording)
                    
            dict_status[key] = status
            dict_action[key] = action
            dict_stop[key] = (high if status == -1 else low)
            # if key == pd.to_datetime('2015-01-22'):
            #     print key, status, high, low
            #     print stop
            #     print dict_stop[key]
            dict_recording_high[key] = high_recording if high_recording else 0
            dict_recording_low[key] = low_recording if low_recording else 0

        tmp = {
            'tc_signal': dict_status,
            'tc_action': dict_action,
            'tc_stop': dict_stop,
            'tc_recording_high': dict_recording_high,
            'tc_recording_low': dict_recording_low,
        }    
        df_tmp = pd.DataFrame(tmp, index=df_nav.index)
        df_tmp.loc[df_tmp['tc_stop'] == float('inf'), 'tc_stop'] = 0
        df_nav = pd.concat([df_nav, df_tmp], axis=1)
             
        # print df_nav.head(5000)
        # print df_nav.tail(60)
        return df_nav
