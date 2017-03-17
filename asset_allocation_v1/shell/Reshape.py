#coding=utf8

import sys
import pandas as pd
import numpy as np
import datetime
import calendar
import logging
import click
from sqlalchemy import *
import random

from db import database

logger = logging.getLogger(__name__)


class Reshape(object):
    
    def __init__(self, interval=2, short_period=20, long_period=252):
        self.interval = interval
        self.short_period = short_period
        self.long_period = long_period

    def reshape(self, df):

        position_datas = []
        position_dates = []

        df['rs_r20'] = df['nav'].rolling(window=self.interval).apply(lambda x: x[-1]/x[0] - 1)
        df.dropna(inplace=True)
        
        df['rs_return'] = df['rs_r20'].rolling(window=self.short_period).mean() # r
        df['rs_risk'] = df['rs_r20'].rolling(window=self.short_period).std()   # risk
        df.dropna(inplace=True)

        df['rs_return_mean'] = df['rs_return'].rolling(window=self.long_period).mean()
        df['rs_return_std'] = df['rs_return'].rolling(window=self.long_period).std()

        df['rs_risk_mean'] = df['rs_risk'].rolling(window=self.long_period).mean()
        df['rs_risk_std'] = df['rs_risk'].rolling(window=self.long_period).std()

        df['rs_r20_quant_high'] = df['rs_return'].rolling(window=self.long_period).quantile(0.9)
        df['rs_r20_quant_low'] = df['rs_return'].rolling(window=self.long_period).quantile(0.1)

        df['rs_risk_quant_high'] = df['rs_risk'].rolling(window=self.long_period).quantile(0.9)
        df['rs_risk_quant_mid'] = df['rs_risk'].rolling(window=self.long_period).quantile(0.5)


        #df.dropna(inplace=True)
        #df.to_csv('nav.csv')

        #df.dropna(inplace=True)

        ps = {}
        last = 1
        dates = df.index
        for i in range(self.long_period, len(dates)):

        #for day, row in df.iterrows():
            row = df.iloc[i,]
            day = dates[i]
            #tmp_rs20_df = df.loc[dates[i - self.long_period] : day, 'rs_r20'].copy()
            #rs20 = df.loc[day, 'rs_r20']
            #print day, rs20, tmp_rs20_df.quantile(0.9)
            signal = 0
            if row['timing']:
                signal = row['timing']
            else:
                logger.warning("missing timing signal: {'date': '%s'}", day.strftime("%Y-%m-%d"))


            train_df = df.iloc[ i - self.long_period : i , ]
            risk = df.loc[day, 'rs_risk']
            #print train_df.index


            look_back = len(train_df)
            loop_num = look_back / 2
            #loop_num = 20
            rep_num = loop_num * (look_back / 4) / look_back
            day_indexs = range(0, look_back) * rep_num
            random.shuffle(day_indexs)
            day_indexs = np.array(day_indexs)
            #print day_indexs
            day_indexs = day_indexs.reshape(len(day_indexs) / (look_back / 4), look_back / 4)
            #print day_indexs
            positions = []
            position = 0.0
            for day_index in day_indexs:
                tmp_df = train_df.iloc[day_index]
                risk_high = tmp_df['rs_risk'].quantile(0.9)
                risk_mid = tmp_df['rs_risk'].quantile(0.5)
                #print day, risk, risk_mid, risk_high
                if risk >= risk_high:
                    position = 0.0
                elif risk <= risk_mid:
                    position = 1.0
                else:
                    position = risk_mid / risk
                positions.append(position)
            position = np.mean(positions)

            #print day_indexs

            #(risk, riskmean, riskstd) = (row['rs_risk'], row['rs_risk_mean'], row['rs_risk_std'])
            #(r, rmean, rstd) = (row['rs_return'], row['rs_return_mean'], row['rs_return_std'])
            #
            # 风险修型规则:
            #
            # 1. 波动率大于等于两个标准差 & 收益率小于一个标准差, 则持有部分仓位
            #
            #        position = risk20_mean / risk20
            #
            # 2. 波动率大于等于两个标准差 & 收益率大于一个标准差 => 空仓
            #
            # 3. 波动率小于波动率均值 则 全仓
            #
            # 4. 其他情况, 则持有部分仓位
            #
            #        position = risk20_mean / risk20
            #
            '''
            if risk >= riskmean + 2 * riskstd and r < rmean - 1 * rstd:
                position = riskmean / risk
            elif risk >= riskmean + 2 * riskstd and r > rmean + 1 * rstd:
                position = 0.0
                #position = riskmean / risk
            elif risk <= riskmean:
                position = 1.0
            else:
                position = riskmean / risk
            '''

            #r20, r20_quant_high, r20_quant_low = row['rs_return'], row['rs_r20_quant_high'], row['rs_r20_quant_low']
            #risk, risk_quant_high, risk_quant_mid = row['rs_risk'], row['rs_risk_quant_high'], row['rs_risk_quant_mid']


            '''
            if r20 >= r20_quant_high:
                position = 0.0
            elif r20 <= r20_quant_low:
                position = 1.0
            else:
                position = last
            '''
            '''
            if risk >= risk_quant_high:
                position = 0.0
            elif risk <= risk_quant_mid:
                position = 1.0
            else:
                position = risk_quant_mid / risk
            '''
            #print day, risk, risk_quant_high, risk_quant_mid, position

            '''

            if risk <= riskmean:
                position = 1.0
            elif risk >= riskmean + 2 * riskstd:
                position = 0.0
            else:
                position = riskmean / risk
            '''
            #
            # 择时调整规则
            #
            # 1. 择时判断空仓 & 本期仓位小于上期仓位的20%或者是上次仓位的一半以下，
            #    则降低仓位至风险修型新算出的仓位
            #    
            # 2. 择时判断持仓 & 本期仓位大于上期仓位的20%或者是上次仓位的一倍以上，
            #    则增加仓位至风险修型新算出的仓位
            #    
            # 3. 否则, 维持原仓位不变
            #
            '''
            if (position <= last * 0.5 or position <= last - 0.2):
                pass
            elif (position >= last * 2 or position >= last + 0.2 or position == 1.0):
                pass
            else:
                position = last

            '''
            if (position <= last * 0.5 or position <= last - 0.2) and signal == -1:
                pass
            elif (position >= last * 2 or position >= last + 0.2 or position == 1.0) and signal == 1:
                pass
            else:
                position = last

            print day, position
            '''
            if position == 0 or position == 1.0:
                pass
            elif position <= last * 0.5 or (position >= last * 2 and last > 0) or abs(position - last) >= 0.3:
                pass
            else:
                position = last
            '''

            #if position < 0.1:
            #    position = 0.0

            ps[day] = last = position

        sr_pos = pd.Series(ps).shift(1).fillna(0.0)
        df['rs_ratio'] = sr_pos

        df.drop('rs_r20_quant_high', axis = 1, inplace = True)
        df.drop('rs_r20_quant_low', axis = 1, inplace = True)
        df.drop('rs_risk_quant_high', axis = 1, inplace = True)
        df.drop('rs_risk_quant_mid', axis = 1, inplace = True)

        df.dropna(inplace=True)
        return df

