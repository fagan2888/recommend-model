#coding=utf8

import sys
import pandas as pd
import numpy as np
import datetime
import calendar
import logging
import click
from sqlalchemy import *

from .db import database

logger = logging.getLogger(__name__)



def periodstdmean(df, period):

    dates = df.index

    meanstd = []
    ds      = []
    for i in range(period, len(dates)):
        d  = dates[i]
        rs = df.iloc[i - period : i, 0]

        meanstd.append([np.std(rs), np.mean(rs)])
        ds.append(d)

    df = pd.DataFrame(meanstd, index=ds, columns=['std','mean'])

    return df


class Reshape(object):
    
    def __init__(self, interval=20, short_period=20, long_period=252):
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

        df.dropna(inplace=True)

        ps = {}
        last = 1
        for day, row in df.iterrows():

            signal = 0
            if row['timing']:
                signal = row['timing']
            else:
                logger.warning("missing timing signal: {'date': '%s'}", day.strftime("%Y-%m-%d"))

            (risk, riskmean, riskstd) = (row['rs_risk'], row['rs_risk_mean'], row['rs_risk_std'])
            (r, rmean, rstd) = (row['rs_return'], row['rs_return_mean'], row['rs_return_std'])
            
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

            if risk >= riskmean + 2 * riskstd and r < rmean - 1 * rstd:
                position = riskmean / risk
            elif risk >= riskmean + 2 * riskstd and r > rmean + 1 * rstd:
                position = 0.0
                #position = riskmean / risk
            elif risk <= riskmean:
                position = 1.0
            else:
                position = riskmean / risk

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

            if (position <= last * 0.5 or position <= last - 0.2) and signal == -1:
                pass
            elif (position >= last * 2 or position >= last + 0.2 or position == 1.0) and signal == 1:
                pass
            else:
                position = last

            #if position < 0.1:
            #    position = 0.0

            ps[day] = last = position

        sr_pos = pd.Series(ps).shift(1).fillna(0.0)
        df['rs_ratio'] = sr_pos

        return df

                



