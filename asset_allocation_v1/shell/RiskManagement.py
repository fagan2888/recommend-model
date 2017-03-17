# -*- coding: utf-8 -*-
"""
Created at Nov 23, 2016
Author: shengyitao
Contact: shengyitao@licaimofang.com
Company: LCMF
"""


import pandas as pd
import datetime
import numpy as np
import utils
import os
import click
import DFUtil
from scipy import stats
import random



def confidence(x, per):
    return stats.norm.ppf(per, x.mean(), x.std(ddof=1))


class RiskManagement(object):

    #def __init__(self):

    def perform(self, asset, df):
        #
        # 计算回撤矩阵 和 0.97, 0.75置信区间
        #
        sr_inc = df['nav'].pct_change().fillna(0.0)
        sr_inc5d = sr_inc.rolling(window = 5).sum()
        sr_inc2d = sr_inc.rolling(window = 2).sum()
        sr_inc3d = sr_inc.rolling(window = 3).sum()
        sr_inc4d = sr_inc.rolling(window = 4).sum()
        #sr_inc2d = sr_inc.rolling(window = 2).sum()
        #sr_inc5d.dropna(inplace = True)
        sr_risk = sr_inc.rolling(20).std()
        sr_risk.dropna(inplace = True)


        long_period = 252
        dates = sr_risk.index
        sr_inc5d = sr_inc5d.loc[dates]


        pos = []
        action = []
        ds = []

        status, empty_days, drawdown_status, drawdown_risk_days = 0, 0, 0, 0
        last_risk_over_date = None

        for i in range(long_period, len(dates)):

            day = dates[i]
            if last_risk_over_date is None:
                last_risk_over_date = day

            timing = df.loc[day, 'timing']
            #print timing

            train_sr_inc5d = sr_inc5d.iloc[ i - long_period : i ]
            train_sr_risk  = sr_risk.iloc[ i - long_period : i ]
            risk = sr_risk.loc[day]
            r = sr_inc5d.loc[day]
            r2 = sr_inc2d.loc[day]
            r3 = sr_inc3d.loc[day]
            r4 = sr_inc4d.loc[day]
            #print day, r
            c99 = confidence(train_sr_inc5d.values, 0.01)

            look_back = len(train_sr_inc5d)
            loop_num = look_back / 2
            #loop_num = 20
            rep_num = loop_num * (look_back / 4) / look_back
            day_indexs = range(0, look_back) * rep_num
            random.shuffle(day_indexs)
            day_indexs = np.array(day_indexs)
            day_indexs = day_indexs.reshape(len(day_indexs) / (look_back / 4), look_back / 4)
            #print len(day_indexs)
            #print day_indexs

            #conf_positions = []
            #risk_positions = []
            c99s = []
            c50s = []
            risk_mids = []
            risk_highs = []
            '''
            for indexs in day_indexs:

                tmp_sr_risk = train_sr_risk.iloc[indexs]
                tmp_sr_inc5d = train_sr_inc5d.iloc[indexs]
                risk_mid = tmp_sr_risk.quantile(0.5)
                risk_high = tmp_sr_risk.quantile(0.9)
                c99 = confidence(tmp_sr_inc5d.values, 0.01)
                c50 = confidence(tmp_sr_inc5d.values, 0.5)
                #c75 = confidence(tmp_sr_inc5d.values, 0.25)

                c99s.append(c99)
                c50s.append(c50)
                risk_mids.append(risk_mid)
                risk_highs.append(risk_high)

            c99 = np.mean(c99s)
            c50 = np.mean(c50s)
            risk_mid = np.mean(risk_mids)
            risk_high = np.mean(risk_highs)
            '''

            #print r, c95, risk, risk_mid, risk_high
            #conf_position = np.mean(conf_positions)
            #risk_position = np.mean(risk_positions)
            #print day, conf_position, risk_position,


            #drawdown_df = df['nav'].loc[last_risk_over_date : day]
            #max_v = max(drawdown_df)
            #last_v = drawdown_df.loc[day]
            #drawdown = 1.0 - last_v / max_v

            position = 0
            '''
            if drawdown >= 0.25 and drawdown_status == 0.0:
                drawdown_status, drawdown_risk_days, position = 1.0, 0, 0.0
            elif drawdown_status == 1.0 and drawdown_risk_days <= 25:
                    drawdown_risk_days += 1
                    position = 0.0
            elif drawdown_status == 1.0 and drawdown_risk_days > 25:
                    if  timing == 1.0:
                        drawdown_status, drawdown_risk_days, position, last_risk_over_date = 0.0, 0.0,  1.0, day
                    else:
                        position = 0.0

            if drawdown_status == 0.0:
            '''

            if (r < c99 and r < -0.03) or r2 <= -0.075 or r3 <= -0.075:
                status, empty_days, position = 1, 0, 0
            elif status == 1:
                if empty_days > 5 and timing == 1.0:
                    status,  empty_days, position = 0, 0, 1.0
                    last_risk_over_date = day
                else:
                    empty_days += 1
                    position = pos[-1]
            elif status == 0:
                position = 1.0

            pos.append(position)
            action.append(0)
            ds.append(day)

            print day, position

        pos = pd.Series(pos, index = ds)
        action = pd.Series(action, index = ds)

        df_result = pd.DataFrame({'rm_pos': pos, 'rm_action': action})
        df_result.index.name = 'rm_date'

        return df_result
