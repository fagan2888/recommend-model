
import pandas as pd
import datetime
import numpy as np
import utils
import os
import sys
import click
import DFUtil
from scipy import stats
import random
from arch.univariate import ARX, GARCH, Normal
from multiprocessing import Pool
from functools import partial
from ipdb import set_trace

def calc_cond_vol(i, inc_interval, interval):
    day = inc_interval.index[i]
    inc = inc_interval.iloc[i::-interval][::-1]
    model = ARX(inc, lags = [1])
    model.volatility = GARCH(1,0,1)
    model.distribution = Normal()
    res = model.fit(disp='off')
    return (day, res.conditional_volatility[-1])

class RiskMgrReshape(object):

    def __init__(self):
        self.interval = 5
        self.ma_period = 20
        self.lookback = 252*2


    def perform(self, asset, df_input, df_vars):
        inc = np.log(df_input['nav']).diff().fillna(0)*100
        inc_2d = inc.rolling(2).sum().fillna(0)
        inc_3d = inc.rolling(3).sum().fillna(0)
        inc_5d = inc.rolling(5).sum().fillna(0)
        pool = Pool()

        inc_interval = inc.rolling(self.interval).sum().dropna()
        calc_cond_vol_picklable = partial(calc_cond_vol, inc_interval=inc_interval, interval=self.interval)
        vol_interval = pd.Series(dict(pool.map(calc_cond_vol_picklable, range(600, len(inc_interval)))))

        return_ma = inc_interval.rolling(self.ma_period).mean().dropna()
        volatility_ma = vol_interval.rolling(self.ma_period).mean().dropna()

        return_mean = return_ma.rolling(self.lookback).mean().dropna()
        return_std = return_ma.rolling(self.lookback).std().dropna()
        volatility_mean = volatility_ma.rolling(self.lookback).mean().dropna()
        volatility_std = volatility_ma.rolling(self.lookback).std().dropna()
        hmm_signal = df_input['hmm_signal']
        hmm_signal = np.sign(hmm_signal)

        #  set_trace()
        df = pd.DataFrame({'inc_interval' : inc_interval,
                        'vol_interval' : vol_interval,
                        'return_ma' : return_ma,
                        'vol_ma' : volatility_ma,
                        'return_mean' : return_mean,
                        'return_std' : return_std,
                        'vol_mean' : volatility_mean,
                        'vol_std' : volatility_std,
                        'hmm_signal' : hmm_signal,
                        'timing': df_input['timing'],
                        'inc2d': inc_2d,
                        'inc3d': inc_3d,
                        'inc5d': inc_5d})
        df = df.join(df_vars)

        pos = {}
        act = {}
        status, position, action = 0, 1, 0
        empty_days = 0

        with click.progressbar(length=len(df.index), label='idk %-20s' % asset) as bar:
            for day, row in df.iterrows():
                bar.update(1)
                if np.isnan(row['vol_mean']):
                    pass
                else:
                    if row['hmm_signal'] == 1 or row['hmm_signal'] == -1:
                        if row['inc2d'] < row['var_2d']:
                            status, empty_days, position, action = 3, 0, 0, 2
                        elif row['inc3d'] < row['var_3d']:
                            status, empty_days, position, action = 3, 0, 0, 3
                        elif row['inc5d'] < row['var_5d']:
                            status, empty_days, position, action = 3, 0, 0, 5


                    elif row['hmm_signal'] == 0:
                        ret, retmean, retstd = row['return_ma'], row['return_mean'], row['return_std']
                        risk, riskmean, riskstd = row['vol_ma'], row['vol_mean'], row['vol_std']
                        #  if status != 2:
                        #  if risk > riskmean + riskstd:
                            #  status, position, action = 2, 0, 2
                        #  if risk <= riskmean + 0.5 * riskstd:
                        if status != 3:
                            if risk <= riskmean:
                                status, position, action = 0, 1, 0
                            else:
                                status, position, action = 1, riskmean/risk, 1
                            #  if status == 2:
                                #  if empty_days <= 5:
                                    #  empty_days += 1
                                #  else:
                                    #  empty_days = 0
                                    #  if risk > riskmean:
                                        #  status, position, action = 1, riskmean/risk, 1
                                    #  else:
                                        #  status, position, action = 0, 1, 0
                    if status == 3:
                        if empty_days <= 5:
                            empty_days += 1
                        else:
                            if row['timing'] == 1.0:
                                if row['hmm_signal'] == 1:
                                    status, position, action = 0, 1, 8
                                if row['hmm_signal'] == -1:
                                    status, position, action = 4, 0.5, 8
                            else:
                                empty_days += 1
                                status, position, action = 3, 0, 7
                    elif status == 1:
                        pass
                    elif status == 4:
                        if row['hmm_signal'] == -1:
                            status, position, action = 4, 0.5, 8
                        else:
                            status, position, action = 0, 1, 8
                    else:
                        status, position, action = 0, 1, 0
                pos[day], act[day] = position, action
        #  sr_pos = pd.Series(ps).shift(1).fillna(1)
        #  return sr_pos
        df_result = pd.DataFrame({'rm_pos':pos, 'rm_action':act})
        df_result.index.name = 'rm_date'
        return df_result

