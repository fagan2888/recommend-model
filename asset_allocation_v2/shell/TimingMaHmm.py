#coding=utf8


import logging
import pandas as pd
import numpy as np
import datetime
import calendar
from sqlalchemy import *
from cal_tech_indic import CalTechIndic as CalTec
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
from scipy.stats import boxcox
from scipy import stats
from utils import day_2_week
import scipy.stats
from sklearn.preprocessing import normalize
from cal_tech_indic import CalTechIndic as CalTec
import warnings
from ipdb import set_trace


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TimingMaHmm(object):

    def __init__(self, ori_data, timing, trade_dates, start_date = '20120727'):

        self.trade_dates = trade_dates.sort_index()
        self.start_date = start_date
        self.state_num = 5
        self.ass_id = timing['tc_index_id']
        ori_data = ori_data.sort_index()
        self.ma_pct = (ori_data.rolling(20).mean() / ori_data.rolling(120).mean() - 1).tc_close.dropna()
        self.rolling_std = ori_data.pct_change().tc_close.rolling(20).std().loc[self.ma_pct.index]
        self.pct = ori_data.pct_change().fillna(0.0).tc_close.loc[self.ma_pct.index]
        self.states = 5

        #print(self.ma_pct.tail(), self.rolling_std.tail())
        self.ob = pd.concat([self.ma_pct, self.rolling_std], axis = 1)
        self.ob_num = self.ob.shape[1]
        #print(self.ob.tail())

    def timing(self):
        """
        :usage: 执行程序
        :return: None
        """
        dates = self.trade_dates.index[self.trade_dates.index >= self.start_date]
        signals = []
        ds = []
        for date in dates:
            ob = self.ob[self.ob.index <= date].values.reshape(-1,self.ob_num)
            #pct = self.pct[self.pct.index <= date]
            #ob = ob[:]
            #pct = pct.iloc[-250 * 5:]
            try:
                model = GaussianHMM(n_components=self.states, covariance_type="diag", n_iter=10000).fit(ob)
            except:
                print('Error HMM')
                continue
            predicts = model.predict(ob)
            #set_trace()
            means_ = model.means_[:,0]
            covars_ = model.covars_[:,0,0]
            lower_bound = means_ - 1 * (covars_ ** 0.5)
            upper_bound = means_ + 1 * (covars_ ** 0.5)
            #sharpes = {}
            #for state in set(predicts):
                #print(predicts == state)
                #state_pct = pct[predicts == state]
                #sharpe = state_pct.mean() / state_pct.std()
                #print(date, state, means_[state], sharpe)
                #sharpes.append(sharpe)
                #sharpes[state] = sharpe
                #print(state, len(state_pct))
                #print(state, )
            #print(lower_bound, upper_bound)
            #print(date, model.means_.reshape(1,self.state_num), model.covars_.reshape(1, self.state_num))
            #print(covars.reshape(1, self.state_num)[0])
            #sharp = means_ / covars_
            #print(date, sharp)
            #lt_zero = len(means_[means_ < 0.00])
            #state_threshold = max(lt_zero, self.state_num / 2 + 1)
            #state_threshold = lt_zero + 1
            #set_trace()
            #state_dict = dict(zip(range(0, len(sharpes)), scipy.stats.rankdata(sharpes)))
            #last_state = state_dict[predicts[-1]]
            #if lower_bound[predicts[-1]] > 0.0:
            #    signals.append(1.0)
            #else:
            #    signals.append(0.0)
            if lower_bound[predicts[-1]] >= 0.0:
                signals.append(1.0)
            elif upper_bound[predicts[-1]] <= 0.0:
                signals.append(0.0)
            else:
                signals.append(0.0)
            ds.append(date)
            #signals.append(last_state)
            #if last_state > self.state_num / 2 + 1:
            #if last_state >= state_threshold:
            #    signals.append(1.0)
            #elif last_date < self.state_num / 2:
            #    signals.append(-1.0)
            #else:
            #    signals.append(0.0)
            u = list(means_)
            u.sort()
            print(date, predicts[-1], means_, signals[-1])
            #print(date, last_state, state_threshold, u, ob[-1], signals[-1])
            #print(date, last_state, u)
            #print(states)
        df = pd.DataFrame(signals, index = ds, columns = ['tc_signal'])
        df.index.name = 'tc_date'

        return df

if __name__ == "__main__":
    pass
