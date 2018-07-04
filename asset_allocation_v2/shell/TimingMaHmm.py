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
        self.ori_data = (ori_data.rolling(20).mean() / ori_data.rolling(120).mean() - 1).tc_close.dropna()

    def timing(self):
        """
        :usage: 执行程序
        :return: None
        """
        dates = self.trade_dates.index[self.trade_dates.index >= self.start_date]
        signals = []
        for date in dates:
            ob = self.ori_data[self.ori_data.index <= date].ravel().reshape(-1,1)
            ob = ob[-250 * 5:]
            model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=10000).fit(ob)
            predicts = model.predict(ob)
            means_ = model.means_.reshape(1,self.state_num)[0]
            #covars_ = model.covars_.reshape(1, self.state_num)[0]
            #print(date, model.means_.reshape(1,self.state_num), model.covars_.reshape(1, self.state_num))
            #print(covars.reshape(1, self.state_num)[0])
            #sharp = means_ / covars_
            #print(date, sharp)
            #lt_zero = len(means_[means_ < 0.00])
            #state_threshold = max(lt_zero, self.state_num / 2 + 1)
            #state_threshold = lt_zero + 1
            state_dict = dict(zip(range(0, len(means_)), scipy.stats.rankdata(means_)))
            last_state = state_dict[predicts[-1]]
            #signals.append(last_state)
            if last_state > self.state_num / 2 + 1:
            #if last_state >= state_threshold:
                signals.append(1.0)
            #elif last_date < self.state_num / 2:
            #    signals.append(-1.0)
            else:
                signals.append(0.0)
            u = list(means_)
            u.sort()
            #print(date, last_state, state_threshold, u, ob[-1], signals[-1])
            print(date, last_state, u)
            #print(states)
        df = pd.DataFrame(signals, index = dates, columns = ['tc_signal'])
        df.index.name = 'tc_date'

        return df

if __name__ == "__main__":
    pass
