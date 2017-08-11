#coding=utf8


import logging
import pandas as pd
import numpy as np
import datetime
import calendar
from sqlalchemy import *
import hmm_incremental
from cal_tech_indic import CalTechIndic as CalTec
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
from scipy.stats import boxcox
from scipy import stats
from utils import sigmoid
from sklearn.preprocessing import normalize
from cal_tech_indic import CalTechIndic as CalTec


logger = logging.getLogger(__name__)


class TimingHmm(object):

    def __init__(self, index_id):

        self.ass_id = index_id
        #self.start_date = start_date
        #self.end_date = end_date

        self.feature_selected = {
            120000001:list(['bias', 'pct_chg', 'priceosc', 'roc']),
            120000002:list(['sobv', 'pct_chg', 'bias', 'pvt']),
            120000013:list(['sobv', 'pct_chg', 'vstd', 'macd']),
            120000014:list(['vstd', 'pct_chg', 'roc', 'wvad']),
            120000015:list(['priceosc', 'pct_chg', 'bias', 'roc']),
            120000028:list(['macd', 'pct_chg', 'atr']),
            120000029:list(['priceosc', 'pct_chg', 'bias', 'roc']),
        }
        # 隐含状态数目
        self.state_num = 5
        self.features = ['macd', 'atr', 'cci', 'rsi', 'sobv', 'mtm', 'roc', \
                        'slowkd', 'pct_chg', 'pvt', 'wvad', 'priceosc', \
                        'bias', 'vma', 'vstd', 'dpo']
        # 模型训练用到的样本数, 149加1即为这个样本数
        self.train_num = 249


    def timing(self, df_nav, trade_dates):



        cal_tec_obj = CalTec(df_nav, trade_dates, data_type=2)
        try:
            self.ori_data = cal_tec_obj.get_indic()
        except Exception, e:
            print  "cal tec indictor exception: %s" % e.message
            return

        self.ori_data.dropna(inplace=True)
        feature_predict = self.feature_selected[self.ass_id]
        all_dates = self.ori_data.index

        p_s_date = all_dates[0]
        p_in_date = all_dates[self.train_num]
        p_e_date = all_dates[-1]
        p_s_num = 0
        p_in_num = self.train_num

        means_arr = []
        all_data = self.ori_data[p_in_date:]
        while p_in_date <= p_e_date:
            p_s_num += 1
            p_in_num += 1
            p_data = self.ori_data[p_s_date:p_in_date]
            ratios = np.array(p_data['pct_chg'])
            try:
                [model, states] = self.training(p_data, list(feature_predict), self.state_num)
            except Exception, e:
                print e
                return (1, "hmm training fail")
            #means = HmmNesc.state_statistic(p_data, self.state_num, states, model)
            means = HmmNesc.cal_stats_pro(model, states, ratios)
            means_arr.append(means)
            if p_in_date != p_e_date:
                p_s_date = all_dates[p_s_num]
                p_in_date = all_dates[p_in_num]
            else:
                p_in_date += datetime.timedelta(days=1)


        tmp = {
            'tc_signal': dict_status,
            'tc_action': dict_action,
        }

        df_tmp = pd.DataFrame(tmp, index=df_nav.index)
        df_tmp.loc[df_tmp['tc_stop_high'] == float('inf'), 'tc_stop_high'] = 0
        df_nav = pd.concat([df_nav, df_tmp], axis=1)

        # print df_nav.head(5000)
        # print df_nav.tail(60)
        return df_nav



    @staticmethod
    def training(t_data, features, state_num):
        """

        :param t_data: 用来训练的数据
        :param features: 训练用到的特征
        :param state_num: 隐形状态数目
        :return: [transit matrix, [mean, variance]], mean and variance 是正态分布的
        """
        # 特征数据
        fea_data = []
        for feature in features:
            fea_data.append(t_data[feature])

        X = np.column_stack(fea_data)
        # X = np.array(X)
        # print X
        # X = sigmoid(normalize(X, axis=0))
        # print X
        # X = boxcox(X)
        model = GaussianHMM(n_components=state_num, covariance_type="diag", n_iter=5000) #, params="st", init_params="st")
        # X = np.nan_to_num(0.0)
        # print X.shape
        # os._exit(0)
        model = model.fit(X)
        # print model.transmat_
        # print model.startprob_
        # if len(features) > 1:
        #     print model.means_
        #     print model.means_weight
        # # print model.covars_
        #     os._exit(0)
        states = model.predict(X)
        # print "Transition matrix"
        # print np.mat(model.transmat_)
        # print np.shape(model.transmat_)
        # print ""

        # print "means and vars of each hidden state"
        # for i in xrange(model.n_components):
        #     print "%dth hidden state" % i
        #     print "mean = ", model.means_[i]
        #     print "var = ", np.diag(model.covars_[i])
        #     print ""
        return [model, states]


    @staticmethod
    def cal_stats_pro(model, states, ratio):
        state_today = states[-1]
        trans_mat_today = model.transmat_[states[-1]]
        mean_today = model.means_[state_today, 1]
        mean_rank_today = sum(model.means_[:, 1] < mean_today)
        
        next_day_state = np.argmax(trans_mat_today)
        next_day_pro = trans_mat_today[next_day_state]
        next_day_mean = model.means_[next_day_state, 1]
        next_day_mean_rank = sum(model.means_[:, 1] < next_day_mean)
        return next_day_mean
