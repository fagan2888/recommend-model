# -*- coding: utf-8 -*-
"""
Created at Mar. 25, 2017
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""
import sys
import warnings
import datetime
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from cal_tech_indic import CalTechIndic as CalTec

sys.path.append('./shell')
warnings.filterwarnings('ignore')

class HmmNesc(object):

    def __init__(self, ori_data, vw_view, start_date = '20120727'):
        self.start_date = start_date
        self.ass_id = vw_view['vw_asset_id']
        self.ori_data = ori_data
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
        # 模型训练最多能用到的样本数, 34为计算技术指标所消耗的样本数
        max_train_num = len(ori_data[:start_date]) - 34 - 1
        # 最多只使用249个(5年)的样本, 避免加入过于久远的数据
        self.train_num = max_train_num if max_train_num < 249 else 249

    def cal_indictor(self):
        cal_tec_obj = CalTec(self.ori_data)
        self.ori_data = cal_tec_obj.get_indic()
        self.ori_data.dropna(inplace=True)

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
        model = GaussianHMM(n_components=state_num, covariance_type="diag", \
                random_state = 0, n_iter=5000) #, params="st", init_params="st")
        model = model.fit(X)
        states = model.predict(X)
        return [model, states]

    def predict(self, p_data, model, features):
        """
        :usage: 样本外预测
        :param p_data: 待预测数据
        :param model: 训练得到的模型
        :return: 收益、最大回撤、胜率等指标
        """
        # 特征数据
        fea_data = []
        for feature in features:
            fea_data.append(p_data[feature])
        X = np.column_stack(fea_data)
        states = model.predict(X)
        return states

    @staticmethod
    def cal_stats_pro(model, states, ratio):
        trans_mat_today = model.transmat_[states[-1]]
        next_day_state = np.argmax(trans_mat_today)
        next_day_mean = model.means_[next_day_state, 1]
        return next_day_mean

    def handle(self):
        """
        :usage: 执行程序
        :return: None
        """
        self.cal_indictor()
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
            [model, states] = self.training(p_data, list(feature_predict), self.state_num)
            #means = HmmNesc.state_statistic(p_data, self.state_num, states, model)
            means = HmmNesc.cal_stats_pro(model, states, ratios)
            means_arr.append(means)
            if p_in_date != p_e_date:
                p_s_date = all_dates[p_s_num]
                p_in_date = all_dates[p_in_num]
            else:
                p_in_date += datetime.timedelta(days=1)

        ####### state statistic
        union_data_tmp = {}
        union_data_tmp["view"] = means_arr
        union_data_tmp = pd.DataFrame(union_data_tmp, index = all_data.index)
        union_data_tmp = union_data_tmp[self.start_date:]
        return union_data_tmp

if __name__ == "__main__":
    ori_data = pd.read_csv('tmp/120000001_ori_week_data.csv', index_col = 0, \
            parse_dates = True)
    vw_view = pd.Series({'vw_asset_id': '120000001'})
    hmm = HmmNesc(ori_data, vw_view, start_date = '20120727')
    result = hmm.handle()
    print(result)
