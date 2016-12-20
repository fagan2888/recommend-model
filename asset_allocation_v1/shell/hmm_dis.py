# -*- coding: UTF-8 -*-
"""
Created at Dec 12, 2016
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""
import pandas as pd
import numpy as np
import datetime
from hmmlearn.hmm import GaussianHMM as ghmm
import os
import utils
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mtlab
from scipy.stats import norm

class HmmDis(object):

    def __init__(self, file_handle):
        """
        初始化输入数据和参数
        :param file_handle: 输入数据路径或者file handle（净值数据）, data format
            date, asset1, asset2, asset3, ......
            2010-01-04, 1, 1, 1, ......
            2010-01-05, 1.1, 1.2, 1.3, ......
        """
        self.nav_data = pd.read_csv(file_handle, index_col=['date'], parse_dates=['date'])
        self.back_days = 61

    def hmm_dis_next_day(self):
        columns = self.nav_data.columns
        dates = self.nav_data.index
        pcts = self.nav_data.pct_change()
        pcts.fillna(0.0)
        date_periods = dates[self.back_days:]
        # print date_periods[0]
        win_ratio_all = 0.0
        for col in columns:
            print col
            win_times = 0.0
            total_num = 0.0
            for next_date in date_periods:
                total_num += 1.0
                data_period = pcts[pcts.index <= next_date][-self.back_days:]
                data_list = list(data_period[col])
                dis_used_data = data_list[:-1]
                next_ratio = data_list[-1]
                model, states = HmmDis.hmm_train(dis_used_data)
                # print len(states)
                # os._exit(0)

                # print model.transmat_
                # HmmDis.plot(dis_used_data, model, states)
                means = model.means_
                variances = model.covars_
                trans_mat = model.transmat_
                cur_state = states[-1]
                tran_pro = trans_mat[cur_state]

                mean_all = np.sum(means * tran_pro)
                std_all = np.sum(variances * tran_pro)
                (mu, sigma) = stats.norm.fit(dis_used_data)
                pro_hmm = norm.cdf(next_ratio, loc=mean_all, scale=std_all)
                pro_mar = norm.cdf(next_ratio, loc=mu, scale=sigma)
                if pro_hmm > pro_mar:
                    win_times += 1.0
                # print win_times
                # tmp_count += 1
                # if tmp_count == 125:
                #     os._exit(0)
                # os._exit(0)
            print win_times, total_num, win_times / total_num
            win_ratio_all += (win_times / total_num)
        print win_ratio_all / len(columns)

    @staticmethod
    def hmm_train(data):
        """
        hmm训练
        :param data:输入数据，list([[], [], [], ...... , []])
        :return:返回model
        """
        x = np.column_stack([data])
        model = ghmm(n_components=3, covariance_type="diag", n_iter=5000).fit(x)
        states = model.predict(x)
        return [model, states]

    @staticmethod
    def plot(ori_data, model, states):
        means = model.means_
        variances = model.covars_
        trans_mat = model.transmat_
        # print means, variances
        (mu, sigma) = stats.norm.fit(ori_data)
        count, bins, ignored = plt.hist(ori_data, len(ori_data), normed=False)
        y_origin = mtlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y_origin, 'g-', label='origin', linewidth=2)

        num = len(means)
        colors = ['r-', 'k-', 'b-']
        for ite in range(num):
            mean = means[ite]
            varian = variances[ite, 0, 0]
            y = mtlab.normpdf(bins, mean, varian)
            plt.plot(bins, y, colors[ite], label="state " + str(ite), linewidth=2)
        cur_state = states[-1]
        tran_pro = trans_mat[cur_state]

        mean_all = np.sum(means * tran_pro)
        std_all = np.sum(variances * tran_pro)
        print cur_state
        print mean_all, std_all
        print mu, sigma
        y_all = mtlab.normpdf(bins, mean_all, std_all)
        plt.plot(bins, y_all, 'k-', label="all", linewidth=2)
        plt.legend(loc='upper left')
        plt.xlabel("ratio")
        plt.ylabel("numbers")
        plt.title("Dis.")
        plt.show()

if __name__ == '__main__':
    HmmDis_obj = HmmDis("../tmp/equalriskasset.csv")

    print HmmDis_obj.hmm_dis_next_day()
