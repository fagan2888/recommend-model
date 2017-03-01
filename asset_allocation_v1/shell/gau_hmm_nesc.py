# -*- coding: utf-8 -*-
"""
Created at Oct. 13, 2016
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""

import datetime
import numpy as np
import pylab as pl
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
import pandas as pd
from matplotlib import pyplot as plt
# from pykalman import KalmanFilter
from scipy.stats import boxcox
import os
from utils import sigmoid
from sklearn.preprocessing import normalize

class HmmNesc(object):

    def __init__(self, file_handle):
        self.ori_data = pd.read_csv(file_handle, index_col='date', parse_dates=["date"], low_memory=False)
        # print self.ori_data
        # 隐形状态数目
        self.state_num = 13
        # 特征集"SI", "SRMI",
        self.features = ["pct_chg", "high", "PVT", "WVAD", "PRICEOSC", "slowKD", "ROC", "MTM", "DPO", "ATR", "CCI",
                         "RSI", "BIAS", "vol_ratio", "VMA", "TAPI", "VSTD", "SOBV", "VMACD"]
        # 训练开始时间
        self.t_start = datetime.datetime(2005, 8, 1)
        # 训练结束时间
        self.t_end = datetime.datetime(2008, 8, 1)

        # 验证开始时间
        #self.v_start = datetime.datetime(2003, 1, 1)
        # 验证结束时间
        #self.v_end = datetime.datetime(2003, 12, 31)

        # 测试开始时间
        self.test_start = datetime.datetime(2009, 1, 9)
        # 测试结束时间
        self.test_end = datetime.datetime(2016, 10, 21)
        # 单边交易费用
        self.fee_ratio = 0.0
        # 多头收益率阈值，用于选取多头状态
        self.filter_ratio = 0.0
        # 市场状态划分阈值
        self.sharpe_ratio = 0.2
        # 特征评价指标，0: 平均排名（收益）, 1: 最大回撤排名, 2: 收益回撤比排名, 3: 平均胜率排名
        self.eva_indic = [0, 3]
        # 选取指标中排名前几的特征，2代表选取某一指标中排名前2的特征作为最终使用的特征
        self.rank_num = 2

    @staticmethod
    def feature_select(t_data, features, state_num, thres, feature_eva=[[0,1], 2]):
        """
        特征选取
        :param t_data: 训练数据
        :param features: 需要比较的特征
        :param state_num: 隐形状态数目
        :param feature_eva: [[指标1, 指标2, ...], 排名前几]
        :return: 指定priority的feature排名
        """
        eva_indic, rank_num = feature_eva
        result = {}
        # 计算每个特征的每个评价指标值
        for feature in features:
            print feature
            # print datetime.datetime.now()
            [model, states] = HmmNesc.training(t_data, [feature], state_num)
            evaluations = HmmNesc.rating(t_data, state_num, states, thres)
            print evaluations
            # print feature, evaluations
            result[feature] = evaluations
        # 最终选择的特征
        feature_selected = set()
        for indx in eva_indic:
            sorted_result = sorted(result.iteritems(), key=lambda item: item[1][indx], reverse=True)
            for ite in range(rank_num):
                feature_selected.add(sorted_result[ite][0])
        print feature_selected
        os._exit(0)
        return feature_selected

    @staticmethod
    def proceed_choose(data, state_num, states, thres):
        """
        :usage: 得到data中最后一天持仓情况
        :param data: 预测的数据
        :param state_num: 状态数目
        :param states: 状态序列
        :param thres: 过滤state阈值
        :return: 当天持仓否
        """
        dates = data.index
        close = list(data["close"])
        pct_change = list(data["pct_chg"])
        # state_num = len(states)
        # result = {}
        #union_data = {}
        #union_data["close"] = close
        # 计算每个状态胜率
        #union_data_tmp = {}
        #union_data_tmp["close"] = close
        #union_data_tmp["ratio"] = pct_change
        # 根据每个状态的收益率得到最终的择时信号，状态在训练的时间内收益率不小于3%则选择为多头信号
        choose_final = np.zeros(len(data))
        for i in range(state_num):
            #result[i] = {}
            state = (states == i)
            # 发出信号为t日，从t+2日开始算收益(与原始算法不同，原始算法为以t+1日开盘价买入）,这里更偏向操作基金
            idx = np.append([0.0, 0.0], state[:-2])
            ratios = np.where(idx == 1, data["pct_chg"], 0)
            nav_list, max_drawdonw_list = HmmNesc.cal_nav_maxdrawdown(ratios)
            #union_data_tmp["signal"] = idx
            #union_data_tmp = pd.DataFrame(union_data_tmp, index=dates)
            #[td_win, td_total, win_ratio, holding_days] = HmmNesc.win_ratio(union_data_tmp)
            #union_data["nav_%d" % i] = nav_list
            #union_data["max_drawdown_%d" % i] = max_drawdonw_list
            #result[i] = [nav_list[-1], min(max_drawdonw_list), abs(nav_list[-1] - 1.0) / abs(min(max_drawdonw_list)),
            #             win_ratio]
            #print [nav_list[-1], min(max_drawdonw_list), abs(nav_list[-1] - 1.0) / abs(min(max_drawdonw_list)),
            #       win_ratio]
            # 择取收益不小于thres的状态作为多头
            if (nav_list[-1] - 1.0) >= thres:
                choose_final += state

        return choose_final[-1]
    @staticmethod
    def market_states(data, state_num, states, thres):
        """
        :usage: 得到data中最后一天持仓情况
        :param data: 预测的数据
        :param state_num: 状态数目
        :param states: 状态序列
        :param thres: 市场(state)划分的阈值，为sharpe ratio
        :return: 当前市场状态， 分为涨、跌、平（1， 0， -1）
        """
        cur_state = states[-1]
        state = (states == cur_state)
        idx = np.append([0], state[:-1])
        ratios = np.where(idx == 1, data["pct_chg"], 0.0)
        # nav_list, max_drawdown = HmmNesc.cal_nav_maxdrawdown(ratios)
        # returns = nav_list[-1] - 1.0
        ratios = ratios / 100.0
        mean_ratio = np.mean(ratios)
        anal_ratio = mean_ratio * 252.0
        return_std = np.std(ratios) * np.sqrt(252.0)
        sharpe = anal_ratio / return_std
        if sharpe >= thres:
            return 1
        elif -thres < sharpe < thres:
            return 0
        else:
            return -1
    @staticmethod
    def rating(data, state_num, states, thres):
        """

        :param data: 用于训练或者择时的原始数据
        :param model: 训练得到的model
        :param states: 训练得到的隐形状态
        :return: [最终净值， 最大回撤， 收益回撤比， 平均胜率]
        """
        dates = data.index
        close = list(data["close"])
        pct_change = list(data["pct_chg"])
        # state_num = len(states)
        result = {}
        union_data = {}
        union_data["close"] = close
        # 计算每个状态胜率
        union_data_tmp = {}
        union_data_tmp["close"] = close
        union_data_tmp["ratio"] = pct_change
        # 根据每个状态的收益率得到最终的择时信号，状态在训练的时间内收益率不小于3%则选择为多头信号
        choose_final = np.zeros(len(data))
        for i in range(state_num):
            result[i] = {}
            state = (states == i)
            # 发出信号为t日，从t+2日开始算收益(与原始算法不同，原始算法为以t+1日开盘价买入）,这里更偏向操作基金
            idx = np.append([0, 0], state[:-2])
            ratios = np.where(idx == 1, data["pct_chg"], 0)
            nav_list, max_drawdonw_list = HmmNesc.cal_nav_maxdrawdown(ratios)
            union_data_tmp["signal"] = idx
            union_data_tmp = pd.DataFrame(union_data_tmp, index=dates)
            [td_win, td_total, win_ratio, holding_days] = HmmNesc.win_ratio(union_data_tmp)
            union_data["nav_%d" % i] = nav_list
            union_data["max_drawdown_%d" % i] = max_drawdonw_list
            result[i] = [nav_list[-1], min(max_drawdonw_list), abs(nav_list[-1] - 1.0) / abs(min(max_drawdonw_list)), win_ratio]
            # print [nav_list[-1], min(max_drawdonw_list), abs(nav_list[-1] - 1.0) / abs(min(max_drawdonw_list)), win_ratio]
            # 择取收益不小于thres的状态作为多头
            if (nav_list[-1] - 1.0) >= thres:
                choose_final += idx

        choose_final = (choose_final > 0)
        f_ratio = np.where(choose_final == 1, data["pct_chg"], 0)
        nav_list, max_drawdonw_list = HmmNesc.cal_nav_maxdrawdown(f_ratio)
        union_data_tmp["signal"] = choose_final
        union_data_tmp = pd.DataFrame(union_data_tmp, index=dates)
        [td_win, td_total, win_ratio, holding_days] = HmmNesc.win_ratio(union_data_tmp)
        # print [nav_list[-1], min(max_drawdonw_list), abs(nav_list[-1] - 1.0) / abs(min(max_drawdonw_list)), win_ratio]
        # print td_total
        # print td_win,td_total,holding_days
        # union_data["signal"] = choose_final
        # union_data["f_nav"] = nav_list
        # union_data["f_max_drawdow"] = max_drawdonw_list
        # large_captial_ratio = data["pct_chg"]
        # lc_nav, lc_max_drawdown = HmmNesc.cal_nav_maxdrawdown(large_captial_ratio)
        # union_data["lc_nav"] = lc_nav
        # union_data["lc_max_drawdown"] = lc_max_drawdown
        # union_data = pd.DataFrame(union_data, index=dates)
        # union_data.to_csv("nesc_hmm_state_nav.csv")
        max_drawdown = 0.0
        if abs(min(max_drawdonw_list)) == 0:
            max_drawdown = 0.0
        else:
            max_drawdown = abs(nav_list[-1] - 1.0) / abs(min(max_drawdonw_list))
        return [nav_list[-1], min(max_drawdonw_list), max_drawdown, win_ratio]

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
        model = GaussianHMM(n_components=state_num, covariance_type="diag", n_iter=5000, params="st", init_params="st")
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
        # covars = []
        # for i in range(self.state_num):
        #     covars.append(np.diagonal(model.covars_[i]))
        # start_pro = [1.0]
        # for i in range(self.state_num-1):
        #     start_pro.append(0.0)
        # start_pro = np.array(start_pro)
        # print start_pro.sum()
        # print model.startprob_
        # os._exit(0)
        # covars = np.array(covars)
        # p_model = GaussianHMM(n_components=self.state_num, covariance_type="diag",
        #                       transmat_prior=model.transmat_, means_prior=model.means_, covars_prior=covars,
        #                       n_iter=100)
        X = np.column_stack(fea_data)

        # p_model = p_model.fit(X)
        # print model.means_
        # print p_model.means_

        states = model.predict(X)

        return states

    def tmp_method_test(self):
        """
        :usage: 测试单个方法运行是否通过
        :return: None
        """
        #print self.ori_data[self.test_start:self.test_end]
        #os._exit(0)
        # feature_selected = HmmNesc.feature_select(self.ori_data[self.t_start:self.t_end], self.features, self.state_num,
        #                                          self.filter_ratio, [self.eva_indic, self.rank_num])
        # print feature_selected
        feature_selected = set(['WVAD', 'pct_chg', 'PVT'])
        print "*********************"
        test_dates = self.ori_data[self.test_start:self.test_end].index
        # print list(test_dates)
        p_s_date = self.test_start
        p_in_date = datetime.datetime(2012, 1, 6)
        p_e_date = datetime.datetime(2012, 10, 26)
        tmp_date = p_in_date
        tmp_e_date = datetime.datetime(2012, 10, 26)
        is_holding = []
        while p_in_date <= p_e_date:
            print p_in_date
            p_data = self.ori_data[p_s_date:p_in_date]
            [model, states] = self.training(p_data, list(feature_selected), self.state_num)
            # print model.means_[2]
            # print np.diag(model.covars_[2])
            # os._exit(0)
            # print model.transmat_
            is_holding.append(HmmNesc.proceed_choose(p_data, self.state_num, states, self.filter_ratio))
            #is_holding.append(HmmNesc.market_states(p_data, self.state_num, states, self.sharpe_ratio))
            p_s_date = HmmNesc.get_move_day(test_dates, p_s_date, 1, previous=False)
            p_in_date = HmmNesc.get_move_day(test_dates, p_in_date, 1, previous=False)
        print "herererere"
        market_states = np.array(is_holding)
        is_holding = (market_states > 0)
        test_data = self.ori_data[tmp_date:tmp_e_date]
        ratios = np.where(is_holding == 1, test_data["pct_chg"], 0)
        # 计算真实的净值（延后两天）
        used_holding = np.append([0.0], is_holding[:-1])
        used_ratio = np.where(used_holding == 1, test_data["pct_chg"], 0) 
        nav_list, max_drawdonw_list = HmmNesc.cal_nav_maxdrawdown(used_ratio)
        union_data_tmp = {}
        union_data_tmp["market"] = market_states
        union_data_tmp["close"] = test_data["close"]
        union_data_tmp["ratio"] = test_data["pct_chg"]
        union_data_tmp["self_ratio"] = used_ratio
        union_data_tmp["signal"] = used_holding
        union_data_tmp["self_ori_ratio"] = ratios
        union_data_tmp["self_ori_signal"] = is_holding
        union_data_tmp["nav"] = nav_list
        union_data_tmp["maxdrawdown"] = max_drawdonw_list
        large_captial_ratio = test_data["pct_chg"]
        lc_nav, lc_max_drawdown = HmmNesc.cal_nav_maxdrawdown(large_captial_ratio)
        union_data_tmp["lc_nav"] = lc_nav
        union_data_tmp["lc_max_drawdown"] = lc_max_drawdown
        union_data_tmp = pd.DataFrame(union_data_tmp, index=test_data.index)
        [td_win, td_total, win_ratio, holding_days] = HmmNesc.win_ratio(union_data_tmp)
        print td_win, td_total, win_ratio, holding_days
        print [nav_list[-1], min(max_drawdonw_list), abs(nav_list[-1] - 1.0) / abs(min(max_drawdonw_list)), win_ratio]
        union_data_tmp.to_csv("market_states_000300_week_20120108_20161014_03_0_3_2.csv")

        # print model.covars_[0]
        # print np.shape(model.covars_[0])
        # print len(model.covars_[0])
        # print model.covars_weight
        # t_states = self.predict(self.ori_data[self.test_start:self.test_end], model, list(feature_selected))
        # v_states = self.predict(self.ori_data[self.v_start:self.v_end], model, list(feature_selected))
        # print "test:"
        # HmmNesc.rating(self.ori_data[self.t_start:self.t_end], self.state_num, states, self.filter_ratio)
        # print "vali:"
        # HmmNesc.rating(self.ori_data[self.v_start:self.v_end], self.state_num, v_states, self.filter_ratio)
        # # self.plot(self.ori_data[self.t_start:self.t_end], model, states)
        # print "test:"
        # HmmNesc.rating(self.ori_data[self.test_start:self.test_end], self.state_num, t_states, self.filter_ratio)
        return None

    def plot(self, data, model, states):
        """
        :usage: 画出状态散点图和状态收益曲线
        :param data:原始数据
        :param model: HMM训练出来的模型
        :param states: 隐形状态
        :return: None
        """
        # print trained parameters and plot
        print "Transition matrix"
        print model.transmat_
        print np.shape(model.transmat_)
        print ""

        print "means and vars of each hidden state"
        for i in xrange(model.n_components):
            print "%dth hidden state" % i
            print "mean = ", model.means_[i]
            print "var = ", np.diag(model.covars_[i])
            print ""

        dates = pd.to_datetime(data.index)
        # print dates
        years = YearLocator()  # every year
        months = MonthLocator()  # every month
        yearsFmt = DateFormatter('%Y')
        close = np.array(data["close"])
        returns = np.array(data["pct_chg"])

        fig = plt.figure(1, figsize=(15, 12))
        ax = fig.add_subplot(211)
        # print len(states)
        # os._exit(0)
        # plt.figure(1)
        # plt.subplot(211)
        for i in xrange(self.state_num):
            # use fancy indexing to plot data in each state
            idx = (states == i)
            # print idx
            # print type(states)
            ax.plot(dates[idx], close[idx], '.')
            ax.legend()
            ax.grid(True)

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.autoscale_view()
        #
        # # format the coords message box
        ax.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax.fmt_ydata = lambda x: '$%1.2f' % x

        # plot state
        date_list = dates  # self.date_ordinal_2_dateformat(self.dates)
        log_return = returns
        latest_seq = states
        # print states
        # os._exit(0)
        data = pd.DataFrame({'datelist': date_list, 'logreturn': log_return, 'state': latest_seq}).set_index('datelist')
        # axs = fig.add_subplot(211)
        # pls = plt.figure(figsize=(15, 8))
        axs = fig.add_subplot(212)
        for i in range(self.state_num):
            state = (latest_seq == i)
            idx = np.append(0, state[:-1])
            # print len(idx)
            # print idx
            ratios = np.where(idx == 1, data["logreturn"], 0)
            nav_list = HmmNesc.cal_nav(ratios)
            # print nav_list
            # os._exit(0)
            data['state %d_return' % i] = data.logreturn.multiply(idx, axis=0)
            axs.plot(dates[idx], nav_list, "-", label="%dth hidden state" % i)
            axs.legend()
            axs.grid(True)

        # format the ticks
        axs.xaxis.set_major_locator(years)
        axs.xaxis.set_major_formatter(yearsFmt)
        axs.xaxis.set_minor_locator(months)
        axs.autoscale_view()

        # format the coords message box
        axs.fmt_xdata = DateFormatter('%Y-%m-%d')
        axs.fmt_ydata = lambda x: '$%1.2f' % x


        # show

        # fig.autofmt_xdate()
        # pl.show()

        # fig.autofmt_xdate()
        plt.show()

    @staticmethod
    def cal_nav_maxdrawdown(return_lsit):
        """
        :usage: 根据收益率列表求净值
        :param retun_lsit: 收益率列表
        :return: 净值列表
        """
        num = len(return_lsit)
        nav_list = []
        max_drawdown_list = []
        cur_nav = 1.0
        max_nav = 1.0
        for i in range(num):
            cur_nav *= (1.0 + return_lsit[i] / 100.0)
            nav_list.append(cur_nav)
            if cur_nav < max_nav:
                drawdown = (cur_nav - max_nav) / max_nav
                max_drawdown_list.append(drawdown)
            else:
                max_drawdown_list.append(0.0)
                max_nav = cur_nav

        return [nav_list, max_drawdown_list]

    @staticmethod
    def win_ratio(union_data):
        """
        :usage: 计算择时胜率
        :param union_data: origin data with TD signal
        :return: ratio of winning within specific
        """
        dates = union_data.index
        data_num = len(union_data)
        cur_num = 0
        # 择时开始收盘价
        start_close = 0.0
        # 择时结束收盘价
        end_close = 0.0
        # 是否持仓
        is_holding = False
        # 是否是择时开始点
        started = True
        # 是否是择时结束点
        ended = False
        # 总的择时信号
        td_total = 0.0
        # 择时成功数目
        td_win = 0.0

        holding_days = 0.0

        for date in dates:
            cur_num += 1
            cur_row = union_data[date:date]
            type = cur_row["signal"].values[0]
            close = cur_row["close"].values[0]
            ratio = cur_row["ratio"].values[0]
            if type == 1:
                is_holding = True
                holding_days += 1
            else:
                is_holding = False

            if cur_num == 1:
                started = True
                start_close = close
            else:
                pre_date = HmmNesc.get_move_day(dates, date, 1)
                pre_type = union_data[pre_date:pre_date]["signal"].values[0]
                if type != pre_type:
                    started = True
                    start_close = close

            if cur_num == data_num:
                ended = True
                end_close = close
            else:
                next_date = HmmNesc.get_move_day(dates, date, 1, previous=False)
                next_type = union_data[next_date:next_date]["signal"].values[0]
                if type != next_type:
                    ended = True
                    end_close = close

            if ended:
                td_total += 1
                if started:
                    if is_holding == True and ratio > 0.0:
                        td_win += 1
                    elif is_holding == False and ratio < 0.0:
                        td_win += 1
                else:
                    if is_holding:
                        if end_close > start_close:
                            td_win += 1
                    else:
                        if end_close < start_close:
                            td_win += 1
            started = False
            ended = False

        # print [td_win, td_total, td_win / td_total, holding_days]
        return [td_win, td_total, td_win / td_total, holding_days]

    @staticmethod
    def get_move_day(dates, cur_date, counts, previous=True):
        """
        :usage: get the previous or next date depend on pre_count
        :param dates: date list(DateIndex)
        :param cur_date: current date
        :param counts: how many date you want to move
        :param previous: if is "True" get previous date, else get next date
        :return: date
        """

        date = (cur_date - datetime.timedelta(days=counts)) if previous else (cur_date + datetime.timedelta(days=counts))
        date_array = list(dates)
        date_num = len(date_array)
        while date not in date_array:
            counts += 1
            date = (cur_date - datetime.timedelta(days=counts)) if previous \
                else (cur_date + datetime.timedelta(days=counts))
            if counts > date_num:
                break

        return date

if __name__ == "__main__":
    sh_000300_all = open("./tmp/hmm_000300_week.csv")
    nesc_hmm = HmmNesc(sh_000300_all)
    nesc_hmm.tmp_method_test()
