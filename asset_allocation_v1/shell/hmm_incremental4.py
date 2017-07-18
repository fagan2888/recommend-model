# -*- coding: utf-8 -*-
"""
Created at Mar. 25, 2017
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""
import sys
sys.path.append('./shell')
import datetime
import numpy as np
#import pylab as pl
#from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
#from hmmlearn.hmm import MultinomialHMM
#from hmmlearn.hmm import GaussianHMM
from pomegranate import MultivariateGaussianDistribution, NormalDistribution, \
        HiddenMarkovModel
#from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import pandas as pd
#from matplotlib import pyplot as plt
# from pykalman import KalmanFilter
#from scipy.stats import boxcox
from scipy import stats
import os
#from utils import sigmoid
#from sklearn.preprocessing import normalize
#from db import caihui_tq_qt_index as load_index
from db import asset_trade_dates as load_td
from db import asset_vw_view as ass_view
from db import asset_vw_view_inc as ass_view_inc
from cal_tech_indic import CalTechIndic as CalTec
import warnings
warnings.filterwarnings('ignore')

class MulGauHmm(object):

    def __init__(self, n_components, ori_data):
        self.ori_data = ori_data
        self.scaler = StandardScaler()
        self.t_data = self.scaler.fit_transform(self.ori_data)
        #np.save('tmp.npy', self.t_data)
        self.state_num = n_components
        self.X = [[tuple(self.t_data[i]) for i in range(len(self.t_data))]]

    @staticmethod
    def reject_outliers(ori_data):
        clf = EllipticEnvelope(contamination = 0.10)
        try:
            clf.fit(ori_data)
        except ValueError, e:
            print e.message+", so we don't clean data"
            return ori_data
        outlier_sample = clf.predict(ori_data)
        cleaned_data = ori_data[outlier_sample == 1]

        return cleaned_data

    def fit(self):
        if self.ori_data.shape[1] > 1:
            model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, \
                    self.state_num, self.X)
            self.model = model
            self.states = np.array(model.predict(self.X[0], algorithm = 'viterbi'))[1:]
            for i in range(self.state_num):
                idx = (self.states == i)
                state_mean = (self.ori_data[idx]).mean(0)
                if i == 0:
                    self.means_ = state_mean
                else:
                    self.means_ = np.vstack([self.means_, state_mean])

            #print self.means_
            self.transmat_ = self.model.dense_transition_matrix()[:self.state_num, :self.state_num]

            sorted_means = np.sort(self.means_[:,1])
            sorted_states_dict = {}
            for i in range(self.state_num):
                sorted_states_dict[i] = sorted_means.tolist().index(self.means_[i, 1])
            sorted_states = [sorted_states_dict[i] for i in self.states]

            #print sorted_means
            self.sorted_means = sorted_means
            self.sorted_states = np.array(sorted_states)

        elif self.ori_data.shape[1] == 1:
            model = HiddenMarkovModel.from_samples(NormalDistribution,\
                    self.state_num, self.X)
            self.model = model
            self.states = np.array(model.predict(self.X[0], algorithm = 'viterbi'))[1:]
            self.means_ = np.zeros([self.state_num, 1])
            for i in range(self.state_num):
                self.means_[i][0] += model.states[i].distribution.parameters[0]
            self.means_ = self.scaler.inverse_transform(self.means_)
            self.transmat_ = self.model.dense_transition_matrix()[:self.state_num, :self.state_num]

            sorted_means = np.sort(self.means_[:,0])
            sorted_states_dict = {}
            for i in range(self.state_num):
                sorted_states_dict[i] = sorted_means.tolist().index(self.means_[i, 0])
            sorted_states = [sorted_states_dict[i] for i in self.states]

            self.sorted_means = np.array(sorted_means)
            self.sorted_states = np.array(self.states)

    def predict(self, p_data):
        p_data = StandardScaler().fit_transform(p_data, 1)
        p_data = [[tuple(p_data[i]) for i in range(len(p_data))]]
        pre_states = self.model.predict(p_data[0])
        return pre_states

    def update(self, new_data):
        self.ori_data = new_data
        self.scaler = StandardScaler()
        self.t_data = self.scaler.fit_transform(self.ori_data)
        self.cleaned_data = self.reject_outliers(self.t_data)
        self.X = [[tuple(self.t_data[i]) for i in range(len(self.t_data))]]
        self.cleaned_X = [[tuple(self.cleaned_data[i]) for i in range(len(self.cleaned_data))]]

        dists = []
        starts = np.zeros(self.state_num)
        for i in range(self.state_num):
            dist_mean = self.model.states[i].distribution.parameters[0]
            dist_cor = self.model.states[i].distribution.parameters[1]
            dist = MultivariateGaussianDistribution(dist_mean, dist_cor)
            dists.append(dist)
        trans_mat = self.model.dense_transition_matrix()
        starts[self.states[-1]] = 1.0
        model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts)
        model.bake()
        model.fit([[self.X[0][-1]]])
        self.model = model
        self.states = np.array(model.predict(self.X[0], algorithm = 'viterbi'))[1:]
        self.means_ = np.zeros([self.state_num, self.ori_data.shape[1]])
        for i in range(self.state_num):
            for j in range(self.ori_data.shape[1]):
                self.means_[i][j] += model.states[i].distribution.parameters[0][j]
        self.means_ = self.scaler.inverse_transform(self.means_)
        self.transmat_ = self.model.dense_transition_matrix()[:self.state_num, :self.state_num]

        sorted_means = np.sort(self.means_[:,1])
        sorted_states_dict = {}
        for i in range(self.state_num):
            sorted_states_dict[i] = sorted_means.tolist().index(self.means_[i, 1])
        sorted_states = [sorted_states_dict[i] for i in self.states]

        self.sorted_means = sorted_means
        self.sorted_states = np.array(sorted_states)


class HmmNesc(object):

    def __init__(self, globalid, start_date=None, end_date=None):
        self.ass_id = globalid
        self.start_date = start_date
        self.end_date = end_date
        self.assets = {
            '120000001':'2070000060', #沪深300
            '120000002':'2070000187', #中证500
            '120000013':'2070006545', #标普500指数
            '120000014':'2070000626', #黄金指数
            '120000015':'2070000076', #恒生指数
            '120000028':'2070006521', #标普高盛原油商品指数收益率
            '120000029':'2070006789', #南华商品指数收益率
        }
        '''
        #day feature
        self.feature_selected = {
            '120000001':['cci', 'pct_chg', 'roc', 'mtm'],
            '120000002':['vma', 'pct_chg', 'roc', 'atr'],
            '120000013':['cci', 'pct_chg', 'mtm'],
            '120000014':['roc', 'pct_chg', 'atr', 'vma'],
            '120000015':['cci', 'pct_chg', 'vma', 'mtm'],
            '120000028':['mtm', 'pct_chg'],
            '120000029':['atr', 'pct_chg', 'roc'],
        }
        '''

        #week feature
        self.feature_selected = {
            '120000001':['dpo', 'pct_chg', 'macd', 'atr', 'vstd'],
            '120000002':['dpo', 'pct_chg', 'atr', 'vma'],
            '120000013':['roc', 'pct_chg', 'atr', 'vma'],
            '120000014':['cci', 'pct_chg', 'atr', 'vstd', 'vma'],
            '120000015':['cci', 'pct_chg', 'atr', 'vstd', 'vma'],
            '120000028':['mtm', 'pct_chg'],
            '120000029':['dpo', 'pct_chg', 'macd', 'vma'],
        }

        # 隐形状态数目
        self.features = ['macd', 'atr', 'cci', 'mtm', 'roc', 'pct_chg', \
                'vma', 'vstd', 'dpo']
        self.state_nums = {
            '120000001':5,
            '120000002':5,
            '120000013':5,
            '120000014':5,
            '120000015':5,
            '120000028':5,
            '120000029':5,
        }
        self.state_num = self.state_nums[self.ass_id]
        # 模型训练用到的样本数, 149加1即为这个样本数
        self.train_num = 249
        # 训练开始时间
        self.t_start = datetime.datetime(2005, 8, 1)
        # 训练结束时间
        self.t_end = datetime.datetime(2012, 8, 1)

        # 验证开始时间
        self.v_start = datetime.datetime(2003, 1, 1)
        # 验证结束时间
        self.v_end = datetime.datetime(2003, 12, 31)

        # 测试开始时间
        self.test_start = datetime.datetime(2009, 1, 1)
        # 测试结束时间
        self.test_end = datetime.datetime(2016, 12, 31)
        # 单边交易费用
        self.fee_ratio = 0.0
        # 多头收益率阈值，用于选取多头状态
        self.filter_ratio = 0.03
        # 市场状态划分阈值
        self.sharpe_ratio = 0.2
        # 特征评价指标，0: 平均排名（收益）, 1: 最大回撤排名, 2: 收益回撤比排名, 3: 平均胜率排名
        self.eva_indic = [0, 3]
        # 选取指标中排名前几的特征，2代表选取某一指标中排名前2的特征作为最终使用的特征
        self.rank_num = 2
    def init_data(self):
        result = self.get_secode()
        if result[0] > 0:
            return result
        result = self.get_view_id()
        if result[0] > 0:
            return result
        self.get_view_newest_date()
        result = self.get_index_origin_data()
        if result[0] > 0:
            return result
        result = self.get_trade_dates()
        if result[0] > 0:
            return result
        result = self.cal_indictor()
        return result
    def get_secode(self):
        if self.assets.has_key(self.ass_id):
            self.secode = self.assets[self.ass_id]
            return (0, 'get data sucess')
        else:
            return (1, "asset id is not in dict:" + self.ass_id)
    def get_view_id(self):
        ass_vw_df = ass_view.get_viewid_by_indexid(self.ass_id)
        if ass_vw_df.empty:
            return (2, "has no view id for asset:" + self.ass_id)
        self.viewid = ass_vw_df['viewid'][0]
        return (0, 'get data sucess')
    def get_view_newest_date(self):
        ass_vw_inc_df = ass_view_inc.get_asset_newest_view(self.viewid)
        # 判断是否已经有历史view
        self.view_newest_date = None
        if not ass_vw_inc_df.empty:
            if ass_vw_inc_df['newest_date'][0] != None:
                self.view_newest_date = pd.Timestamp(ass_vw_inc_df['newest_date'][0])
    def get_index_origin_data(self):
        self.ori_data = self.load_offline_data(self.ass_id)
        # load_index.load_index_daily_data(self.secode, \
        #                 self.start_date, self.end_date)
        if self.ori_data.empty:
            return (3, 'has no data for secode:' + self.secode)
        for col in self.ori_data.columns:
            self.ori_data[col].replace(to_replace=0, method='ffill', inplace=True)
        return (0, 'get data sucess')
    def get_trade_dates(self):
        self.trade_dates = load_td.load_trade_dates()
        if self.trade_dates.empty:
            return (4, 'has no data for trade dates')
        return (0, 'get data sucess')
    def cal_indictor(self):
        cal_tec_obj = CalTec(self.ori_data, self.trade_dates, data_type=2)
        try:
            self.ori_data = cal_tec_obj.get_indic()
        except Exception, e:
            return (5, "cal tec indictor exception:" + e.message)
        self.ori_data.dropna(inplace=True)
        return (0, 'get data sucess')
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
            evaluations += [evaluations[0]*evaluations[3]]
            print evaluations
            # print feature, evaluations
            result[feature] = evaluations
        # 最终选择的特征
        feature_selected = set()
        print result
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
        #dates = data.index
        #close = list(data["close"])
        #pct_change = list(data["pct_chg"])
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
            idx = np.append([0.0], state[:-1])
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
    def state_statistic(data, state_num, states, model):
        cur_state = states[-1]
        # state = (states == cur_state)
        # idx = np.append([], state)
        # ratios = np.where(idx == 1, data["pct_chg"], -1)
        # ratios = ratios[ratios != -1]
        # means = ratios.mean()
        # stds = ratios.std(ddof=1)
        means_arr = []
        #stds_arr = []
        for ite in range(state_num):
            state = (states == ite)
            idx = np.append([], state)
            ratios = np.where(idx == 1, data["pct_chg"], -100)
            ratios = ratios[ratios != -100]
            ratios = ratios / 100.0
            if len(ratios) == 0:
                ratios = np.array([0])
            means_arr.append(ratios.mean())
            # stds_arr.append(ratios.std(ddof=1))
        trans_mat = model.transmat_[cur_state]
        used_trans = np.where( trans_mat > 0.0, trans_mat, 0.0)
        means = np.dot(used_trans, means_arr)
        #print cur_state, data['pct_chg'][-1], (100*np.sort(means_arr)).round(2), means
        print means
        return means
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
        # state_hold_days = float(sum(idx))
        ratios = np.where(idx == 1, data["pct_chg"], -100)
        ratios = ratios[ratios != -100]
        # nav_list, max_drawdown = HmmNesc.cal_nav_maxdrawdown(ratios)
        # returns = nav_list[-1] - 1.0
        ratios = ratios / 100.0
        mean_ratio = np.mean(ratios)
        anal_ratio = mean_ratio * 252
        #return_std = np.std(ratios) * np.sqrt(252)
        # thres 为sharpe
        # sharpe = anal_ratio / return_std
        # if sharpe >= thres:
        #     return 1
        # elif -thres < sharpe < thres:
        #     return 0
        # else:
        #     return -1

        # thres 为年化收益率
        if anal_ratio >= thres:
            return 1
        elif -thres < anal_ratio < thres:
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
            idx = np.append([0], state[:-1])
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
        model = MulGauHmm(state_num, X)
        #, params="st", init_params="st")
        # X = np.nan_to_num(0.0)
        # print X.shape
        # os._exit(0)
        model.fit()
        # print model.transmat_
        # print model.startprob_
        # if len(features) > 1:
        #     print model.means_
        #     print model.means_weight
        # # print model.covars_
        #     os._exit(0)
        states = model.states
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
    def update_model(t_data, features, model):
        fea_data = []
        for feature in features:
            fea_data.append(t_data[feature])

        X = np.column_stack(fea_data)
        model.update(X)
        states = model.states
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
    @staticmethod
    def cal_stats_pro(model, states, ratio):
        #state_today = states[-1]
        trans_mat_today = model.transmat_[states[-1]]
        #mean_today = model.means_[state_today, 1]
        #mean_rank_today = sum(model.means_[:, 1] < mean_today)

        next_day_state = np.argmax(trans_mat_today)
        #next_day_pro = trans_mat_today[next_day_state]
        next_day_mean = model.sorted_means[next_day_state]
        #print states[-1], next_day_state, next_day_mean
        print model.sorted_means
        #next_day_mean_rank = sum(model.means_[:, 1] < next_day_mean)
        #print next_day_mean
        return next_day_mean
    @staticmethod
    def statistic_win_ratio(ratios, means_arr, stds_arr):
        ratio_num = len(ratios)
        win_num = 0.0
        for ite in range(ratio_num):
            interval_95 = stats.norm.interval(0.95, loc=means_arr[ite], scale=stds_arr[ite])
            if ratios[ite] >= min(interval_95) and ratios[ite] <= max(interval_95):
                win_num += 1.0
        print "win ratio:", win_num / ratio_num

    def handle(self):
        """
        :usage: 执行程序
        :return: None
        """
        #feature_predict = self.feature_select(self.ori_data[self.t_start:self.t_end], \
        #        self.features, self.state_num, thres = 0, feature_eva = [[4], 4])

        feature_predict = self.feature_selected[self.ass_id]
        #feature_predict = self.features

        all_dates = self.ori_data.index
        self.view_newest_date = None
        if self.view_newest_date == None:
            p_s_date = all_dates[0]
            p_in_date = all_dates[self.train_num]
            p_e_date = all_dates[-1]
            p_s_num = 0
            p_in_num = self.train_num
        else:
            newest_date_pos = np.argwhere(all_dates == self.view_newest_date)[0,0]
            if newest_date_pos == len(all_dates) - 1:
                return (0, "newest view in database, no need to update")
            p_s_date = all_dates[newest_date_pos - self.train_num]
            p_in_date = all_dates[newest_date_pos+1]
            p_e_date = all_dates[-1]
            p_s_num = newest_date_pos - self.train_num
            p_in_num = newest_date_pos + 1
        means_arr = []
        all_data = self.ori_data[p_in_date:]
        while p_in_date <= p_e_date:
            p_s_num += 1
            p_in_num += 1
            p_data = self.ori_data[p_s_date:p_in_date]
            #ratios = np.array(p_data['pct_chg'])
            print p_data.pct_chg[-1]
            try:
                [model, states] = self.training(p_data, list(feature_predict), self.state_num)
            except Exception, e:
                print e
                return (1, "hmm training fail")
            '''

            if p_s_num <= 2:
                [model, states] = self.training(p_data, list(feature_predict), self.state_num)
            else:
                [model, states] = self.update_model(p_data, list(feature_predict), model)
                '''
            #[model, states] = self.training(p_data, list(feature_predict), self.state_num)
            means = HmmNesc.state_statistic(p_data, self.state_num, states, model)
            #print self.rating(p_data, self.state_num, states, self.sharpe_ratio)
            #means = HmmNesc.cal_stats_pro(model, states, ratios)
            means_arr.append(means)
            if p_in_date != p_e_date:
                p_s_date = all_dates[p_s_num]
                p_in_date = all_dates[p_in_num]
            else:
                p_in_date += datetime.timedelta(days=1)

        #insample_model, insample_states = self.training(self.ori_data.loc[all_data.index, :], \
        #        list(feature_predict), self.state_num)
                ####### state statistic
        union_data_tmp = {}
        union_data_tmp["means"] = np.array(means_arr)
        union_data_tmp["dates"] = all_data.index
        union_data_tmp["ids"] = np.repeat(self.viewid, len(means_arr))
        union_data_tmp['create_time'] = np.repeat(datetime.datetime.now(),len(means_arr))
        union_data_tmp['update_time'] = np.repeat(datetime.datetime.now(),len(means_arr))
        union_data_tmp = pd.DataFrame(union_data_tmp)
        print self.cal_sig_wr(self.ori_data.loc[all_data.index,:], means_arr, show_num = True)
        union_data_tmp.loc[:,['dates', 'means']].to_csv('../tmp/tmp_result.csv')
        #result = ass_view_inc.insert_predict_pct(union_data_tmp)
        return union_data_tmp

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
    def load_offline_data(self, ass_id):
        df = pd.read_csv(ass_id + "_ori_day_data.csv", index_col=['date'], \
            parse_dates=['date'])
        return df

    @staticmethod
    def cal_sig_wr(ori_data, means, show_num = False):
        signal = np.sign(means)
        win_num = 0.0
        total_num = 0.0
        signal_nav = 1.0
        t_nav = 1.0

        for i in range(len(means)-1):
            current = signal[i]
            next_ = signal[i+1]
            signal_nav *= (1.0 + ori_data['pct_chg'][i+1]/100)
            if current > 0:
                t_nav *= (1.0 + ori_data['pct_chg'][i+1]/100)

            if current != next_:
                #print signal_nav, current
                if (signal_nav - 1.0)*current >= 0:
                    win_num += 1.0
                total_num += 1.0
                signal_nav = 1.0

        if total_num == 0:
            if (signal_nav - 1.0)*current >= 0:
                return 1
            else:
                return 0
        if show_num:
            print 'win_num: ', win_num
            print 'total_num: ', total_num
            print 't_nav: ', t_nav
        win_ratio = win_num / total_num
        return win_ratio

if __name__ == "__main__":
    view_ass = ['120000001', '120000002', '120000013', '120000014', \
                '120000015', '120000029']
    view_ass = ['120000001']
    for v_ass in view_ass:
        print v_ass
        nesc_hmm = HmmNesc(v_ass, '20050101')
        result = nesc_hmm.init_data()
        if result[0] == 0:
            result_handle = nesc_hmm.handle()
            #print result_handle
        else:
            print result
