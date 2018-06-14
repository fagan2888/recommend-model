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
from sklearn.preprocessing import normalize
from cal_tech_indic import CalTechIndic as CalTec
import warnings


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TimingHmm(object):

    def __init__(self, ori_data, timing, trade_dates, start_date = '20120727'):

        self.start_date = start_date
        self.state_num = 5
        self.ass_id = timing['tc_index_id']
        self.ori_data = self.preprocess_data(ori_data, timing['tc_index_id'],  trade_dates)
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
        self.features = ['macd', 'atr', 'cci', 'rsi', 'sobv', 'mtm', 'roc', \
                        'slowkd', 'pct_chg', 'pvt', 'wvad', 'priceosc', \
                        'bias', 'vma', 'vstd', 'dpo']
        # 模型训练最多能用到的样本数, 34为计算技术指标所消耗的样本数
        max_train_num = len(ori_data[:start_date]) - 34 - 1
        # 最多只使用249个(5年)的样本, 避免加入过于久远的数据
        self.train_num = max_train_num if max_train_num < 249 else 249


    def preprocess_data(self, df_nav, asset_id, trade_dates):
        if asset_id == '120000013':
            self.state_num = 3
        asset_id = int(asset_id)
        av_selection = {120000001: 'volume', 120000002:'volume', 120000013:'volume',
                         120000014:'volume', 120000015:'amount', 120000028:'volume',
                            120000029:'volume'}
        df_nav = df_nav.rename(columns={'tc_open':'open', 'tc_high':'high', 'tc_close':'close', 'tc_low':'low', 'tc_volume':'volume', 'tc_amount':'amount'})
        df_nav['volume'] = df_nav[av_selection[asset_id]]
        columns = ['open', 'high', 'low', 'close', 'volume']
        df_nav = df_nav[columns]
        trade_dates.columns = ['trade_type']
        trade_dates.index.name = 'date'
        df_nav['volume'] = df_nav['volume'].replace(0, np.nan)
        df_nav = df_nav.fillna(method = 'ffill').fillna(method = 'bfill')
        df_nav = day_2_week(df_nav, trade_dates)
        return df_nav


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

    def timing(self):
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
            means = self.cal_stats_pro(model, states, ratios)
            #print p_in_date, np.sign(means)
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
        union_data_tmp.index.name = 'tc_date'
        union_data_tmp['tc_signal'] = np.sign(union_data_tmp['view'])
        return union_data_tmp

if __name__ == "__main__":
    ori_data = pd.read_csv('tmp/120000001_ori_week_data.csv', index_col = 0, \
            parse_dates = True)
    vw_view = pd.Series({'vw_asset_id': '120000001'})
    hmm = HmmNesc(ori_data, vw_view, start_date = '20120727')
    result = hmm.handle()
    print(result)
