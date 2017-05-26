# -*- encoding:utf-8 -*-
from __future__ import division
import sys
sys.path.append('./shell')
import os
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import datetime

import itertools

from db import asset_vw_view as ass_view
from db import caihui_tq_qt_index as load_index
from db import asset_trade_dates as load_td
from cal_tech_indic import CalTechIndic as CalTec

class svm_relative_veiw(object):
    def __init__(self, asset_pair, start_date=None, end_date=None):
        self.ass1_id = asset_pair[0]
        self.ass2_id = asset_pair[1]
        self.start_date = start_date
        self.end_date = end_date
        self.train_num = 250
        # 资产全局id:财汇secode
        self.assets = {
            '120000001':'2070000060', #沪深300
            '120000002':'2070000187', #中证500
            '120000013':'2070006545', #标普500指数
            '120000014':'2070000626', #黄金指数
            '120000015':'2070000076', #恒生指数
            '120000028':'2070006521', #标普高盛原油商品指数收益率（财汇没有数据）
            '120000029':'2070006789', #南华商品指数
        }

        self.best_params = {
            '120000001:120000002':[1000, 0.01, 0, 1],
            '120000001:120000015':[0.1, 10, 0, 2],
            '120000001:120000029':[1000, 1, 0, 3],
            '120000001:120000013':[1000, 0.1, 0, 4],
            '120000001:120000014':[100, 0.1, 0, 5],
            '120000002:120000015':[0.1, 100, 1, 2],
            '120000002:120000029':[960, 0.1, 1, 3],
            '120000002:120000013':[1, 1, 1, 4],
            '120000002:120000014':[10, 1, 1, 5],
            '120000015:120000029':[95, 9.5, 2, 3],
            '120000015:120000013':[10, 10, 2, 4],
            '120000015:120000014':[1000, 0.01, 2, 5],
            '120000029:120000013':[0.1, 10, 3, 4],
            '120000029:120000014':[10, 0.1, 3, 5],
            '120000013:120000014':[100, 0.001, 4, 5]
        }
        self.init_data()

        self.get_relative_view(2,3)
    def init_data(self):
        result = self.get_view_id()
        if result[0] == 2:
            return result

        result = self.get_index_origin_data()
        if result[0] == 3:
            return result

        result = self.get_trade_dates()
        if result[0] == 4:
            return result

        result = self.cal_indictor()
        if result[0] == 5:
            return result
    # get trade_dates
    def get_trade_dates(self):
        self.trade_dates = load_td.load_trade_dates()
        if self.trade_dates.empty:
            return (4, 'has no data for trade dates')
        return (0, 'get data sucess')
    # 计算技术指标
    def cal_indictor(self):
        cal_tec_obj1 = CalTec(self.ori_data1, self.trade_dates, data_type=2)
        cal_tec_obj2 = CalTec(self.ori_data2, self.trade_dates, data_type=2)
        try:
            self.ori_data1 = cal_tec_obj1.get_indic()
            self.ori_data2 = cal_tec_obj2.get_indic()
        except Exception, e:
            return (5, "cal tec indictor exception:" + e.message)
        self.ori_data1.dropna(inplace=True)
        self.ori_data2.dropna(inplace=True)
        return (0, 'get data sucess')
    # training
    def svm_train(data1, data2, c, g):
        svc = SVC(C = c, gamma = g, probability = True)
        svc.fit(x_train, y_train)
        pre_state = svc.predict(x_test)[0]
        return pre_state
    # get relative view
    def get_relative_view(self, c, g):
        pct_chg_1 = self.ori_data1[:250]['pct_chg']
        pct_chg_2 = self.ori_data2[:250]['pct_chg']

        predict_start_date = max(self.ori_data1.index[250],
            self.ori_data2.index[250])

        feature 
        print predict_start_date
    # 从财汇得到原始数据
    def get_index_origin_data(self):
        secode1 = self.assets[self.ass1_id]
        secode2 = self.assets[self.ass2_id]
        self.ori_data1 = load_index.load_index_daily_data(secode1, \
                        self.start_date, self.end_date)
        self.ori_data2 = load_index.load_index_daily_data(secode2, \
                        self.start_date, self.end_date)
        if self.ori_data1.empty:
            return (3, 'has no data for secode:' + secode1)

        if self.ori_data2.empty:
            return (3, 'has no data for secode:' + secode2)
        for col in self.ori_data1.columns:
            self.ori_data1[col].replace(to_replace=0, method='ffill', inplace=True)
        for col in self.ori_data2.columns:
            self.ori_data2[col].replace(to_replace=0, method='ffill', inplace=True)
        return (0, 'load origin data sucessfully from caihui')
    # 得到两个资产相对view的id
    def get_view_id(self):
        ass_vw_df = ass_view.get_viewid_by_assids(self.ass1_id, self.ass2_id)
        if ass_vw_df.empty:
            return (2, "has no relative view id for assets:" \
                + self.ass1_id + " " + self.ass2_id)
        self.view_id = ass_vw_df['viewid'][0]
        return (0, "get viewid success")
    def feature_select(self):
        return 0
    def train_para(self):
        return 0
    def get_view(self):
        return 0
    def to_db(self):
        return 0

if __name__ == "__main__":
    view_ass = ['120000001', '120000002', '120000013', '120000014', \
                '120000015', '120000029']
    ass_com_two = itertools.combinations(view_ass, 2)
    start_date = '20050101'
    end_date = '20170531'
    for ass_pair in ass_com_two:
        svm_obj = svm_relative_veiw(('120000001', '120000014'), start_date, end_date)
        os._exit(0)
