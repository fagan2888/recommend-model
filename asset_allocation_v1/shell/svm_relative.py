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

import itertools

from db import asset_vw_view as ass_view
from db import caihui_tq_qt_index as load_index

class svm_relative_veiw(object):
    def __init__(self, asset_pair, start_date=None, end_date=None):
        self.ass1_id = asset_pair[0]
        self.ass2_id = asset_pair[1]
        self.start_date = start_date
        self.end_date = end_date
        # 资产全局id:财汇secode
        self.assets = {
            '120000001':'2070000060', #沪深300
            '120000002':'2070000187', #中证500
            '120000013':'2070006545', #标普500指数
            '120000014':'2070000626', #黄金指数
            '120000015':'2070000076', #恒生指数
            '120000028':'2070006789', #标普高盛原油商品指数收益率（财汇没有数据）
            '120000029':'2070006521', #南华商品指数
        }
        self.init_data()
    def init_data(self):
        result = self.get_view_id()
        if result[0] == 2:
            return result

        result = self.get_index_origin_data()
        if result[0] == 3:
            return result
        print self.ori_data1

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
        for col in self.ori_data.columns:
            self.ori_data[col].replace(to_replace=0, method='ffill', inplace=True)
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
    for ass_pair in ass_com_two:
        svm_obj = svm_relative_veiw(ass_pair)
        os._exit(0)
