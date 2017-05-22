# -*- encoding:utf-8 -*-
from __future__ import division
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import pandas as pd
import numpy as np

import itertools

class svm_relative_veiw(object):
    def __init__(self):
        return 0
    def init_data(self):
        return 0
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
    for item in ass_com_two:
        print str(item)
