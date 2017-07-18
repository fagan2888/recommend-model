# -*- coding: utf-8 -*-
"""
Created at Mar. 25, 2017
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""
import sys
sys.path.append('./shell')
#import datetime
import numpy as np
from pomegranate import HiddenMarkovModel, MultivariateGaussianDistribution
#import os

class HmmModel(object):
    def __init__(self, x, n_components = 5, n_iter = 5000):
        self.samples = [[tuple(i) for i in x]]
        self.state_num = n_components
        self.n_iter = n_iter
    def fit(self, x):
        x = [[tuple(i) for i in x]]
        model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, \
            self.state_num, X=x, max_iterations = self.n_iter)
        self.model = model
        s_means = []
        for i in range(self.state_num):
            s_means.append(self.model.states[i].distribution.parameters[0])
        self.means_ = s_means
        self.transmat_ = self.model.dense_transition_matrix() \
            [:self.state_num, :self.state_num]
        return self
    def predict(self, x):
        x = [[tuple(i) for i in x]]
        self.states = np.array(self.model.predict(x[0]))
        return self.states
    # def get_means(self):
    #     s_means = []
    #     for i in range(self.state_num):
    #         s_means.append(self.model.states[i].distribution.parameters[0])
    #     self.means_ = s_means
    # def get_trans(self):
    #     self.transmat_ = self.dense_transition_matrix()[:self.state_num, :self.state_num]

