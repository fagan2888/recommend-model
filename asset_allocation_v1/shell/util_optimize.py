#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ipdb import set_trace
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('seaborn')


class ConstrRegression(object):

    def __init__(self, cons = None):

        if cons is None:
            self.cons = (
                # {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: -x + 1.5},
                {'type': 'ineq', 'fun': lambda x:  x + 1.5},
            )

    def fit(self, x, y):

        fea_num = x.shape[1]
        w0 = [1.0 / fea_num] * fea_num
        self.res = minimize(ConstrRegression.loss, w0, args = [x, y], method = 'SLSQP', constraints=self.cons, options={'disp':False, 'eps':0.01})
        self.w = self.res.x

        tmp_fitted_y = np.multiply(x, self.w)
        self.contrib = []
        for i in range(fea_num):
            fea_contrib = np.corrcoef(y, tmp_fitted_y[:, i])[1,0]
            self.contrib.append(fea_contrib)
        self.contrib = np.array(self.contrib)


    @staticmethod
    def loss(w, pars):

        x = pars[0]
        y = pars[1]

        mse = ((np.dot(x, w) - y)**2).sum()

        return mse

    def score(self, x, y):

        fitted_y = np.dot(x, self.w)
        return np.corrcoef(y, fitted_y)[1,0]
