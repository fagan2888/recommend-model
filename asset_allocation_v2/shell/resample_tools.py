# -*- coding: utf-8 -*-
"""
Created on Tue May 7 16:33:03 2019

@author: Boyang ZHOU
"""

import logging
import scipy as sp
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, gaussian_kde
import statsmodels as sm
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
# import seaborn.apionly as sns
from sklearn.preprocessing import StandardScaler
import random
import time
from datetime import timedelta, datetime
from functools import partial
from multiprocessing import Pool
from ipdb import set_trace


logger = logging.getLogger(__name__)


class GaussianCopula:

    def __init__(self):

        pass

    @staticmethod
    def ecdf(arr):

        yvals = np.arange(1, len(arr) + 1) / len(arr)

        return yvals

    @staticmethod
    def filter_data_by_nan(data, na_tolerance_quantile):

        tolerate_index = []
        for i in range(data.shape[1]):
            if data[data.columns[i]].count() < (1 - na_tolerance_quantile) * data.shape[0]:
                tolerate_index.append(data.columns[i])
        data = data.drop(columns=tolerate_index)

        return data

    @staticmethod
    def generate_simu_data(data, simulation_cdf):

        '''
        Function inverse copula data to original data_arbitrage_distributed simulated data
        '''

        simu_data = np.array([])

        scaler = StandardScaler().fit(data.reshape(1, -1))
        log_ret = scaler.transform(data.reshape(1, -1)).reshape(-1)

        inverse_ecdf = sp.interpolate.interp1d(
            GaussianCopula.ecdf(data),
            data,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        simu_data_uni_distri_inv_ecdf = inverse_ecdf(simulation_cdf)

        return simu_data_uni_distri_inv_ecdf

    @staticmethod
    def gaussian_copula(df_nav, windows_size, num_MC, i):

        if not isinstance(df_nav, pd.DataFrame):
            raise ValueError("Input Cls data matrix is not a DataFrame")

        df_nav = df_nav.iloc[i:windows_size+i]

        arr_log_ret = np.log(df_nav / df_nav.shift(1)).iloc[1:].fillna(0.0).values
        # arr_log_ret = df_nav.pct_change().iloc[1:].values

        '''
        Generate values from a multivariate normal distribution with specified mean vector and covariance matrix and the time is the same in histroy
        '''

        log_ret_corr = np.corrcoef(arr_log_ret.T)
        log_ret_mean = np.mean(arr_log_ret, axis=0)
        cholesky_deco_corr_log_ret = np.linalg.cholesky(log_ret_corr)

        arr_gaussian_copula_simulation = log_ret_mean + np.dot(cholesky_deco_corr_log_ret, np.random.normal(size=[arr_log_ret.shape[1], num_MC])).T

        arr_gaussian_copula_simulation_cdf = sp.stats.norm.cdf(arr_gaussian_copula_simulation)

        v_generate_simu_data = np.vectorize(GaussianCopula.generate_simu_data, signature='(m),(n)->(n)')
        simulated_data_by_gaussian_copula = v_generate_simu_data(arr_log_ret.T, arr_gaussian_copula_simulation_cdf.T).T

        return simulated_data_by_gaussian_copula


if __name__ == '__main__':

    df_nav = pd.read_csv('a.csv', index_col=[0], parse_dates=[0])
    df_nav = df_nav[['H50', 'H300', 'H500']]
    df_log_ret = np.log(df_nav / df_nav.shift(1)).iloc[1:].fillna(0.0).values

    Copula = gaussian_copula(df_nav=df_nav, windows_size=120, num_MC=int(1e+5), i=1)

    # df_log_ret.iloc[:1001, :].plot(kind='kde')
    # Copula.plot(kind='kde')

    print(Copula.mean(axis=0))
    print(np.exp(Copula)[888])
    print(Copula[888] / df_log_ret[:1001].mean(axis=0) - 1)

    set_trace()

