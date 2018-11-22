# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:52:52 2018

@author: Boyang ZHOU
"""

import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, gaussian_kde
from collections import Counter
import pylab
import statsmodels as sm
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
import seaborn.apionly as sns
import random
import time
from datetime import timedelta, datetime

from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from statsmodels.stats.diagnostic import unitroot_adf
from statsmodels.multivariate import pca
from sklearn import decomposition

from functools import partial
from multiprocess import Pool


##########################################################################


def z_score_normalization(A):
    AB = (A - np.mean(A)) / np.std(A)
    return AB

    
def inverse_z_score_normalization(input_A, original_data):
    result = input_A * np.std(original_data) + np.mean(original_data)
    return result

##########################################################################


def ecdf(input_array):
    yvals = np.arange(1, len(sorted(input_array)) + 1) / \
        float(len(sorted(input_array)))
    return yvals

##########################################################################


def filter_data_by_nan(data, na_tolerance_quantile):
    tolerate_index = []
    for i in range(data.shape[1]):
        if data[data.columns[i]].count() < (1 - na_tolerance_quantile) * data.shape[0]:
            tolerate_index.append(data.columns[i])
    data = data.drop(columns=tolerate_index)
    return data

##########################################################################


def normalization_maxmin(A):

    A = 0 + (A - np.min(A)) / (np.max(A) - np.min(A)) * (1 - 0)
    AAA = A / np.sum(A)
    return AAA

##########################################################################


def Two_Return_Cum(input_dataframe):
    input_dataframe = pd.DataFrame(input_dataframe)

    log_return = np.log(input_dataframe / input_dataframe.shift(1)).dropna()
    log_return_sum = log_return.cumsum()

    arithmetic_return = input_dataframe.pct_change().dropna()
    arithmetic_return_cumproduct = (arithmetic_return + 1).cumprod() - 1

    diff = log_return_sum - arithmetic_return_cumproduct
    summary = pd.concat(
        [log_return_sum, arithmetic_return_cumproduct, diff], axis=1, join='inner')
    summary.columns = ['log_ret_cumsum',
                       'archimetric_ret_cumprod', 'difference']
    return summary


##########################################################################

"Function inverse copula data to original data_arbitrage_distributed simulated data"


def Tranfer_simu2orig_data(log_ret, input_data):
    ''
    if log_ret.shape[1] != input_data.shape[1]:
        raise ValueError(
            'Something Wrong about the shape of input Simulated data')

    Simu_data = []
    for i in range(log_ret.shape[1]):
        data_test = log_ret.iloc[:, i].tolist()

        'Extend log return data to (0,1]'
        'Z Score'
        data_test_norm = z_score_normalization(normalization_maxmin(data_test))
        inverse_ECDF_data_test = sp.interpolate.interp1d(ecdf(
            data_test_norm), data_test_norm, kind='linear', bounds_error=False, fill_value='extrapolate')

        input_data1 = input_data.iloc[:, i].tolist()

        Simu_data_uni_distri_inv_ecdf = inverse_ECDF_data_test(input_data1)
        Simu_data.append(inverse_z_score_normalization(
            Simu_data_uni_distri_inv_ecdf, data_test).T)

    Simu_data_All = pd.DataFrame(data=np.stack(
        Simu_data).T, columns=log_ret.columns)

    return Simu_data_All

##########################################################################


def Gaussian_Copula(data, data_corr, Windows_size, suspension_tolerance_filtered_level, Nb_MC, Confidence_level,i):
    
    data=data.iloc[:Windows_size + i, :]
    data_corr=data_corr.iloc[:Windows_size + i, :]
    
    data_tolerance_filtered = filter_data_by_nan(
        data, suspension_tolerance_filtered_level)
    data_tolerance_filtered = data_tolerance_filtered.fillna(method='bfill')
    'Differernce between the log return and the arithmetic return'
    log_ret = np.log(data_tolerance_filtered /
                     data_tolerance_filtered.shift(1))

    log_ret = log_ret.iloc[1:, :]
    log_ret = log_ret.fillna(value=0)

    
    'Generate values from a multivariate normal distribution with specified mean vector and covariance matrix and the time is the same in histroy'
    data_corr=data_corr[data_corr.index==data_corr.index[-1]]
    data_corr=data_corr.values.reshape(5,5)
    cholesky_deco_corr_log_ret = np.linalg.cholesky(data_corr)
#    cholesky_deco_corr_log_ret = np.linalg.cholesky(log_ret.corr())

    Gaussian_Copula_Simulation = [(np.mean(log_ret) + np.dot(cholesky_deco_corr_log_ret, [
                                   np.random.normal() for i in range(log_ret.shape[1])])).values.T for i in range(int(Nb_MC))]
   
    Gaussian_Copula_Simulation = pd.DataFrame(data=np.stack(
        Gaussian_Copula_Simulation), columns=log_ret.columns)
    Gaussian_Copula_Simulation_cdf = pd.DataFrame(data=sp.stats.norm.cdf(
        Gaussian_Copula_Simulation), columns=log_ret.columns)

    Simulated_data_by_Gaussian_Copula = Tranfer_simu2orig_data(
        log_ret, Gaussian_Copula_Simulation_cdf)

    return Simulated_data_by_Gaussian_Copula


##########################################################################


def Algo_summary(input_dataframe):
    'The input dataframe should be the daily return of the portfolio, where need to be compared'
    summary = []
    Risk_free = 0.035
    for k in range(input_dataframe.shape[1]):
        input_dataset = input_dataframe.iloc[:, k]

        Std = np.std(input_dataset)
        Ann_Std = Std * np.sqrt(252)

        input_dataset = np.cumsum(input_dataset)

        Ret = input_dataset[-1]
        Ann_Ret = (1 + Ret)**(252 / len(input_dataset)) - 1

        SharpR = (Ann_Ret - Risk_free) / Ann_Std

        i = np.argmax(np.maximum.accumulate(input_dataset) -
                      input_dataset)  # end of the period
        j = np.argmax(input_dataset[:i])  # start of period

        summary.append([Ret, Ann_Ret, Std, Ann_Std, SharpR, j,
                        i, input_dataset[j] - input_dataset[i]])

    Algo_summary = pd.DataFrame(summary, index=input_dataframe.columns, columns=[
                                'Ret', 'Ann Ret', 'Daily_Ret_Std', 'Ann Std', 'Sharp Ratio', 'MDD_Start_Date', 'MDD_End_Date', 'MDD'])
    return(Algo_summary)


##########################################################################
##########################################################################
##########################################################################


if __name__ == '__main__':
    
#    Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\Multi_Mkt_Indices.csv",index_col=[0])
    Index_data = pd.read_csv(r"Multi_Mkt_Indices.csv",index_col=[0])
    columns_name = [ '000001.SH', '399001.SZ','HSI.HI','SP500.SPI','NDX.GI']
    Index_data.columns = columns_name
    
    Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))
    Index_data=Index_data.dropna()
    
#    Index_data_corr=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\Multi_Mkt_Indices_corr_matrix.csv",index_col=[0])
    Index_data_corr=pd.read_csv(r"Multi_Mkt_Indices_corr_matrix.csv",index_col=[0])
    
    Index_data=Index_data[Index_data.index.isin(Index_data_corr.index)]
    Index_data_Chg = Index_data.pct_change()
    
#    Copula = Gaussian_Copula(data=Index_data,data_corr=Index_data_corr, suspension_tolerance_filtered_level=0.1, Nb_MC=1e+5, Confidence_level=0.99)
    
    q = 0.01
#    Copula1 = Copula.quantile(q)
    
    ##########################################################################
    'What is the length of backtesting'
    
    ##########################################################################
    'Rolling or '
    
    
    Windows_size = 1000
    Windows_length = Index_data.shape[0] - Windows_size - 1 - 4000
    
#    Timeseries_MC = [Gaussian_Copula(data=Index_data.iloc[:Windows_size + i, :], data_corr=Index_data_corr.iloc[:Windows_size + i, :], suspension_tolerance_filtered_level=0.1, Nb_MC=1e+5, Confidence_level=0.99) for i in range(Windows_length)]
    
    
    
    
    func = partial(Gaussian_Copula, Index_data, Index_data_corr, Windows_size, 0.1, 1e5, 0.99,)
    pool = Pool(32)
    Timeseries_MC = pool.map(func, range(Windows_length))
    pool.close()
    pool.join()
    
    
   
    Timeseries_VaR_Summary = np.stack([Timeseries_MC[i].quantile(q).T for i in range(len(Timeseries_MC))])
    
    #Timeseries_VaR=[Gaussian_Copula(Index_data.iloc[:Windows_size+i,:],0.1,1e+4,0.99).quantile(q) for i in range(Windows_length)]
    
    Timeseries_VaR_Summary = pd.DataFrame(data=Timeseries_VaR_Summary, columns=Index_data.columns, index=Index_data.index[
                                          Windows_size:Windows_size + Windows_length])
    
    Index_backtesting = Index_data_Chg[
        Index_data_Chg.index.isin(Timeseries_VaR_Summary.index)]
    Index_backtesting_Indicator = Index_backtesting <= Timeseries_VaR_Summary
    Index_backtesting_Indicator_1D = pd.DataFrame(
        data=Index_backtesting_Indicator.values, columns=Index_backtesting_Indicator.columns, index=Index_backtesting_Indicator.index + timedelta(days=1))
    
    Index_data_Chg_Triggered = Index_data_Chg[
        Index_backtesting_Indicator_1D].fillna(value=0)
    Index_data_Chg_Triggered.to_csv(r'MC_result1121.csv')
#    Index_data_Chg_Triggered1 = Index_data_Chg_Triggered.cumsum()
    
