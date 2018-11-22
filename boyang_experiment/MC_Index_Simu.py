# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:39:02 2018

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

from statsmodels.stats.diagnostic import unitroot_adf
from statsmodels.multivariate import pca
from sklearn import decomposition
from statsmodels.stats.sandwich_covariance import cov_hac

# MC_Index_data = pd.read_csv(
#     r"C:\Users\yshlm\Desktop\licaimofang\data\MC_result1.csv", parse_dates=[0], index_col=0)

MC_Index_data = pd.read_csv(
    r"C:\Users\yshlm\Desktop\licaimofang\data\MC_result_q_05.csv", parse_dates=[0], index_col=0)


Index_data = pd.read_csv(
    r"C:\Users\yshlm\Desktop\licaimofang\data\ra_index_nav_CN_US_HK.csv")
columns_name = ['Index_Code', 'Trd_dt', 'Index_Cls']
Index_data.columns = columns_name
Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))

'Expand the ret data to [0,1]'


def Clean_data(input_data):

    Index_Code_unique = Index_data[
        ~Index_data.Index_Code.duplicated()].Index_Code.tolist()
    Data = pd.DataFrame()
    for i in range(len(Index_Code_unique)):
        Index_1 = Index_data[Index_data.Index_Code == Index_Code_unique[i]]
        Index_1 = pd.DataFrame(data=Index_1.Index_Cls.values, columns=[
                               Index_Code_unique[i]], index=Index_1.Trd_dt)
        Data = pd.merge(Data, Index_1, right_index=True,
                        left_index=True, how='outer')

    return Data

Index_data = Clean_data(Index_data)
Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))
Index_data = Index_data.fillna(method='bfill')
del Index_data['399006']

Index_data_Chg = Index_data.pct_change()
Index_data_Chg_Acc = Index_data_Chg.cumsum()
Index_data_Chg_Acc = Index_data_Chg_Acc[
    Index_data_Chg_Acc.index > '2000-01-01']
##########################################################################
##########################################################################


# Index_data_Chg.plot()
# MC_Index_data.plot()

Index_backtesting = Index_data_Chg[
    Index_data_Chg.index.isin(MC_Index_data.index)]
Index_backtesting_Indicator = Index_backtesting <= MC_Index_data
Index_backtesting_Indicator_1D = pd.DataFrame(
    data=Index_backtesting_Indicator.values, columns=Index_backtesting_Indicator.columns, index=Index_backtesting_Indicator.index + timedelta(days=0))

Index_data_Chg_Triggered = Index_data_Chg[
    Index_backtesting_Indicator_1D].fillna(value=0)
Index_data_Chg_Triggered1 = Index_data_Chg_Triggered.cumsum()
Index_data_Chg_Triggered1.plot()

# Count the number of trigger in the algorithm

NB_Tri = pd.DataFrame(data=[Counter(Index_backtesting_Indicator.iloc[:, i]) for i in range(
    Index_backtesting_Indicator.shape[1])], index=Index_backtesting_Indicator.columns)


def Index_Backtesting_indicator(Index_backtesting_Indicator, lag_days):
    Index_backtesting_Indicator_1D = pd.DataFrame(
        data=Index_backtesting_Indicator.values, columns=Index_backtesting_Indicator.columns, index=Index_backtesting_Indicator.index + timedelta(days=lag_days))

    Index_data_Chg_Triggered = Index_data_Chg[
        Index_backtesting_Indicator_1D].fillna(value=0)
    Index_data_Chg_Triggered1 = Index_data_Chg_Triggered.cumsum()
#    Index_data_Chg_Triggered1.plot()
    return Index_data_Chg_Triggered1

Index_Backtesting_indicator(Index_backtesting_Indicator, 0).plot()


##########################################################################
#Subplot the cummultative return of triggered cut risk
plt.figure(1)
plt_trig_lag_start = -3
plt_trig_lag_length = 25
for i in range(plt_trig_lag_length):

    #    nb_subplt=4401+i
    plt.subplot(np.sqrt(plt_trig_lag_length),
                np.sqrt(plt_trig_lag_length), i + 1)
    x = Index_Backtesting_indicator(
        Index_backtesting_Indicator, i + plt_trig_lag_start).index
    y = Index_Backtesting_indicator(
        Index_backtesting_Indicator, i + plt_trig_lag_start)

    plt.plot(x, y)
    plt.legend = True
    plt.mark_right = True

    plt.title('%d days after the triggered day' % (i + plt_trig_lag_start))


plt.show()
##########################################################################
'''
Extreme value theory
'''
#Step I
#The hypothesis test for independency: maybe Chi_square test?
#Target: for canceling the autocorrelation effect in the covariance matrix
#Methodology: Newey-West 
#Demean for each correlation matrix

'Reference: Newey, W. K. and K. D. West (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, Vol. 55(3), 703 – 708.'
Ret =  Index_data.pct_change()

Ret = Ret.iloc[1:, :]
Ret = Ret.fillna(value=0)
#demean
Ret=Ret.sub(Ret.mean())

Ini_Windows_size=1000
Cov_nlags=10

#Time series Covariance matrix
Cov_ret_TS_gamma_0=pd.DataFrame()
for i in range(Ret.shape[0]-Ini_Windows_size-Cov_nlags):    
    Cov_ret=Ret.iloc[:Ini_Windows_size+i,:].cov()
    Cov_ret_TS_gamma_0=Cov_ret.add(Cov_ret_TS_gamma_0,fill_value=0)
#Gamma 0 Matrix
Cov_ret_TS_gamma_0=Cov_ret_TS_gamma_0.divide(Ret.shape[0]-Ini_Windows_size-Cov_nlags)

Cov_ret_TS_gamma_i=pd.DataFrame()
for i in range(Cov_nlags):
    Cov_ret=Ret.iloc[:-Cov_nlags+i,:].cov()
    omega_weight=1-(i/(1+Cov_nlags))
    Cov_ret_weighted=omega_weight*Cov_ret
    Cov_ret_TS_gamma_i=Cov_ret_weighted.add(Cov_ret_TS_gamma_i,fill_value=0)

#Cov_ret_TS_gamma_i.T
Cov_ret_TS=Cov_ret_TS_gamma_0+Cov_ret_TS_gamma_i
