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

MC_Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\MC_result1121.csv", parse_dates=[0], index_col=0)
#MC_Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\MC_result1121_q05.csv", parse_dates=[0], index_col=0)
#MC_Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\MC_result1121_q10.csv", parse_dates=[0], index_col=0)


Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\Multi_Mkt_Indices.csv",index_col=[0])
columns_name = [ '000001.SH', '399001.SZ','HSI.HI','SP500.SPI','NDX.GI']
Index_data.columns = columns_name

Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))
Index_data=Index_data.dropna()

Index_data_corr=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\Multi_Mkt_Indices_corr_matrix.csv",index_col=[0])
#Index_data_corr=pd.read_csv(r"Multi_Mkt_Indices_corr_matrix.csv",index_col=[0])

Index_data=Index_data[Index_data.index.isin(Index_data_corr.index)]
Index_data_Chg = Index_data.pct_change()

##########################################################################
##########################################################################

#Index_data_Chg.plot()
#MC_Index_data.plot()

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

NB_Tri = pd.DataFrame(data=[sum(MC_Index_data.iloc[:,i].apply(lambda x: 1 if x!=0 else x)) for i in range(
    Index_backtesting_Indicator.shape[1])], index=Index_backtesting_Indicator.columns, columns=['NB Trigger'])


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
plt_trig_lag_start = -10
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


'Calibrate the optimized result in different q'
N_days_lags=[]
for i in range(250):  
    N_days_lags.append(Index_Backtesting_indicator(Index_backtesting_Indicator, i).iloc[-1:,:])
N_days_lags=pd.DataFrame(data=np.vstack(N_days_lags),columns=MC_Index_data.columns)

N_triggered_days_PL_summary=pd.DataFrame()
for k in range(N_days_lags.shape[1]):   
    N_triggered_days_summary1=N_days_lags.iloc[:,k]
    N_triggered_days_summary2=[]
    for i in range(1,250):
        for j in range(i+1,250):
            Acc_PL1=N_triggered_days_summary1[j]-N_triggered_days_summary1[i]
            Acc_PL=[Acc_PL1,i,j]
            N_triggered_days_summary2.append(Acc_PL)
    
    N_triggered_days_summary1=pd.DataFrame(data=N_triggered_days_summary2,columns=[N_days_lags.columns[k],'Start_dt','End_dt'])
    N_triggered_days_PL_summary=pd.merge(N_triggered_days_PL_summary,N_triggered_days_summary1,right_index=True,left_index=True,how='outer')

