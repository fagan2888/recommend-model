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

MC_Index_data = pd.read_csv(r"MC_VaR.csv", parse_dates=[0], index_col=0)
Index_data = pd.read_csv(r"Multi_Indices_20181206.csv", parse_dates=[0], index_col=0)

Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))
Index_data=Index_data.dropna()

Index_data_corr=pd.read_csv(r"Multi_Indices_corr_matrix.csv",index_col=[0])
Index_data_corr.index = Index_data_corr.index.map(lambda x: pd.Timestamp(x))


Index_trad_dt=pd.read_csv(r"trade_dates.csv",index_col=[0])

Index_data=Index_data[Index_data.index.isin(Index_data_corr.index)]


Index_data=Index_data[Index_data.index.isin(Index_trad_dt.index)]
Index_data_corr=Index_data_corr[Index_data_corr.index.isin(Index_trad_dt.index)]

Index_data_Chg=Index_data.pct_change()
##########################################################################
##########################################################################


Index_backtesting = Index_data_Chg[
    Index_data_Chg.index.isin(MC_Index_data.index)]
Index_backtesting_Indicator = Index_backtesting <= MC_Index_data
Index_backtesting_Indicator_1D = pd.DataFrame(
    data=Index_backtesting_Indicator.values, columns=Index_backtesting_Indicator.columns, index=Index_backtesting_Indicator.index + timedelta(days=0))


Index_data_Chg_Triggered = Index_data_Chg[
    Index_backtesting_Indicator_1D].fillna(value=0)

# Count the number of trigger in the algorithm

NB_Tri = pd.DataFrame(data=[sum(Index_data_Chg_Triggered.iloc[:,i].apply(lambda x: 1 if x!=0 else x)) for i in range(
    Index_backtesting_Indicator.shape[1])], index=Index_backtesting_Indicator.columns, columns=['NB Trigger'])

'The lag is calender date instead of trading date'
def Index_Backtesting_indicator(Index_backtesting_Indicator, lag_days):
    Index_backtesting_Indicator_1D = pd.DataFrame(
        data=Index_backtesting_Indicator.values, columns=Index_backtesting_Indicator.columns, index=Index_backtesting_Indicator.index + timedelta(days=lag_days))

    Index_data_Chg_Triggered = Index_data_Chg[
        Index_backtesting_Indicator_1D].fillna(value=0)
    Index_data_Chg_Triggered1 = Index_data_Chg_Triggered.cumsum()
    return Index_data_Chg_Triggered1


[Index_backtesting_Indicator.iloc[:,i].tolist().count(True) for i in range(Index_backtesting_Indicator.shape[1])]

#Index_data_Chg[Index_backtesting_Indicator.shift(2).fillna(value=False)].fillna(value=0).cumsum().plot()

##########################################################################
'Subplot the cummultative return of triggered cut risk'


def Cumsum_ret_lag_days(Index_backtesting_Indicator, plt_trig_lag_start, plt_trig_lag_length):
    plt.figure(1)
#    plt_trig_lag_start = -10
#    plt_trig_lag_length = 25
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

# Cumsum_ret_lag_days(Index_backtesting_Indicator,plt_trig_lag_start=-10,plt_trig_lag_length=25)
##########################################################################

def N_triggered_days_PL_summary_func(Index_backtesting_Indicator,start_trigger_limitation,end_trigger_limitation):
    'Calibrate the optimized result in different q'
    N_days_lags=[]
    for i in range(250):  
        N_days_lags.append(Index_Backtesting_indicator(Index_backtesting_Indicator, i).iloc[-1:,:])
    N_days_lags=pd.DataFrame(data=np.vstack(N_days_lags),columns=Index_backtesting_Indicator.columns)
    
    
    'Find the optimized the result of liquidated position and re-long position timeseries statistically'
    N_triggered_days_PL_summary=pd.DataFrame()
    for k in range(N_days_lags.shape[1]):   
        N_triggered_days_summary1=N_days_lags.iloc[:,k]
        N_triggered_days_summary2=[]
        for i in range(1,start_trigger_limitation):
            'The upper bound of j is the limitation of the flat position period in calender date'
            for j in range(i+1,end_trigger_limitation):
    #            Acc_PL1=np.sum(N_triggered_days_summary1[i:j])-N_triggered_days_summary1[i]
                Acc_PL1=np.sum(N_triggered_days_summary1[i+1:j])
                Acc_PL=[Acc_PL1,i,j]
                N_triggered_days_summary2.append(Acc_PL)
        
        N_triggered_days_summary1=pd.DataFrame(data=N_triggered_days_summary2,columns=[N_days_lags.columns[k],'Start_dt','End_dt'])
        N_triggered_days_summary2=N_triggered_days_summary1.sort_values(by=N_days_lags.columns[k],ascending=True).iloc[:25,:]
        N_triggered_days_summary2=N_triggered_days_summary2.reset_index(drop=True)
    #    N_triggered_days_PL_summary=pd.concat(N_triggered_days_PL_summary,N_triggered_days_summary2)
        N_triggered_days_PL_summary=pd.merge(N_triggered_days_PL_summary,N_triggered_days_summary2,right_index=True,left_index=True,how='outer')

    return N_triggered_days_PL_summary


N_triggered_days_PL_summary=N_triggered_days_PL_summary_func(Index_backtesting_Indicator,30,90)




'Time series liquidated position time lags in days and the long position lags in days; More SOLID!'
Index_backtesting_Indicator_Index=[Index_backtesting_Indicator[Index_backtesting_Indicator.iloc[:,i]==True].index.tolist() for i in range(Index_backtesting_Indicator.shape[1])]
Index_backtesting_Indicator_Index=np.hstack(Index_backtesting_Indicator_Index)
Index_backtesting_Indicator_Index=list(Index_backtesting_Indicator_Index)
Index_backtesting_Indicator_Index=list(set(Index_backtesting_Indicator_Index))
Index_backtesting_Indicator_Index.sort()


N_triggered_days_PL_summary_TS=[]
for i in Index_backtesting_Indicator_Index[20:]:
     N_triggered_days_PL_summary_assis=pd.DataFrame(N_triggered_days_PL_summary_func(Index_backtesting_Indicator[Index_backtesting_Indicator.index<=i],30,90).iloc[0,:])
     N_triggered_days_PL_summary_TS.append(N_triggered_days_PL_summary_assis)
     
N_triggered_days_PL_summary_TS_1=np.hstack(N_triggered_days_PL_summary_TS)
N_triggered_days_PL_summary_TS_1=N_triggered_days_PL_summary_TS_1.T
N_triggered_days_PL_summary_TS_1=pd.DataFrame(N_triggered_days_PL_summary_TS_1)
N_triggered_days_PL_summary_TS_1.index=Index_backtesting_Indicator_Index[20:]
N_triggered_days_PL_summary_TS_1.columns=N_triggered_days_PL_summary_assis.index
#N_triggered_days_PL_summary_TS_1.to_csv(r'N_triggered_days_PL_summary_TS_2.csv')


##########################################################################
##########################################################################
##########################################################################    
