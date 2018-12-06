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

MC_Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\MC_VaR1122q001.csv", parse_dates=[0], index_col=0)
#MC_Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\MC_VaR1122q005.csv", parse_dates=[0], index_col=0)
#MC_Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\MC_VaR1122q010.csv", parse_dates=[0], index_col=0)
#MC_Index_data=MC_Index_data.replace(0,1)
#MC_Index_data.plot(title='VaR')
#plt.show()

Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\Multi_Mkt_Indices.csv")
columns_name = [ 'Trd_dt','000001.SH', '399001.SZ','HSI.HI','SP500.SPI','NDX.GI']
Index_data.columns = columns_name
Index_data.index=Index_data.Trd_dt
Index_data=Index_data.drop(columns='Trd_dt')
Index_data=Index_data.dropna()

Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))
Index_data=Index_data.fillna(method='bfill')

Index_data_corr=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\Multi_Mkt_Indices_corr_matrix.csv",index_col=[0])
#Index_data_corr=pd.read_csv(r"Multi_Mkt_Indices_corr_matrix.csv",index_col=[0])

Index_data=Index_data[Index_data.index.isin(Index_data_corr.index)]
Index_data_Chg = Index_data.pct_change()

##########################################################################
##########################################################################

#Index_data_Chg.cumsum().plot()
#MC_Index_data.plot()

Index_backtesting = Index_data_Chg[
    Index_data_Chg.index.isin(MC_Index_data.index)]
Index_backtesting_Indicator = Index_backtesting <= MC_Index_data
Index_backtesting_Indicator_1D = pd.DataFrame(
    data=Index_backtesting_Indicator.values, columns=Index_backtesting_Indicator.columns, index=Index_backtesting_Indicator.index + timedelta(days=0))


Index_data_Chg_Triggered = Index_data_Chg[
    Index_backtesting_Indicator_1D].fillna(value=0)
#Index_data_Chg_Triggered1 = Index_data_Chg_Triggered.cumsum()
#Index_data_Chg_Triggered1.plot()

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
#    Index_data_Chg_Triggered1.plot()
    return Index_data_Chg_Triggered1


[Index_backtesting_Indicator.iloc[:,i].tolist().count(True) for i in range(Index_backtesting_Indicator.shape[1])]
#Index_Backtesting_indicator(Index_backtesting_Indicator, 3).plot()
#
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
N_triggered_days_PL_summary_TS_1.to_csv(r'C:\Users\yshlm\Desktop\licaimofang\data\N_triggered_days_PL_summary_TS_2.csv')


##########################################################################
##########################################################################
##########################################################################    
#def Index_backtesting_Indicator_Trad_dt(Index_backtesting_Indicator,start_lag_days,end_lag_days):
#    Index_backtesting_Indicator_Trad_dt=pd.DataFrame()
#    for k in range(Index_backtesting_Indicator.shape[1]):
#        A=Index_backtesting_Indicator.iloc[:,k]
#        A=A[A.values==True]
#        A=A.index
#        'If the liquidated indicators are even smaller than start lag days, it means the intensive signals remarks in a short period, I decide to liquidated the position immediately after 1 trading day (since the first day after the drawdown always has a decent rebound)'
#        'If the trigger pushed forcely, the length of liquidation will be the difference with the end days lag and the start days lag data timestamp point'
#        C=[ A[j] + timedelta(days=start_lag_days+i) if A[j+1]-A[j] > timedelta(days=start_lag_days) else A[j] + timedelta(days=1+i) for i in range(end_lag_days-start_lag_days) for j in range(len(A)-1)]
#        C.extend([A[-1]+timedelta(days=start_lag_days+i) for i in range(end_lag_days-start_lag_days)])
#        C=list(set(C))
#        C.sort()
#        C=pd.DataFrame(data=C)
#        C=C.reset_index(drop=True)
#        
#        Index_backtesting_Indicator_Trad_dt=pd.merge(Index_backtesting_Indicator_Trad_dt,C,right_index=True,left_index=True,how='outer')
#    Index_backtesting_Indicator_Trad_dt.columns=Index_backtesting_Indicator.columns
#
#    return Index_backtesting_Indicator_Trad_dt
# 
#
#
#
#def Index_backtesting_Result(N_triggered_days_PL_summary,Index_backtesting_Indicator,Index_backtesting,id_index):
##    start_lag_days=Counter(N_triggered_days_PL_summary.iloc[:,id_index*3+1]).most_common()[0][0]
##    end_lag_days=Counter(N_triggered_days_PL_summary.iloc[:,id_index*3+2]).most_common()[0][0]
#    'The very overfitting point'
#    start_lag_days=int(N_triggered_days_PL_summary.iloc[:,id_index*3+1][0])
#    end_lag_days=int(N_triggered_days_PL_summary.iloc[:,id_index*3+2][0])
#
#    
#    if start_lag_days>=end_lag_days:
#        ValueError("Stupid input")
#    
#           
#    Index_backtesting_Indicator_Trad_dt1=Index_backtesting_Indicator_Trad_dt(Index_backtesting_Indicator,start_lag_days,end_lag_days)
#    
##    'Check if there is any zero return of the index'
##    print([Index_backtesting.iloc[:,i].tolist().count(0) for i in range(Index_backtesting.shape[1])])
#    
#    Index_backtesting_Trad_dt_Summary=pd.DataFrame()
#    for i in range(Index_backtesting.shape[1]):
#        Index_backtesting_Trad_dt=Index_backtesting.iloc[:,i]
#        Index_backtesting_Trad_dt=Index_backtesting_Trad_dt[~Index_backtesting_Trad_dt.index.isin(Index_backtesting_Indicator_Trad_dt1.iloc[:,i].values)]
#        Index_backtesting_Trad_dt=pd.DataFrame(Index_backtesting_Trad_dt)
#        Index_backtesting_Trad_dt_Summary=pd.merge(Index_backtesting_Trad_dt_Summary,Index_backtesting_Trad_dt,right_index=True,left_index=True,how='outer')
#    #Index_backtesting_Trad_dt_Summary.cumsum().plot()
#    #Index_backtesting.cumsum().plot()
#    
#    Index_backtesting_Result=pd.merge(pd.DataFrame(Index_backtesting.iloc[:,id_index]),pd.DataFrame(Index_backtesting_Trad_dt_Summary.iloc[:,id_index]),right_index=True,left_index=True,how='outer')
#    Index_backtesting_Result=Index_backtesting_Result.fillna(0)
#    Index_backtesting_Result.columns=[str(Index_backtesting.columns[id_index]),str(Index_backtesting.columns[id_index]+ ' Algo')]
#    Index_backtesting_Result.cumsum().plot(title=str('Compare Algorithm and Original data of ' + Index_backtesting.columns[id_index]))
#    plt.show()
#    
#    return Index_backtesting_Result
#
#
#
#for i in range(Index_backtesting.shape[1]):
#    
#    Index_backtesting_Result(N_triggered_days_PL_summary,Index_backtesting_Indicator,Index_backtesting,i)
##Index_backtesting.iloc[:,0].cumsum().plot()
