# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:39:02 2018

@author: Boyang ZHOU
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, gaussian_kde
from collections import Counter
import pylab
import statsmodels as sm

import time
from datetime import timedelta, datetime


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
Index_data_Chg_Triggered1 = Index_data_Chg_Triggered.cumsum()
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


#A=[]
#B=[]
#for i in range(Index_backtesting_Indicator.shape[0]):
#    if Index_backtesting_Indicator.iloc[i,:].sum()!=0:
#        A.append(Index_backtesting_Indicator.iloc[i,:])
#        B.append(Index_backtesting_Indicator.index[i])
#        
#A=pd.DataFrame(data=np.vstack(A),index=B,columns=Index_backtesting_Indicator.columns)    
#A.to_csv(r'C:\Users\yshlm\Desktop\licaimofang\data\Index_backtesting_Indicator__.csv')


#Index_Backtesting_indicator(Index_backtesting_Indicator, 3).plot()
#
#Index_data_Chg[Index_backtesting_Indicator.shift(2).fillna(value=False)].fillna(value=0).cumsum().plot()

##########################################################################
'Subplot the cummultative return of triggered cut risk'
def Cumsum_ret_lag_days(Index_backtesting_Indicator,plt_trig_lag_start,plt_trig_lag_length):
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

#Cumsum_ret_lag_days(Index_backtesting_Indicator,plt_trig_lag_start=-10,plt_trig_lag_length=25)
##########################################################################

'Calibrate the optimized result in different q'
N_days_lags=[]
for i in range(250):  
    N_days_lags.append(Index_Backtesting_indicator(Index_backtesting_Indicator, i).iloc[-1:,:])
N_days_lags=pd.DataFrame(data=np.vstack(N_days_lags),columns=MC_Index_data.columns)


'Find the optimized the result of liquidated position and re-long position timeseries statistically'
N_triggered_days_PL_summary=pd.DataFrame()
for k in range(N_days_lags.shape[1]):   
    N_triggered_days_summary1=N_days_lags.iloc[:,k]
    N_triggered_days_summary2=[]
    'Start to flat positions at most 30 calender days'
    for i in range(1,30):
        'Start to re-long positions at most 90 trading days'
        for j in range(i+1,90):
#            Acc_PL1=np.sum(N_triggered_days_summary1[i:j])-N_triggered_days_summary1[i]
            Acc_PL1=np.sum(N_triggered_days_summary1[i+1:j])
            Acc_PL=[Acc_PL1,i,j]
            N_triggered_days_summary2.append(Acc_PL)
    
    N_triggered_days_summary1=pd.DataFrame(data=N_triggered_days_summary2,columns=[N_days_lags.columns[k],'Start_dt','End_dt'])
    N_triggered_days_summary2=N_triggered_days_summary1.sort_values(by=N_days_lags.columns[k],ascending=True).iloc[:25,:]
    N_triggered_days_summary2=N_triggered_days_summary2.reset_index(drop=True)
#    N_triggered_days_PL_summary=pd.concat(N_triggered_days_PL_summary,N_triggered_days_summary2)
    N_triggered_days_PL_summary=pd.merge(N_triggered_days_PL_summary,N_triggered_days_summary2,right_index=True,left_index=True,how='outer')



'The timeseries parameters of start lags and the end lags'
N_triggered_days_PL_summary_TS_1=pd.read_csv(r'C:\Users\yshlm\Desktop\licaimofang\data\N_triggered_days_PL_summary_TS_2.csv',index_col=[0])
N_triggered_days_PL_summary_TS_1.index=N_triggered_days_PL_summary_TS_1.index.map(lambda x:pd.Timestamp(x))
N_triggered_days_PL_summary_TS_1=N_triggered_days_PL_summary_TS_1[N_triggered_days_PL_summary_TS_1.index>=pd.Timestamp('2002-01-28')]

'''
STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT 
STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT 
STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT STYLE INPUT 
'''
#N_triggered_days_PL_Logical_Style_Indicator=['Tight','Tight','Tight','Loose','Loose']
N_triggered_days_PL_Logical_Style_Indicator=['Loose','Loose','Loose','Loose','Loose']

N_triggered_days_PL_summary_TS_Style=pd.DataFrame()
for id_index in range(Index_data_Chg.shape[1]):
 
    N_triggered_days_PL_summary_TS_start=[N_triggered_days_PL_summary_TS_1.iloc[0,id_index*3+1]]
    N_triggered_days_PL_summary_TS_end=[N_triggered_days_PL_summary_TS_1.iloc[0,id_index*3+2]]
    
    if N_triggered_days_PL_Logical_Style_Indicator[id_index]=='Tight':

        for i in range(1,N_triggered_days_PL_summary_TS_1.shape[0]):
            
            if N_triggered_days_PL_summary_TS_1.iloc[i,id_index*3+2]-N_triggered_days_PL_summary_TS_1.iloc[i,id_index*3+1] >=30:
                N_triggered_days_PL_summary_TS_start.append(N_triggered_days_PL_summary_TS_1.iloc[i,id_index*3+1])
                N_triggered_days_PL_summary_TS_end.append(N_triggered_days_PL_summary_TS_1.iloc[i,id_index*3+2])
            else:
                N_triggered_days_PL_summary_TS_start.append(np.nan)
                N_triggered_days_PL_summary_TS_end.append(np.nan)
                    
    else:
        
        for i in range(1,N_triggered_days_PL_summary_TS_1.shape[0]):

            if N_triggered_days_PL_summary_TS_1.iloc[i,id_index*3+2]-N_triggered_days_PL_summary_TS_1.iloc[i,id_index*3+1] <30:
                N_triggered_days_PL_summary_TS_start.append(N_triggered_days_PL_summary_TS_1.iloc[i,id_index*3+1])
                N_triggered_days_PL_summary_TS_end.append(N_triggered_days_PL_summary_TS_1.iloc[i,id_index*3+2])
            else:
                N_triggered_days_PL_summary_TS_start.append(np.nan)
                N_triggered_days_PL_summary_TS_end.append(np.nan)
             
    N_triggered_days_PL_summary_TS_start=pd.DataFrame(N_triggered_days_PL_summary_TS_start,index=N_triggered_days_PL_summary_TS_1.index,columns=[Index_data_Chg.columns[id_index]+str(' start')])
    N_triggered_days_PL_summary_TS_end=pd.DataFrame(N_triggered_days_PL_summary_TS_end,index=N_triggered_days_PL_summary_TS_1.index,columns=[Index_data_Chg.columns[id_index]+str(' end')])
    
    N_triggered_days_PL_summary_TS_Style=pd.merge(N_triggered_days_PL_summary_TS_Style,N_triggered_days_PL_summary_TS_start,right_index=True,left_index=True,how='right')
    N_triggered_days_PL_summary_TS_Style=pd.merge(N_triggered_days_PL_summary_TS_Style,N_triggered_days_PL_summary_TS_end,right_index=True,left_index=True,how='outer')


N_triggered_days_PL_summary_TS_Style1=Index_data_Chg[Index_data_Chg.index>=N_triggered_days_PL_summary_TS_1.index[0]]
N_triggered_days_PL_summary_TS_Style1=pd.merge(N_triggered_days_PL_summary_TS_Style1,N_triggered_days_PL_summary_TS_Style,right_index=True,left_index=True,how='outer')
N_triggered_days_PL_summary_TS_Style1=N_triggered_days_PL_summary_TS_Style1.drop(columns=N_triggered_days_PL_summary_TS_Style1.columns[:Index_data_Chg.shape[1]])
N_triggered_days_Style_TS=N_triggered_days_PL_summary_TS_Style1.fillna(method='ffill')

#N_triggered_days_Style_TS.to_csv(r'C:\Users\yshlm\Desktop\licaimofang\data\N_triggered_days_Style_TS.csv')

'Find the time point change the parameters for reducing the computational burden'
N_triggered_days_Style_TS_TP=pd.DataFrame()
for id_index in range(Index_data_Chg.shape[1]):
    N_triggered_days_Style_TS_TP_Assis=[N_triggered_days_Style_TS.index[0]]
    for i in range(1,N_triggered_days_Style_TS.shape[0]-1):
        if N_triggered_days_Style_TS.iloc[i,id_index*2+0]!=N_triggered_days_Style_TS.iloc[i+1,id_index*2+0] or N_triggered_days_Style_TS.iloc[i,id_index*2+1]!=N_triggered_days_Style_TS.iloc[i+1,id_index*2+1]:
            N_triggered_days_Style_TS_TP_Assis.append(N_triggered_days_Style_TS.index[i+1])
    N_triggered_days_Style_TS_TP_Assis.append(Index_data_Chg.index[-1])
    N_triggered_days_Style_TS_TP_Assis=pd.DataFrame(N_triggered_days_Style_TS_TP_Assis,columns=[Index_data_Chg.columns[id_index]])
#    N_triggered_days_Style_TS_TP_Assis=N_triggered_days_Style_TS_TP_Assis.reset_index()
    
    N_triggered_days_Style_TS_TP=pd.merge(N_triggered_days_Style_TS_TP,N_triggered_days_Style_TS_TP_Assis,right_index=True,left_index=True,how='outer')

##########################################################################



def Index_backtesting_Indicator_Trad_dt_func(Index_backtesting_Indicator,start_lag_days,end_lag_days):
    Index_backtesting_Indicator_Trad_dt_1=pd.DataFrame()
    Index_backtesting_Indicator_Trad_dt_2=pd.DataFrame()
    
    for k in range(Index_backtesting_Indicator.shape[1]):
        A=Index_backtesting_Indicator.iloc[:,k]
        A=A[A.values==True]
        A=A.index
        'If the liquidated indicators are even smaller than start lag days, it means the intensive signals remarks in a short period, I decide to liquidated the position immediately after 1 trading day (since the first day after the drawdown always has a decent rebound)'
        'If the trigger pushed forcely, the length of liquidation will be the difference with the end days lag and the start days lag data timestamp point'
    
        C=[ A[j] + timedelta(days=start_lag_days+i) if A[j+1]-A[j] > timedelta(days=start_lag_days) else A[j] + timedelta(days=1+i) for i in range(end_lag_days-start_lag_days) for j in range(len(A)-1)]
        C.extend([A[-1]+timedelta(days=start_lag_days+i) for i in range(end_lag_days-start_lag_days)])
        
        D=pd.DataFrame(Counter(C).most_common(),columns=[Index_backtesting_Indicator.columns[k],str(Index_backtesting_Indicator.columns[k]+' Triggered Times')])
        D=D.reset_index(drop=True)
        
        Index_backtesting_Indicator_Trad_dt_2=pd.merge(Index_backtesting_Indicator_Trad_dt_2,D,right_index=True,left_index=True,how='outer')
        
        C=list(set(C))   
        C.sort()
        C=pd.DataFrame(data=C)
        C=C.reset_index(drop=True)
        
        Index_backtesting_Indicator_Trad_dt_1=pd.merge(Index_backtesting_Indicator_Trad_dt_1,C,right_index=True,left_index=True,how='outer')
    Index_backtesting_Indicator_Trad_dt_1.columns=Index_backtesting_Indicator.columns
    'Index_backtesting_Indicator_Trad_dt_1 is the Indicator; Index_backtesting_Indicator_Trad_dt_2 is the indicators sorted by times'
    return Index_backtesting_Indicator_Trad_dt_1,Index_backtesting_Indicator_Trad_dt_2


##########test################################################################
Index_backtesting_Indicator_Timeseries_Signal=pd.DataFrame()

for id_index in range(Index_data_Chg.shape[1]):
    #start_lag_days=int(N_triggered_days_PL_summary.iloc[:,id_index*3+1][0])
    #end_lag_days=int(N_triggered_days_PL_summary.iloc[:,id_index*3+2][0])
    
    #start_lag_days=24
    #end_lag_days=36
    
    N_triggered_days_Style_TS_TP_Assis=N_triggered_days_Style_TS_TP.iloc[:,id_index].dropna()
    
    Index_backtesting_Indicator_Trad_dt=pd.DataFrame()
    for t in range(len(N_triggered_days_Style_TS_TP_Assis)-1):
        
        TP=N_triggered_days_Style_TS_TP_Assis[t]
        start_lag_days=int(N_triggered_days_Style_TS[N_triggered_days_Style_TS.index==TP].iloc[:,id_index*2+0].values)
        end_lag_days=int(N_triggered_days_Style_TS[N_triggered_days_Style_TS.index==TP].iloc[:,id_index*2+1].values)
    
    #    Index_backtesting_Indicator_TP=Index_backtesting_Indicator[(Index_backtesting_Indicator.index>=N_triggered_days_Style_TS_TP_Assis[t]) & (Index_backtesting_Indicator.index<N_triggered_days_Style_TS_TP_Assis[t+1])]
    
        if start_lag_days>=end_lag_days:
            ValueError("Stupid input")
    
    
        Index_backtesting_Indicator_Trad_dt_1,Index_backtesting_Indicator_Trad_dt_2=Index_backtesting_Indicator_Trad_dt_func(Index_backtesting_Indicator,start_lag_days,end_lag_days)
    
    
        AAtest=Index_backtesting_Indicator_Trad_dt_2.iloc[:,(2*id_index+0):(2*id_index+2)]
        AAtest=AAtest.set_index(AAtest.columns[0],drop=True)
        'Trading days'
        Index_backtesting_Indicator_Trad_dt_2_assis=AAtest[AAtest.index.isin(Index_backtesting.index)]
        'Different risk affine kernel function: (lambda x: 1/(2**x)) (lambda x: np.pi/2/(np.exp(-x)+np.exp(x))) (lambda x: (np.exp(-x)-1)/(-x))'
        'Kernel funciton #1'
        #Index_backtesting_Indicator_Trad_dt_2_assis=Index_backtesting_Indicator_Trad_dt_2_assis.apply(lambda x: 1/(2**x))
        'Kernel funciton #2'
        #Index_backtesting_Indicator_Trad_dt_2_assis=Index_backtesting_Indicator_Trad_dt_2_assis.apply(lambda x: np.pi/2/(np.exp(-x)+np.exp(x)))
        'Kernel funciton #3'
        Index_backtesting_Indicator_Trad_dt_2_assis=Index_backtesting_Indicator_Trad_dt_2_assis.apply(lambda x: (np.exp(-x)-1)/(-x))
        'Kernel funciton #4 full liquidated'
        # Index_backtesting_Indicator_Trad_dt_2_assis=Index_backtesting_Indicator_Trad_dt_2_assis.apply(lambda x: x*0)
    
    
        Index_backtesting_Indicator_A=Index_backtesting_Indicator.iloc[:,id_index]
        Index_backtesting_Indicator_Trad_dt_=Index_backtesting_Indicator_A[~Index_backtesting_Indicator_A.index.isin(Index_backtesting_Indicator_Trad_dt_2_assis.index)].apply(lambda x:1)
        Index_backtesting_Indicator_Trad_dt_=pd.DataFrame(Index_backtesting_Indicator_Trad_dt_)
        
        
        Index_backtesting_Indicator_Trad_dt_=Index_backtesting_Indicator_Trad_dt_.append(Index_backtesting_Indicator_Trad_dt_2_assis)
        Index_backtesting_Indicator_Trad_dt_=Index_backtesting_Indicator_Trad_dt_.fillna(1)
        Index_backtesting_Indicator_Trad_dt_=Index_backtesting_Indicator_Trad_dt_.drop([Index_backtesting_Indicator_Trad_dt_.columns[0]],axis=1)
        Index_backtesting_Indicator_Trad_dt_=Index_backtesting_Indicator_Trad_dt_.sort_index()
        Index_backtesting_Indicator_Trad_dt_=Index_backtesting_Indicator_Trad_dt_[(Index_backtesting_Indicator_Trad_dt_.index>=N_triggered_days_Style_TS_TP_Assis[t]) & (Index_backtesting_Indicator_Trad_dt_.index<N_triggered_days_Style_TS_TP_Assis[t+1])]
        Index_backtesting_Indicator_Trad_dt=Index_backtesting_Indicator_Trad_dt.append(Index_backtesting_Indicator_Trad_dt_)
    
    Index_backtesting_Indicator_Timeseries_Signal=pd.merge(Index_backtesting_Indicator_Timeseries_Signal,Index_backtesting_Indicator_Trad_dt,right_index=True,left_index=True,how='outer')
#    Index_backtesting_Indicator_Timeseries_Signal.to_csv(r'C:\Users\yshlm\Desktop\licaimofang\data\Index_backtesting_Indicator_Timeseries_Signal_Loose.csv')

    Test_RETURN=pd.DataFrame(Index_data_Chg[Index_data_Chg.index.isin(Index_backtesting_Indicator_Trad_dt.index)].iloc[:,id_index])
    Test_RETURN_A=np.hstack(Index_backtesting_Indicator_Trad_dt.values)
    
    Test_RETURN_B=np.hstack(Test_RETURN.values)
    
    Test_RETURN1=Test_RETURN_A*Test_RETURN_B
    Test_RETURN1=pd.DataFrame(Test_RETURN1,index=Test_RETURN.index,columns=Test_RETURN.columns)
    
    Index_backtesting_Result=pd.merge(pd.DataFrame(Index_data_Chg.iloc[:,id_index]),pd.DataFrame(Test_RETURN1),right_index=True,left_index=True,how='inner')
    Index_backtesting_Result=Index_backtesting_Result.fillna(0)
    Index_backtesting_Result.columns=[str(Index_backtesting.columns[id_index]),str(Index_backtesting.columns[id_index]+ ' Algo')]
    Index_backtesting_Result=Index_backtesting_Result.cumsum()
    #Index_backtesting_Result=Index_backtesting_Result[Index_backtesting_Result.index>=pd.Timestamp('2008-10-21')]
    Index_backtesting_Result.plot(title=str('Compare Algorithm data and Original data of ' + Index_backtesting.columns[id_index]))
    plt.show()


##########Mkt###Analysis#############################################################
##########Mkt###Analysis#############################################################
##########Mkt###Analysis#############################################################
##########Mkt###Analysis#############################################################
##########Mkt###Analysis#############################################################
##########Mkt###Analysis#############################################################
##########Mkt###Analysis#############################################################
##########Mkt###Analysis#############################################################
##########Mkt###Analysis#############################################################


###########################################################
Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\Multi_Mkt_Indices.csv")
columns_name = [ 'Trd_dt','000001.SH', '399001.SZ','HSI.HI','SP500.SPI','NDX.GI']
Index_data.columns = columns_name
Index_data.index=Index_data.Trd_dt
Index_data=Index_data.drop(columns='Trd_dt')
Index_data=Index_data.dropna()

Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))
Index_data=Index_data.fillna(method='bfill')

Ret=np.log(Index_data.div(Index_data.shift(1)))
Ret=Ret.dropna()

cum_returns = (1 + Ret).cumprod()
drawdown =  cum_returns.div(cum_returns.cummax()) -1
#drawdown.plot()
#plt.show()
#drawdown.quantile(0.05)



###########################################################


consecutive_negative_ret_signal=pd.DataFrame()
for k in range(Ret.shape[1]):  
 
    consecutive_negative_ret_indicator=[]
    for i in range(Ret.shape[0]):
        if Ret.iloc[i,k]<0:        
            while True:            
                consecutive_negative_ret_indicator.append(1)
                if Ret.iloc[i,k]<0:
                    break
        else:
            consecutive_negative_ret_indicator.append(0)
    
    consecutive_negative_ret_indicator1=[]
    for i in range(1,Ret.shape[0]-1):
        if consecutive_negative_ret_indicator[i]==1:
            if consecutive_negative_ret_indicator[i+1]==1 or consecutive_negative_ret_indicator[i-1]==1:
                consecutive_negative_ret_indicator1.append(1)
            else:
                consecutive_negative_ret_indicator1.append(0)
        else:
            consecutive_negative_ret_indicator1.append(0)
    
    consecutive_negative_ret_indicator1.insert(0,0)
    consecutive_negative_ret_indicator1.insert(-1,0)
    consecutive_negative_ret_indicator1=pd.DataFrame(consecutive_negative_ret_indicator1,index=Ret.index)
        
    consecutive_negative_ret_signal=pd.merge(consecutive_negative_ret_signal,consecutive_negative_ret_indicator1,right_index=True,left_index=True,how='outer')

consecutive_negative_ret_signal.columns=Ret.columns
consecutive_negative_ret_signal.plot()


###########################################################

'chenggonglv shouyilv quxian from Mr. Ma'

Ret=Index_data.pct_change()
Ret=Ret.drop(Ret.index[0])
consecutive_negative_ret=consecutive_negative_ret_signal*Ret

'Replace'
def consecutive_negative_ret_cumsum(consecutive_negative_ret):
    consecutive_negative_ret_cumsum=pd.DataFrame()
    for k in range(consecutive_negative_ret.shape[1]):
        consecutive_negative_ret_assis=[]
        i=0
        while i <= consecutive_negative_ret.shape[0]-1:
            if consecutive_negative_ret.iloc[i,k]!=0:
                for j in range(consecutive_negative_ret.shape[0]-i):
                    if consecutive_negative_ret.iloc[i+j,k]!=0:
                        consecutive_negative_ret_assis.append(float(consecutive_negative_ret.iloc[i:i+j+1,k].cumsum()[-1]))
#                        print(i,j)
    
                    else:
                        i+=j-1
                        break
                            
            elif consecutive_negative_ret.iloc[i,k]==0:
                consecutive_negative_ret_assis.append(consecutive_negative_ret.iloc[i,k])
            i+=1
        
        consecutive_negative_ret_assis=pd.DataFrame(consecutive_negative_ret_assis)
        consecutive_negative_ret_cumsum=pd.merge(consecutive_negative_ret_cumsum,consecutive_negative_ret_assis,right_index=True,left_index=True,how='outer')
    
    consecutive_negative_ret_cumsum.index=consecutive_negative_ret.index
    consecutive_negative_ret_cumsum.columns=consecutive_negative_ret.columns
    
    return consecutive_negative_ret_cumsum

    
consecutive_negative_ret_cumsum=consecutive_negative_ret_cumsum(consecutive_negative_ret)
consecutive_negative_ret_cumsum.plot(title='Consecutive_Negative_Cumsum_Ret')

###########################################################

def consecutive_negative_ret_maxcumsum(consecutive_negative_ret_cumsum):
    consecutive_negative_ret_maxcumsum=pd.DataFrame()
    for k in range(consecutive_negative_ret_cumsum.shape[1]):
        consecutive_negative_ret_maxcumsum_assis=[]
        for i in range(1,consecutive_negative_ret_cumsum.shape[0]):
            if consecutive_negative_ret_cumsum.iloc[i,k]==0 and consecutive_negative_ret_cumsum.iloc[i-1,k]!=0:
                consecutive_negative_ret_maxcumsum_assis.append(consecutive_negative_ret_cumsum.iloc[i-1,k])
                
        consecutive_negative_ret_maxcumsum_assis=pd.DataFrame(consecutive_negative_ret_maxcumsum_assis)    
        consecutive_negative_ret_maxcumsum=pd.merge(consecutive_negative_ret_maxcumsum,consecutive_negative_ret_maxcumsum_assis,right_index=True,left_index=True,how='outer')
    
    consecutive_negative_ret_maxcumsum.columns=consecutive_negative_ret_cumsum.columns
    return consecutive_negative_ret_maxcumsum
consecutive_negative_ret_maxcumsum=consecutive_negative_ret_maxcumsum(consecutive_negative_ret_cumsum)

consecutive_negative_ret_maxcumsum.plot(kind='density',title='Maximum Cumsum Consecutive Negative Ret')
#consecutive_negative_ret_maxcumsum.quantile(q=0.05)
consecutive_negative_ret_maxcumsum.describe()
###########precision##########recall######################################

Index_backtesting_Indicator_Timeseries_Signal_1=Index_backtesting_Indicator_Timeseries_Signal*1
consecutive_negative_ret_cumsum_1=consecutive_negative_ret_cumsum*1

Index_backtesting_Indicator_Timeseries_Signal_1[Index_backtesting_Indicator_Timeseries_Signal_1<1]=0
consecutive_negative_ret_cumsum_1=consecutive_negative_ret_cumsum_1[consecutive_negative_ret_cumsum_1.index.isin(Index_backtesting_Indicator_Timeseries_Signal_1.index)]
consecutive_negative_ret_cumsum_1[consecutive_negative_ret_cumsum_1>=0]=1
consecutive_negative_ret_cumsum_1[consecutive_negative_ret_cumsum_1<0]=0

from sklearn.metrics import classification_report

[print(classification_report(consecutive_negative_ret_cumsum_1.iloc[:,i], Index_backtesting_Indicator_Timeseries_Signal_1.iloc[:,i], digits = 6)) for i in range(consecutive_negative_ret_cumsum.shape[1])]
#
#A=consecutive_negative_ret_cumsum_1.iloc[:,0][consecutive_negative_ret_cumsum_1.iloc[:,0]==0]
#B=Index_backtesting_Indicator_Timeseries_Signal_1.iloc[:,0][Index_backtesting_Indicator_Timeseries_Signal_1.iloc[:,0]==0]
#np.sum(A.index.isin(B.index))/B.shape[0]
