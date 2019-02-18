# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:21:54 2018

@author: zhouboyang
"""

import numpy as np
#import scipy as sp 
import pandas as pd 
import scipy.stats 

"Algo Function"
######################################FIXED##############################################################################
#################################FIXED###################################################################################
#####################################################FIXED###############################################################

def algo_result(algo_percentage,algo_input_array,input_aes,Annulize,underlying_px_matrix,Backtesting_time_percetage):
#    algo_percentage means the quantile of the cut in the underlying 
    "pattern sorted result index ascending is 1, desending is -1, otherwise the input will be not correct"    
    Backtesting_sorted_index=np.argsort(algo_input_array)[::input_aes]  
    "from the most important underlyings to the last important ones"
    Backtesting_time_lenth=round(Backtesting_time_percetage*underlying_px_matrix.shape[0])
#####################        
    algo_interval_ret_result_step=[(underlying_px_matrix[Backtesting_time_lenth+1+i:,0:]/underlying_px_matrix[Backtesting_time_lenth+i:-1,0:]-1) for i in range(underlying_px_matrix.shape[0]-Backtesting_time_lenth-1)][0]
    algo_ret_result=[np.prod((algo_interval_ret_result_step+np.ones(len(algo_interval_ret_result_step[0])))[:,i])-1 for i in range(underlying_px_matrix.shape[1])]
    "maybe more efficient"   
#   algo_ret_result=(underlying_px_matrix[-1,0:]/underlying_px_matrix[Backtesting_time_lenth,0:]-1).tolist() 
#####################    
#   desending
    Backtesting_long_position_result=[algo_ret_result[Backtesting_sorted_index[-i]] for i in range(1,(round(algo_percentage*len(algo_input_array)))+1)]
#   ascending
    Backtesting_short_position_result=[algo_ret_result[Backtesting_sorted_index[i]] for i in range(round(algo_percentage*len(algo_input_array)))]            
#####################   
    Algo_long_result=np.average(Backtesting_long_position_result)*Annulize
    Algo_short_result=np.average(Backtesting_short_position_result)*Annulize
    Algo_long_result_std=np.std(Backtesting_long_position_result)*Annulize
    Algo_short_result_std=np.std(Backtesting_short_position_result)*Annulize
    "result as long, shot, std long, std short"
#    return [Algo_long_result,Algo_short_result,Backtesting_sorted_index,Backtesting_long_position_result]
    return [Algo_long_result,Algo_short_result,Algo_long_result_std,Algo_short_result_std,algo_ret_result]

######################################FIXED##############################################################################
#################################FIXED###################################################################################
#####################################################FIXED###############################################################

data=pd.read_csv(r"C:\Users\zhouboyang\Desktop\csv data\PX_MSCI222_SH300_CN10YB_Mkt_Cap_Weighted_Index_Yield_20060601_20180116.csv", parse_dates=[0], index_col=0)

start='20161201'
end='20180101'
data1= data[data.index>start]
data1= data1[data1.index<end]
data2=data1.fillna(data1.mean())

#data2=data.fillna(method='ffill')
cp=np.array(data2)[:,:-3]
hs300=np.array(data2)[:,-3]
#risk free rate CN10YB YIELD
Rf=np.array(data2)[:,-2]
mkt_Cap_Weighted_Index_Yield=np.array(data2)[:,-1]
"input"
#quantile of backtesting time lenth
R=12
H=3
Backtesting_time_percetage=R/(R+H)
Annual_Trading_Day=252
Transaction_fee=0.0016*2

Backtesting_time_lenth=round(Backtesting_time_percetage*cp.shape[0])
Algo_time_lenth=int(cp.shape[0]-Backtesting_time_lenth)
#Return in underlying, market and China goveronment 10 year treasury bond yield
Ret=cp[1:Backtesting_time_lenth,:]/cp[:Backtesting_time_lenth-1,:]-1
hs300_backtesting_daily_ret=hs300[1:Backtesting_time_lenth]/hs300[:Backtesting_time_lenth-1,]-1
hs300_algo_ret=hs300[-1:]/hs300[Backtesting_time_lenth]-1
Rf_ret=Rf[-1]

#cp2=np.log(cp.shift(1)/cp)
Annulized_parameter=Annual_Trading_Day/Algo_time_lenth
####################################################################################################################

"Cross Section Momentum"
####################################################################################################################

Momentum_position_percentage=0.3
Momentum_cp_sum=Ret.sum(axis=0).tolist()
Momentum_cp_sum_algo_step=(cp[Backtesting_time_lenth+1:,0:]/cp[Backtesting_time_lenth:-1,0:]-1).tolist()
Momentum_cp_sum_algo=(cp[-1:,:]/cp[Backtesting_time_lenth:,:]-1).sum(axis=0).tolist()

#######################
"Equal Weight Result"
#Weighted arithmetic mean
MSCI_Ret=np.average(Momentum_cp_sum_algo)*Annual_Trading_Day/Algo_time_lenth
#Mkt_value_weight
#######################
Cross_Section_Momentum_result=algo_result(Momentum_position_percentage,Momentum_cp_sum,1,Annulized_parameter,cp,Backtesting_time_percetage)

Cross_Section_Momentum_result_TREND=Cross_Section_Momentum_result[-1]
#######################
print("Rf is %f %%" %(Rf[-1]))
print("Annulized no compounding Return of HS300 is %f %%" %(np.average(hs300_algo_ret)*Annual_Trading_Day/Algo_time_lenth*100))
print("Annulized no compounding Return of Equal Weighted market is %f %%" %(MSCI_Ret))
print("Ann no compounding Ret of EW Method Momentum long position gain is %f %%, short position gain is %f %%, and total gain is %f %%" %(Cross_Section_Momentum_result[0]*100,-Cross_Section_Momentum_result[1]*100,(Cross_Section_Momentum_result[0]-Cross_Section_Momentum_result[1])*100))


####################################################################################################################

"Roll_yield"
####################################################################################################################
#data range
Roll_yield_data=cp
Roll_yield_position_percentage=0.3
#Around one Month Trading day, better to be one can divided by "Algo_time_lenth"
Roll_yield_period=round((len(Roll_yield_data)/(R+H)-1))

Roll_yield_ret_period=[]
for i in range(1,round(Backtesting_time_lenth/Roll_yield_period)+1):
    Roll_yield_ret_period+=((Roll_yield_data[(i-1)*Roll_yield_period+1:i*Roll_yield_period+1,0:]/Roll_yield_data[(i-1)*Roll_yield_period:i*Roll_yield_period,0:]-1).sum(axis=0).tolist())

Roll_yield_ret_period=np.asarray(Roll_yield_ret_period).reshape(int(round(Backtesting_time_lenth/Roll_yield_period)),int(len(Roll_yield_ret_period)/round(Backtesting_time_lenth/Roll_yield_period)))
Roll_yield_ret_sum=Roll_yield_ret_period.sum(axis=0).tolist()
Roll_Yield_Algo=Momentum_cp_sum_algo
#######################
Roll_Yield_result=algo_result(Roll_yield_position_percentage,Roll_yield_ret_sum,1,Annulized_parameter,cp,Backtesting_time_percetage)
#######################
print("Ann no compounding Ret of EW Method Roll Yield Strategy long position gain is %f %%, short position gain is %f %%, and total gain is %f %%" %(Roll_Yield_result[0]*100,-Roll_Yield_result[1]*100,(Roll_Yield_result[0]-Roll_Yield_result[1])*100))
####################################################################################################################
####################################################################################################################
####################################################################################################################

"Skewness"
####################################################################################################################
Skew_cp_skewness=scipy.stats.skew(cp[:Backtesting_time_lenth,:]).tolist()

Skew_position_percentage=0.3
Skew_Algo=Momentum_cp_sum_algo
#######################
Skew_result=algo_result(Skew_position_percentage,Skew_cp_skewness,1,Annulized_parameter,cp,Backtesting_time_percetage)
#######################
print("Ann no compounding Ret of EW Method Skewness Strategy long position gain is %f %%, short position gain is %f %%, and total gain is %f %%" %(Skew_result[0]*100,-Skew_result[1]*100,(Skew_result[0]-Skew_result[1])*100))
####################################################################################################################
####################################################################################################################
####################################################################################################################

"Low risk PX_Volatility based"
####################################################################################################################
#Out of sample
PX_Volatility_based=[np.std(cp[:,i]) for i in range(cp.shape[1])]

PX_Volatility_based_position_percentage=0.3
PX_Volatility_based_Algo=Momentum_cp_sum_algo
#######################
PX_Volatility_based_result=algo_result(PX_Volatility_based_position_percentage,PX_Volatility_based,1,Annulized_parameter,cp,Backtesting_time_percetage)
#######################
print("Ann no compounding Ret of EW Method PX Volatility Strategy long position gain is %f %%, short position gain is %f %%, and total gain is %f %%" %(PX_Volatility_based_result[0]*100,-PX_Volatility_based_result[1]*100,(PX_Volatility_based_result[0]-PX_Volatility_based_result[1])*100))
####################################################################################################################
####################################################################################################################
####################################################################################################################
"Low risk Ret_Volatility based"
####################################################################################################################
####################################################################################################################
####################################################################################################################
#out ofsample
Ret_Volatility_based=[np.std(Ret[:Backtesting_time_lenth,i]) for i in range(Ret.shape[1])]

Ret_Volatility_based_percentage=0.3
Ret_Volatility_based_Algo=Momentum_cp_sum_algo
#######################
Ret_Volatility_result=algo_result(Ret_Volatility_based_percentage,Ret_Volatility_based,1,Annulized_parameter,cp,Backtesting_time_percetage)
#######################
print("Ann no compounding Ret of EW Method Ret Volatility Strategy long position gain is %f %%, short position gain is %f %%, and total gain is %f %%" %(Ret_Volatility_result[0]*100,-Ret_Volatility_result[1]*100,(Ret_Volatility_result[0]-Ret_Volatility_result[1])*100))
####################################################################################################################
####################################################################################################################
####################################################################################################################
"Low risk BETA based"
####################################################################################################################
####################################################################################################################
####################################################################################################################
#out of sample
Beta_based_cov=[np.cov(Ret[:,i],mkt_Cap_Weighted_Index_Yield[1:Backtesting_time_lenth])[0,1] for i in range(Ret.shape[1])]
Beta_based_mkt_std=np.std(mkt_Cap_Weighted_Index_Yield[1:Backtesting_time_lenth])
#Beta_based_underlying=[np.std(Ret[:,i]) for i in range (Ret.shape[1])]
Beta_based_beta=[Beta_based_cov[i]/Beta_based_mkt_std for i in range(len(Beta_based_cov))]

Beta_based_percentage=0.3
Beta_based_Algo=Momentum_cp_sum_algo
#######################
Beta_based_result=algo_result(Beta_based_percentage,Beta_based_beta,1,Annulized_parameter,cp,Backtesting_time_percetage)
#######################
print("Ann no compounding Ret of EW Method Ret Beta Strategy long position gain is %f %%, short position gain is %f %%, and total gain is %f %%" %(Beta_based_result[0]*100,-Beta_based_result[1]*100,(Beta_based_result[0]-Beta_based_result[1])*100))
####################################################################################################################
####################################################################################################################
####################################################################################################################
"Idiosyncratic volatility portfolio, Ivol signal"
####################################################################################################################
####################################################################################################################
####################################################################################################################
#long low Ivol short high Ivol
# Rst - Rf = beta_zero + beta_one(ret_hs300 - ret_BK) + epsilon
# Ivol = std(spsilon)
Ivol_beta_zero=np.cov(mkt_Cap_Weighted_Index_Yield[1:Backtesting_time_lenth])/Beta_based_mkt_std
#Ivol_abc=[Beta_based_beta[i]*(hs300_backtesting_daily_ret[i]-mkt_Cap_Weighted_Index_Yield[i]) for i in range(Backtesting_time_lenth-1)]
Ivol_espilon=[[Ret[i,j]-Rf[i]-Ivol_beta_zero-Beta_based_beta[i]*(hs300_backtesting_daily_ret[i]-mkt_Cap_Weighted_Index_Yield[i]) for i in range(Backtesting_time_lenth-1)] for j in range(Ret.shape[1])]
Ivol_std=[np.std(Ivol_espilon[i]) for i in range(Ret.shape[1])]

Ivol_based_percentage=0.3
Ivol_based_Algo=Momentum_cp_sum_algo
#######################
Ivol_based_result=algo_result(Ivol_based_percentage,Ivol_std,1,Annulized_parameter,cp,Backtesting_time_percetage)
#######################
print("Ann no compounding Ret of EW Ivol Strategy long position gain is %f %%, short position gain is %f %%, and total gain is %f %%" %(Ivol_based_result[0]*100,-Ivol_based_result[1]*100,(Ivol_based_result[0]-Ivol_based_result[1])*100))
####################################################################################################################
####################################################################################################################
####################################################################################################################
"Low PB"
####################################################################################################################
####################################################################################################################
####################################################################################################################

"Liquidity Ratio Risk Premia_______Szymanowska et al 2014"
####################################################################################################################
####################################################################################################################
####################################################################################################################



####################################################################################################################
####################################################################################################################
####################################################################################################################
"Liquidity Ratio Risk Premia_______Szymanowska et al 2014_Average two months ofthe daily ratio of volume to absolute return"
####################################################################################################################
####################################################################################################################
####################################################################################################################

####################################################################################################################
####################################################################################################################
####################################################################################################################

#result_SUMMARY=[Rf[-1],np.average(hs300_algo_ret)*Annual_Trading_Day/Algo_time_lenth*100,MSCI_Ret*100]
#print(result_SUMMARY)