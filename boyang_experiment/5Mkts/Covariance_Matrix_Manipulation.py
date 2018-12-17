# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:23:25 2018

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
import arch

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



#Cov_nlags=10
##Time series Covariance matrix
#Cov_ret_TS_gamma_0=pd.DataFrame()
#for i in range(Ret.shape[0]-Ini_Windows_size-Cov_nlags):    
#    Cov_ret=Ret.iloc[:Ini_Windows_size+i,:].cov()
#    Cov_ret_TS_gamma_0=Cov_ret.add(Cov_ret_TS_gamma_0,fill_value=0)
##Gamma 0 Matrix
#Cov_ret_TS_gamma_0=Cov_ret_TS_gamma_0.divide(Ret.shape[0]-Ini_Windows_size-Cov_nlags)
#
#Cov_ret_TS_gamma_i=pd.DataFrame()
#for i in range(Cov_nlags):
#    Cov_ret=Ret.iloc[:-Cov_nlags+i,:].cov()
#    omega_weight=(i/(1+Cov_nlags))
#    Cov_ret_weighted=omega_weight*Cov_ret
#    Cov_ret_TS_gamma_i=Cov_ret_weighted.add(Cov_ret_TS_gamma_i,fill_value=0)
#
##Cov_ret_TS_gamma_i.T
#Cov_ret_TS=(Cov_ret_TS_gamma_0+Cov_ret_TS_gamma_i)/2

##########################################################################

Cov_ret_TS=[]
for i in range(Ret.shape[0]-Ini_Windows_size):    
    Cov_ret=Ret.iloc[i:Ini_Windows_size+i,:].cov()
    Cov_ret_1=Cov_ret.values.reshape(1,Cov_ret.size)
    Cov_ret_TS.append(Cov_ret_1)
Cov_ret_TS=pd.DataFrame(data=np.vstack(Cov_ret_TS),index=Ret.index[Ini_Windows_size:])
'''
Unit root test: Hypothesis test be rejected 
'''
ADF_AIC_result=[sm.tsa.stattools.adfuller(Cov_ret_TS.iloc[:,i], maxlag=None, regression='c', autolag='AIC', store=False, regresults=False) for i in range(Cov_ret_TS.shape[1])]
'''
Autocorrelation
'''
Ljung_Box_result = [sm.stats.diagnostic.acorr_ljungbox(Cov_ret_TS.iloc[:,i]) for i in range(Cov_ret_TS.shape[1])]


Cov_ret_TS.iloc[:,6].plot()
Cov_ret_TS.plot()
'''
Co-integration
'''
Cov_ret_TS_Co_Inte=np.log(Cov_ret_TS/Cov_ret_TS.shift(1)).dropna()

ADF_AIC_result_Co_Inte=[sm.tsa.stattools.adfuller(Cov_ret_TS_Co_Inte.iloc[:,i], maxlag=None, regression='c', autolag='AIC', store=False, regresults=False) for i in range(Cov_ret_TS_Co_Inte.shape[1])]
Ljung_Box_result_Co_Inte=[sm.stats.diagnostic.acorr_ljungbox(Cov_ret_TS_Co_Inte.iloc[:,i]) for i in range(Cov_ret_TS_Co_Inte.shape[1])]
#'GARCH Constant Mean, student-t as distribuition cause the heavy tail'
#GARCH_Cov_ret_Co_Inte = arch.arch_model(Cov_ret_TS_Co_Inte.iloc[:,0], p=1, o=1, q=1, power=1.0, dist='StudentsT')
#GARCH_Cov_ret_Co_Inte_fit = GARCH_Cov_ret_Co_Inte.fit(update_freq=10)
#GARCH_Cov_ret_Co_Inte_fit.summary()
#fig = GARCH_Cov_ret_Co_Inte_fit.plot(annualize='D')
#A=GARCH_Cov_ret_Co_Inte_fit.forecast()
#print(A)
#A.residual_variance.iloc[-3:]
#GARCH_Cov_ret_Co_Inte_fit.conditional_volatility.plot()
#Cov_ret_TS_Co_Inte.iloc[:,0].plot()

'''
Since the cointegrated function shows the autocorrelation, then I basiclly set the model as ACF
and PACF or MA and ARCH model
'''


