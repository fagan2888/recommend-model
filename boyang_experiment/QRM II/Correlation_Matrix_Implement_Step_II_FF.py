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
#import pylab
import statsmodels as sm
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
import seaborn.apionly as sns
#import random
import time
from datetime import timedelta, datetime
#import arch
import itertools
import warnings

#from statsmodels.stats.diagnostic import unitroot_adf
#from statsmodels.multivariate import pca
#from sklearn import decomposition
#from statsmodels.stats.sandwich_covariance import cov_hac
#from statsmodels.tsa import statespace
#import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


Index_data = pd.read_csv(r"Multi_Indices_20181206.csv", parse_dates=[0], index_col=0)

Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))
Index_data=Index_data.dropna()
##########################################################################
##########################################################################

# The hypothesis test for independency: maybe Chi_square test?
# Target: for canceling the autocorrelation effect in the covariance matrix
# Methodology: Newey-West
# Demean for each correlation matrix

'Reference: Newey, W. K. and K. D. West (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, Vol. 55(3), 703 – 708.'
Ret = Index_data.pct_change()

Ret = Ret.iloc[1:, :]
Ret = Ret.fillna(value=0)

# demean
#Ret = Ret.sub(Ret.mean())

Ini_Windows_size = 100

##########################################################################
##########################################################################

Corr_ret_TS = []
for i in range(Ret.shape[0] - Ini_Windows_size):
    'Change the parameter from Covariance to Correlation'
    Corr_ret = Ret.iloc[i:Ini_Windows_size + i, :].corr()
    Corr_ret_1 = Corr_ret.values.reshape(1, Corr_ret.size)
    Corr_ret_TS.append(Corr_ret_1)

Corr_ret_names = []
for i in range(Corr_ret.shape[0]):
    for j in range(Corr_ret.shape[1]):
        Corr_ret_name = ('%s and %s' % (Corr_ret.index[i], Corr_ret.columns[j]))
        Corr_ret_names.append(Corr_ret_name)
        
Corr_ret_TS = pd.DataFrame(data=np.vstack(Corr_ret_TS),index=Ret.index[Ini_Windows_size:],columns=Corr_ret_names)


'''
Co-integration
'''
Corr_ret_TS_Co_Inte = np.log(Corr_ret_TS / Corr_ret_TS.shift(1))

Corr_ret_TS_Co_Inte=Corr_ret_TS_Co_Inte.iloc[1:]
Corr_ret_TS_Co_Inte=Corr_ret_TS_Co_Inte.fillna(method='bfill')
Corr_ret_TS_Co_Inte=Corr_ret_TS_Co_Inte.drop(columns=[Corr_ret_TS_Co_Inte.columns[i*(int(np.sqrt(Corr_ret_TS.shape[1]))+1)] for i in range(int(np.sqrt(Corr_ret_TS.shape[1])))])
Corr_ret_names_Na_0=Corr_ret_names
Corr_ret_names_Na_01=[Corr_ret_names[(int(np.sqrt(Corr_ret_TS.shape[1]))+1)*i] for i in range(int(np.sqrt(len(Corr_ret_names))))]
Corr_ret_names_Na_0=[x for x in Corr_ret_names if x not in Corr_ret_names_Na_01]

    
 
  
y=Corr_ret_TS_Co_Inte.iloc[:,1]
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

ARIMA_calibra_paras=[]
ARIMA_calibra_paras_INT=[]
ARIMA_calibra_paras_AIC=[]

'STAR HERE: The season params in step I'
for j in range(4):
    
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], j+1) for x in list(itertools.product(p, d, q))]
    
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
#                    mod = SARIMAX(y, order=param,
#                                                    seasonal_order=param_seasonal,
#                                                    enforce_stationarity=False,
#                                                    enforce_invertibility=False)
#        
#                    results = mod.fit()
    
    #            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                ARIMA_calibra_paras.append('ARIMA{}x{}'.format(param, param_seasonal))
                ARIMA_calibra_paras_INT.append(list(itertools.chain.from_iterable([param,param_seasonal])))
#                    ARIMA_calibra_paras_AIC.append(results.aic)
    
            except:
                continue
        
#    ARIMA_calibra_paras_Summary1=pd.DataFrame(data=ARIMA_calibra_paras_AIC,index=ARIMA_calibra_paras)
#    ARIMA_calibra_paras_Summary=pd.merge(ARIMA_calibra_paras_Summary,ARIMA_calibra_paras_Summary1,right_index=True,left_index=True,how='outer')
#
#ARIMA_calibra_paras_Summary.columns=Corr_ret_names
#ARIMA_calibra_paras_Summary.to_csv('ARIMA_calibra_paras_Summary.csv')

# ARIMA_calibra_paras_Summary=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\ARIMA_calibra_paras_Summary0.csv",index_col=0)
# ARIMA_calibra_paras_Summary.columns=Corr_ret_names_Na_0
# ARIMA_calibra_paras_Summary.index=ARIMA_calibra_paras
'''
Finally, params calibrated as ARIMA parameters
'''
#

def index_corr_pred(Corr_ret_TS_Co_Inte,ARIMA_calibra_paras_argmin):
    
    Corr_pred=pd.DataFrame()
    for i in range(Corr_ret_TS_Co_Inte.shape[1]):
        order= tuple(ARIMA_calibra_paras_argmin[i][:3])
        seasonal_order=tuple(ARIMA_calibra_paras_argmin[i][-4:])
        
        mod = SARIMAX(Corr_ret_TS_Co_Inte.iloc[:,i], trend='c', order=order, seasonal_order=seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
        res = mod.fit(disp=False)
        
        pred = res.get_prediction()
        pred_ci = pred.conf_int()
        pred_ci=pred_ci.drop(pred_ci.index[:seasonal_order[3]+1])
        pred_ci_mean=pred_ci.mean(axis=1)
        pred_ci_mean=pd.DataFrame(pred_ci_mean)
        
        Corr_pred=pd.merge(Corr_pred,pred_ci_mean,right_index=True,left_index=True,how='outer')
    
    for i in range(5):       
        Corr_pred_Matrix=Corr_pred.insert(loc=i*6,column=str(i),value=pd.Series(np.ones(Corr_ret_TS_Co_Inte.shape[0])))
    
    return Corr_pred_Matrix

#########################



#########################
ARIMA_calibra_paras_ = pd.read_csv(r"ARIMA_calibra_paras_Summary.csv",index_col=[0])
ARIMA_calibra_paras_.index=np.asarray(ARIMA_calibra_paras_INT)
ARIMA_calibra_paras_.columns=Corr_ret_names_Na_0
#ARIMA_calibra_paras_.to_csv(r"ARIMA_calibra_paras_seasonal_order_3.csv")

ARIMA_calibra_paras_argmin=[np.argmin(ARIMA_calibra_paras_.iloc[:,i]) for i in range(ARIMA_calibra_paras_.shape[1])]




'Fitting the timeseries model'
Corr_pred=pd.DataFrame()
for i in range(Corr_ret_TS_Co_Inte.shape[1]):
    order= tuple([int(ARIMA_calibra_paras_argmin[i][j]) for j in range(3)])
    'STAR HERE: The season params in step I'
    seasonal_order=tuple([int(ARIMA_calibra_paras_argmin[i][3+j]) for j in range(4)])
    
#    data_ori=Corr_ret_TS_Co_Inte.iloc[:,i]
    mod = SARIMAX(Corr_ret_TS_Co_Inte.iloc[:,i], trend='c', order=order, seasonal_order=seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
    res = mod.fit(disp=False)
    
    pred = res.get_prediction()
    pred_ci = pred.conf_int()
    pred_ci=pred_ci.drop(pred_ci.index[:seasonal_order[3]+1])
    pred_ci_mean=pred_ci.mean(axis=1)
    pred_ci_mean=pd.DataFrame(pred_ci_mean)
    
    Corr_pred=pd.merge(Corr_pred,pred_ci_mean,right_index=True,left_index=True,how='outer')


Corr_pred_Matrix=Corr_pred*1
for i in range(Index_data.shape[1]):       
    Corr_pred_Matrix.insert(loc=i*(Index_data.shape[1]+1),column=str(i),value=pd.Series(np.ones(Corr_pred.shape[0])))

Corr_pred_Matrix=Corr_pred_Matrix.fillna(method='bfill')
Corr_pred_Matrix=Corr_pred_Matrix.fillna(value=1)

Corr_pred_Matrix_=pd.DataFrame()
for i in range(Corr_pred_Matrix.shape[1]):
    'Inverse function of kernel function'
    Reverse_Co_Inte=Corr_ret_TS.iloc[:,i]*(Corr_pred_Matrix.iloc[:,i].map(lambda x: np.exp(x)))
    'For the UNEXPECTED correlation parameters, I cannot find some solid/rigious solution to fix it, I set the value to nan here'
    Reverse_Co_Inte=Reverse_Co_Inte.apply(lambda x: float('nan') if x >= 1 else (float('nan') if x<=-1 else x))
    Reverse_Co_Inte=pd.DataFrame(Reverse_Co_Inte)
    Corr_pred_Matrix_=pd.merge(Corr_pred_Matrix_,Reverse_Co_Inte,right_index=True,left_index=True,how='outer')
 
Corr_pred_Matrix_.columns=Corr_ret_names
Corr_pred_Matrix_=Corr_pred_Matrix_.fillna(method='bfill')
Corr_pred_Matrix_=Corr_pred_Matrix_.fillna(value=1)
Corr_pred_Matrix_.to_csv(r'Multi_Indices_corr_matrix.csv')

'Test whether the matrix is full ranked and try reduce the collnearity'
Corr_pred_Matrix_final=[]
for i in range(Corr_pred_Matrix_.shape[0]):
    Corr_pred_Matrix_1=pd.DataFrame(Corr_pred_Matrix_.iloc[i,:].reshape(Index_data.shape[1],Index_data.shape[1]))
    Corr_pred_Matrix_1.columns=Corr_ret.columns
    Corr_pred_Matrix_1.index=Corr_ret.columns
    Corr_pred_Matrix_final.append(Corr_pred_Matrix_1)
    
#Corr_pred_Matrix_final=pd.DataFrame(Corr_pred_Matrix_final)
#Corr_pred_Matrix_final.index=Corr_pred_Matrix_.index
A=[]
#AAQ=[]
for i in range(len(Corr_pred_Matrix_final)):
#    print(Corr_pred_Matrix_.index[i])
    try:
        A.append(np.linalg.cholesky(Corr_pred_Matrix_final[i]))
    except:
        print(Corr_pred_Matrix_.index[i])
        A.append(np.nan)
#        AAQ.append(Corr_pred_Matrix_.index[i])
        pass
A=pd.DataFrame(A,index=Corr_pred_Matrix_.index,columns=['Cholesky']).fillna(method='ffill')    





