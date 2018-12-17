# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:23:25 2018

@author: Boyang ZHOU
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

import time
from datetime import timedelta, datetime
import itertools
import warnings

#import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from functools import partial
from multiprocess import Pool



def ARIMA_CALI_PARAMS(Corr_ret_TS_Co_Inte,i):
    

    y=Corr_ret_TS_Co_Inte.iloc[:,i]
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)
    
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    
    ARIMA_calibra_paras=[]
    ARIMA_calibra_paras_AIC=[]
    for j in range(30):
        
        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], j+1) for x in list(itertools.product(p, d, q))]
        
        warnings.filterwarnings("ignore") # specify to ignore warning messages
    
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = SARIMAX(y, order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
        
                    results = mod.fit()
        
        #            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                    ARIMA_calibra_paras.append('ARIMA{}x{}'.format(param, param_seasonal))          
                    ARIMA_calibra_paras_AIC.append(results.aic)
        
                except:
                    continue
            
    ARIMA_calibra_paras_Summary1=pd.DataFrame(data=ARIMA_calibra_paras_AIC,index=ARIMA_calibra_paras)
#        ARIMA_calibra_paras_Summary=pd.merge(ARIMA_calibra_paras_Summary,ARIMA_calibra_paras_Summary1,right_index=True,left_index=True,how='outer')
        
#    return ARIMA_calibra_paras_Summary   
    return ARIMA_calibra_paras_Summary1

    
if __name__ == '__main__':
    
#    Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\QRM TEST\Multi_Indices_20181206.csv", parse_dates=[0], index_col=0)
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

    
    
    ##########################################################################
    ##########################################################################
    ##########################################################################
    '''
    Since the cointegrated function shows the autocorrelation, then I basiclly set the model as ACF
    and PACF or MA and AR model.
    For the moving avarage process, finally I prefer the ARIMA model for improving the predictability
    of the Corrariance matrix
    '''
    '''
    Corr_ret_TS_Co_Inte[1].plot()
    # Fit the model in one sample
    order=(2,1,0)
    seasonal_order=(1,1,0,12)
    mod = SARIMAX(Corr_ret_TS_Co_Inte[1], trend='c', order=order, seasonal_order=seasonal_order)
    res = mod.fit(disp=False)
    print(res.summary())
    
    
    
    #########################
    # Calibrate the parameters
    '''
    
    
    func=partial(ARIMA_CALI_PARAMS,Corr_ret_TS_Co_Inte,)
    pool=Pool(32)
    ARIMA_calibra_paras_Summary=pool.map(func,range(Corr_ret_TS_Co_Inte.shape[1]))
    pool.close()
    pool.join()
    
    ARIMA_calibra_paras_Summary=np.hstack(ARIMA_calibra_paras_Summary)
    ARIMA_calibra_paras_Summary=pd.DataFrame(ARIMA_calibra_paras_Summary)
    ARIMA_calibra_paras_Summary.to_csv('ARIMA_calibra_paras_Summary.csv')
