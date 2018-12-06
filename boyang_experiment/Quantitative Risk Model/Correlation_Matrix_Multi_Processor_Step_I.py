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

import time
from datetime import timedelta, datetime
import arch
import itertools
import warnings

#import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from functools import partial
from multiprocess import Pool


#def ARIMA_CALI_PARAMS(Corr_ret_TS_Co_Inte):
#    
#    
#    for i in range(Corr_ret_TS_Co_Inte.shape[1]):
#    #    
#        y=Corr_ret_TS_Co_Inte.iloc[:,i]
#        # Define the p, d and q parameters to take any value between 0 and 2
#        p = d = q = range(0, 2)
#        
#        # Generate all different combinations of p, q and q triplets
#        pdq = list(itertools.product(p, d, q))
#        
#        ARIMA_calibra_paras=[]
#        ARIMA_calibra_paras_AIC=[]
#        for j in range(60):
#            
#            # Generate all different combinations of seasonal p, q and q triplets
#            seasonal_pdq = [(x[0], x[1], x[2], j+1) for x in list(itertools.product(p, d, q))]
#            
#            warnings.filterwarnings("ignore") # specify to ignore warning messages
#        
#            for param in pdq:
#                for param_seasonal in seasonal_pdq:
#                    try:
#                        mod = SARIMAX(y, order=param,
#                                                        seasonal_order=param_seasonal,
#                                                        enforce_stationarity=False,
#                                                        enforce_invertibility=False)
#            
#                        results = mod.fit()
#            
#            #            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
#                        ARIMA_calibra_paras.append('ARIMA{}x{}'.format(param, param_seasonal))          
#                        ARIMA_calibra_paras_AIC.append(results.aic)
#            
#                    except:
#                        continue
#            
#        ARIMA_calibra_paras_Summary1=pd.DataFrame(data=ARIMA_calibra_paras_AIC,index=ARIMA_calibra_paras)
#        ARIMA_calibra_paras_Summary=pd.merge(ARIMA_calibra_paras_Summary,ARIMA_calibra_paras_Summary1,right_index=True,left_index=True,how='outer')
#        
#    return ARIMA_calibra_paras_Summary

def ARIMA_CALI_PARAMS(Corr_ret_TS_Co_Inte,i):
    
    
#    for i in range(Corr_ret_TS_Co_Inte.shape[1]):
    #    
#    y=Corr_ret_TS_Co_Inte.iloc[:,i]
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
    
    # Index_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\Multi_Mkt_Indices.csv")
    Index_data = pd.read_csv(r"Multi_Mkt_Indices.csv")

    columns_name = ['Trd_dt', '000001.SH', '399001.SZ','HSI.HI','SP500.SPI','NDX.GI']
    Index_data.columns = columns_name
    Index_data.index=Index_data['Trd_dt']
    del Index_data['Trd_dt']
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
    Unit root test: Hypothesis test be rejected and result does not solid
    '''
    #ADF_AIC_result = [sm.tsa.stattools.adfuller(Corr_ret_TS.iloc[:, i], maxlag=None, regression='c', autolag='AIC', store=False, regresults=False) for i in range(Corr_ret_TS.shape[1])]
    '''
    Autocorrelation
    '''
    #Ljung_Box_result = [sm.stats.diagnostic.acorr_ljungbox(Corr_ret_TS.iloc[:, i]) for i in range(Corr_ret_TS.shape[1])]
    
    '''
    Plot Corrariance para time series
    '''
    #for i in range(Corr_ret_TS.shape[1]):
    #    plt.subplot(np.sqrt(Corr_ret_TS.shape[1]),np.sqrt(Corr_ret_TS.shape[1]),i+1)
    #    x=Corr_ret_TS.index
    #    y=Corr_ret_TS.iloc[:,i]
    #
    #    plt.plot(x, y)
    #    plt.legend = True
    #    plt.mark_right = True
    #
    #    plt.title('%s %d smpl  Corr para' % (Corr_ret_names[i],Ini_Windows_size))
    #    
    #plt.show()
    
    '''
    Co-integration
    '''
#    Corr_ret_TS_Co_Inte = np.log(Corr_ret_TS.shift(1) / Corr_ret_TS)
    Corr_ret_TS_Co_Inte = np.log(Corr_ret_TS / Corr_ret_TS.shift(1))

    Corr_ret_TS_Co_Inte=Corr_ret_TS_Co_Inte.iloc[1:]
    Corr_ret_TS_Co_Inte=Corr_ret_TS_Co_Inte.fillna(method='bfill')
    Corr_ret_TS_Co_Inte=Corr_ret_TS_Co_Inte.drop(columns=[Corr_ret_TS_Co_Inte.columns[i*6] for i in range(int(np.sqrt(Corr_ret_TS.shape[1])))])
    Corr_ret_names_Na_0=Corr_ret_names
    Corr_ret_names_Na_01=[Corr_ret_names[6*i] for i in range(int(np.sqrt(len(Corr_ret_names))))]
    Corr_ret_names_Na_0=[x for x in Corr_ret_names if x not in Corr_ret_names_Na_01]


    #
    #ADF_AIC_result_Co_Inte = [sm.tsa.stattools.adfuller(Corr_ret_TS_Co_Inte.iloc[:, i], maxlag=None, regression='c', autolag='AIC', store=False, regresults=False) for i in range(Corr_ret_TS_Co_Inte.shape[1])]
    #Ljung_Box_result_Co_Inte = [sm.stats.diagnostic.acorr_ljungbox(Corr_ret_TS_Co_Inte.iloc[:, i]) for i in range(Corr_ret_TS_Co_Inte.shape[1])]
    
    '''
    Plot new kernel time series
    '''
    #for i in range(Corr_ret_TS_Co_Inte.shape[1]):
    #    plt.subplot(np.sqrt(Corr_ret_TS_Co_Inte.shape[1]), np.sqrt(
    #        Corr_ret_TS_Co_Inte.shape[1]), i + 1)
    #    x = Corr_ret_TS_Co_Inte.index
    #    y = Corr_ret_TS_Co_Inte.iloc[:,i]
    #
    #    # plt.hist(y, bins=100,density=True)
    #    plt.plot(x, y)
    #    plt.legend = True
    #    plt.mark_right = True
    #
    #    plt.title('%s %d smpl prjcd Corr para' % (Corr_ret_names[i],Ini_Windows_size))
    #
    #plt.show()
    
    
    #########################
    
    #'GARCH Constant Mean, student-t as distribuition cause the heavy tail'
    #GARCH_Corr_ret_Co_Inte = arch.arch_model(Corr_ret_TS_Co_Inte.iloc[:,0], p=1, o=1, q=1, power=1.0, dist='StudentsT')
    #GARCH_Corr_ret_Co_Inte_fit = GARCH_Corr_ret_Co_Inte.fit(update_freq=10)
    # GARCH_Corr_ret_Co_Inte_fit.summary()
    #fig = GARCH_Corr_ret_Co_Inte_fit.plot(annualize='D')
    # A=GARCH_Corr_ret_Co_Inte_fit.forecast()
    # print(A)
    # A.residual_variance.iloc[-3:]
    # GARCH_Corr_ret_Co_Inte_fit.conditional_volatility.plot()
    #Corr_ret_TS_Co_Inte.iloc[:,9].plot(kind='kde',title='PDF of Projected Corrariance of W00007 and HSCI.HI')
    
    
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
    #Step I: Calibrate the seasonal parameters:
    #y=Corr_ret_TS_Co_Inte.iloc[:,2]
    #warnings.filterwarnings("ignore") # specify to ignore warning messages
    #
    #for i in range(60):
    #    param=(1,1,0)
    #    param_seasonal=(0,1,0,i+1)
    #    try:
    #        mod = SARIMAX(y, order=param,
    #                                        seasonal_order=param_seasonal,
    #                                        enforce_stationarity=False,
    #                                        enforce_invertibility=False)
    #    
    #        results = mod.fit()
    #    
    #        print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
    #    except:
    #        continue
    #
    
    #Step II: Calibrate the parameters
#    ARIMA_calibra_paras_Summary=pd.DataFrame()
    
#    ARIMA_calibra_paras_Summary=ARIMA_CALI_PARAMS(Corr_ret_TS_Co_Inte)
#    A=[ARIMA_CALI_PARAMS]
    
    func=partial(ARIMA_CALI_PARAMS,Corr_ret_TS_Co_Inte,)
    pool=Pool(32)
    ARIMA_calibra_paras_Summary=pool.map(func,range(Corr_ret_TS_Co_Inte.shape[1]))
    pool.close()
    pool.join()
    
    ARIMA_calibra_paras_Summary=np.hstack(ARIMA_calibra_paras_Summary)
    ARIMA_calibra_paras_Summary=pd.DataFrame(ARIMA_calibra_paras_Summary)
    ARIMA_calibra_paras_Summary.to_csv('ARIMA_calibra_paras_Summary.csv')
#    ARIMA_calibra_paras_Summary.columns=Corr_ret_names_Na_0
#    ARIMA_calibra_paras_Summary.index
#    ARIMA_calibra_paras_Summary.to_csv('ARIMA_calibra_paras_Summary.csv')
    

  
#A=[ARIMA_CALI_PARAMS(Corr_ret_TS_Co_Inte.iloc[:,i]) for i in range(3)]
#AA=np.hstack(A)

    '''
    Finally, params calibrated as ARIMA(0, 1, 0)x(0, 1, 0, 1)
    '''
    #
    #order=(0,1,0)
    #seasonal_order=(0,1,0,1)
    #mod = SARIMAX(Corr_ret_TS_Co_Inte.iloc[:,1], trend='c', order=order, seasonal_order=seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
    #res = mod.fit(disp=False)
    #print(res.summary())
    #res.plot_diagnostics(figsize=(15, 12))
    #plt.show()
    #
    ##########################
    #'''
    #Prediction effect
    #'''
    ##pred = res.get_prediction(start=pd.to_datetime('2008-01-01'), dynamic=False)
    #pred = res.get_prediction()
    #
    #pred_ci = pred.conf_int()
    #
    #pred_ci_mean=pred_ci.mean(axis=1)
    #eg_pre_ori_merge=pd.merge(pd.DataFrame(pred_ci_mean,columns=['Pred_Data']),pd.DataFrame(Corr_ret_TS_Co_Inte[1]),right_index=True,left_index=True,how='outer')
    #eg_pre_ori_merge.columns=['Pred_Data','Ori_Data']
    #eg_pre_ori_merge.plot(kind='line',title='e.g. Corr Performance')
