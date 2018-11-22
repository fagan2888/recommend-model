# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:07:14 2018

@author: Boyang
"""

import scipy as sp
import numpy as np
import pandas as pd

import statsmodels as sm
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter

from datetime import timedelta, datetime

from statsmodels.stats.diagnostic import unitroot_adf
from statsmodels.multivariate import pca
from sklearn import decomposition, linear_model

import arch

data = pd.read_csv(
    r"C:\Users\yshlm\Desktop\licaimofang\data\Cleaned up Data.csv")

data.index = data.iloc[:, 0]
del data[data.columns[0]]

data.index = data.index.map(lambda x: pd.Timestamp(x))

print(data.dtypes)
'I take the 1st order partial derivative here, you can adjust it to any order you want'
K_USD_1_Order = data.pct_change().dropna()

data_x = data.iloc[:, 1:]
##########################################################################
'Part I'
'PCA to find out the most primum elements'
PCA_result = decomposition.PCA(n_components=data_x.shape[1])
PCA_result.fit(data_x)
PCA_result.components_
PCA_result.explained_variance_
PCA_result.explained_variance_ratio_
print(PCA_result.explained_variance_ratio_)



'Remove FX Mkt components '
Data_non_FX = pd.DataFrame(data=data_x.values,columns=data_x.columns,index=data_x.index)
del Data_non_FX[data_x.columns[0]]
del Data_non_FX[data_x.columns[1]]
del Data_non_FX[data_x.columns[2]]
del Data_non_FX[data_x.columns[3]]


PCA_result_non_FX = decomposition.PCA(n_components=Data_non_FX.shape[1])
PCA_result_non_FX.fit(Data_non_FX)
PCA_result_non_FX.components_
PCA_result_non_FX.explained_variance_
PCA_result_non_FX.explained_variance_ratio_
PCA_result_non_FX.n_components_

A=pca.PCA(Data_non_FX, standardize=True)
A1=A.factors
A4=A.coeff


print(PCA_result_non_FX.explained_variance_ratio_)

##########################################################################
'Part II'
'Step I: Multi variable Regression'

Regression_window_size = 100

Multi_reg_result_para_collect = pd.DataFrame()

for i in range(data.shape[0] - Regression_window_size):
    i = i + Regression_window_size
    Multi_reg_result = linear_model.LinearRegression()
    Multi_reg_result.fit(data.iloc[:i, 1:], data.iloc[:i, 0])
    Multi_reg_result_para = Multi_reg_result.coef_
    Multi_reg_result_para = pd.DataFrame(data=Multi_reg_result_para)

    Multi_reg_result_para_collect = pd.merge(
        Multi_reg_result_para_collect, Multi_reg_result_para, left_index=True, right_index=True, how='outer')


Multi_reg_result_para_collect = Multi_reg_result_para_collect.T
Multi_reg_result_para_collect.columns = data.columns[1:]
Multi_reg_result_para_collect.index = data.index[Regression_window_size:]
"The result of time series regression"
print(Multi_reg_result_para_collect)


'Step II: fit the time series model such like GARCH or anyother one, you name it'
'You HAVE to make the data pass the unit-root test'
Reg_Diagno_AIC_result = sm.tsa.stattools.adfuller(
    Multi_reg_result_para_collect.iloc[:, 0], maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)


'Step III: Fit GARCH other time series model for volatility'
Data_ARCH_res_para = pd.DataFrame()
for i in range(Multi_reg_result_para_collect.shape[1]):
    Data_ARCH = arch.arch_model(
        Multi_reg_result_para_collect.iloc[:, i], p=1, q=1)
    Data_ARCH_res = Data_ARCH.fit(update_freq=10)
    Data_ARCH_res.summary()
    Data_ARCH_res_para = pd.DataFrame(Data_ARCH_res.params)
    Data_ARCH_res_para = pd.merge(
        Data_ARCH_res_para, Data_ARCH_res_para, left_index=True, right_index=True, how='outer')
    'If'
    'The message is: Inequality constraints incompatible See scipy.optimize.fmin_slsqp for code meaning.'
    ', then you need to co-integration'
#    Data_ARCH_res_para.index=Multi_reg_result_para_collect.index
    'In this case, if you need to find it out the solution, then just try one kernel function and re-do it again and again until pass the ADF test'
