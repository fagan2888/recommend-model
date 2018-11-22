# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:52:52 2018

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


from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from statsmodels.stats.diagnostic import unitroot_adf
from statsmodels.multivariate import pca
from sklearn import decomposition

####################################################################################################################


def z_score_normalization(A):
    AB = (A-np.mean(A))/np.std(A)
    return AB


def inverse_z_score_normalization(input_A, original_data):
    result = input_A*np.std(original_data)+np.mean(original_data)
    return result

####################################################################################################################


def ecdf(input_array):
    yvals = np.arange(1, len(sorted(input_array))+1) / \
        float(len(sorted(input_array)))
    return yvals

####################################################################################################################


def filter_data_by_nan(data, na_tolerance_quantile):
    tolerate_index = []
    for i in range(data.shape[1]):
        if data[data.columns[i]].count() < (1-na_tolerance_quantile)*data.shape[0]:
            tolerate_index.append(data.columns[i])
    data = data.drop(columns=tolerate_index)
    return data

####################################################################################################################


def normalization_maxmin(A):

    A = 0+(A-np.min(A))/(np.max(A)-np.min(A))*(1-0)
    AAA = A/np.sum(A)
    return AAA

####################################################################################################################


def Two_Return_Cum(input_dataframe):
    input_dataframe = pd.DataFrame(input_dataframe)

    log_return = np.log(input_dataframe/input_dataframe.shift(1)).dropna()
    log_return_sum = log_return.cumsum()

    arithmetic_return = input_dataframe.pct_change().dropna()
    arithmetic_return_cumproduct = (arithmetic_return+1).cumprod()-1

    diff = log_return_sum-arithmetic_return_cumproduct
    summary = pd.concat(
        [log_return_sum, arithmetic_return_cumproduct, diff], axis=1, join='inner')
    summary.columns = ['log_ret_cumsum',
                       'archimetric_ret_cumprod', 'difference']
    return summary


####################################################################################################################

"Function inverse copula data to original data_arbitrage_distributed simulated data"


def Tranfer_simu2orig_data(log_ret, input_data):
    ''
    if log_ret.shape[1] != input_data.shape[1]:
        raise ValueError(
            'Something Wrong about the shape of input Simulated data')

    Simu_data = []
    for i in range(log_ret.shape[1]):
        data_test = log_ret.iloc[:, i].tolist()

        'Extend log return data to (0,1]'
        'Z Score'
        data_test_norm = z_score_normalization(normalization_maxmin(data_test))
        inverse_ECDF_data_test = sp.interpolate.interp1d(ecdf(
            data_test_norm), data_test_norm, kind='linear', bounds_error=False, fill_value='extrapolate')

        input_data1 = input_data.iloc[:, i].tolist()

        Simu_data_uni_distri_inv_ecdf = inverse_ECDF_data_test(input_data1)
        Simu_data.append(inverse_z_score_normalization(
            Simu_data_uni_distri_inv_ecdf, data_test).T)

    Simu_data_All = pd.DataFrame(data=np.stack(
        Simu_data).T, columns=log_ret.columns)

    return Simu_data_All

####################################################################################################################


def portflio_optimization_by_qrisk(data, suspension_tolerance_filtered_level, Nb_MC, Confidence_level):

    data_tolerance_filtered = filter_data_by_nan(
        data, suspension_tolerance_filtered_level)
    data_tolerance_filtered = data_tolerance_filtered.fillna(method='bfill')
    log_ret = np.log(data_tolerance_filtered/data_tolerance_filtered.shift(1))

    log_ret = log_ret.iloc[1:, :]
    log_ret = log_ret.fillna(value=0)

    'Generate values from a multivariate normal distribution with specified mean vector and covariance matrix and the time is the same in histroy'

    cholesky_deco_corr_log_ret = np.linalg.cholesky(log_ret.corr())
    Gaussian_Copula_Simulation = [(np.mean(log_ret)+np.dot(cholesky_deco_corr_log_ret, [
                                   np.random.normal() for i in range(log_ret.shape[1])])).values.T for i in range(int(Nb_MC))]
    Gaussian_Copula_Simulation = pd.DataFrame(data=np.stack(
        Gaussian_Copula_Simulation), columns=log_ret.columns)
    Gaussian_Copula_Simulation_cdf = pd.DataFrame(data=sp.stats.norm.cdf(
        Gaussian_Copula_Simulation), columns=log_ret.columns)

    Simulated_data_by_Gaussian_Copula = Tranfer_simu2orig_data(
        log_ret, Gaussian_Copula_Simulation_cdf)

#    "Modified part below in 31.07.2018"
#
#    VaR_Gaussian = pd.DataFrame(data=Simulated_data_by_Gaussian_Copula,
#                                columns=Simulated_data_by_Gaussian_Copula.columns).quantile(1-Confidence_level)
#    VaR_Gaussian = pd.DataFrame(
#        data=VaR_Gaussian.values, index=Simulated_data_by_Gaussian_Copula.columns, columns=['VaR'])
#
#    CVaR_Gaussian = [Simulated_data_by_Gaussian_Copula.iloc[:, i].nsmallest(n=round(
#        Simulated_data_by_Gaussian_Copula.shape[0]*(1-Confidence_level))) for i in range(Simulated_data_by_Gaussian_Copula.shape[1])]
#    CVaR_Gaussian = pd.DataFrame(data=np.stack(
#        CVaR_Gaussian).T, columns=Simulated_data_by_Gaussian_Copula.columns)
#    CVaR_Gaussian = pd.DataFrame(data=CVaR_Gaussian.mean(
#    ), index=CVaR_Gaussian.columns, columns=['CVaR'])
#
#    VaR_CVaR_Gaussian = pd.merge(
#        VaR_Gaussian, CVaR_Gaussian, left_index=True, right_index=True, how='inner')
#
#    VaR_CVaR_Gaussian_Weight = (np.ones([VaR_CVaR_Gaussian.shape[0], VaR_CVaR_Gaussian.shape[1]]))/(
#        VaR_CVaR_Gaussian/np.array(VaR_CVaR_Gaussian.sum()).T)
#    VaR_CVaR_Gaussian_Weight = VaR_CVaR_Gaussian_Weight / \
#        VaR_CVaR_Gaussian_Weight.sum().T

    return Simulated_data_by_Gaussian_Copula


####################################################################################################################


def Algo_summary(input_dataframe):
    'The input dataframe should be the daily return of the portfolio, where need to be compared'
    summary = []
    Risk_free = 0.035
    for k in range(input_dataframe.shape[1]):
        input_dataset = input_dataframe.iloc[:, k]

        Std = np.std(input_dataset)
        Ann_Std = Std*np.sqrt(252)

        input_dataset = np.cumsum(input_dataset)

        Ret = input_dataset[-1]
        Ann_Ret = (1+Ret)**(252/len(input_dataset))-1

        SharpR = (Ann_Ret-Risk_free)/Ann_Std

        i = np.argmax(np.maximum.accumulate(input_dataset) -
                      input_dataset)  # end of the period
        j = np.argmax(input_dataset[:i])  # start of period

        summary.append([Ret, Ann_Ret, Std, Ann_Std, SharpR, j,
                        i, input_dataset[j]-input_dataset[i]])

    Algo_summary = pd.DataFrame(summary, index=input_dataframe.columns, columns=[
                                'Ret', 'Ann Ret', 'Daily_Ret_Std', 'Ann Std', 'Sharp Ratio', 'MDD_Start_Date', 'MDD_End_Date', 'MDD'])
    return(Algo_summary)


####################################################################################################################
####################################################################################################################
####################################################################################################################


Index_data=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\data\SH_SP_HS.csv")
columns_name=['Index_Code','Trd_dt','Index_Cls']
Index_data.columns=columns_name


Index_Code_unique=Index_data[~Index_data.Index_Code.duplicated()].Index_Code.tolist()

Index_1=Index_data[Index_data.Index_Code==Index_Code_unique[0]]
#Index_2=Index_data[Index_data.Index_Code=='HSCI.HI']
#Index_3=Index_data[Index_data.Index_Code=='SP500.SPI']
Index_1=pd.DataFrame(data=Index_1.Index_Cls.values,columns=[Index_Code_unique[0]],index=Index_1.Trd_dt)
#Index_2.index=Index_2.Trd_dt
#Index_3.index=Index_3.Trd_dt

Index_2=Index_data[Index_data.Index_Code==Index_Code_unique[1]]
#Index_2=Index_data[Index_data.Index_Code=='HSCI.HI']
#Index_3=Index_data[Index_data.Index_Code=='SP500.SPI']
Index_2=pd.DataFrame(data=Index_2.Index_Cls.values,columns=[Index_Code_unique[1]],index=Index_2.Trd_dt)

A=pd.merge(Index_1,Index_2,right_index=True,left_index=True,how='outer')

def Clean_data(input_data):
    
    Index_Code_unique=Index_data[~Index_data.Index_Code.duplicated()].Index_Code.tolist()
    Data=pd.DataFrame()
    for i in range(len(Index_Code_unique)):
        Index_1=Index_data[Index_data.Index_Code==Index_Code_unique[i]]
        Index_1=pd.DataFrame(data=Index_1.Index_Cls.values,columns=[Index_Code_unique[i]],index=Index_1.Trd_dt)
        Data=pd.merge(Data,Index_1,right_index=True,left_index=True,how='outer')
    
    return Data
    
AA=Clean_data(Index_data)
