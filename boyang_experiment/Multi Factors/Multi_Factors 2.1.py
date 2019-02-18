# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:33:12 2019

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


import linearmodels
from statsmodels.datasets import grunfeld
from linearmodels import IV2SLS, IVLIML, IVGMM, IVGMMCUE,PanelOLS,BetweenOLS
###########################################################
'Example Data'
stock_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\cls_px.csv",index_col=[0])

stock_data.index = stock_data.index.map(lambda x: pd.Timestamp(str(int(x))))
stock_data=stock_data.apply(lambda x: np.nan if type(x)==float else np.float64(x))
stock_data.groupby(stock_data.index.strftime('%Y-%m')).last()



stock_mkt_cap=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\mkt_cap.csv",index_col=[0])
stock_mkt_cap.index = stock_mkt_cap.index.map(lambda x: pd.Timestamp(str(int(x))))

PE_ratio=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\PE_ratio.csv",index_col=[0])
PE_ratio.index = PE_ratio.index.map(lambda x: pd.Timestamp(str(int(x))))

PS_ratio=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\PSales_ratio.csv",index_col=[0])
PS_ratio.index=PS_ratio.index.map(lambda x: pd.Timestamp(str(int(x))))
PS_ratio=PS_ratio[PS_ratio.index.isin(stock_data.index)]

USDCNY=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\usdcny.csv",index_col=[0])
USDCNY.index=USDCNY.index.map(lambda x: pd.Timestamp(str(x)))
#USDCNY=USDCNY[USDCNY.index.isin(stock_data.index)]
Fin_data=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\caihui_tq_fin_procfsqsubjects_50.csv",index_col=[0])
Fin_data.PUBLISHDATE=Fin_data.FIRSTPUBLISHDATE.map(lambda x: pd.Timestamp(str(x)))


start_date=pd.Timestamp('2004-01-01')
stock_data=stock_data[stock_data.index>=start_date]
stock_mkt_cap=stock_mkt_cap[stock_mkt_cap.index>=start_date]
PE_ratio=PE_ratio[PE_ratio.index>=start_date]

PS_ratio=PS_ratio[PS_ratio.index>=start_date]
Fin_data=Fin_data[Fin_data.PUBLISHDATE>=start_date]
#PE_ratio=PE_ratio.applymap(lambda x: 1000 if type(x)==np.str else np.float64(x))
PE_ratio=PE_ratio.replace('--',np.nan)
PS_ratio=PS_ratio.replace('--',np.nan)
'filter nan'
#PE_ratio.isnull().sum()/PE_ratio.shape[0]
#print(Counter(Fin_data.isnull().sum()/Fin_data.shape[0]).most_common()[0])
#print(Counter(stock_mkt_cap.isnull().sum()/stock_mkt_cap.shape[0]).most_common()[0])
#print(Counter(PE_ratio.isnull().sum()/PE_ratio.shape[0]).most_common()[0])

AA=Fin_data.isnull().sum()

valid_columns=[stock_data.columns[i] if stock_data.iloc[:,i].isnull().sum()/stock_data.shape[0] <0.1 else np.nan for i in range(stock_data.shape[1])]
valid_columns=[x for x in valid_columns if x==x]

stock_data=stock_data[valid_columns].ffill()
stock_mkt_cap=stock_mkt_cap[valid_columns].ffill()
PE_ratio=PE_ratio[valid_columns].ffill()
PS_ratio=PS_ratio[valid_columns].ffill()

PE_ratio=PE_ratio.apply(lambda x:np.float64(x))
PS_ratio=PS_ratio.apply(lambda x:np.float64(x))
'Arithmetic return instead of logarithm return according to FMB93'
Return=stock_data.pct_change()
Return=Return.dropna()

Test_individual_underlying=Fin_data[Fin_data.Ticker==Return.columns[1]]

#'HAC cov'
#Covariance=sm.stats.sandwich_covariance.cov_hac_simple(Return.cov())
#


'Mkt Cap weighted index'
Indexing=np.sum(Return*(stock_mkt_cap.divide(stock_mkt_cap.sum(axis=1).values,axis=0)),axis=1).fillna(value=0).add(1,axis=0).cumprod()
Ret_indexing=pd.DataFrame(Indexing,columns=['Indexing']).pct_change().dropna()
Mkt_weight=stock_mkt_cap.divide(stock_mkt_cap.sum(axis=1).values,axis=0)
stock_mkt_cap_log=np.log(stock_mkt_cap)

######################################################################################################################
######################################################################################################################

'''
Statistical hypothesis test

I: Autocorrelation Test

Methodologies:
    
        Ljung Box Q Test: 
           H0: The data are independently distributed
           H1: Serial correlateion effect
           Pitfall: Small sample size; All regressors are "strictly exogenous"    
           
        Durbin Watson Test:
           H0: No serial correlateion effect
           H1: (Only) First order autoregressive
           Pitfall: The data must show an autocorrelation process
           
        Breusch Godfrey Test:
           H0: No seial correlation effect
           H1: Any order up to p autoregessive      
           
           
Reference: 
Maddala (2001) "Introduction to Econometrics (3d edition), ch 6.7, and 13. 5 p 528.


II: Normality test

Methodologies:
    
        Jarque Bera Test: 
           H0: A joint hypothesis of the skewness being zero and the excess kurtosis being zero
           H1: Serial correlateion effect
           Pitfall: Test depends on the skewness and kurtosis of matching a normal distribution (central moments based): The sensitivity of chi-squared approximation of small sample size is high  
           
        Kolmogorov Smirnov Test:
           H0: A good fit of the empirical distribution and the theoretical cdf
           H1: Alternative H0
           Pitfall: The distributions considered under the null hypothesis are continuous distributions but are otherwise unrestricted
           
        Anderson Darling Test:
           H0: A good fit of the empirical distribution and the theoretical cdf
           H1: Alternative H0
           Pitfall: Essentially the same test statistic can be used in the test of fit of a family of distributions, but then it must be compared against the critical values appropriate to that family of theoretical distributions and dependent also on the method used for parameter estimation.
     
Reference:
Jarque, Carlos M.; Bera, Anil K. (1987). "A test for normality of observations and regression residuals". International Statistical Review. 55 (2): 163–172
Scholz, F. W.; Stephens, M. A. (1987). "K-sample Anderson–Darling Tests". Journal of the American Statistical Association. 82 (399): 918–924.        
'''
def Auto_Corr_Normality_multi_tests(residuals_regression):
    
    #40 lags for Ljungbox
    A1=sm.stats.diagnostic.acorr_ljungbox(residuals_regression)[1]
    A2=sm.stats.stattools.durbin_watson(residuals_regression)
    #QQ plop
    sp.stats.probplot(residuals_regression,dist='norm',plot=pylab)
    #Normality test
    B1=sm.stats.stattools.jarque_bera(residuals_regression,axis=0)[1]
    B2=sm.stats.diagnostic.kstest_normal(residuals_regression,dist='norm',pvalmethod='approx')[0]
    B3=sm.stats.diagnostic.normal_ad(residuals_regression, axis=0)[1]
    plt.show()
    
    summary=pd.DataFrame({
            'Ljung Box': [A1],
            'Durbin Watson': [A2],
            'Jarque Bera': [B1],
            'Komlogorov Smirnov': [B2],
            'Anderson Darling': [B3]},
            index=['P value like'])
    print(summary)
    return summary
    
##########################
'''
Test I: Different Regression Model Comparision'
        
        Portfolio: Full filtered dataset
        Factors: PE_ratio
        Factors type: Single factor
        Factors Joint Methodology: Equally Weighted
        
'''

'Regression Type I: OLS'
def OLS_compary_summary(X,Y):
    
    X=pd.DataFrame(X)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]

    OLS1=sm.regression.linear_model.OLS(X,Y).fit()
    OLS1.summary()
    OLS1_resid=OLS1.resid
    #40 lags for Ljungbox
    sm.stats.diagnostic.acorr_ljungbox(OLS1_resid)
    sm.stats.stattools.durbin_watson(OLS1_resid)
    sm.stats.diagnostic.acorr_breusch_godfrey(OLS1)[2]
    #QQ plop
    sp.stats.probplot(OLS1_resid,dist='norm',plot=pylab)
    #Normality test
    sm.stats.stattools.jarque_bera(OLS1_resid,axis=0)[1]
    sm.stats.diagnostic.kstest_normal(OLS1_resid,dist='norm',pvalmethod='approx')[0]
    sm.stats.diagnostic.normal_ad(OLS1_resid, axis=0)[1]
    
    
    'Homoskedastic estimator'
    print('OLS Homoskedastic estimator')
    OLS1=IV2SLS(X,Y,None,None).fit(cov_type='unadjusted')
    print(OLS1.summary)
    OLS1_resid=OLS1._resid
    #40 lags for Ljungbox
    sm.stats.diagnostic.acorr_ljungbox(OLS1_resid)
    sm.stats.stattools.durbin_watson(OLS1_resid)
    #QQ plop
    sp.stats.probplot(OLS1_resid,dist='norm',plot=pylab)
    #Normality test
    sm.stats.stattools.jarque_bera(OLS1_resid,axis=0)[1]
    sm.stats.diagnostic.kstest_normal(OLS1_resid,dist='norm',pvalmethod='approx')[0]
    sm.stats.diagnostic.normal_ad(OLS1_resid, axis=0)[1]
    
    'Robust to heteroskedasticity'
    print('OLS Robust to heteroskedasticity')
    OLS1=IV2SLS(X,Y,None,None).fit(cov_type='robust')
    print(OLS1.summary)
    OLS1_resid=OLS1._resid
    #40 lags for Ljungbox
    sm.stats.diagnostic.acorr_ljungbox(OLS1_resid)
    sm.stats.stattools.durbin_watson(OLS1_resid)
    #QQ plop
    sp.stats.probplot(OLS1_resid,dist='norm',plot=pylab)
    #Normality test
    sm.stats.stattools.jarque_bera(OLS1_resid,axis=0)[1]
    sm.stats.diagnostic.kstest_normal(OLS1_resid,dist='norm',pvalmethod='approx')[0]
    sm.stats.diagnostic.normal_ad(OLS1_resid, axis=0)[1]
    
    'One- or two-way clustering to account for additional sources of dependence between the model scores'
    print('OLS Cluster')
    OLS1=IV2SLS(X,Y,None,None).fit(cov_type='clustered')
    print(OLS1.summary)
    OLS1_resid=OLS1._resid
    #40 lags for Ljungbox
    sm.stats.diagnostic.acorr_ljungbox(OLS1_resid)
    sm.stats.stattools.durbin_watson(OLS1_resid)
    #QQ plop
    sp.stats.probplot(OLS1_resid,dist='norm',plot=pylab)
    #Normality test
    sm.stats.stattools.jarque_bera(OLS1_resid,axis=0)[1]
    sm.stats.diagnostic.kstest_normal(OLS1_resid,dist='norm',pvalmethod='approx')[0]
    sm.stats.diagnostic.normal_ad(OLS1_resid, axis=0)[1]
    
    'A heteroskedasticity-autocorrelation robust covariance estimator'
    print('OLS HAC')
    OLS1=IV2SLS(X,Y,None,None).fit(cov_type='kernel')
    print(OLS1.summary)
    OLS1_resid=OLS1._resid
    #40 lags for Ljungbox
    sm.stats.diagnostic.acorr_ljungbox(OLS1_resid)
    sm.stats.stattools.durbin_watson(OLS1_resid)
    #QQ plop
    sp.stats.probplot(OLS1_resid,dist='norm',plot=pylab)
    #Normality test
    sm.stats.stattools.jarque_bera(OLS1_resid,axis=0)[1]
    sm.stats.diagnostic.kstest_normal(OLS1_resid,dist='norm',pvalmethod='approx')[0]
    sm.stats.diagnostic.normal_ad(OLS1_resid, axis=0)[1]



OLS_compary_summary(Ret_indexing,PE_ratio.mean(axis=1))


'Example Factor'




GLS=sm.regression.linear_model.GLS(Indexing,stock_data)
GLS_results=GLS.fit()
print(GLS_results.summary())
GLS_results._results.resid
A=Auto_Corr_Normality_multi_tests(GLS_results._results.resid)

ivolsmod = IV2SLS(Indexing, stock_data, None, None)
res_ols = ivolsmod.fit()
print(res_ols)
res_ols._params
res_ols._resid
def stat_summary_compare_by_fit(Y,X):
    X=pd.DataFrame(X)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]
    
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit()
    print(res_ols)
    #res_ols.resids
    'unadjusted: the classic homoskedastic estimator'
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='unadjusted')
    print(res_ols)
    'robust: robust to heteroskedasticity'
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='robust')
    print(res_ols)
    'clustered: one- or two-way clustering to account for additional sources of dependence between the model scores'
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='clustered')
    print(res_ols)
    'kernel: a heteroskedasticity-autocorrelation robust covariance estimator'
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='kernel')
    print(res_ols)


#stat_summary_compare_by_fit(Ret_indexing,(Return*Mkt_weight).sum(axis=1))
#
#
#stat_summary_compare_by_fit(Ret_indexing,stock_mkt_cap_log)
#
#stat_summary_compare_by_fit(Ret_indexing,(PE_ratio*Mkt_weight).sum(axis=1))
#stat_summary_compare_by_fit(Ret_indexing,(PS_ratio*Mkt_weight).sum(axis=1))
#
#stat_summary_compare_by_fit(Ret_indexing,PS_ratio)
stat_summary_compare_by_fit(Ret_indexing,PE_ratio)
#
def stats_in_m(Y,X):
    X=pd.DataFrame(X)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]

    X=X.groupby(X.index.strftime('%Y-%m')).last()
    Y=Y.groupby(Y.index.strftime('%Y-%m')).last()
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='kernel')
    print(res_ols)
    return res_ols

stats_in_m(Ret_indexing,Return)
A=stats_in_m(Ret_indexing,USDCNY.pct_change())
A.tstats.values[0]
A._params.values[0]
A.resids



stat_summary_compare_by_fit(Ret_indexing,USDCNY.pct_change())

def stats_ts(Y,X):
    
    X=pd.DataFrame(X)
    Y=pd.DataFrame(Y)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]
    
#    ivolsmod = IV2SLS(Y, X, None, None)
#    res_ols = ivolsmod.fit(cov_type='kernel')
    tstats=[]
    _params=[]
    for i in range(Y.shape[0]-1000-1):
        
        ivolsmod = IV2SLS(Y.iloc[:1000+i,:], X.iloc[:1000+i,:], None, None)        
        res_ols = ivolsmod.fit(cov_type='kernel')
        
        tstats.append(res_ols.tstats.values[0])
        _params.append(res_ols._params.values[0])
        
        
    return tstats,_params


def GLS(Y,X):
#    new_index = X.index.intersect(Y.index)
#    X = X.reindex(new_index)
#    Y = Y.reindex(new_index)
    
    ##
#    Intersect_index=set(X.index.tolist(),Y.index.tolist())
    X=pd.DataFrame(X)
    Y=pd.DataFrame(Y)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]

    GLS=sm.regression.linear_model.GLS(Y,X)
    GLS_results=GLS.fit()
    print(GLS_results.summary())
    return GLS_results

#GLS(Ret_indexing,USDCNY.pct_change())

GLS_TEST=GLS(Ret_indexing,PS_ratio)
GLS_TEST_params=GLS_TEST._results.params
GLS_TEST_resid=GLS_TEST._results.resid
GLS_TEST_params.mean()

GLS(Ret_indexing,PS_ratio.mean(axis=1))



#df = pd.concat([USDCNY.pct_change(),Ret_indexing],1)
#df=df.dropna()
#df=df.groupby(df.index.strftime('%Y-%m')).last()
#plt.scatter(df['USDCNY'], df['Indexing'])
#

def indexed_data(Y,X):
    X=pd.DataFrame(X)
    Y=pd.DataFrame(Y)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]

    return X,Y


#linearmodels.asset_pricing.LinearFactorModelGMM(PE_ratio,PS_ratio).fit()
    
PanelOLS(Ret_indexing, pd.DataFrame(PS_ratio[PS_ratio.index.isin(Ret_indexing.index)].sum(axis=1)),entity_effects=True).fit(cov_type='clustered')

PanelOLS(Ret_indexing, pd.DataFrame(Return[Return.index.isin(Ret_indexing.index)].sum(axis=1))).fit(cov_type='unadjusted')

A=pd.merge(pd.DataFrame(Return[Return.index.isin(Ret_indexing.index)].sum(axis=1)), pd.DataFrame(PS_ratio[PS_ratio.index.isin(Ret_indexing.index)].sum(axis=1)),right_index=True,left_index=True,how='outer')

PanelOLS(Ret_indexing, A,entity_effects=True).fit(cov_type='unadjusted')



pd.concat(pd.DataFrame(PE_ratio[PE_ratio.index.isin(Ret_indexing.index)].sum(axis=1)),pd.DataFrame(PS_ratio[PS_ratio.index.isin(Ret_indexing.index)].sum(axis=1)))


np.sort(list(set(Fin_data.index)))[1]
Fin_data[Fin_data.index==np.sort(list(set(Fin_data.index)))[1]].ROEAVGPRE.isnull().sum()


